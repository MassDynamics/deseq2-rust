//! `nbinomWaldTest` and `nbinomLRT` (DESeq2 `R/core.R`, `betaPrior = FALSE`) plus Cook's
//! distances (`calculateCooksDistance`, `robustMethodOfMomentsDisp`, `recordMaxCooks`).
//!
//! Every per-gene vector of [`TestFit`] covers all rows (all-zero rows are NaN / `None`, as
//! `buildDataFrameWithNARows` leaves them), and every matrix is row-major genes x samples.

use crate::design::{moment_cells, n_or_more_in_cell, Design};
use crate::disp::normalized;
use crate::ext;
use crate::glm::{fit_nbinom_glms, GlmFit, GlmOptions};
use crate::la::Mat;

/// Which test was run.
#[derive(Clone, Debug, PartialEq)]
pub enum TestKind {
    /// Wald test of every coefficient.
    Wald,
    /// Likelihood ratio test of the full design against the design without its first term.
    Lrt {
        /// Degrees of freedom (`ncol(full) - ncol(reduced)`).
        df: usize,
    },
}

/// The results columns `nbinomWaldTest` / `nbinomLRT` add to `mcols(dds)`, plus the assays
/// `mu`, `H` and `cooks`.
#[derive(Clone, Debug)]
pub struct TestFit {
    /// The test.
    pub kind: TestKind,
    /// `resultsNames(dds)`.
    pub coef_names: Vec<String>,
    /// The full model matrix (also the `dispModelMatrix`).
    pub x: Mat,
    /// Rows excluded from the fit (`allZero`).
    pub all_zero: Vec<bool>,
    /// Coefficients on the log2 scale (`n x p`).
    pub beta: Vec<f64>,
    /// Standard errors (`n x p`).
    pub se: Vec<f64>,
    /// Wald: `WaldStatistic_*` (`n x p`); LRT: `LRTStatistic` (`n`).
    pub stat: Vec<f64>,
    /// Wald: `WaldPvalue_*` (`n x p`); LRT: `LRTPvalue` (`n`).
    pub pvalue: Vec<f64>,
    /// `betaConv` (Wald) or `fullBetaConv` (LRT).
    pub conv: Vec<Option<bool>>,
    /// `reducedBetaConv` (LRT only).
    pub reduced_conv: Vec<Option<bool>>,
    /// `betaIter` (NaN on all-zero rows).
    pub iter: Vec<f64>,
    /// `deviance` (`-2 logLike` of the full model).
    pub deviance: Vec<f64>,
    /// `maxCooks`.
    pub max_cooks: Vec<f64>,
    /// Assay `mu` (`n x m`).
    pub mu: Vec<f64>,
    /// Assay `H` (`n x m`).
    pub hat: Vec<f64>,
    /// Assay `cooks` (`n x m`).
    pub cooks: Vec<f64>,
}

impl TestFit {
    /// Number of coefficients.
    pub fn p(&self) -> usize {
        self.coef_names.len()
    }

    /// Index of a coefficient by name.
    pub fn coef_index(&self, name: &str) -> Option<usize> {
        self.coef_names.iter().position(|c| c == name)
    }

    /// Column `k` of an `n x p` matrix field.
    pub fn column(v: &[f64], p: usize, k: usize) -> Vec<f64> {
        v.iter().skip(k).step_by(p).copied().collect()
    }
}

/// R's `pmax(a, b)` for scalars (NA propagates).
fn r_pmax(a: f64, b: f64) -> f64 {
    if a.is_nan() || b.is_nan() {
        f64::NAN
    } else if a >= b {
        a
    } else {
        b
    }
}

/// `max()` over a slice (NA propagates; `-Inf` when empty).
fn r_max(v: &[f64]) -> f64 {
    let mut m = f64::NEG_INFINITY;
    for &x in v {
        if x.is_nan() {
            return f64::NAN;
        }
        if x > m {
            m = x;
        }
    }
    m
}

/// `cut(n, breaks = c(0, 3.5, 23.5, Inf))` as a 0-based bin.
fn trim_bin(n: usize) -> usize {
    let n = n as f64;
    if n <= 3.5 {
        0
    } else if n <= 23.5 {
        1
    } else {
        2
    }
}

/// `robustMethodOfMomentsDisp` for one row of normalized counts, given the cell of each sample
/// (`None` for samples outside cells with three or more members) or `None` for the
/// `trimmedVariance` path.
fn robust_disp_row(row: &[f64], cells: Option<&(Vec<Option<usize>>, usize)>) -> f64 {
    let v = match cells {
        Some((cell, ncell)) => {
            const TRIM: [f64; 3] = [1.0 / 3.0, 1.0 / 4.0, 1.0 / 8.0];
            const SCALE: [f64; 3] = [2.04, 1.86, 1.51];
            let mut best = f64::NEG_INFINITY;
            let mut any_nan = false;
            for c in 0..*ncell {
                let vals: Vec<f64> = (0..row.len())
                    .filter(|&j| cell[j] == Some(c))
                    .map(|j| row[j])
                    .collect();
                if vals.is_empty() {
                    continue;
                }
                let b = trim_bin(vals.len());
                let cm = ext::trimmed_mean(&vals, TRIM[b]);
                let sq: Vec<f64> = vals.iter().map(|v| (v - cm) * (v - cm)).collect();
                let ve = SCALE[b] * ext::trimmed_mean(&sq, TRIM[b]);
                if ve.is_nan() {
                    any_nan = true;
                } else if ve > best {
                    best = ve;
                }
            }
            if any_nan {
                f64::NAN
            } else {
                best
            }
        }
        None => {
            let rm = ext::trimmed_mean(row, 1.0 / 8.0);
            let sq: Vec<f64> = row.iter().map(|v| (v - rm) * (v - rm)).collect();
            1.51 * ext::trimmed_mean(&sq, 1.0 / 8.0)
        }
    };
    let m = ext::row_mean(row);
    r_pmax((v - m) / (m * m), 0.04)
}

/// The cells `robustMethodOfMomentsDisp` uses: `Some((cell per sample, ncells))` when some cell
/// has three or more samples (cells with fewer are dropped), `None` otherwise.
fn robust_cells(x: &Mat) -> Option<(Vec<Option<usize>>, usize)> {
    if !n_or_more_in_cell(x, 3).iter().any(|b| *b) {
        return None;
    }
    let cells = moment_cells(x);
    let ncell = cells.iter().max().map_or(0, |c| c + 1);
    let mut size = vec![0usize; ncell];
    for &c in &cells {
        size[c] += 1;
    }
    Some((
        cells
            .iter()
            .map(|&c| if size[c] >= 3 { Some(c) } else { None })
            .collect(),
        ncell,
    ))
}

/// `calculateCooksDistance(object, H, modelMatrix)` for the rows of `counts` (`n x m`), with
/// fitted means `mu` and hat diagonals `hat`.
pub fn cooks_distance(counts: &[f64], sf: &[f64], mu: &[f64], hat: &[f64], x: &Mat) -> Vec<f64> {
    let m = sf.len();
    let p = x.ncol as f64;
    let n = counts.len() / m;
    let norm = normalized(counts, sf);
    let cells = robust_cells(x);
    let mut out = vec![0.0; n * m];
    for g in 0..n {
        let d = robust_disp_row(&norm[g * m..(g + 1) * m], cells.as_ref());
        for j in 0..m {
            let i = g * m + j;
            let mu_ = mu[i];
            let v = mu_ + d * (mu_ * mu_);
            let r = counts[i] - mu_;
            let prs = r * r / v;
            let h = hat[i];
            let omh = 1.0 - h;
            out[i] = prs / p * h / (omh * omh);
        }
    }
    out
}

/// `recordMaxCooks(design, colData, modelMatrix, cooks, numRow)`.
pub fn max_cooks(cooks: &[f64], x: &Mat) -> Vec<f64> {
    let m = x.nrow;
    let n = cooks.len() / m;
    let s = n_or_more_in_cell(x, 3);
    if !(m > x.ncol && s.iter().any(|b| *b)) {
        return vec![f64::NAN; n];
    }
    (0..n)
        .map(|g| {
            let v: Vec<f64> = (0..m).filter(|&j| s[j]).map(|j| cooks[g * m + j]).collect();
            r_max(&v)
        })
        .collect()
}

fn expand_rows(v: &[f64], width: usize, all_zero: &[bool]) -> Vec<f64> {
    let mut out = Vec::with_capacity(all_zero.len() * width);
    let mut k = 0;
    for &z in all_zero {
        if z {
            out.extend(std::iter::repeat_n(f64::NAN, width));
        } else {
            out.extend_from_slice(&v[k * width..(k + 1) * width]);
            k += 1;
        }
    }
    out
}

fn expand_opt(v: &[bool], all_zero: &[bool]) -> Vec<Option<bool>> {
    let mut it = v.iter();
    all_zero
        .iter()
        .map(|z| if *z { None } else { it.next().copied() })
        .collect()
}

/// `2 * pnorm(abs(z), lower.tail = FALSE)`.
pub fn wald_pvalue(z: f64) -> f64 {
    2.0 * rnum::nmath::pnorm::pnorm(z.abs(), 0.0, 1.0, false, false)
}

/// `nbinomWaldTest(dds)` (`betaPrior = FALSE`) or, with `lrt`, `nbinomLRT(dds, full = design,
/// reduced = design without its first term)`. `counts` (`n x m`) are the counts the fit uses,
/// `all_zero` excludes rows and `dispersion` holds `dispersions(dds)` for every row (only the
/// non-all-zero entries are read).
pub fn nbinom_test(
    counts: &[f64],
    sf: &[f64],
    all_zero: &[bool],
    dispersion: &[f64],
    design: &Design,
    lrt: bool,
) -> Result<TestFit, String> {
    let m = sf.len();
    let (x, coef_names) = design.deseq_matrix();
    let p = x.ncol;
    let nz: Vec<bool> = all_zero.iter().map(|z| !z).collect();
    let cnz = crate::disp::subset_rows(counts, m, &nz);
    let alpha: Vec<f64> = (0..all_zero.len())
        .filter(|&g| nz[g])
        .map(|g| dispersion[g])
        .collect();
    let full = fit_nbinom_glms(&cnz, sf, &x, &alpha, &GlmOptions::standard(p))?;
    let cooks = cooks_distance(&cnz, sf, &full.mu, &full.hat, &x);
    let mc = max_cooks(&cooks, &x);
    let nnz = alpha.len();
    let deviance: Vec<f64> = full.log_like.iter().map(|l| -2.0 * l).collect();
    let (kind, stat, pvalue, reduced_conv) = if lrt {
        let (xr, _) = design.drop_first().deseq_matrix();
        let df = p - xr.ncol;
        if df < 1 {
            return Err("less than one degree of freedom, perhaps full and reduced models are not in the correct order".into());
        }
        let red: GlmFit = fit_nbinom_glms(&cnz, sf, &xr, &alpha, &GlmOptions::standard(xr.ncol))?;
        let st: Vec<f64> = (0..nnz)
            .map(|g| 2.0 * (full.log_like[g] - red.log_like[g]))
            .collect();
        let pv: Vec<f64> = st
            .iter()
            .map(|s| rnum::nmath::pgamma::pchisq(*s, df as f64, false, false))
            .collect();
        (
            TestKind::Lrt { df },
            expand_rows(&st, 1, all_zero),
            expand_rows(&pv, 1, all_zero),
            expand_opt(&red.conv, all_zero),
        )
    } else {
        let st: Vec<f64> = (0..nnz * p).map(|i| full.beta[i] / full.se[i]).collect();
        let pv: Vec<f64> = st.iter().map(|z| wald_pvalue(*z)).collect();
        (
            TestKind::Wald,
            expand_rows(&st, p, all_zero),
            expand_rows(&pv, p, all_zero),
            vec![None; all_zero.len()],
        )
    };
    let iter: Vec<f64> = full.iter.iter().map(|i| *i as f64).collect();
    Ok(TestFit {
        kind,
        coef_names,
        all_zero: all_zero.to_vec(),
        beta: expand_rows(&full.beta, p, all_zero),
        se: expand_rows(&full.se, p, all_zero),
        stat,
        pvalue,
        conv: expand_opt(&full.conv, all_zero),
        reduced_conv,
        iter: expand_rows(&iter, 1, all_zero),
        deviance: expand_rows(&deviance, 1, all_zero),
        max_cooks: expand_rows(&mc, 1, all_zero),
        mu: expand_rows(&full.mu, m, all_zero),
        hat: expand_rows(&full.hat, m, all_zero),
        cooks: expand_rows(&cooks, m, all_zero),
        x,
    })
}
