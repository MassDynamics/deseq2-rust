//! `DESeq()` as production runs it (`betaPrior = FALSE`, `minReplicatesForReplace = 7`):
//! size factors, dispersions, the Wald or LRT fit, and `refitWithoutOutliers` (Cook's outlier
//! replacement by the trimmed mean and the refit of the replaced rows).
//!
//! Per-gene vectors cover every row of the count matrix given to [`deseq`]; matrices are
//! row-major genes x samples.

use crate::design::{n_or_more_in_cell, Design};
use crate::disp::{self, BaseStats, Dispersions};
use crate::ext;
use crate::nbtest::{max_cooks, nbinom_test, TestFit};

/// The dispersion columns of `mcols(dds)` over every row (NaN / `None` on all-zero rows).
#[derive(Clone, Debug)]
pub struct DispColumns {
    /// `dispGeneEst`.
    pub gene_est: Vec<f64>,
    /// `dispGeneIter`.
    pub gene_iter: Vec<f64>,
    /// `dispFit`.
    pub fit: Vec<f64>,
    /// `dispersion`.
    pub dispersion: Vec<f64>,
    /// `dispIter`.
    pub iter: Vec<f64>,
    /// `dispOutlier`.
    pub outlier: Vec<Option<bool>>,
    /// `dispMAP`.
    pub map: Vec<f64>,
}

impl DispColumns {
    fn from_stage(d: &Dispersions) -> DispColumns {
        let az = &d.base.all_zero;
        let f = |v: &[usize]| disp::expand(&v.iter().map(|i| *i as f64).collect::<Vec<_>>(), az);
        let mut it = d.map.outlier.iter();
        DispColumns {
            gene_est: disp::expand(&d.gene.disp, az),
            gene_iter: f(&d.gene.iter),
            fit: disp::expand(&d.disp_fit, az),
            dispersion: disp::expand(&d.map.dispersion, az),
            iter: f(&d.map.iter),
            outlier: az
                .iter()
                .map(|z| if *z { None } else { it.next().copied() })
                .collect(),
            map: disp::expand(&d.map.disp_map, az),
        }
    }
}

/// What `replaceOutliers` / `refitWithoutOutliers` did.
#[derive(Clone, Debug)]
pub struct Replacement {
    /// `qf(.99, p, m - p)`.
    pub cutoff: f64,
    /// `dds$replaceable`: samples in cells with at least 7 members.
    pub replaceable: Vec<bool>,
    /// `mcols(dds)$replace` (NA on rows that were all zero before replacement).
    pub replace: Vec<Option<bool>>,
    /// `assays(dds)$replaceCounts` (`n x m`), the counts the refit used.
    pub counts: Vec<f64>,
    /// `sum(replace, na.rm = TRUE)`.
    pub n_replaced: usize,
    /// Whether any row was refitted.
    pub refit: bool,
}

/// A fitted `DESeqDataSet`.
#[derive(Clone, Debug)]
pub struct DeseqFit {
    /// The design.
    pub design: Design,
    /// Original counts (`n x m`).
    pub counts: Vec<f64>,
    /// Size factors.
    pub sf: Vec<f64>,
    /// The dispersion stage as first run (before any replacement).
    pub dispersions: Dispersions,
    /// The initial Wald / LRT fit (before replacement).
    pub initial: TestFit,
    /// `baseMean`, `baseVar`, `allZero` after replacement.
    pub base: BaseStats,
    /// Dispersion columns after replacement.
    pub disp: DispColumns,
    /// The results columns after replacement; the assays `mu`, `H`, `cooks` stay the initial
    /// ones, as `refitWithoutOutliers` restores them.
    pub test: TestFit,
    /// Replacement details when some cell had 7 or more samples.
    pub replacement: Option<Replacement>,
}

impl DeseqFit {
    /// The counts `getContrast` uses: `replaceCounts` when present, else the original counts.
    pub fn fit_counts(&self) -> &[f64] {
        match &self.replacement {
            Some(r) if r.n_replaced > 0 => &r.counts,
            _ => &self.counts,
        }
    }
}

/// R's `any(row > cutoff)` with NA semantics.
fn r_any_gt(row: &[f64], cutoff: f64) -> Option<bool> {
    if row.iter().any(|v| *v > cutoff) {
        Some(true)
    } else if row.iter().any(|v| v.is_nan()) {
        None
    } else {
        Some(false)
    }
}

fn set_rows<T: Clone>(dst: &mut [T], src: &[T], width: usize, rows: &[usize]) {
    for (k, &g) in rows.iter().enumerate() {
        dst[g * width..(g + 1) * width].clone_from_slice(&src[k * width..(k + 1) * width]);
    }
}

/// `DESeq(dds)` (Wald) or `DESeq(dds, test = "LRT", reduced = <design without its first
/// term>)` on integer counts `counts` (`n x m`).
pub fn deseq(counts: &[f64], design: &Design, lrt: bool) -> Result<DeseqFit, String> {
    let m = design.n;
    let n = counts.len() / m;
    let (x, _) = design.model_matrix();
    let p = x.ncol;
    let sf = disp::size_factors(counts, m)?;
    // checkForExperimentalReplicates (estimateDispersions)
    if m == p {
        return Err("\n\n  The design matrix has the same number of samples and coefficients to fit,\n  so estimation of dispersion is not possible. Treating samples\n  as replicates was deprecated in v1.20 and no longer supported since v1.22.\n\n".into());
    }
    let d = disp::estimate_dispersions(counts, &sf, &x)?;
    let initial = nbinom_test(
        counts,
        &sf,
        &d.base.all_zero,
        &d.dispersion_all(),
        design,
        lrt,
    )?;
    let mut fit = DeseqFit {
        design: design.clone(),
        counts: counts.to_vec(),
        sf: sf.clone(),
        base: d.base.clone(),
        disp: DispColumns::from_stage(&d),
        test: initial.clone(),
        initial,
        dispersions: d,
        replacement: None,
    };
    let replaceable = n_or_more_in_cell(&x, 7);
    if !replaceable.iter().any(|b| *b) || m <= p {
        return Ok(fit);
    }

    // replaceOutliers
    let cutoff = rnum::nmath::f::qf(0.99, p as f64, (m - p) as f64, true, false);
    let cooks = &fit.initial.cooks;
    let replace: Vec<Option<bool>> = (0..n)
        .map(|g| r_any_gt(&cooks[g * m..(g + 1) * m], cutoff))
        .collect();
    let norm = disp::normalized(counts, &sf);
    let mut new_counts = counts.to_vec();
    for g in 0..n {
        let tbm = ext::trimmed_mean(&norm[g * m..(g + 1) * m], 0.2);
        for j in 0..m {
            let i = g * m + j;
            if replaceable[j] && cooks[i] > cutoff {
                new_counts[i] = (tbm * sf[j]).trunc();
            }
        }
    }
    let n_replaced = replace.iter().filter(|r| **r == Some(true)).count();
    let mut rep = Replacement {
        cutoff,
        replaceable: replaceable.clone(),
        replace,
        counts: new_counts,
        n_replaced,
        refit: false,
    };
    if n_replaced == 0 {
        fit.replacement = Some(rep);
        return Ok(fit);
    }

    // refitWithoutOutliers
    let base = disp::base_stats(&rep.counts, &sf);
    let new_all_zero: Vec<usize> = (0..n)
        .filter(|&g| rep.replace[g] == Some(true) && base.all_zero[g])
        .collect();
    if n_replaced > new_all_zero.len() {
        rep.refit = true;
        let rows: Vec<usize> = (0..n)
            .filter(|&g| rep.replace[g] == Some(true) && !base.all_zero[g])
            .collect();
        let mut keep = vec![false; n];
        for &g in &rows {
            keep[g] = true;
        }
        let csub = disp::subset_rows(&rep.counts, m, &keep);
        let bm: Vec<f64> = rows.iter().map(|&g| base.base_mean[g]).collect();
        let bv: Vec<f64> = rows.iter().map(|&g| base.base_var[g]).collect();
        let ds = &fit.dispersions;
        let ge = disp::gene_est(&csub, &sf, &bm, &bv, &x)?;
        let disp_fit = ds.function.eval(&bm);
        let map = disp::map_est(&csub, &x, &ge, &disp_fit, ds.prior_var, ds.var_log)?;
        let sub = nbinom_test(
            &csub,
            &sf,
            &vec![false; rows.len()],
            &map.dispersion,
            design,
            lrt,
        )?;

        let dc = &mut fit.disp;
        let f = |v: &[usize]| v.iter().map(|i| *i as f64).collect::<Vec<_>>();
        set_rows(&mut dc.gene_est, &ge.disp, 1, &rows);
        set_rows(&mut dc.gene_iter, &f(&ge.iter), 1, &rows);
        set_rows(&mut dc.fit, &disp_fit, 1, &rows);
        set_rows(&mut dc.dispersion, &map.dispersion, 1, &rows);
        set_rows(&mut dc.iter, &f(&map.iter), 1, &rows);
        let out: Vec<Option<bool>> = map.outlier.iter().map(|b| Some(*b)).collect();
        set_rows(&mut dc.outlier, &out, 1, &rows);
        set_rows(&mut dc.map, &map.disp_map, 1, &rows);

        let t = &mut fit.test;
        let sw = if t.stat.len() == n { 1 } else { p };
        set_rows(&mut t.beta, &sub.beta, p, &rows);
        set_rows(&mut t.se, &sub.se, p, &rows);
        set_rows(&mut t.stat, &sub.stat, sw, &rows);
        set_rows(&mut t.pvalue, &sub.pvalue, sw, &rows);
        set_rows(&mut t.conv, &sub.conv, 1, &rows);
        set_rows(&mut t.reduced_conv, &sub.reduced_conv, 1, &rows);
        set_rows(&mut t.iter, &sub.iter, 1, &rows);
        set_rows(&mut t.deviance, &sub.deviance, 1, &rows);
        set_rows(&mut t.max_cooks, &sub.max_cooks, 1, &rows);
        for &g in &new_all_zero {
            for k in 0..p {
                t.beta[g * p + k] = f64::NAN;
                t.se[g * p + k] = f64::NAN;
            }
            for k in 0..sw {
                t.stat[g * sw + k] = f64::NAN;
                t.pvalue[g * sw + k] = f64::NAN;
            }
            t.conv[g] = None;
            t.reduced_conv[g] = None;
            t.iter[g] = f64::NAN;
            t.deviance[g] = f64::NAN;
            t.max_cooks[g] = f64::NAN;
        }
        if replaceable.iter().all(|b| *b) {
            t.max_cooks = vec![f64::NAN; n];
        } else {
            let mut rc = fit.initial.cooks.clone();
            for g in 0..n {
                for j in 0..m {
                    if replaceable[j] {
                        rc[g * m + j] = 0.0;
                    }
                }
            }
            t.max_cooks = max_cooks(&rc, &x);
        }
    }
    fit.base = base;
    fit.test.all_zero = fit.base.all_zero.clone();
    fit.replacement = Some(rep);
    Ok(fit)
}
