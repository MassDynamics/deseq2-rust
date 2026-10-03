//! `results()` (DESeq2 `R/results.R`) for the calls production makes: a coefficient by name,
//! a factor contrast (coefficient, negated coefficient, or a difference of coefficients via
//! `getContrast`), the LRT statistic, the Cook's filter with the two-group rescue, the
//! all-zero-after-replacement rule, and `pvalueAdjustment` (independent filtering on
//! `baseMean` with the rejection-curve rule, or plain BH).

use crate::cpp::{fit_beta, FitBetaControl};
use crate::design::{make_name, Design, Var};
use crate::ext;
use crate::glm::{ln2, log2e};
use crate::la::Mat;
use crate::nbtest::{wald_pvalue, TestFit, TestKind};

/// What `results()` reads from a fitted `DESeqDataSet`.
#[derive(Clone, Copy, Debug)]
pub struct ResultsData<'a> {
    /// `design(dds)` (factor levels decide the contrast path and the rescue).
    pub design: &'a Design,
    /// The results columns of `mcols(dds)` and the assays `cooks`.
    pub test: &'a TestFit,
    /// `mcols(dds)$baseMean`.
    pub base_mean: &'a [f64],
    /// `mcols(dds)$allZero`.
    pub all_zero: &'a [bool],
    /// `mcols(dds)$replace`, when replacement ran.
    pub replace: Option<&'a [Option<bool>]>,
    /// `counts(dds)` (`n x m`).
    pub counts: &'a [f64],
    /// The counts `getContrast` fits: `replaceCounts` when present, else `counts(dds)`.
    pub fit_counts: &'a [f64],
    /// Size factors.
    pub sf: &'a [f64],
    /// `dispersions(dds)` for every row.
    pub dispersion: &'a [f64],
}

/// Which result to extract.
#[derive(Clone, Debug)]
pub enum Which {
    /// `results(dds)`: the last coefficient (for an LRT fit: the LRT statistic).
    Last,
    /// `results(dds, name = ...)`.
    Name(String),
    /// `results(dds, contrast = c(factor, numerator, denominator))`.
    Contrast {
        /// Factor name.
        factor: String,
        /// Numerator level.
        num: String,
        /// Denominator level.
        den: String,
    },
}

/// The independent-filtering detail `pvalueAdjustment` stores.
#[derive(Clone, Debug)]
pub struct Filtering {
    /// The 50 quantile levels.
    pub theta: Vec<f64>,
    /// `quantile(baseMean, theta)`.
    pub cutoffs: Vec<f64>,
    /// Rejections at `alpha` per cutoff.
    pub num_rej: Vec<f64>,
    /// `lowess(numRej ~ theta, f = 1/5)$y`.
    pub lowess_y: Vec<f64>,
    /// Chosen index (1-based, as R reports it).
    pub j: usize,
    /// `filterThreshold`.
    pub threshold: f64,
}

/// A `results()` table plus the intermediate flags.
#[derive(Clone, Debug)]
pub struct ResultsTable {
    /// `baseMean`.
    pub base_mean: Vec<f64>,
    /// `log2FoldChange`.
    pub lfc: Vec<f64>,
    /// `lfcSE`.
    pub se: Vec<f64>,
    /// `stat`.
    pub stat: Vec<f64>,
    /// `pvalue`.
    pub pvalue: Vec<f64>,
    /// `padj`.
    pub padj: Vec<f64>,
    /// `qf(.99, p, m - p)`.
    pub cooks_cutoff: f64,
    /// `maxCooks > cutoff` after the rescue (NA where `maxCooks` is NA).
    pub cooks_outlier: Vec<Option<bool>>,
    /// Rows the two-group rescue cleared.
    pub cooks_rescued: Vec<bool>,
    /// Rows all zero after replacement.
    pub now_zero: Vec<bool>,
    /// The independent-filtering detail, when it ran.
    pub filtering: Option<Filtering>,
    /// `"coef"`, `"negated_coef"` or `"contrast"`.
    pub path: &'static str,
}

/// `p.adjust(p, "BH")` (NA entries stay NA and do not count in `n`).
pub fn p_adjust_bh(p: &[f64]) -> Vec<f64> {
    rnum::linalg::p_adjust_bh(p)
}

/// The rejection-curve rule of `pvalueAdjustment` with `filter = baseMean`.
pub fn independent_filtering(filter: &[f64], p: &[f64], alpha: f64) -> (Vec<f64>, Filtering) {
    let n = filter.len();
    let lower = ext::mean_count(
        filter.iter().filter(|f| **f == 0.0).count() as u64,
        n as u64,
    );
    let upper = if lower < 0.95 { 0.95 } else { 1.0 };
    let theta = crate::prior_var::seq_len_out(lower, upper, 50);
    let cutoffs = rnum::linalg::quantile7(filter, &theta);
    let mut padj_cols: Vec<Vec<f64>> = Vec::with_capacity(theta.len());
    let mut num_rej = Vec::with_capacity(theta.len());
    for &c in &cutoffs {
        let use_: Vec<usize> = (0..n).filter(|&i| filter[i] >= c).collect();
        let mut col = vec![f64::NAN; n];
        if !use_.is_empty() {
            let sub: Vec<f64> = use_.iter().map(|&i| p[i]).collect();
            for (k, v) in p_adjust_bh(&sub).into_iter().enumerate() {
                col[use_[k]] = v;
            }
        }
        num_rej.push(col.iter().filter(|v| **v < alpha).count() as f64);
        padj_cols.push(col);
    }
    let (_, lo_y) = rnum::lowess::lowess(&theta, &num_rej, 1.0 / 5.0, 3, None);
    let max_rej = num_rej.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let j = if max_rej <= 10.0 {
        0
    } else {
        let residual: Vec<f64> = if num_rej.iter().all(|v| *v == 0.0) {
            vec![0.0]
        } else {
            (0..num_rej.len())
                .filter(|&i| num_rej[i] > 0.0)
                .map(|i| num_rej[i] - lo_y[i])
                .collect()
        };
        let max_fit = lo_y.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let sq: Vec<f64> = residual.iter().map(|r| r * r).collect();
        let thresh = max_fit - ext::mean(&sq).sqrt();
        let first = |t: f64| num_rej.iter().position(|v| *v > t);
        first(thresh)
            .or_else(|| first(0.9 * max_fit))
            .or_else(|| first(0.8 * max_fit))
            .unwrap_or(0)
    };
    let padj = padj_cols[j].clone();
    let f = Filtering {
        threshold: cutoffs[j],
        theta,
        cutoffs,
        num_rej,
        lowess_y: lo_y,
        j: j + 1,
    };
    (padj, f)
}

/// `getContrast(dds, contrast)`: `fitBeta` at `maxit = 0` from the stored betas, giving
/// `(log2FoldChange, lfcSE, stat, pvalue)` per row (NaN on all-zero rows).
pub fn get_contrast(d: &ResultsData, contrast: &[f64]) -> Result<[Vec<f64>; 4], String> {
    let t = d.test;
    let x: &Mat = &t.x;
    let m = x.nrow;
    let p = x.ncol;
    let n = d.all_zero.len();
    let l2 = ln2();
    let l2e = log2e();
    let lambda = 1.0 / (l2 * l2 * 1e6);
    let ctl = FitBetaControl {
        lambda: vec![lambda; p],
        contrast: contrast.to_vec(),
        tol: 1e-8,
        maxit: 0,
        minmu: 0.5,
    };
    let mut out = [
        vec![f64::NAN; n],
        vec![f64::NAN; n],
        vec![f64::NAN; n],
        vec![f64::NAN; n],
    ];
    for g in 0..n {
        if d.all_zero[g] {
            continue;
        }
        let beta: Vec<f64> = (0..p).map(|k| l2 * t.beta[g * p + k]).collect();
        let r = fit_beta(
            &d.fit_counts[g * m..(g + 1) * m],
            d.sf,
            x,
            d.dispersion[g],
            &beta,
            &ctl,
        )?;
        let est = l2e * r.contrast_num;
        let se = l2e * r.contrast_denom;
        let st = est / se;
        out[0][g] = est;
        out[1][g] = se;
        out[2][g] = st;
        out[3][g] = wald_pvalue(st);
    }
    Ok(out)
}

/// R's `which.max` (first maximum, NA skipped).
fn which_max(v: &[f64]) -> Option<usize> {
    let mut best: Option<usize> = None;
    for (i, &x) in v.iter().enumerate() {
        if x.is_nan() {
            continue;
        }
        if best.is_none_or(|b| x > v[b]) {
            best = Some(i);
        }
    }
    best
}

/// `results()`: `stopifnot(alpha > 0 & alpha < 1)` (results.R, before the contrast checks).
pub fn check_alpha(alpha: f64) -> Result<(), String> {
    if alpha > 0.0 && alpha < 1.0 {
        Ok(())
    } else {
        Err("alpha > 0 & alpha < 1 is not TRUE".into())
    }
}

/// `checkContrast`: the numerator and denominator levels must differ.
pub fn check_levels_differ(num: &str, den: &str) -> Result<(), String> {
    if num == den {
        Err(format!("{num} and {den} should be different level names"))
    } else {
        Ok(())
    }
}

/// `results(dds, ..., independentFiltering, alpha)`.
pub fn results(
    d: &ResultsData,
    which: &Which,
    filter: bool,
    alpha: f64,
) -> Result<ResultsTable, String> {
    check_alpha(alpha)?;
    let t = d.test;
    let p = t.p();
    let n = d.all_zero.len();
    let m = t.x.nrow;
    let lrt = matches!(t.kind, TestKind::Lrt { .. });
    let col = |v: &[f64], k: usize| TestFit::column(v, p, k);
    let coef = |name: &str, sign: f64| -> Result<[Vec<f64>; 4], String> {
        let k = t
            .coef_index(name)
            .ok_or_else(|| format!("'{name}' is not a coefficient: {}", t.coef_names.join(", ")))?;
        let lfc: Vec<f64> = col(&t.beta, k).iter().map(|v| sign * v).collect();
        let se = col(&t.se, k);
        let (stat, pv) = if lrt {
            (t.stat.clone(), t.pvalue.clone())
        } else {
            (
                col(&t.stat, k).iter().map(|v| sign * v).collect(),
                col(&t.pvalue, k),
            )
        };
        Ok([lfc, se, stat, pv])
    };
    let mut path = "coef";
    let [mut lfc, mut se, mut stat, mut pv] = match which {
        Which::Last => coef(&t.coef_names[p - 1].clone(), 1.0)?,
        Which::Name(nm) => coef(nm, 1.0)?,
        Which::Contrast { factor, num, den } => {
            check_levels_differ(num, den)?;
            let f = d
                .design
                .vars
                .iter()
                .find_map(|v| match v {
                    Var::Factor(f) if f.name == *factor => Some(f),
                    _ => None,
                })
                .ok_or_else(|| format!("'{factor}' is not a factor of the design"))?;
            let base = &f.levels[0];
            let has = |name: &String| t.coef_names.contains(name);
            let mut r = if den == base {
                let name = make_name(&format!("{factor}_{num}_vs_{den}"));
                if !has(&name) {
                    return Err(format!("as {den} is the reference level, was expecting {name} to be present in 'resultsNames(object)'"));
                }
                coef(&name, 1.0)?
            } else if num == base {
                path = "negated_coef";
                let name = make_name(&format!("{factor}_{den}_vs_{num}"));
                if !has(&name) {
                    return Err(format!("as {num} is the reference level, was expecting {name} to be present in 'resultsNames(object)'"));
                }
                coef(&name, -1.0)?
            } else {
                path = "contrast";
                let cn = make_name(&format!("{factor}_{num}_vs_{base}"));
                let cd = make_name(&format!("{factor}_{den}_vs_{base}"));
                if !(has(&cn) && has(&cd)) {
                    return Err(format!("{num} and {den} should be levels of {factor} such that {cn} and {cd} are contained in 'resultsNames(object)'"));
                }
                let cv: Vec<f64> = t
                    .coef_names
                    .iter()
                    .map(|c| {
                        if *c == cn {
                            1.0
                        } else if *c == cd {
                            -1.0
                        } else {
                            0.0
                        }
                    })
                    .collect();
                let mut r = get_contrast(d, &cv)?;
                if lrt {
                    r[2] = t.stat.clone();
                    r[3] = t.pvalue.clone();
                }
                r
            };
            let grp: Vec<bool> = f.values().iter().map(|v| *v == num || *v == den).collect();
            let ngrp = grp.iter().filter(|b| **b).count();
            for g in 0..n {
                let zeros = (0..m)
                    .filter(|&j| grp[j] && d.counts[g * m + j] == 0.0)
                    .count();
                if zeros == ngrp && !d.all_zero[g] {
                    r[0][g] = 0.0;
                    r[2][g] = 0.0;
                    r[3][g] = 1.0;
                }
            }
            if lrt {
                r[2] = t.stat.clone();
                r[3] = t.pvalue.clone();
            }
            r
        }
    };

    // Cook's filter and the rescue.
    let dmm = &t.x;
    let cutoff = rnum::nmath::f::qf(
        0.99,
        dmm.ncol as f64,
        (dmm.nrow - dmm.ncol) as f64,
        true,
        false,
    );
    let mut outlier: Vec<Option<bool>> = t
        .max_cooks
        .iter()
        .map(|c| if c.is_nan() { None } else { Some(*c > cutoff) })
        .collect();
    let mut rescued = vec![false; n];
    if outlier.contains(&Some(true)) && d.design.single_two_level_factor() {
        for g in 0..n {
            if outlier[g] != Some(true) {
                continue;
            }
            let row = &t.cooks[g * m..(g + 1) * m];
            if let Some(jmax) = which_max(row) {
                let out_count = d.counts[g * m + jmax];
                let above = (0..m).filter(|&j| d.counts[g * m + j] > out_count).count();
                if above >= 3 {
                    rescued[g] = true;
                    outlier[g] = Some(false);
                }
            }
        }
    }
    for g in 0..n {
        if outlier[g] == Some(true) {
            pv[g] = f64::NAN;
        }
    }

    // Rows whose replaced counts are all zero.
    let mut now_zero = vec![false; n];
    if let Some(rep) = d.replace {
        if rep.contains(&Some(true)) {
            for g in 0..n {
                if rep[g] == Some(true) && d.base_mean[g] == 0.0 {
                    now_zero[g] = true;
                    lfc[g] = 0.0;
                    se[g] = 0.0;
                    stat[g] = 0.0;
                    pv[g] = 1.0;
                }
            }
        }
    }

    let (padj, filtering) = if filter {
        let (pa, f) = independent_filtering(d.base_mean, &pv, alpha);
        (pa, Some(f))
    } else {
        (p_adjust_bh(&pv), None)
    };
    Ok(ResultsTable {
        base_mean: d.base_mean.to_vec(),
        lfc,
        se,
        stat,
        pvalue: pv,
        padj,
        cooks_cutoff: cutoff,
        cooks_outlier: outlier,
        cooks_rescued: rescued,
        now_zero,
        filtering,
        path,
    })
}
