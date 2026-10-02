//! Size factors and dispersion estimation (DESeq2 `R/core.R`): `estimateSizeFactorsForMatrix`,
//! `getBaseMeansAndVariances`, `estimateDispersionsGeneEst`, `estimateDispersionsFit`
//! (parametric `a + b/mean` via `glm(family = Gamma("identity"))`, with the local `locfit`
//! fallback), the dispersion-function setter (`dispFit`, `varLogDispEsts`),
//! `estimateDispersionsPriorVar` and `estimateDispersionsMAP`.
//!
//! Count matrices are row-major genes x samples. Functions that work on the non-all-zero rows
//! (`objectNZ` in R) say so; callers subset with [`BaseStats::all_zero`].

use crate::cpp::{fit_disp, fit_disp_grid, DispPrior, FitDispControl};
use crate::design::n_groups;
use crate::ext;
use crate::glm::{fit_nbinom_glms, linear_model_mu, GlmOptions};
use crate::gnm::trigamma;
use crate::la::Mat;
use crate::linpack::qr_decompose;
use rnum::glibm::{exp, ln};

/// DESeq2's `minDisp`.
pub const MIN_DISP: f64 = 1e-8;

/// `estimateSizeFactorsForMatrix(counts)` (median-of-ratios, `type = "ratio"`).
pub fn size_factors(counts: &[f64], m: usize) -> Result<Vec<f64>, String> {
    let n = counts.len() / m;
    let logc: Vec<f64> = counts.iter().map(|&c| ln(c)).collect();
    let lgm: Vec<f64> = (0..n).map(|g| ext::row_mean(&logc[g * m..(g + 1) * m])).collect();
    if lgm.iter().all(|v| v.is_infinite()) {
        return Err(
            "every gene contains at least one zero, cannot compute log geometric means".into(),
        );
    }
    Ok((0..m)
        .map(|j| {
            let r: Vec<f64> = (0..n)
                .filter(|&g| lgm[g].is_finite() && counts[g * m + j] > 0.0)
                .map(|g| logc[g * m + j] - lgm[g])
                .collect();
            exp(ext::median(&r))
        })
        .collect())
}

/// `counts(dds, normalized = TRUE)`: `t(t(counts) / sf)`.
pub fn normalized(counts: &[f64], sf: &[f64]) -> Vec<f64> {
    let m = sf.len();
    counts.iter().enumerate().map(|(i, c)| c / sf[i % m]).collect()
}

/// matrixStats `rowVars` for one row: double mean, one refinement pass, then the sum of
/// squared deviations over `n - 1`.
pub fn row_var(x: &[f64]) -> f64 {
    let n = x.len();
    if n <= 1 {
        return f64::NAN;
    }
    let mut s = 0.0;
    for &v in x {
        s += v;
    }
    let mut mu = s / n as f64;
    if mu.is_finite() {
        let mut r = 0.0;
        for &v in x {
            r += v - mu;
        }
        mu += r / n as f64;
    }
    let mut ss = 0.0;
    for &v in x {
        let d = v - mu;
        ss += d * d;
    }
    ss / (n - 1) as f64
}

/// `baseMean`, `baseVar`, `allZero` per gene (`getBaseMeansAndVariances`).
#[derive(Clone, Debug)]
pub struct BaseStats {
    /// Mean of normalized counts.
    pub base_mean: Vec<f64>,
    /// Variance of normalized counts.
    pub base_var: Vec<f64>,
    /// `rowSums(counts) == 0`.
    pub all_zero: Vec<bool>,
}

/// [`BaseStats`] for a count matrix and size factors.
pub fn base_stats(counts: &[f64], sf: &[f64]) -> BaseStats {
    let m = sf.len();
    let n = counts.len() / m;
    let norm = normalized(counts, sf);
    BaseStats {
        base_mean: (0..n).map(|g| ext::row_mean(&norm[g * m..(g + 1) * m])).collect(),
        base_var: (0..n).map(|g| row_var(&norm[g * m..(g + 1) * m])).collect(),
        all_zero: (0..n).map(|g| ext::row_sum(&counts[g * m..(g + 1) * m]) == 0.0).collect(),
    }
}

/// Keep the rows of a row-major matrix where `keep` is true.
pub fn subset_rows(x: &[f64], m: usize, keep: &[bool]) -> Vec<f64> {
    let mut out = Vec::new();
    for (g, &k) in keep.iter().enumerate() {
        if k {
            out.extend_from_slice(&x[g * m..(g + 1) * m]);
        }
    }
    out
}

fn r_rank(x: &Mat) -> usize {
    qr_decompose(&x.data, x.nrow, x.ncol, 1e-7).rank
}

/// `checkFullRank` (the error text DESeq2 gives for a rank-deficient design).
pub fn check_full_rank(x: &Mat) -> Result<(), String> {
    if r_rank(x) < x.ncol {
        return Err("the model matrix is not full rank, so the model cannot be fit as specified.\n  One or more variables or interaction terms in the design formula are linear\n  combinations of the others and must be removed.".into());
    }
    Ok(())
}

/// Gene-wise dispersion estimates over the non-all-zero rows (`estimateDispersionsGeneEst`).
#[derive(Clone, Debug)]
pub struct GeneEst {
    /// Bounded rough/moments starting values (`alpha_init`).
    pub alpha_init: Vec<f64>,
    /// `dispGeneEst`.
    pub disp: Vec<f64>,
    /// `dispGeneIter`.
    pub iter: Vec<usize>,
    /// Fitted means used for the dispersion fit (`assays(dds)$mu`, clamped at 0.5), `nz x m`.
    pub mu: Vec<f64>,
    /// `max(10, m)`.
    pub max_disp: f64,
    /// Whether the linear-model shortcut was used for `mu` (`linearMu`).
    pub linear_mu: bool,
}

/// `estimateDispersionsGeneEst` on the non-all-zero rows. `counts` is `nz x m`; `base_mean`
/// and `base_var` are those rows' values; `x` is the model matrix.
pub fn gene_est(
    counts: &[f64],
    sf: &[f64],
    base_mean: &[f64],
    base_var: &[f64],
    x: &Mat,
) -> Result<GeneEst, String> {
    let m = sf.len();
    let p = x.ncol;
    check_full_rank(x)?;
    if m == p {
        return Err("the number of samples and the number of model coefficients are equal,\n  i.e., there are no replicates to estimate the dispersion.\n  use an alternate design formula".into());
    }
    let n = base_mean.len();
    let norm = normalized(counts, sf);

    // roughDispEstimate
    let lmu = linear_model_mu(&norm, m, x)?;
    let mut rough = vec![0.0; n];
    for g in 0..n {
        let terms: Vec<f64> = (0..m)
            .map(|j| {
                let mu = lmu[g * m + j].max(1.0);
                let y = norm[g * m + j];
                let d = y - mu;
                (d * d - mu) / (mu * mu)
            })
            .collect();
        rough[g] = (ext::row_sum(&terms) / (m - p) as f64).max(0.0);
    }
    // momentsDispEstimate
    let inv_sf: Vec<f64> = sf.iter().map(|s| 1.0 / s).collect();
    let xim = ext::mean(&inv_sf);
    let max_disp = 10f64.max(m as f64);
    let alpha_init: Vec<f64> = (0..n)
        .map(|g| {
            let bm = base_mean[g];
            let moments = (base_var[g] - xim * bm) / (bm * bm);
            r_pmin(r_pmax(MIN_DISP, r_pmin(rough[g], moments)), max_disp)
        })
        .collect();

    let linear_mu = n_groups(x) == p;
    let mut mu = if linear_mu {
        let lm = linear_model_mu(&norm, m, x)?;
        lm.iter().enumerate().map(|(i, v)| v * sf[i % m]).collect::<Vec<f64>>()
    } else {
        let mut opt = GlmOptions::standard(p);
        opt.log_like = false;
        fit_nbinom_glms(counts, sf, x, &alpha_init, &opt)?.mu
    };
    for v in mu.iter_mut() {
        if *v < 0.5 {
            *v = 0.5;
        }
    }

    let ctl = FitDispControl {
        min_log_alpha: ln(MIN_DISP / 10.0),
        kappa_0: 1.0,
        tol: 1e-6,
        maxit: 100,
        use_cr: true,
    };
    let mut disp = vec![0.0; n];
    let mut iter = vec![0usize; n];
    for g in 0..n {
        let y = &counts[g * m..(g + 1) * m];
        let mug = &mu[g * m..(g + 1) * m];
        let la = ln(alpha_init[g]);
        let prior = DispPrior {
            mean: la,
            sigmasq: 1.0,
            use_prior: false,
        };
        let r = fit_disp(y, mug, x, la, prior, ctl)?;
        iter[g] = r.iter;
        let mut a = r_pmin(exp(r.log_alpha), max_disp);
        // noIncrease (niter == 1): fall back to the starting value.
        if r.last_lp < r.initial_lp + r.initial_lp.abs() / 1e6 {
            a = alpha_init[g];
        }
        let conv = r.iter < ctl.maxit && r.iter != 1;
        if !conv && a > MIN_DISP * 10.0 {
            let prior0 = DispPrior {
                mean: 0.0,
                sigmasq: 1.0,
                use_prior: false,
            };
            a = fit_disp_grid(y, mug, x, prior0, true)?;
        }
        disp[g] = r_pmin(r_pmax(a, MIN_DISP), max_disp);
    }
    Ok(GeneEst {
        alpha_init,
        disp,
        iter,
        mu,
        max_disp,
        linear_mu,
    })
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

/// R's `pmin(a, b)` for scalars (NA propagates).
fn r_pmin(a: f64, b: f64) -> f64 {
    if a.is_nan() || b.is_nan() {
        f64::NAN
    } else if a <= b {
        a
    } else {
        b
    }
}

/// The fitted dispersion-mean trend (`dispersionFunction(dds)`).
#[derive(Clone, Debug)]
pub enum DispFunction {
    /// `asymptDisp + extraPois / mean`.
    Parametric {
        /// `asymptDisp`.
        asympt_disp: f64,
        /// `extraPois`.
        extra_pois: f64,
    },
    /// `exp(predict(locfit(log disp ~ log mean, weights = mean), log mean))`.
    Local(Box<rnum::locfit::Locfit>),
}

impl DispFunction {
    /// Evaluate the trend at the given means (`dispFit`).
    pub fn eval(&self, means: &[f64]) -> Vec<f64> {
        match self {
            DispFunction::Parametric {
                asympt_disp,
                extra_pois,
            } => means.iter().map(|q| asympt_disp + extra_pois / q).collect(),
            DispFunction::Local(fit) => means.iter().map(|q| exp(fit.predict(ln(*q)))).collect(),
        }
    }

    /// `"parametric"` or `"local"` (`attr(dispFunction, "fitType")`).
    pub fn fit_type(&self) -> &'static str {
        match self {
            DispFunction::Parametric { .. } => "parametric",
            DispFunction::Local(_) => "local",
        }
    }
}

/// `glm.fit(x, y, start, family = Gamma(link = "identity"))` as `parametricDispersionFit`
/// reaches it: returns `(coefficients, converged)`, or an error where R would stop (including
/// a rank-deficient fit, whose `NA` coefficient makes the caller's `if` fail).
fn glm_gamma_identity(x: &Mat, y: &[f64], start: &[f64]) -> Result<(Vec<f64>, bool), String> {
    let n = y.len();
    let p = x.ncol;
    let eta_of = |b: &[f64]| -> Vec<f64> {
        (0..n)
            .map(|i| {
                let mut s = 0.0;
                for k in 0..p {
                    s += x.at(i, k) * b[k];
                }
                s
            })
            .collect()
    };
    let dev_of = |mu: &[f64]| -> f64 {
        let r: Vec<f64> = (0..n)
            .map(|i| {
                let yi = y[i];
                let l = if yi == 0.0 { 0.0 } else { ln(yi / mu[i]) };
                -2.0 * (l - (yi - mu[i]) / mu[i])
            })
            .collect();
        ext::sum(&r)
    };
    let valid = |mu: &[f64]| mu.iter().all(|v| v.is_finite() && *v > 0.0);

    let mut start = start.to_vec();
    let mut coefold = start.clone();
    let mut eta = eta_of(&start);
    let mut mu = eta.clone();
    if !valid(&mu) {
        return Err("cannot find valid starting values: please specify some".into());
    }
    let mut devold = dev_of(&mu);
    let mut conv = false;
    let mut coef: Option<Vec<f64>> = None;
    let mut rank = p;
    for _iter in 0..25 {
        let z: Vec<f64> = (0..n).map(|i| eta[i] + (y[i] - mu[i])).collect();
        let w: Vec<f64> = (0..n).map(|i| (1.0 / (mu[i] * mu[i])).sqrt()).collect();
        let mut xw = vec![0.0; n * p];
        for k in 0..p {
            for i in 0..n {
                xw[k * n + i] = x.at(i, k) * w[i];
            }
        }
        let zw: Vec<f64> = (0..n).map(|i| z[i] * w[i]).collect();
        let q = qr_decompose(&xw, n, p, 1e-11);
        rank = q.rank;
        let qty = q.qty(&zw);
        let mut cf = vec![0.0; p];
        if q.rank > 0 {
            let b = q.coef_pivoted(&qty)?;
            cf[..q.rank].copy_from_slice(&b);
        }
        if cf.iter().any(|v| !v.is_finite()) {
            break;
        }
        for k in 0..p {
            start[q.pivot[k]] = cf[k];
        }
        eta = eta_of(&start);
        mu = eta.clone();
        let mut dev = dev_of(&mu);
        if !dev.is_finite() {
            let mut ii = 1;
            while !dev.is_finite() {
                if ii > 25 {
                    return Err("inner loop 1; cannot correct step size".into());
                }
                ii += 1;
                for k in 0..p {
                    start[k] = (start[k] + coefold[k]) / 2.0;
                }
                eta = eta_of(&start);
                mu = eta.clone();
                dev = dev_of(&mu);
            }
        }
        if !valid(&mu) {
            let mut ii = 1;
            while !valid(&mu) {
                if ii > 25 {
                    return Err("inner loop 2; cannot correct step size".into());
                }
                ii += 1;
                for k in 0..p {
                    start[k] = (start[k] + coefold[k]) / 2.0;
                }
                eta = eta_of(&start);
                mu = eta.clone();
            }
            dev = dev_of(&mu);
        }
        if (dev - devold).abs() / (dev.abs() + 0.1) < 1e-8 {
            conv = true;
            coef = Some(start.clone());
            break;
        }
        devold = dev;
        coefold = start.clone();
        coef = Some(start.clone());
    }
    let coef = coef.ok_or("object 'coef' not found")?;
    if rank < p {
        return Err("rank-deficient Gamma fit (NA coefficient)".into());
    }
    Ok((coef, conv))
}

/// `parametricDispersionFit(means, disps)`: returns `(asymptDisp, extraPois)` or the error
/// that makes DESeq2 fall back to the local fit.
pub fn parametric_dispersion_fit(means: &[f64], disps: &[f64]) -> Result<(f64, f64), String> {
    let mut coefs = vec![0.1, 1.0];
    let mut iter = 0;
    loop {
        let good: Vec<usize> = (0..means.len())
            .filter(|&i| {
                let r = disps[i] / (coefs[0] + coefs[1] / means[i]);
                r > 1e-4 && r < 15.0
            })
            .collect();
        if good.is_empty() {
            return Err("no usable points for the parametric fit".into());
        }
        let mut x = Mat::zeros(good.len(), 2);
        let mut y = vec![0.0; good.len()];
        for (r, &i) in good.iter().enumerate() {
            *x.at_mut(r, 0) = 1.0;
            *x.at_mut(r, 1) = 1.0 / means[i];
            y[r] = disps[i];
        }
        let (new, converged) = glm_gamma_identity(&x, &y, &coefs)?;
        let old = std::mem::replace(&mut coefs, new);
        if !coefs.iter().all(|c| *c > 0.0) {
            return Err("parametric dispersion fit failed".into());
        }
        let sq: Vec<f64> = (0..2)
            .map(|k| {
                let l = ln(coefs[k] / old[k]);
                l * l
            })
            .collect();
        if ext::sum(&sq) < 1e-6 && converged {
            break;
        }
        iter += 1;
        if iter > 10 {
            return Err("dispersion fit did not converge".into());
        }
    }
    Ok((coefs[0], coefs[1]))
}

/// `estimateDispersionsFit(fitType = "parametric")` over the non-all-zero rows, with the local
/// fallback. `base_mean` and `disp_gene_est` are the non-all-zero rows' values.
pub fn fit_trend(base_mean: &[f64], disp_gene_est: &[f64]) -> Result<DispFunction, String> {
    let use_for_fit: Vec<usize> = (0..disp_gene_est.len())
        .filter(|&i| disp_gene_est[i] > 100.0 * MIN_DISP)
        .collect();
    if use_for_fit.is_empty() {
        return Err("all gene-wise dispersion estimates are within 2 orders of magnitude\n  from the minimum value, and so the standard curve fitting techniques will not work.\n  One can instead use the gene-wise estimates as final estimates:\n  dds <- estimateDispersionsGeneEst(dds)\n  dispersions(dds) <- mcols(dds)$dispGeneEst\n  ...then continue with testing using nbinomWaldTest or nbinomLRT".into());
    }
    let means: Vec<f64> = use_for_fit.iter().map(|&i| base_mean[i]).collect();
    let disps: Vec<f64> = use_for_fit.iter().map(|&i| disp_gene_est[i]).collect();
    if let Ok((a, b)) = parametric_dispersion_fit(&means, &disps) {
        return Ok(DispFunction::Parametric {
            asympt_disp: a,
            extra_pois: b,
        });
    }
    // localDispersionFit: every useForFit disp is > 1e-6 >= minDisp * 10.
    let lx: Vec<f64> = means.iter().map(|v| ln(*v)).collect();
    let ly: Vec<f64> = disps.iter().map(|v| ln(*v)).collect();
    let fit = rnum::locfit::locfit(&lx, &ly, Some(&means), &rnum::locfit::LocfitOptions::default())
        .map_err(|e| format!("locfit: {e}"))?;
    Ok(DispFunction::Local(Box::new(fit)))
}

/// R's `mad(x)` (`1.4826 * median(|x - median(x)|)`).
pub fn mad(x: &[f64]) -> f64 {
    let c = ext::median(x);
    let d: Vec<f64> = x.iter().map(|v| (v - c).abs()).collect();
    1.4826 * ext::median(&d)
}

/// `varLogDispEsts` (the dispersion-function setter): `mad(log ge - log fit)^2` over the
/// estimates `>= 1e-6`. `None` when there are none.
pub fn var_log_disp_ests(disp_gene_est: &[f64], disp_fit: &[f64]) -> Option<f64> {
    let r: Vec<f64> = (0..disp_gene_est.len())
        .filter(|&i| disp_gene_est[i] >= MIN_DISP * 100.0)
        .map(|i| ln(disp_gene_est[i]) - ln(disp_fit[i]))
        .collect();
    if r.is_empty() {
        return None;
    }
    let v = mad(&r);
    Some(v * v)
}

/// `estimateDispersionsPriorVar` for `m` samples and `p` coefficients.
pub fn prior_var(
    disp_gene_est: &[f64],
    disp_fit: &[f64],
    var_log: f64,
    m: usize,
    p: usize,
) -> Result<f64, String> {
    let resid: Vec<f64> = (0..disp_gene_est.len())
        .filter(|&i| disp_gene_est[i] >= MIN_DISP * 100.0)
        .map(|i| ln(disp_gene_est[i]) - ln(disp_fit[i]))
        .collect();
    if resid.is_empty() {
        return Err("no data found which is greater than minDisp".into());
    }
    if m > p && m - p <= 3 {
        Ok(crate::prior_var::prior_var_simulation(&resid, (m - p) as f64)?.prior_var)
    } else if m > p {
        let v = var_log - trigamma((m - p) as f64 / 2.0);
        Ok(if v >= 0.25 { v } else { 0.25 })
    } else {
        Ok(var_log)
    }
}

/// MAP dispersions over the non-all-zero rows (`estimateDispersionsMAP`).
#[derive(Clone, Debug)]
pub struct MapEst {
    /// `dispMAP`.
    pub disp_map: Vec<f64>,
    /// `dispIter`.
    pub iter: Vec<usize>,
    /// `dispOutlier`.
    pub outlier: Vec<bool>,
    /// `dispersion` (the gene-wise estimate for outliers, the MAP otherwise).
    pub dispersion: Vec<f64>,
}

/// `estimateDispersionsMAP` given the gene-wise fit, the trend values and the prior.
pub fn map_est(
    counts: &[f64],
    x: &Mat,
    ge: &GeneEst,
    disp_fit: &[f64],
    prior_var: f64,
    var_log: f64,
) -> Result<MapEst, String> {
    let m = x.nrow;
    let n = disp_fit.len();
    let ctl = FitDispControl {
        min_log_alpha: ln(MIN_DISP / 10.0),
        kappa_0: 1.0,
        tol: 1e-6,
        maxit: 100,
        use_cr: true,
    };
    let mut out = MapEst {
        disp_map: vec![0.0; n],
        iter: vec![0; n],
        outlier: vec![false; n],
        dispersion: vec![0.0; n],
    };
    let thresh_sd = 2.0 * var_log.sqrt();
    for g in 0..n {
        let y = &counts[g * m..(g + 1) * m];
        let mu = &ge.mu[g * m..(g + 1) * m];
        let gd = ge.disp[g];
        let mut init = if gd > 0.1 * disp_fit[g] { gd } else { disp_fit[g] };
        if init.is_nan() {
            init = disp_fit[g];
        }
        let prior = DispPrior {
            mean: ln(disp_fit[g]),
            sigmasq: prior_var,
            use_prior: true,
        };
        let r = fit_disp(y, mu, x, ln(init), prior, ctl)?;
        let mut a = exp(r.log_alpha);
        out.iter[g] = r.iter;
        if r.iter >= ctl.maxit {
            a = fit_disp_grid(y, mu, x, prior, true)?;
        }
        let a = r_pmin(r_pmax(a, MIN_DISP), ge.max_disp);
        out.disp_map[g] = a;
        let o = ln(gd) > ln(disp_fit[g]) + thresh_sd;
        out.outlier[g] = o;
        out.dispersion[g] = if o { gd } else { a };
    }
    Ok(out)
}

/// The whole dispersion stage (`estimateDispersions` as `DESeq()` runs it) for all rows.
#[derive(Clone, Debug)]
pub struct Dispersions {
    /// Base statistics for every row.
    pub base: BaseStats,
    /// Gene-wise estimates over the non-all-zero rows.
    pub gene: GeneEst,
    /// The trend.
    pub function: DispFunction,
    /// `dispFit` over the non-all-zero rows.
    pub disp_fit: Vec<f64>,
    /// `varLogDispEsts`.
    pub var_log: f64,
    /// `dispPriorVar`.
    pub prior_var: f64,
    /// MAP estimates over the non-all-zero rows.
    pub map: MapEst,
}

impl Dispersions {
    /// `dispersions(dds)` expanded to all rows (NaN for all-zero rows).
    pub fn dispersion_all(&self) -> Vec<f64> {
        expand(&self.map.dispersion, &self.base.all_zero)
    }
}

/// Expand non-all-zero values to every row, NaN where `all_zero`.
pub fn expand(v: &[f64], all_zero: &[bool]) -> Vec<f64> {
    let mut it = v.iter();
    all_zero
        .iter()
        .map(|z| if *z { f64::NAN } else { *it.next().unwrap() })
        .collect()
}

/// Size factors given, run base statistics, gene-wise, trend, prior variance and MAP.
pub fn estimate_dispersions(counts: &[f64], sf: &[f64], x: &Mat) -> Result<Dispersions, String> {
    let m = sf.len();
    let base = base_stats(counts, sf);
    let nz: Vec<bool> = base.all_zero.iter().map(|z| !z).collect();
    let cnz = subset_rows(counts, m, &nz);
    let bm: Vec<f64> = (0..nz.len()).filter(|&g| nz[g]).map(|g| base.base_mean[g]).collect();
    let bv: Vec<f64> = (0..nz.len()).filter(|&g| nz[g]).map(|g| base.base_var[g]).collect();
    let gene = gene_est(&cnz, sf, &bm, &bv, x)?;
    let function = fit_trend(&bm, &gene.disp)?;
    let disp_fit = function.eval(&bm);
    let var_log = var_log_disp_ests(&gene.disp, &disp_fit)
        .ok_or("variance of dispersion residuals not estimated")?;
    let prior_var = prior_var(&gene.disp, &disp_fit, var_log, m, x.ncol)?;
    let map = map_est(&cnz, x, &gene, &disp_fit, prior_var, var_log)?;
    Ok(Dispersions {
        base,
        gene,
        function,
        disp_fit,
        var_log,
        prior_var,
        map,
    })
}
