//! `ashr::ash(betahat, sebetahat, mixcompdist = "normal", method = "shrink")` (ashr 2.2-63)
//! as DESeq2 1.50.2 `lfcShrink(type = "ashr")` calls it: a zero-centred scale mixture of
//! normals with no point mass, uniform prior, grid from `autoselect.mixsd`, weights by
//! mix-SQP, results computed on `prune(g, 1e-10)`.

use crate::dense::Mat;
use crate::mixsqp::{mixsqp_ashr, MixsqpResult};
use crate::xld::{r_cumsum, r_sum};
use crate::ShrinkError;
use rnum::glibm;
use rnum::nmath::{dnorm, pnorm, qnorm};

/// Per-observation output, in input order (one row per input row).
#[derive(Clone, Debug, Default)]
pub struct AshrTable {
    pub posterior_mean: Vec<f64>,
    pub posterior_sd: Vec<f64>,
    pub negative_prob: Vec<f64>,
    pub zero_prob: Vec<f64>,
    pub lfsr: Vec<f64>,
    pub svalue: Vec<f64>,
    /// `PosteriorMean -/+ qnorm(0.975) * PosteriorSD` (production `deseq2StatsFun.R`).
    pub cri_left: Vec<f64>,
    pub cri_right: Vec<f64>,
}

/// Everything the ashr path computes, kept for the golden tests.
#[derive(Clone, Debug)]
pub struct AshrFit {
    /// Mixture standard deviations (`autoselect.mixsd`), ascending.
    pub mixsd: Vec<f64>,
    pub excluded: Vec<bool>,
    /// `exp(llik - rowmax)` over the non-excluded observations, all grid columns.
    pub lik: Mat,
    /// Row maxima of the log-likelihood (`lnorm`), non-excluded observations.
    pub lnorm: Vec<f64>,
    pub nonzero_cols: Vec<bool>,
    /// mix-SQP result on the non-zero columns (None when fewer than two columns).
    pub mixsqp: Option<MixsqpResult>,
    /// `g$pi` after `pmax(pihat, 0)`, all grid columns.
    pub pi: Vec<f64>,
    pub table: AshrTable,
}

/// `get_exclusions`: `s == 0 | s == Inf | is.na(x) | is.na(s)`.
pub fn exclusions(x: &[f64], s: &[f64]) -> Vec<bool> {
    x.iter().zip(s).map(|(x, s)| *s == 0.0 || *s == f64::INFINITY || x.is_nan() || s.is_nan()).collect()
}

/// `autoselect.mixsd(data, mult = sqrt(2), mode = 0)`.
pub fn autoselect_mixsd(x: &[f64], s: &[f64], excluded: &[bool]) -> Result<Vec<f64>, ShrinkError> {
    let b: Vec<f64> = (0..x.len()).filter(|&i| !excluded[i]).map(|i| x[i] - 0.0).collect();
    let se: Vec<f64> = (0..x.len()).filter(|&i| !excluded[i]).map(|i| s[i]).collect();
    if b.is_empty() {
        return Err(ShrinkError::InvalidInput("ashr: no non-excluded observations".into()));
    }
    let smin = se.iter().cloned().fold(f64::INFINITY, f64::min) / 10.0;
    let smax = if b.iter().zip(&se).all(|(b, s)| b * b <= s * s) {
        8.0 * smin
    } else {
        let mx = b.iter().zip(&se).map(|(b, s)| b * b - s * s).fold(f64::NEG_INFINITY, f64::max);
        2.0 * mx.sqrt()
    };
    let mult = 2f64.sqrt();
    let npoint = ((smax / smin).log2() / mult.log2()).ceil();
    if !npoint.is_finite() || !(0.0..=1e6).contains(&npoint) {
        return Err(ShrinkError::Numerical(format!("ashr: grid size not finite ({npoint})")));
    }
    let np = npoint as i64;
    Ok((-np..=0).map(|k| r_pow(mult, k as f64) * smax).collect())
}

/// R's `R_POW` for finite arguments: `y == 2 ? x*x : R_pow(x, y)`; `R_pow` returns 1 for
/// `y == 0` and libm `pow` otherwise.
fn r_pow(x: f64, y: f64) -> f64 {
    if y == 2.0 {
        x * x
    } else if x == 1.0 || y == 0.0 {
        1.0
    } else {
        x.powf(y)
    }
}

/// `log_comp_dens_conv.normalmix` for one observation and one component:
/// `dnorm(x / sqrt(s^2 + sd^2), log = TRUE) - log(sqrt(s^2 + sd^2))`.
#[inline]
fn lcd(x: f64, s: f64, sd: f64) -> f64 {
    let sdm = (s * s + sd * sd).sqrt();
    dnorm((x - 0.0) / sdm, 0.0, 1.0, true) - glibm::ln(sdm)
}

/// The full ashr path. `x` = MLE log2 fold changes, `s` = their standard errors.
pub fn ash_shrink(x: &[f64], s: &[f64]) -> Result<AshrFit, ShrinkError> {
    if x.len() != s.len() {
        return Err(ShrinkError::InvalidInput("ashr: betahat and sebetahat lengths differ".into()));
    }
    let n_all = x.len();
    let excluded = exclusions(x, s);
    let mixsd = autoselect_mixsd(x, s, &excluded)?;
    let k = mixsd.len();
    let keep: Vec<usize> = (0..n_all).filter(|&i| !excluded[i]).collect();
    let n = keep.len();

    // matrix_llik (n x k), lnorm, matrix_lik
    let mut lik = Mat::zeros(n, k);
    for c in 0..k {
        for (r, &i) in keep.iter().enumerate() {
            lik.set(r, c, lcd(x[i], s[i], mixsd[c]));
        }
    }
    let mut lnorm = vec![f64::NEG_INFINITY; n];
    for c in 0..k {
        for r in 0..n {
            let v = lik.at(r, c);
            if v > lnorm[r] {
                lnorm[r] = v;
            }
        }
    }
    for c in 0..k {
        for r in 0..n {
            let v = glibm::exp(lik.at(r, c) - lnorm[r]);
            lik.set(r, c, v);
        }
    }
    let nonzero_cols: Vec<bool> = (0..k).map(|c| lik.col(c).iter().cloned().fold(f64::NEG_INFINITY, f64::max) > 0.0).collect();
    let nz: Vec<usize> = (0..k).filter(|&c| nonzero_cols[c]).collect();

    let (mixsqp, pihat) = if nz.len() > 1 {
        let mut sub = Mat::zeros(n, nz.len());
        for (cc, &c) in nz.iter().enumerate() {
            sub.col_mut(cc).copy_from_slice(lik.col(c));
        }
        let res = mixsqp_ashr(&sub)?;
        // ashr:::mixSQP: pihat = x / sum(x)
        let sx = r_sum(res.x.iter().cloned());
        let p: Vec<f64> = res.x.iter().map(|v| v / sx).collect();
        (Some(res), p)
    } else {
        // gradient check branch with a single component: pihat = 1
        (None, vec![1.0; nz.len()])
    };
    let mut pi = vec![0.0; k];
    for (cc, &c) in nz.iter().enumerate() {
        pi[c] = pihat[cc].max(0.0);
    }

    let table = posterior_table(x, s, &excluded, &mixsd, &pi);
    Ok(AshrFit { mixsd, excluded, lik, lnorm, nonzero_cols, mixsqp, pi, table })
}

/// `prune(g, 1e-10)` then the result columns (`calc_pm`, `calc_psd`, `calc_np`,
/// `calc_lfdr`, `calc_lfsr`, `calc_svalue`) for a zero-centred normal mixture with weights
/// `pi` on standard deviations `mixsd` (all > 0, so `ZeroProb` is 0).
pub fn posterior_table(x: &[f64], s: &[f64], excluded: &[bool], mixsd: &[f64], pi: &[f64]) -> AshrTable {
    let kept: Vec<usize> = (0..pi.len()).filter(|&c| pi[c] > 1e-10).collect();
    let spi = r_sum(kept.iter().map(|&c| pi[c]));
    let pk: Vec<f64> = kept.iter().map(|&c| pi[c] / spi).collect();
    let sdk: Vec<f64> = kept.iter().map(|&c| mixsd[c]).collect();
    let logpi: Vec<f64> = pk.iter().map(|&p| glibm::ln(p)).collect();
    let kk = pk.len();
    let n = x.len();
    let mut t = AshrTable::default();
    let qn = qnorm(0.975, 0.0, 1.0, true, false);

    // Excluded observations get the prior's moments.
    let mixmean = r_sum(pk.iter().map(|p| p * 0.0));
    let mixmean2 = r_sum((0..kk).map(|c| pk[c] * (0.0 * 0.0 + sdk[c] * sdk[c])));
    let mixsd_prior = (mixmean2 - mixmean * mixmean).sqrt();
    let mixcdf0: f64 = (0..kk).map(|c| pk[c] * pnorm(0.0, 0.0, sdk[c], true, false)).sum();

    let mut lpost = vec![0.0; kk];
    let mut pp = vec![0.0; kk];
    let mut pm = vec![0.0; kk];
    let mut psd = vec![0.0; kk];
    for i in 0..n {
        let (pmean, psdv, np) = if excluded[i] {
            (mixmean, mixsd_prior, mixcdf0)
        } else {
            let (xi, si) = (x[i], s[i]);
            let s2 = si * si;
            for c in 0..kk {
                lpost[c] = lcd(xi, si, sdk[c]) + logpi[c];
            }
            let lmax = lpost.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
            for c in 0..kk {
                pp[c] = glibm::exp(lpost[c] - lmax);
            }
            let rs = r_sum(pp.iter().cloned());
            for c in 0..kk {
                pp[c] /= rs;
                let sd2 = sdk[c] * sdk[c];
                pm[c] = (s2 * 0.0 + xi * sd2) / (s2 + sd2);
                psd[c] = (s2 * sd2 / (s2 + sd2)).sqrt();
            }
            let mean = r_sum((0..kk).map(|c| pp[c] * pm[c]));
            let mut m2 = r_sum((0..kk).map(|c| pp[c] * (psd[c] * psd[c] + pm[c] * pm[c])));
            if m2 < 0.0 {
                m2 = 0.0;
            }
            let mut var = m2 - mean * mean;
            if var < 0.0 {
                var = 0.0;
            }
            let cdf = r_sum((0..kk).map(|c| pp[c] * pnorm(0.0, pm[c], psd[c], true, false)));
            (mean, var.sqrt(), cdf - 0.0)
        };
        let np = if np < 0.0 { 0.0 } else { np };
        let zp = 0.0;
        let lfsr = if np > 0.5 * (1.0 - zp) { 1.0 - np } else { np + zp };
        let lfsr = if lfsr < 0.0 { 0.0 } else { lfsr };
        t.posterior_mean.push(pmean);
        t.posterior_sd.push(psdv);
        t.negative_prob.push(np);
        t.zero_prob.push(zp);
        t.lfsr.push(lfsr);
        t.cri_left.push(pmean - qn * psdv);
        t.cri_right.push(pmean + qn * psdv);
    }
    t.svalue = qval_from_lfdr(&t.lfsr);
    t
}

/// `qval.from.lfdr`: `qvalue[order(l)] = cumsum(sort(l)) / (1:n)` (stable order, no NAs).
pub fn qval_from_lfdr(l: &[f64]) -> Vec<f64> {
    let n = l.len();
    let mut o: Vec<usize> = (0..n).collect();
    o.sort_by(|&a, &b| l[a].partial_cmp(&l[b]).unwrap_or(std::cmp::Ordering::Equal));
    let sorted: Vec<f64> = o.iter().map(|&i| l[i]).collect();
    let cs = r_cumsum(&sorted);
    let mut q = vec![0.0; n];
    for (r, &i) in o.iter().enumerate() {
        q[i] = cs[r] / ((r + 1) as f64);
    }
    q
}
