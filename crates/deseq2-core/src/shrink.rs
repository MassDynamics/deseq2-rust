//! `lfcShrink(type = "normal")` as production calls it on a `betaPrior = FALSE` fit:
//! `estimateBetaPriorVar` (the weighted upper-quantile rule, `Hmisc::wtd.quantile` with
//! `normwt = TRUE`) and the penalised refit `nbinomWaldTest(betaPrior = TRUE, betaPriorVar)`,
//! which reuses the MLE fit's `mu` and `H` and only changes the coefficients and their SEs.

use crate::design::{r_num_string, Design};
use crate::ext::F80;
use crate::glm::{fit_nbinom_glms, GlmOptions};
use crate::nbtest::TestFit;

/// R's `sum` (long-double accumulation).
fn ld_sum(x: &[f64]) -> f64 {
    crate::ext::sum(x)
}

/// R's `cumsum` (long-double accumulator, each partial sum rounded to double).
fn ld_cumsum(x: &[f64]) -> Vec<f64> {
    let mut s = F80::from_f64(0.0);
    x.iter()
        .map(|v| {
            s = s.add_f64(*v);
            s.to_f64()
        })
        .collect()
}

/// `approx(x, y, xout, method = "constant", f = 1, rule = 2)` for one point, `x` increasing.
fn approx_constant_f1(x: &[f64], y: &[f64], v: f64) -> f64 {
    let n = x.len();
    let (mut i, mut j) = (0usize, n - 1);
    if v < x[i] {
        return y[0];
    }
    if v > x[j] {
        return y[n - 1];
    }
    while i + 1 < j {
        let ij = (i + j) / 2;
        if v < x[ij] {
            j = ij;
        } else {
            i = ij;
        }
    }
    if v == x[j] {
        return y[j];
    }
    if v == x[i] {
        return y[i];
    }
    y[j] * 1.0
}

/// `Hmisc.wtd.quantile(x, weights, prob, normwt = TRUE)` (DESeq2's copy, `type = "quantile"`).
pub fn wtd_quantile_normwt(x: &[f64], weights: &[f64], prob: f64) -> f64 {
    // Drop NA / zero weights, then NA x or weights (`na.rm`).
    let keep: Vec<usize> = (0..x.len())
        .filter(|&i| !(weights[i].is_nan() || weights[i] == 0.0 || (x[i] + weights[i]).is_nan()))
        .collect();
    let xs: Vec<f64> = keep.iter().map(|&i| x[i]).collect();
    let ws: Vec<f64> = keep.iter().map(|&i| weights[i]).collect();
    let n = xs.len() as f64;
    let sw = ld_sum(&ws);
    let ws: Vec<f64> = ws.iter().map(|w| w * n / sw).collect();
    // order(x): stable ascending.
    let mut o: Vec<usize> = (0..xs.len()).collect();
    o.sort_by(|&a, &b| xs[a].partial_cmp(&xs[b]).unwrap());
    let xs: Vec<f64> = o.iter().map(|&i| xs[i]).collect();
    let ws: Vec<f64> = o.iter().map(|&i| ws[i]).collect();
    let dup = xs.windows(2).any(|w| w[0] == w[1]);
    let (xv, wv) = if dup {
        // tapply(weights, x, sum), then x <- as.numeric(names(weights)).
        let mut xv = Vec::new();
        let mut wv = Vec::new();
        let mut k = 0;
        while k < xs.len() {
            let mut e = k;
            while e < xs.len() && xs[e] == xs[k] {
                e += 1;
            }
            xv.push(r_num_string(xs[k]).parse::<f64>().unwrap());
            wv.push(ld_sum(&ws[k..e]));
            k = e;
        }
        (xv, wv)
    } else {
        (xs, ws)
    };
    let tot = ld_sum(&wv);
    let order = 1.0 + (tot - 1.0) * prob;
    let low = order.floor().max(1.0);
    let high = (low + 1.0).min(tot);
    let frac = order - order.floor();
    let cs = ld_cumsum(&wv);
    let ql = approx_constant_f1(&cs, &xv, low);
    let qh = approx_constant_f1(&cs, &xv, high);
    (1.0 - frac) * ql + frac * qh
}

/// `estimateBetaPriorVar(dds)` for a standard model matrix: per coefficient the variance of
/// the zero-centred normal whose 97.5% quantile matches the weighted 95% quantile of |MLE beta|
/// (weights `1 / (1 / baseMean + dispFit)`, betas with |beta| >= 10 dropped), and `1e6` for the
/// intercept. `beta` is the `n x p` MLE (log2) over every row; only non-all-zero rows are read.
pub fn beta_prior_var(
    beta: &[f64],
    coef_names: &[String],
    all_zero: &[bool],
    base_mean: &[f64],
    disp_fit: &[f64],
) -> Vec<f64> {
    let p = coef_names.len();
    let nz: Vec<usize> = (0..all_zero.len()).filter(|&g| !all_zero[g]).collect();
    let weights: Vec<f64> = nz
        .iter()
        .map(|&g| 1.0 / (1.0 / base_mean[g] + disp_fit[g]))
        .collect();
    let qn = rnum::nmath::qnorm(1.0 - 0.05 / 2.0, 0.0, 1.0, true, false);
    let mut out: Vec<f64> = (0..p)
        .map(|k| {
            let x: Vec<f64> = nz.iter().map(|&g| beta[g * p + k]).collect();
            if x.len() <= 1 {
                return x.first().map_or(f64::NAN, |b| b * b);
            }
            let use_: Vec<usize> = (0..x.len()).filter(|&i| x[i].abs() < 10.0).collect();
            if use_.is_empty() {
                return 1e6;
            }
            let ax: Vec<f64> = use_.iter().map(|&i| x[i].abs()).collect();
            let w: Vec<f64> = use_.iter().map(|&i| weights[i]).collect();
            let sd = wtd_quantile_normwt(&ax, &w, 1.0 - 0.05) / qn;
            sd * sd
        })
        .collect();
    for (k, c) in coef_names.iter().enumerate() {
        if c == "Intercept" || c == "(Intercept)" {
            out[k] = 1e6;
        }
    }
    out
}

/// The normal-prior fit: the prior variances and the penalised coefficients and SEs (log2,
/// `n x p`, NaN on all-zero rows).
#[derive(Clone, Debug)]
pub struct NormalShrink {
    /// `betaPriorVar`, one per coefficient.
    pub beta_prior_var: Vec<f64>,
    /// Shrunken coefficients.
    pub beta: Vec<f64>,
    /// Their standard errors.
    pub se: Vec<f64>,
}

/// `nbinomWaldTest(dds, betaPrior = TRUE, betaPriorVar = estimateBetaPriorVar(dds))` on the
/// MLE fit `mle` (whose design is `design`): `counts` (`n x m`) are `counts(dds)`,
/// `dispersion`, `base_mean`, `disp_fit` the `mcols` columns over every row.
#[allow(clippy::too_many_arguments)]
pub fn shrink_normal(
    counts: &[f64],
    sf: &[f64],
    all_zero: &[bool],
    dispersion: &[f64],
    base_mean: &[f64],
    disp_fit: &[f64],
    design: &Design,
    mle: &TestFit,
) -> Result<NormalShrink, String> {
    let m = sf.len();
    let p = mle.p();
    let bpv = beta_prior_var(&mle.beta, &mle.coef_names, all_zero, base_mean, disp_fit);
    if bpv.contains(&0.0) {
        return Err("beta prior variances are equal to zero for some variables".into());
    }
    let (x, _) = design.deseq_matrix();
    let nz: Vec<bool> = all_zero.iter().map(|z| !z).collect();
    let cnz = crate::disp::subset_rows(counts, m, &nz);
    let alpha: Vec<f64> = (0..all_zero.len())
        .filter(|&g| nz[g])
        .map(|g| dispersion[g])
        .collect();
    let opt = GlmOptions {
        lambda: bpv.iter().map(|v| 1.0 / v).collect(),
        maxit: 100,
        log_like: false,
    };
    let fit = fit_nbinom_glms(&cnz, sf, &x, &alpha, &opt)?;
    let n = all_zero.len();
    let mut beta = vec![f64::NAN; n * p];
    let mut se = vec![f64::NAN; n * p];
    let mut k = 0;
    for g in 0..n {
        if all_zero[g] {
            continue;
        }
        beta[g * p..(g + 1) * p].copy_from_slice(&fit.beta[k * p..(k + 1) * p]);
        se[g * p..(g + 1) * p].copy_from_slice(&fit.se[k * p..(k + 1) * p]);
        k += 1;
    }
    Ok(NormalShrink {
        beta_prior_var: bpv,
        beta,
        se,
    })
}
