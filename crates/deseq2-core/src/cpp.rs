//! DESeq2's C++ (`src/DESeq2.cpp`): the dispersion line search `fitDisp`, its grid fallback
//! `fitDispGrid`, and the IRLS beta fit `fitBeta` (QR path), without observation weights
//! (production never passes `weights`).
//!
//! Every libm call goes through the glibc ports (`rnum::glibm*`) and the R nmath copies in
//! `crate::gnm`, and every matrix product through [`crate::la`], so the operation order and the
//! last bit match the reference image. Rcpp sugar `sum` is a sequential double sum.

use crate::gnm::{arith::r_pow_di, digamma, dnbinom_mu, lgammafn, G};
use crate::la::{self, Mat};

#[inline]
fn ln(x: f64) -> f64 {
    rnum::glibm::ln(x)
}
#[inline]
fn exp(x: f64) -> f64 {
    rnum::glibm::exp(x)
}
#[inline]
fn pow(x: f64, y: f64) -> f64 {
    x.gpow(y)
}

/// `pow(pow(mu, -1) + alpha, -1)` in the Cox-Reid weights: GCC expands `pow(x, n)` for a
/// constant integer `n` in [-1, 2] once the sugar exponent is inlined, so there `-1` compiles
/// to `1.0 / x`. In the `dlog_posterior` sum the same sugar `pow(.., -1)` stays a glibc `pow`
/// call, and `-2` is always a glibc call. Both measured against the image on every row of the
/// airway run at three log alphas (lp and dlp bit-identical only with this split).
fn recip(x: f64) -> f64 {
    1.0 / x
}

/// Prior on log dispersion used by the line search and the grid.
#[derive(Clone, Copy, Debug)]
pub struct DispPrior {
    /// `log_alpha_prior_mean` for the row.
    pub mean: f64,
    /// `log_alpha_prior_sigmasq` (shared by all rows).
    pub sigmasq: f64,
    /// `usePrior`.
    pub use_prior: bool,
}

/// `log_posterior` in `DESeq2.cpp` (no weights).
pub fn log_posterior(
    log_alpha: f64,
    y: &[f64],
    mu: &[f64],
    x: &Mat,
    prior: DispPrior,
    use_cr: bool,
) -> Result<f64, String> {
    let alpha = exp(log_alpha);
    let cr_term = if use_cr {
        let w: Vec<f64> = mu.iter().map(|&m| recip(recip(m) + alpha)).collect();
        let b = la::xtwx(x, &w);
        -0.5 * ln(la::det(&b))
    } else {
        0.0
    };
    let a1 = r_pow_di(alpha, -1);
    let lg_a1 = lgammafn(a1);
    let mut ll = 0.0;
    for (&yi, &mi) in y.iter().zip(mu) {
        ll += lgammafn(yi + a1) - lg_a1 - yi * ln(mi + a1) - a1 * ln(1.0 + mi * alpha);
    }
    let prior_part = if prior.use_prior {
        -0.5 * r_pow_di(log_alpha - prior.mean, 2) / prior.sigmasq
    } else {
        0.0
    };
    Ok(ll + prior_part + cr_term)
}

/// `dlog_posterior` in `DESeq2.cpp` (no weights): derivative with respect to log alpha.
pub fn dlog_posterior(
    log_alpha: f64,
    y: &[f64],
    mu: &[f64],
    x: &Mat,
    prior: DispPrior,
    use_cr: bool,
) -> Result<f64, String> {
    let alpha = exp(log_alpha);
    let cr_term = if use_cr {
        let w: Vec<f64> = mu.iter().map(|&m| recip(recip(m) + alpha)).collect();
        let dw: Vec<f64> = mu
            .iter()
            .map(|&m| -1.0 * pow(recip(m) + alpha, -2.0))
            .collect();
        let b = la::xtwx(x, &w);
        let db = la::xtwx(x, &dw);
        let bi = la::inv(&b).map_err(|e| format!("inv(): {}", e.0))?;
        let ddetb = la::det(&b) * la::trace_mul(&bi, &db);
        -0.5 * ddetb / la::det(&b)
    } else {
        0.0
    };
    let a1 = r_pow_di(alpha, -1);
    let a2 = r_pow_di(alpha, -2);
    let dg_a1 = digamma(a1);
    let mut s = 0.0;
    for (&yi, &mi) in y.iter().zip(mu) {
        s += dg_a1 + ln(1.0 + mi * alpha) - mi * alpha * pow(1.0 + mi * alpha, -1.0)
            - digamma(yi + a1)
            + yi * pow(mi + a1, -1.0);
    }
    let ll = a2 * s;
    let prior_part = if prior.use_prior {
        -1.0 * (log_alpha - prior.mean) / prior.sigmasq
    } else {
        0.0
    };
    Ok((ll + cr_term) * alpha + prior_part)
}

/// Control of the dispersion line search (`fitDisp` arguments).
#[derive(Clone, Copy, Debug)]
pub struct FitDispControl {
    /// `min_log_alpha`.
    pub min_log_alpha: f64,
    /// `kappa_0`.
    pub kappa_0: f64,
    /// `tol`.
    pub tol: f64,
    /// `maxit`.
    pub maxit: usize,
    /// `useCR`.
    pub use_cr: bool,
}

/// One row's result of [`fit_disp`].
#[derive(Clone, Copy, Debug)]
pub struct FitDispRow {
    /// Final log alpha.
    pub log_alpha: f64,
    /// Iterations used.
    pub iter: usize,
    /// Log posterior at the start.
    pub initial_lp: f64,
    /// Log posterior at the end.
    pub last_lp: f64,
}

/// `fitDisp` for one row: backtracking line search on log alpha starting at `log_alpha`.
pub fn fit_disp(
    y: &[f64],
    mu: &[f64],
    x: &Mat,
    log_alpha: f64,
    prior: DispPrior,
    ctl: FitDispControl,
) -> Result<FitDispRow, String> {
    let epsilon = 1.0e-4;
    let lpf = |a: f64| log_posterior(a, y, mu, x, prior, ctl.use_cr);
    let dlpf = |a: f64| dlog_posterior(a, y, mu, x, prior, ctl.use_cr);
    let mut a = log_alpha;
    let mut lp = lpf(a)?;
    let mut dlp = dlpf(a)?;
    let mut kappa = ctl.kappa_0;
    let initial_lp = lp;
    let mut iter = 0usize;
    let mut iter_accept = 0usize;
    for _t in 0..ctl.maxit {
        iter += 1;
        let a_propose = a + kappa * dlp;
        if a_propose < -30.0 {
            kappa = (-30.0 - a) / dlp;
        }
        if a_propose > 10.0 {
            kappa = (10.0 - a) / dlp;
        }
        let theta_kappa = -1.0 * lpf(a + kappa * dlp)?;
        let theta_hat_kappa = -1.0 * lp - kappa * epsilon * r_pow_di(dlp, 2);
        if theta_kappa <= theta_hat_kappa {
            iter_accept += 1;
            a += kappa * dlp;
            let lpnew = lpf(a)?;
            let change = lpnew - lp;
            if change < ctl.tol {
                lp = lpnew;
                break;
            }
            if a < ctl.min_log_alpha {
                break;
            }
            lp = lpnew;
            dlp = dlpf(a)?;
            kappa = (kappa * 1.1).min(ctl.kappa_0);
            if iter_accept % 5 == 0 {
                kappa /= 2.0;
            }
        } else {
            kappa /= 2.0;
        }
    }
    Ok(FitDispRow {
        log_alpha: a,
        iter,
        initial_lp,
        last_lp: lp,
    })
}

/// R's `seq(from, to, length.out = n)` (and Armadillo's `linspace`, which is the same loop).
pub fn seq_len_out(from: f64, to: f64, n: usize) -> Vec<f64> {
    crate::prior_var::seq_len_out(from, to, n)
}

/// Armadillo `index_max`: first index of the largest value (NaN never wins).
fn index_max(v: &[f64]) -> usize {
    let mut best = 0;
    let mut bv = f64::NEG_INFINITY;
    for (i, &x) in v.iter().enumerate() {
        if x > bv {
            bv = x;
            best = i;
        }
    }
    best
}

/// `fitDispGridWrapper` for one row: coarse then fine grid search; returns alpha
/// (`exp(log_alpha)`), as the R wrapper does. `ncol` is the number of samples.
pub fn fit_disp_grid(
    y: &[f64],
    mu: &[f64],
    x: &Mat,
    prior: DispPrior,
    use_cr: bool,
) -> Result<f64, String> {
    let ncol = y.len();
    let grid = seq_len_out(ln(1e-8), ln(10f64.max(ncol as f64)), 20);
    let delta = grid[1] - grid[0];
    let mut lpv = vec![0.0; grid.len()];
    for (t, &a) in grid.iter().enumerate() {
        lpv[t] = log_posterior(a, y, mu, x, prior, use_cr)?;
    }
    let a_hat = grid[index_max(&lpv)];
    let fine = seq_len_out(a_hat - delta, a_hat + delta, grid.len());
    for (t, &a) in fine.iter().enumerate() {
        lpv[t] = log_posterior(a, y, mu, x, prior, use_cr)?;
    }
    Ok(exp(fine[index_max(&lpv)]))
}

/// One row's result of [`fit_beta`].
#[derive(Clone, Debug)]
pub struct FitBetaRow {
    /// Coefficients on the natural log scale.
    pub beta: Vec<f64>,
    /// Diagonal of the sandwich covariance.
    pub beta_var: Vec<f64>,
    /// Iterations (`maxit` when the fit diverged).
    pub iter: usize,
    /// Hat matrix diagonal per sample.
    pub hat: Vec<f64>,
    /// `contrast' beta`.
    pub contrast_num: f64,
    /// `sqrt(contrast' Sigma contrast)`.
    pub contrast_denom: f64,
    /// Final deviance.
    pub deviance: f64,
}

/// Control of [`fit_beta`].
#[derive(Clone, Debug)]
pub struct FitBetaControl {
    /// Ridge `lambda` per coefficient (natural log scale).
    pub lambda: Vec<f64>,
    /// Contrast vector.
    pub contrast: Vec<f64>,
    /// `tol`.
    pub tol: f64,
    /// `maxit`.
    pub maxit: usize,
    /// `minmu`.
    pub minmu: f64,
}

/// `fitBeta` (useQR = TRUE, no weights) for one row.
pub fn fit_beta(
    y: &[f64],
    nf: &[f64],
    x: &Mat,
    alpha: f64,
    beta_init: &[f64],
    ctl: &FitBetaControl,
) -> Result<FitBetaRow, String> {
    let m = x.nrow;
    let p = x.ncol;
    let large = 30.0;
    let mut beta = beta_init.to_vec();
    let mu_of = |beta: &[f64]| -> Vec<f64> {
        let eta = la::mul_vec(x, beta);
        (0..m).map(|j| (nf[j] * exp(eta[j])).max(ctl.minmu)).collect()
    };
    let mut mu = mu_of(&beta);
    let sqrt_ridge: Vec<f64> = ctl.lambda.iter().map(|l| l.sqrt()).collect();
    let mut dev = 0.0;
    let mut dev_old = 0.0;
    let mut iter = 0usize;
    for t in 0..ctl.maxit {
        iter += 1;
        let w: Vec<f64> = mu.iter().map(|&mi| mi / (1.0 + alpha * mi)).collect();
        let ws: Vec<f64> = w.iter().map(|v| v.sqrt()).collect();
        let mut wxr = Mat::zeros(m + p, p);
        for c in 0..p {
            for j in 0..m {
                *wxr.at_mut(j, c) = x.at(j, c) * ws[j];
            }
            // sqrt(diagmat(lambda)): zero off the diagonal.
            *wxr.at_mut(m + c, c) = sqrt_ridge[c];
        }
        let (q, r) = la::qr_econ(&wxr);
        let mut big = vec![0.0; m + p];
        for j in 0..m {
            let z = ln(mu[j] / nf[j]) + (y[j] - mu[j]) / mu[j];
            big[j] = z * w[j].sqrt();
        }
        let gamma = la::tmul_vec(&q, &big);
        beta = la::solve_upper(&r, &gamma);
        if beta.iter().any(|b| b.abs() > large) {
            iter = ctl.maxit;
            break;
        }
        mu = mu_of(&beta);
        dev = 0.0;
        for j in 0..m {
            dev += -2.0 * dnbinom_mu(y[j], 1.0 / alpha, mu[j], true);
        }
        let conv = (dev - dev_old).abs() / (dev.abs() + 0.1);
        if conv.is_nan() {
            iter = ctl.maxit;
            break;
        }
        if t > 0 && conv < ctl.tol {
            break;
        }
        dev_old = dev;
    }
    let w: Vec<f64> = mu.iter().map(|&mi| mi / (1.0 + alpha * mi)).collect();
    let ws: Vec<f64> = w.iter().map(|v| v.sqrt()).collect();
    let mut xw = x.clone();
    for c in 0..p {
        for j in 0..m {
            *xw.at_mut(j, c) = x.at(j, c) * ws[j];
        }
    }
    let mut xtwx_r = la::xtwx(x, &w);
    for c in 0..p {
        *xtwx_r.at_mut(c, c) += ctl.lambda[c];
    }
    let inv = la::inv(&xtwx_r).map_err(|e| format!("inv(): {}", e.0))?;
    let mut hat = vec![0.0; m];
    for (jp, h) in hat.iter_mut().enumerate() {
        for i1 in 0..p {
            for i2 in 0..p {
                *h += xw.at(jp, i1) * (xw.at(jp, i2) * inv.at(i2, i1));
            }
        }
    }
    // sigma = inv * x' * (x % w) * inv, evaluated by Armadillo as (inv * (x' M)) * inv.
    let mut xm = x.clone();
    for c in 0..p {
        for j in 0..m {
            *xm.at_mut(j, c) = x.at(j, c) * w[j];
        }
    }
    let tmp = la::tmul(x, &xm);
    let sigma = la::mul(&la::mul(&inv, &tmp), &inv);
    let contrast_num = la::direct_dot(&ctl.contrast, &beta);
    // contrast' * sigma * contrast: (c' Sigma) then (.) c, both sequential sums.
    let cs = la::tmul_vec(&sigma, &ctl.contrast);
    let mut q = 0.0;
    for k in 0..p {
        q += cs[k] * ctl.contrast[k];
    }
    let contrast_denom = q.sqrt();
    let beta_var = (0..p).map(|k| sigma.at(k, k)).collect();
    Ok(FitBetaRow {
        beta,
        beta_var,
        iter,
        hat,
        contrast_num,
        contrast_denom,
        deviance: dev,
    })
}
