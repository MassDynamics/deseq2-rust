//! `fitNbinomGLMs` (DESeq2 `R/fitNbinomGLMs.R`) for the paths production reaches: the
//! intercept-only shortcut (LRT reduced `~ 1`) and the IRLS fit `fitBeta` started from the QR
//! least-squares betas; plus `linearModelMu`. Matrices are row-major genes x samples.
//!
//! Rows that did not converge, or with non-finite betas or non-positive variances, go through
//! `fitNbinomGLMsOptim`: R's `optim(method = "L-BFGS-B")` on the negative log posterior
//! ([`crate::lbfgsb`]), then the ridge sandwich standard errors.

use crate::cpp::{fit_beta, FitBetaControl};
use crate::ext;
use crate::gnm::consts::M_LN_SQRT_2PI;
use crate::gnm::{arith::r_pow, dnbinom_mu};
use crate::la::{dgetrf2, solve_upper, Mat};
use crate::lbfgsb::optim_lbfgsb;
use crate::linpack::qr_decompose;

/// `log2(exp(1))` as R computes it (glibc `exp` then `log2`).
pub fn log2e() -> f64 {
    rnum::glibm_log2::log2(rnum::glibm::exp(1.0))
}

/// `log(2)` (glibc).
pub fn ln2() -> f64 {
    rnum::glibm::ln(2.0)
}

/// R's `qr(x)` pieces for a full-rank `x`: `qr.Q` (`m x p`) and `qr.R` (`p x p`), or `None`
/// when the LINPACK rank (tol 1e-7) is below `p`.
pub fn qr_q_r(x: &Mat) -> Option<(Mat, Mat)> {
    let (m, p) = (x.nrow, x.ncol);
    let q = qr_decompose(&x.data, m, p, 1e-7);
    if q.rank < p {
        return None;
    }
    let mut qm = Mat::zeros(m, p);
    for k in 0..p {
        let mut e = vec![0.0; m];
        e[k] = 1.0;
        let col = q.qy(&e);
        for i in 0..m {
            *qm.at_mut(i, k) = col[i];
        }
    }
    let mut r = Mat::zeros(p, p);
    for j in 0..p {
        for i in 0..=j {
            *r.at_mut(i, j) = q.qr[j * m + i];
        }
    }
    Some((qm, r))
}

/// `linearModelMu(y, x) = (y %*% Q) %*% t(x %*% solve(R))` for row-major `y` (`n x m`).
pub fn linear_model_mu(y: &[f64], m: usize, x: &Mat) -> Result<Vec<f64>, String> {
    let p = x.ncol;
    let (q, r) = qr_q_r(x).ok_or("linearModelMu: model matrix is not full rank")?;
    // Rinv = solve(R): back-substitution on each identity column.
    let mut rinv = Mat::zeros(p, p);
    for k in 0..p {
        let mut e = vec![0.0; p];
        e[k] = 1.0;
        let c = solve_upper(&r, &e);
        for i in 0..p {
            *rinv.at_mut(i, k) = c[i];
        }
    }
    // xr = x %*% Rinv (m x p), sequential sums.
    let mut xr = Mat::zeros(m, p);
    for j in 0..m {
        for k in 0..p {
            let mut s = 0.0;
            for l in 0..p {
                s += x.at(j, l) * rinv.at(l, k);
            }
            *xr.at_mut(j, k) = s;
        }
    }
    let n = y.len() / m;
    let mut out = vec![0.0; n * m];
    let mut yq = vec![0.0; p];
    for g in 0..n {
        let row = &y[g * m..(g + 1) * m];
        for (k, v) in yq.iter_mut().enumerate() {
            let mut s = 0.0;
            for j in 0..m {
                s += row[j] * q.at(j, k);
            }
            *v = s;
        }
        for j in 0..m {
            let mut s = 0.0;
            for k in 0..p {
                s += yq[k] * xr.at(j, k);
            }
            out[g * m + j] = s;
        }
    }
    Ok(out)
}

/// Result of [`fit_nbinom_glms`]; per-gene matrices are row-major.
#[derive(Clone, Debug)]
pub struct GlmFit {
    /// Number of coefficients.
    pub p: usize,
    /// Coefficients on the log2 scale (`n x p`).
    pub beta: Vec<f64>,
    /// Standard errors on the log2 scale (`n x p`).
    pub se: Vec<f64>,
    /// Fitted means `sf * exp(X beta)` (`n x m`, not clamped).
    pub mu: Vec<f64>,
    /// Hat matrix diagonals (`n x m`).
    pub hat: Vec<f64>,
    /// Log likelihood per gene (when requested).
    pub log_like: Vec<f64>,
    /// IRLS iterations per gene.
    pub iter: Vec<usize>,
    /// `betaConv`.
    pub conv: Vec<bool>,
}

/// Options for [`fit_nbinom_glms`].
#[derive(Clone, Debug)]
pub struct GlmOptions {
    /// Ridge penalty per coefficient on the log2 scale (`lambda`; default `1e-6`).
    pub lambda: Vec<f64>,
    /// `maxit` (100).
    pub maxit: usize,
    /// Compute the log likelihood with the row dispersions.
    pub log_like: bool,
}

impl GlmOptions {
    /// The defaults for `p` coefficients: `lambda = 1e-6`, `maxit = 100`, log likelihood on.
    pub fn standard(p: usize) -> GlmOptions {
        GlmOptions {
            lambda: vec![1e-6; p],
            maxit: 100,
            log_like: true,
        }
    }
}

/// `fitNbinomGLMs(object, modelMatrix = x, alpha_hat = alpha, lambda)` with size factors `sf`
/// (`useQR = TRUE`, `betaTol = 1e-8`, `minmu = 0.5`). `counts` is row-major `n x m`.
pub fn fit_nbinom_glms(
    counts: &[f64],
    sf: &[f64],
    x: &Mat,
    alpha: &[f64],
    opt: &GlmOptions,
) -> Result<GlmFit, String> {
    let m = x.nrow;
    let p = x.ncol;
    let n = alpha.len();
    assert_eq!(counts.len(), n * m);
    for j in 0..p {
        if x.col(j).iter().all(|v| *v == 0.0) {
            return Err("all(colSums(abs(modelMatrix)) > 0) is not TRUE".into());
        }
    }
    let l2e = log2e();
    let norm: Vec<f64> = (0..n * m).map(|i| counts[i] / sf[i % m]).collect();

    let just_intercept = p == 1 && x.data.iter().all(|v| *v == 1.0);
    if just_intercept && opt.lambda.iter().all(|l| *l <= 1e-6) {
        let mut fit = GlmFit {
            p: 1,
            beta: vec![0.0; n],
            se: vec![0.0; n],
            mu: vec![0.0; n * m],
            hat: vec![0.0; n * m],
            log_like: vec![f64::NAN; n],
            iter: vec![1; n],
            conv: vec![true; n],
        };
        for g in 0..n {
            let b = rnum::glibm_log2::log2(ext::row_mean(&norm[g * m..(g + 1) * m]));
            fit.beta[g] = b;
            let two_b = r_pow(2.0, b);
            let mut w = vec![0.0; m];
            let mut ll = vec![0.0; m];
            for j in 0..m {
                let mu = sf[j] * two_b;
                fit.mu[g * m + j] = mu;
                ll[j] = dnbinom_mu(counts[g * m + j], 1.0 / alpha[g], mu, true);
                w[j] = r_pow(r_pow(mu, -1.0) + alpha[g], -1.0);
            }
            fit.log_like[g] = ext::row_sum(&ll);
            let xtwx = ext::row_sum(&w);
            let sigma = r_pow(xtwx, -1.0);
            fit.se[g] = l2e * sigma.sqrt();
            for j in 0..m {
                fit.hat[g * m + j] = w[j] * r_pow(xtwx, -1.0);
            }
        }
        return Ok(fit);
    }

    // Initial betas (natural log): QR least squares on log(norm + 0.1) when full rank.
    let qr = qr_q_r(x);
    let mut beta_init = vec![0.0; n * p];
    for g in 0..n {
        let row = &norm[g * m..(g + 1) * m];
        match &qr {
            Some((q, r)) => {
                let ly: Vec<f64> = row.iter().map(|v| rnum::glibm::ln(v + 0.1)).collect();
                let qty: Vec<f64> = (0..p)
                    .map(|k| {
                        let mut s = 0.0;
                        for j in 0..m {
                            s += q.at(j, k) * ly[j];
                        }
                        s
                    })
                    .collect();
                beta_init[g * p..(g + 1) * p].copy_from_slice(&solve_upper(r, &qty));
            }
            None => {
                // "Intercept" is always the first column here.
                beta_init[g * p] = rnum::glibm::ln(ext::row_mean(row));
            }
        }
    }
    let ln2 = ln2();
    let lambda_nat: Vec<f64> = opt.lambda.iter().map(|l| l / (ln2 * ln2)).collect();
    let mut contrast = vec![0.0; p];
    contrast[0] = 1.0;
    let ctl = FitBetaControl {
        lambda: lambda_nat,
        contrast,
        tol: 1e-8,
        maxit: opt.maxit,
        minmu: 0.5,
    };
    let mut fit = GlmFit {
        p,
        beta: vec![0.0; n * p],
        se: vec![0.0; n * p],
        mu: vec![0.0; n * m],
        hat: vec![0.0; n * m],
        log_like: vec![f64::NAN; n],
        iter: vec![0; n],
        conv: vec![true; n],
    };
    for g in 0..n {
        let y = &counts[g * m..(g + 1) * m];
        let r = fit_beta(y, sf, x, alpha[g], &beta_init[g * p..(g + 1) * p], &ctl)?;
        let stable = r.beta.iter().all(|b| !b.is_nan());
        // rowVarPositive is `sum(row <= 0) == 0`: NA when a variance is NaN, and `which()`
        // then drops the row unless betaConv or rowStable already selects it.
        let var_nonpos = if r.beta_var.iter().any(|v| v.is_nan()) {
            None
        } else {
            Some(r.beta_var.iter().any(|v| *v <= 0.0))
        };
        let conv = r.iter < opt.maxit;
        if !conv || !stable || var_nonpos == Some(true) {
            let beta_log2: Vec<f64> = r.beta.iter().map(|b| l2e * b).collect();
            let start: &[f64] = if stable && beta_log2.iter().all(|b| b.abs() < 30.0) {
                &beta_log2
            } else {
                &beta_init[g * p..(g + 1) * p]
            };
            let o = optim_row(y, sf, x, alpha[g], &opt.lambda, &ctl.lambda, start, l2e)?;
            fit.beta[g * p..(g + 1) * p].copy_from_slice(&o.beta);
            fit.se[g * p..(g + 1) * p].copy_from_slice(&o.se);
            fit.mu[g * m..(g + 1) * m].copy_from_slice(&o.mu);
            fit.hat[g * m..(g + 1) * m].copy_from_slice(&r.hat);
            if opt.log_like {
                fit.log_like[g] = o.log_like;
            }
            fit.iter[g] = r.iter;
            fit.conv[g] = conv || o.converged;
            continue;
        }
        for j in 0..m {
            let mut eta = 0.0;
            for k in 0..p {
                eta += x.at(j, k) * r.beta[k];
            }
            fit.mu[g * m + j] = sf[j] * rnum::glibm::exp(eta);
            fit.hat[g * m + j] = r.hat[j];
        }
        if opt.log_like {
            let ll: Vec<f64> = (0..m)
                .map(|j| dnbinom_mu(y[j], 1.0 / alpha[g], fit.mu[g * m + j], true))
                .collect();
            fit.log_like[g] = ext::row_sum(&ll);
        }
        for k in 0..p {
            fit.beta[g * p + k] = l2e * r.beta[k];
            fit.se[g * p + k] = l2e * r.beta_var[k].max(0.0).sqrt();
        }
        fit.iter[g] = r.iter;
        fit.conv[g] = conv;
    }
    Ok(fit)
}

/// One row of `fitNbinomGLMsOptim`.
struct OptimRow {
    beta: Vec<f64>,
    se: Vec<f64>,
    mu: Vec<f64>,
    log_like: f64,
    converged: bool,
}

/// `x %*% b` as R runs it for a matrix times a vector (reference `dgemv('N')`).
fn dgemv(x: &Mat, b: &[f64]) -> Vec<f64> {
    let mut y = vec![0.0; x.nrow];
    for (k, bk) in b.iter().enumerate() {
        for (i, yi) in y.iter_mut().enumerate() {
            *yi += bk * x.at(i, k);
        }
    }
    y
}

/// `dnorm(x, 0, sigma, log = TRUE)` (R 4.5 `dnorm4`).
fn dnorm0_log(x: f64, sigma: f64) -> f64 {
    if x.is_nan() || sigma.is_nan() {
        return x + sigma;
    }
    if sigma < 0.0 {
        return f64::NAN;
    }
    if !sigma.is_finite() {
        return f64::NEG_INFINITY;
    }
    if sigma == 0.0 {
        return if x == 0.0 {
            f64::INFINITY
        } else {
            f64::NEG_INFINITY
        };
    }
    let z = (x - 0.0) / sigma;
    if !z.is_finite() {
        return f64::NEG_INFINITY;
    }
    let z = z.abs();
    if z >= 2.0 * f64::MAX.sqrt() {
        return f64::NEG_INFINITY;
    }
    -(M_LN_SQRT_2PI + 0.5 * z * z + rnum::glibm::ln(sigma))
}

/// Reference `dgemm('N','N')`: `C(i,j) = sum_l A(i,l) * B(l,j)`, summed in `l` order from 0.
fn dgemm(a: &Mat, b: &Mat) -> Mat {
    let mut c = Mat::zeros(a.nrow, b.ncol);
    for j in 0..b.ncol {
        for l in 0..a.ncol {
            let temp = b.at(l, j);
            for i in 0..a.nrow {
                *c.at_mut(i, j) += temp * a.at(i, l);
            }
        }
    }
    c
}

/// R `solve(a)`: `La_solve` with `b = diag(n)`, i.e. LAPACK `dgesv` (`dgetrf`, which is
/// `dgetrf2` below the block size 64, then `dgetrs`: row interchanges, unit lower and upper
/// `dtrsm`). The `dgecon` "computationally singular" check R runs afterwards is not ported.
fn r_solve(a: &Mat) -> Result<Mat, String> {
    let n = a.nrow;
    let mut lu = a.clone();
    let mut ipiv = vec![0usize; n];
    let info = dgetrf2(&mut lu, 0, 0, n, n, &mut ipiv);
    if info > 0 {
        return Err(format!(
            "Lapack routine dgesv: system is exactly singular: U[{info},{info}] = 0"
        ));
    }
    let mut b = Mat::zeros(n, n);
    for i in 0..n {
        *b.at_mut(i, i) = 1.0;
    }
    for i in 0..n {
        let ip = ipiv[i];
        if ip != i {
            for j in 0..n {
                let t = b.at(i, j);
                *b.at_mut(i, j) = b.at(ip, j);
                *b.at_mut(ip, j) = t;
            }
        }
    }
    for j in 0..n {
        for k in 0..n {
            let bkj = b.at(k, j);
            if bkj != 0.0 {
                for i in (k + 1)..n {
                    *b.at_mut(i, j) -= bkj * lu.at(i, k);
                }
            }
        }
        for k in (0..n).rev() {
            if b.at(k, j) != 0.0 {
                *b.at_mut(k, j) /= lu.at(k, k);
                let bkj = b.at(k, j);
                for i in 0..k {
                    *b.at_mut(i, j) -= bkj * lu.at(i, k);
                }
            }
        }
    }
    Ok(b)
}

/// `fitNbinomGLMsOptim` for one row (no weights, normalization factors = size factors):
/// L-BFGS-B on `-(sum dnbinom + sum dnorm(beta, 0, sqrt(1/lambda)))` within `[-30, 30]`, then
/// `mu = sf * 2^(x %*% par)`, and the sandwich `solve(X'WX + ridge) X'WX solve(X'WX + ridge)`
/// with `mu` clamped at `minmu = 0.5` for the weights and the log likelihood.
#[allow(clippy::too_many_arguments)]
fn optim_row(
    y: &[f64],
    sf: &[f64],
    x: &Mat,
    alpha: f64,
    lambda: &[f64],
    lambda_nat: &[f64],
    start: &[f64],
    l2e: f64,
) -> Result<OptimRow, String> {
    let m = x.nrow;
    let p = x.ncol;
    let size = 1.0 / alpha;
    let sigma: Vec<f64> = lambda.iter().map(|l| (1.0 / l).sqrt()).collect();
    let mut objective = |b: &[f64]| -> f64 {
        let eta = dgemv(x, b);
        let ll: Vec<f64> = (0..m)
            .map(|j| dnbinom_mu(y[j], size, sf[j] * r_pow(2.0, eta[j]), true))
            .collect();
        let log_like = ext::row_sum(&ll);
        let prior: Vec<f64> = (0..p).map(|k| dnorm0_log(b[k], sigma[k])).collect();
        let log_prior = ext::row_sum(&prior);
        let neg = -(log_like + log_prior);
        if neg.is_finite() {
            neg
        } else {
            1e300
        }
    };
    let o = optim_lbfgsb(&mut objective, start, &vec![-30.0; p], &vec![30.0; p])?;
    let eta = dgemv(x, &o.par);
    let mu: Vec<f64> = (0..m).map(|j| sf[j] * r_pow(2.0, eta[j])).collect();
    let mu_c: Vec<f64> = mu.iter().map(|v| if *v < 0.5 { 0.5 } else { *v }).collect();
    let w: Vec<f64> = mu_c
        .iter()
        .map(|v| r_pow(r_pow(*v, -1.0) + alpha, -1.0))
        .collect();
    // t(x) %*% diag(w) %*% x, left to right, both through dgemm.
    let mut tx = Mat::zeros(p, m);
    let mut wm = Mat::zeros(m, m);
    for j in 0..m {
        for k in 0..p {
            *tx.at_mut(k, j) = x.at(j, k);
        }
        *wm.at_mut(j, j) = w[j];
    }
    let xtwx = dgemm(&dgemm(&tx, &wm), x);
    let mut a = xtwx.clone();
    for k in 0..p {
        *a.at_mut(k, k) += lambda_nat[k];
    }
    let inv = r_solve(&a)?;
    let sig = dgemm(&dgemm(&inv, &xtwx), &inv);
    let mut se = vec![0.0; p];
    for k in 0..p {
        let v = sig.at(k, k);
        let v = if v.is_nan() || v > 0.0 { v } else { 0.0 };
        se[k] = l2e * v.sqrt();
        if se[k].is_nan() {
            return Err("!any(is.na(betaSE)) is not TRUE".into());
        }
    }
    let ll: Vec<f64> = (0..m)
        .map(|j| dnbinom_mu(y[j], size, mu_c[j], true))
        .collect();
    Ok(OptimRow {
        beta: o.par,
        se,
        mu,
        log_like: ext::row_sum(&ll),
        converged: o.convergence == 0,
    })
}
