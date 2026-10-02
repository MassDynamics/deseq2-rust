//! `fitNbinomGLMs` (DESeq2 `R/fitNbinomGLMs.R`) for the paths production reaches: the
//! intercept-only shortcut (LRT reduced `~ 1`) and the IRLS fit `fitBeta` started from the QR
//! least-squares betas; plus `linearModelMu`. Matrices are row-major genes x samples.
//!
//! The `fitNbinomGLMsOptim` fallback (rows that did not converge, or with non-finite betas or
//! non-positive variances) is not ported: such rows return an error naming the row.

use crate::cpp::{fit_beta, FitBetaControl};
use crate::ext;
use crate::gnm::{arith::r_pow, dnbinom_mu};
use crate::la::{solve_upper, Mat};
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
        let var_pos = r.beta_var.iter().all(|v| *v > 0.0);
        let conv = r.iter < opt.maxit;
        if !(conv && stable && var_pos) {
            return Err(format!(
                "fitNbinomGLMs: row {} needs the optim fallback (betaConv {conv}, finite betas {stable}, positive variances {var_pos}), which is not ported",
                g + 1
            ));
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
