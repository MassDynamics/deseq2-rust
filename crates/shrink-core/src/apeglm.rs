//! `lfcShrink(type = "apeglm")`: apeglm 1.32.0 with `method = "nbinomCR"` (DESeq2 1.50.2's
//! call), Laplace posterior intervals.
//!
//! The pipeline, in R's order:
//!
//! 1. `priorVar(mle)`: Efron-Morris moment estimate of the prior variance of the natural-log
//!    MLE, by `uniroot` on `[0.001^2, 20^2]`; the Cauchy prior scale is `min(sqrt(pv), 1)`.
//! 2. `nbinomGLM` (C++): every non-zero row is fitted twice by LBFGSpp (RcppNumerical
//!    0.6) from `init1 = (.1, -.1, ..)` and `init2 = (-.1, .1, ..)` on the objective scaled by
//!    `cnst`; rows whose two fits differ by more than 0.01 are flagged as not converged.
//! 3. Row loop (`optimNbinomHess`): the posterior Hessian at the C++ MAP by `optimHess` on
//!    R's `nbinomFn` / `nbinomGr`; rows flagged in (2), or with a non-positive variance
//!    estimate, are refitted by R's `optim(method = "BFGS", hessian = TRUE)`.
//! 4. `sd`, interval, `fsr`, `svalue`; DESeq2 reports `log2(e) * map[coef]`, `log2(e) * sd[coef]`.
//!
//! The C++ objective goes through [`crate::eigen`] so the LBFGSpp path is R's, evaluation for
//! evaluation; the R objective uses R's own operation order (`dgemv`, long-double `sum`).
//! Every `exp` / `log` / `log1p` either side goes through [`rnum::glibm`] and
//! [`rnum::glibm_log1p`] (the reference R's glibc), so the result does not depend on the host libm.
//! Observation weights are not supported (DESeq2 passes them only from a `weights` assay).

#![allow(
    clippy::needless_range_loop,
    clippy::neg_multiply,
    clippy::assign_op_pattern
)]

use crate::dense::{getrf, getrs, rcond_from_lu, Mat};
use crate::eigen;
use crate::xld::{r_cumsum, r_sum};
use crate::ShrinkError;
use rnum::glibm;
use rnum::glibm_log1p::log1p;
use rnum::lbfgsb::{optim_bfgs, optimhess, OptimControl};
use rnum::nmath::{pnorm, qnorm};
use rnum::optim::uniroot;

/// `prior.control$prior.no.shrink.scale`.
pub const SIGMA: f64 = 15.0;

/// `priorVar(mle)` and how it was reached.
#[derive(Debug, Clone)]
pub struct PriorVar {
    pub prior_var: f64,
    pub prior_scale: f64,
    /// `objective(.001^2)`; when negative the prior variance is the lower bound.
    pub objective_at_min: f64,
    /// `uniroot` `iter`, `estim.prec`, `f.root` (0 / NaN / NaN when not run).
    pub iter: i32,
    pub estim_prec: f64,
    pub f_root: f64,
    /// Rows with a non-NA MLE.
    pub kept: usize,
}

/// `apeglm:::priorVar(mle)` on the natural-log MLE and SE. `None` where `uniroot` stops.
pub fn prior_var(mle: &[f64], se: &[f64]) -> Option<PriorVar> {
    let keep: Vec<usize> = (0..mle.len()).filter(|&i| !mle[i].is_nan()).collect();
    let d: Vec<f64> = keep.iter().map(|&i| se[i] * se[i]).collect();
    let s: Vec<f64> = keep.iter().map(|&i| mle[i] * mle[i]).collect();
    let objective = |a: f64| -> f64 {
        let ii: Vec<f64> = d
            .iter()
            .map(|&dk| {
                let t = a + dk;
                1.0 / (2.0 * (t * t))
            })
            .collect();
        let num = r_sum((0..d.len()).map(|k| (s[k] - d[k]) * ii[k]));
        let den = r_sum(ii.iter().copied());
        num / den - a
    };
    let min_var = 0.001 * 0.001;
    let max_var = 20.0 * 20.0;
    let obj_min = objective(min_var);
    let (pv, iter, estim_prec, f_root) = if obj_min < 0.0 {
        (min_var, 0, f64::NAN, f64::NAN)
    } else {
        let f_upper = objective(max_var);
        let z = uniroot(
            objective,
            min_var,
            max_var,
            obj_min,
            f_upper,
            f64::EPSILON.powf(0.25),
            1000,
        )?;
        (z.root, z.iter, z.estim_prec, objective(z.root))
    };
    Some(PriorVar {
        prior_var: pv,
        prior_scale: pv.sqrt().min(1.0),
        objective_at_min: obj_min,
        iter,
        estim_prec,
        f_root,
        kept: keep.len(),
    })
}

/// One row's negative-binomial posterior: design `x` (`n x p`, column-major), counts `y`,
/// `size = 1 / dispersion`, `offset = log(sizeFactors)`, normal prior with scale
/// [`SIGMA`] on `no_shrink` and Cauchy with scale `s` on `shrink` (0-based indices).
pub struct NbRow<'a> {
    pub x: &'a Mat,
    pub y: &'a [f64],
    pub size: f64,
    pub offset: &'a [f64],
    pub s: f64,
    pub no_shrink: &'a [usize],
    pub shrink: &'a [usize],
}

impl NbRow<'_> {
    /// apeglm's C++ `optimFun::f_grad` (`src/nbinomGLM.cpp`), Eigen evaluation order.
    pub fn f_grad_cpp(&self, beta: &[f64], cnst: f64, grad: &mut [f64]) -> f64 {
        let n = self.x.nrow;
        let p = self.x.ncol;
        let sigma2 = SIGMA * SIGMA;
        let s2 = self.s * self.s;
        let size = self.size;
        let xbeta = eigen::gemv_colmajor(&self.x.data, n, p, beta);
        let xbeta_off: Vec<f64> = (0..n).map(|i| xbeta[i] + self.offset[i]).collect();
        let e = eigen::array_exp(&xbeta_off);
        let a: Vec<f64> = self.y.iter().map(|&yi| yi + size).collect();
        let b: Vec<f64> = e.iter().map(|&ei| ei + size).collect();
        let cw: Vec<f64> = (0..n)
            .map(|i| self.y[i] - (a[i] * e[i]) * (1.0 / b[i]))
            .collect();
        let logb = eigen::array_log(&b);
        let dw: Vec<f64> = (0..n)
            .map(|i| self.y[i] * xbeta[i] - a[i] * logb[i])
            .collect();
        let mut neg_prior = 0.0;
        let mut d_neg_prior = vec![0.0; p];
        for &k in self.no_shrink {
            neg_prior += (beta[k] * beta[k]) / (2.0 * sigma2);
            d_neg_prior[k] = beta[k] / sigma2;
        }
        for &k in self.shrink {
            neg_prior += log1p((beta[k] * beta[k]) / s2);
            d_neg_prior[k] = 2.0 * beta[k] / (s2 + beta[k] * beta[k]);
        }
        let f = -1.0 * eigen::sum(&dw) / cnst + neg_prior / cnst + 10.0;
        let d_neg_lik = eigen::gemv_t_rowmajor(&self.x.data, n, p, &cw, -1.0);
        for k in 0..p {
            grad[k] = d_neg_lik[k] / cnst + d_neg_prior[k] / cnst;
        }
        f
    }

    /// `x %*% beta` as R computes it (reference `dgemv`, or the same loop when non-finite).
    fn xbeta_r(&self, beta: &[f64]) -> Vec<f64> {
        let n = self.x.nrow;
        let mut y = vec![0.0; n];
        for (j, &bj) in beta.iter().enumerate() {
            let col = self.x.col(j);
            for i in 0..n {
                y[i] += bj * col[i];
            }
        }
        y
    }

    /// apeglm's R `nbinomFn` (negative log posterior plus `cnst`).
    pub fn nbinom_fn(&self, beta: &[f64], cnst: f64) -> f64 {
        let xbeta = self.xbeta_r(beta);
        let size = self.size;
        let s2 = self.s * self.s;
        let two_sigma2 = 2.0 * (SIGMA * SIGMA);
        let prior = r_sum(
            self.no_shrink
                .iter()
                .map(|&k| -(beta[k] * beta[k]) / two_sigma2),
        ) + r_sum(
            self.shrink
                .iter()
                .map(|&k| -log1p((beta[k] * beta[k]) / s2)),
        );
        let lik = r_sum((0..xbeta.len()).map(|i| {
            let yi = self.y[i];
            yi * xbeta[i] - (yi + size) * glibm::ln(size + glibm::exp(xbeta[i] + self.offset[i]))
        }));
        -lik - prior + cnst
    }

    /// apeglm's R `nbinomGr`.
    pub fn nbinom_gr(&self, beta: &[f64]) -> Vec<f64> {
        let xbeta = self.xbeta_r(beta);
        let n = xbeta.len();
        let p = beta.len();
        let size = self.size;
        let s2 = self.s * self.s;
        let sigma2 = SIGMA * SIGMA;
        let mut prior = vec![0.0; p];
        for &k in self.no_shrink {
            prior[k] = -beta[k] / sigma2;
        }
        for &k in self.shrink {
            prior[k] = -2.0 * beta[k] / (s2 + beta[k] * beta[k]);
        }
        let v: Vec<f64> = (0..n)
            .map(|i| {
                let e = glibm::exp(xbeta[i] + self.offset[i]);
                self.y[i] - (self.y[i] + size) * e / (size + e)
            })
            .collect();
        let mut g = vec![0.0; p];
        for (i, &vi) in v.iter().enumerate() {
            for k in 0..p {
                g[k] += vi * -self.x.at(i, k);
            }
        }
        (0..p).map(|k| g[k] - prior[k]).collect()
    }
}

// ---- LBFGSpp (RcppNumerical 0.6 `optim_lbfgs`) ------------------------------------------

/// What `optim_lbfgs` leaves behind: the parameters and value at exit, the status (0, or
/// -1 when LBFGSpp threw) and the number of objective evaluations.
#[derive(Debug, Clone)]
pub struct LbfgsOut {
    pub x: Vec<f64>,
    pub fx: f64,
    pub status: i32,
    pub nevals: usize,
}

struct BfgsMat {
    m: usize,
    theta: f64,
    s: Vec<Vec<f64>>,
    y: Vec<Vec<f64>>,
    ys: Vec<f64>,
    alpha: Vec<f64>,
    ncorr: usize,
    ptr: usize,
}

impl BfgsMat {
    fn new(n: usize, m: usize) -> BfgsMat {
        BfgsMat {
            m,
            theta: 1.0,
            s: vec![vec![0.0; n]; m],
            y: vec![vec![0.0; n]; m],
            ys: vec![0.0; m],
            alpha: vec![0.0; m],
            ncorr: 0,
            ptr: m,
        }
    }

    fn add_correction(&mut self, s: &[f64], y: &[f64]) {
        let loc = self.ptr % self.m;
        self.s[loc].copy_from_slice(s);
        self.y[loc].copy_from_slice(y);
        let ys = eigen::dot(&self.s[loc], &self.y[loc]);
        self.ys[loc] = ys;
        self.theta = eigen::squared_norm(&self.y[loc]) / ys;
        if self.ncorr < self.m {
            self.ncorr += 1;
        }
        self.ptr = loc + 1;
    }

    fn apply_hv(&mut self, v: &[f64], a: f64, res: &mut [f64]) {
        let m = self.m;
        for i in 0..v.len() {
            res[i] = a * v[i];
        }
        let mut j = self.ptr % m;
        for _ in 0..self.ncorr {
            j = (j + m - 1) % m;
            self.alpha[j] = eigen::dot(&self.s[j], res) / self.ys[j];
            for i in 0..res.len() {
                res[i] -= self.alpha[j] * self.y[j][i];
            }
        }
        for r in res.iter_mut() {
            *r /= self.theta;
        }
        for _ in 0..self.ncorr {
            let beta = eigen::dot(&self.y[j], res) / self.ys[j];
            for i in 0..res.len() {
                res[i] += (self.alpha[j] - beta) * self.s[j][i];
            }
            j = (j + 1) % m;
        }
    }
}

struct LsThrow;

const FTOL: f64 = 1e-4;
const WOLFE: f64 = 0.9;
const MAX_LINESEARCH: usize = 100;

fn quad_interp(step_lo: f64, step_hi: f64, fx_lo: f64, fx_hi: f64, dg_lo: f64) -> f64 {
    let fdiff = fx_hi - fx_lo;
    let sdiff = step_hi - step_lo;
    let smid = (step_hi + step_lo) / 2.0;
    let mut cand = fdiff * step_lo - smid * sdiff * dg_lo;
    cand = cand / (fdiff - sdiff * dg_lo);
    let nan = !cand.is_finite();
    let end_dist = (cand - step_lo).abs().min((cand - step_hi).abs());
    let near_end = end_dist < 0.01 * sdiff.abs();
    let bisect = nan || cand <= step_lo.min(step_hi) || cand >= step_lo.max(step_hi) || near_end;
    if bisect {
        smid
    } else {
        cand
    }
}

/// `LineSearchNocedalWright::LineSearch`. `x`, `fx`, `grad`, `dg`, `step` are the solver's
/// own variables (C++ references); an `Err` is a thrown exception, with them as left.
#[allow(clippy::too_many_arguments)]
fn line_search<F: FnMut(&[f64], &mut [f64]) -> f64>(
    f: &mut F,
    xp: &[f64],
    drt: &[f64],
    step: &mut f64,
    fx: &mut f64,
    grad: &mut Vec<f64>,
    dg: &mut f64,
    x: &mut Vec<f64>,
) -> Result<(), LsThrow> {
    if *step <= 0.0 {
        return Err(LsThrow);
    }
    let fx_init = *fx;
    let dg_init = *dg;
    if dg_init > 0.0 {
        return Err(LsThrow);
    }
    let test_decr = FTOL * dg_init;
    let test_curv = -WOLFE * dg_init;
    let mut step_hi;
    let mut fx_hi;
    let mut step_lo = 0.0;
    let mut fx_lo = fx_init;
    let mut dg_lo = dg_init;
    let mut x_lo = xp.to_vec();
    let mut grad_lo = grad.clone();
    let mut iter = 0;
    let n = xp.len();
    loop {
        for i in 0..n {
            x[i] = xp[i] + *step * drt[i];
        }
        *fx = f(x, grad);
        *dg = eigen::dot(grad, drt);
        if *fx - fx_init > *step * test_decr || (0.0 < step_lo && *fx >= fx_lo) {
            step_hi = *step;
            fx_hi = *fx;
            break;
        }
        if dg.abs() <= test_curv {
            return Ok(());
        }
        step_hi = step_lo;
        fx_hi = fx_lo;
        step_lo = *step;
        fx_lo = *fx;
        dg_lo = *dg;
        std::mem::swap(&mut x_lo, x);
        std::mem::swap(&mut grad_lo, grad);
        if *dg >= 0.0 {
            break;
        }
        iter += 1;
        if iter >= MAX_LINESEARCH {
            std::mem::swap(x, &mut x_lo);
            std::mem::swap(grad, &mut grad_lo);
            return Ok(());
        }
        *step *= 2.0;
    }
    loop {
        *step = quad_interp(step_lo, step_hi, fx_lo, fx_hi, dg_lo);
        for i in 0..n {
            x[i] = xp[i] + *step * drt[i];
        }
        *fx = f(x, grad);
        *dg = eigen::dot(grad, drt);
        if *fx - fx_init > *step * test_decr || *fx >= fx_lo {
            if *step == step_hi {
                return Err(LsThrow);
            }
            step_hi = *step;
            fx_hi = *fx;
        } else {
            if dg.abs() <= test_curv {
                return Ok(());
            }
            if *dg * (step_hi - step_lo) >= 0.0 {
                step_hi = step_lo;
                fx_hi = fx_lo;
            }
            if *step == step_lo {
                return Err(LsThrow);
            }
            step_lo = *step;
            fx_lo = *fx;
            dg_lo = *dg;
            std::mem::swap(&mut x_lo, x);
            std::mem::swap(&mut grad_lo, grad);
        }
        iter += 1;
        if iter >= MAX_LINESEARCH {
            if step_lo <= 0.0 {
                return Err(LsThrow);
            }
            *step = step_lo;
            *fx = fx_lo;
            *dg = dg_lo;
            std::mem::swap(x, &mut x_lo);
            std::mem::swap(grad, &mut grad_lo);
            return Ok(());
        }
    }
}

/// `LBFGSSolver::minimize`; `x` and `fx` are the caller's (C++ references).
fn minimize<F: FnMut(&[f64], &mut [f64]) -> f64>(
    f: &mut F,
    x: &mut Vec<f64>,
    fx: &mut f64,
    maxit: usize,
    eps_f: f64,
    eps_g: f64,
) -> Result<(), LsThrow> {
    let n = x.len();
    let mut bfgs = BfgsMat::new(n, 6);
    let mut grad = vec![0.0; n];
    *fx = f(x, &mut grad);
    let gnorm = eigen::norm(&grad);
    let mut m_fx = *fx;
    if gnorm <= eps_g || gnorm <= eps_g * eigen::norm(x) {
        return Ok(());
    }
    let mut drt: Vec<f64> = grad.iter().map(|g| -g).collect();
    let mut step = 1.0 / eigen::norm(&drt);
    let mut k = 1usize;
    let mut vecs = vec![0.0; n];
    let mut vecy = vec![0.0; n];
    loop {
        let xp = x.clone();
        let gradp = grad.clone();
        let mut dg = eigen::dot(&grad, &drt);
        line_search(f, &xp, &drt, &mut step, fx, &mut grad, &mut dg, x)?;
        let gnorm = eigen::norm(&grad);
        if gnorm <= eps_g || gnorm <= eps_g * eigen::norm(x) {
            return Ok(());
        }
        let fxd = m_fx;
        if (fxd - *fx).abs() <= eps_f * fx.abs().max(fxd.abs()).max(1.0) {
            return Ok(());
        }
        m_fx = *fx;
        if maxit != 0 && k >= maxit {
            return Ok(());
        }
        for i in 0..n {
            vecs[i] = x[i] - xp[i];
            vecy[i] = grad[i] - gradp[i];
        }
        if eigen::dot(&vecs, &vecy) > f64::EPSILON * eigen::squared_norm(&vecy) {
            bfgs.add_correction(&vecs, &vecy);
        }
        bfgs.apply_hv(&grad, -1.0, &mut drt);
        step = 1.0;
        k += 1;
    }
}

/// RcppNumerical `optim_lbfgs(f, x, fx_opt, maxit, eps_f, eps_g)`: LBFGSpp's
/// `LBFGSSolver<double, LineSearchNocedalWright>` with `m = 6`, `past = 1`,
/// `delta = eps_f`, `epsilon = epsilon_rel = eps_g`, `max_linesearch = 100`. A thrown
/// exception gives status -1 with `x` and `fx` as the solver left them.
pub fn optim_lbfgs<F: FnMut(&[f64], &mut [f64]) -> f64>(
    mut f: F,
    init: &[f64],
    maxit: usize,
    eps_f: f64,
    eps_g: f64,
) -> LbfgsOut {
    let mut nevals = 0usize;
    let mut fc = |x: &[f64], g: &mut [f64]| {
        nevals += 1;
        f(x, g)
    };
    let mut x = init.to_vec();
    let mut fx = f64::NAN;
    let r = minimize(&mut fc, &mut x, &mut fx, maxit, eps_f, eps_g);
    LbfgsOut {
        x,
        fx,
        status: if r.is_ok() { 0 } else { -1 },
        nevals,
    }
}

// ---- the row pass and the public entry point ------------------------------------------

/// The C++ prefit of one non-zero row (`nbinomCppRoutine`).
#[derive(Debug, Clone)]
pub struct PrefitRow {
    /// `nbinomFn(rep(0, p), cnst = 0)` and its floor at 1.
    pub cnst_raw: f64,
    pub cnst: f64,
    pub fit1: LbfgsOut,
    pub fit2: LbfgsOut,
    /// `max |fit1 - fit2|`; `conv` is fit1's status, or -1 where `delta > 0.01`.
    pub delta: f64,
    pub conv: i32,
}

/// `optimNbinomHess` for one non-zero row.
#[derive(Debug, Clone)]
pub struct RowPass {
    pub nan_prefit: bool,
    pub init: Vec<f64>,
    pub cnst2: f64,
    /// `-optimHess` at the prefit and `diag(-solve(.))`, when the prefit converged.
    pub hess: Option<Vec<f64>>,
    pub var_est: Option<Vec<f64>>,
    pub fallback: bool,
    /// The R `optim(method = "BFGS")` refit: counts, convergence, value.
    pub fb_fncount: Option<i32>,
    pub fb_grcount: Option<i32>,
    pub fb_conv: Option<i32>,
    pub fb_value: Option<f64>,
    /// The Hessian the posterior SD comes from (column-major `p x p`).
    pub final_hess: Vec<f64>,
}

/// `lfcShrink(type = "apeglm")` output: the apeglm fit (natural log) and DESeq2's columns.
#[derive(Debug, Clone)]
pub struct ApeglmFit {
    pub prior: PriorVar,
    /// 0-based; the shrunk coefficient is the only one not in this list.
    pub no_shrink: Vec<usize>,
    pub basemean: Vec<f64>,
    pub prefit: Vec<Option<PrefitRow>>,
    pub rows: Vec<Option<RowPass>>,
    /// `G x p`, column-major.
    pub map: Mat,
    pub sd: Mat,
    pub interval_lo: Vec<f64>,
    pub interval_hi: Vec<f64>,
    pub fsr: Vec<f64>,
    pub svalue: Vec<f64>,
    pub diag_conv: Vec<f64>,
    pub diag_count: Vec<f64>,
    /// DESeq2's `log2FoldChange`, `lfcSE`; the interval on the log2 scale.
    pub log2_fold_change: Vec<f64>,
    pub lfc_se: Vec<f64>,
    pub cri_left: Vec<f64>,
    pub cri_right: Vec<f64>,
}

/// R `solve(a)` (`dgesv` against the identity, then the `dgecon` singularity check).
fn r_solve(a: &Mat) -> Result<Mat, ShrinkError> {
    let n = a.nrow;
    let (f, ipiv, info) = getrf(a);
    if info > 0 {
        return Err(ShrinkError::Numerical(format!(
            "Lapack routine dgesv: system is exactly singular: U[{info},{info}] = 0"
        )));
    }
    let rcond = rcond_from_lu(a, &f, &ipiv);
    if rcond < f64::EPSILON {
        return Err(ShrinkError::Numerical(format!(
            "system is computationally singular: reciprocal condition number = {rcond:e}"
        )));
    }
    let mut inv = Mat::zeros(n, n);
    for j in 0..n {
        let mut e = vec![0.0; n];
        e[j] = 1.0;
        getrs(&f, &ipiv, &mut e);
        inv.col_mut(j).copy_from_slice(&e);
    }
    Ok(inv)
}

fn neg(v: &[f64]) -> Vec<f64> {
    v.iter().map(|x| -1.0 * x).collect()
}

fn optim_err(e: rnum::lbfgsb::OptimError) -> ShrinkError {
    ShrinkError::Numerical(e.to_string())
}

/// `apeglm:::svalue(lfsr)`.
pub fn svalue(lfsr: &[f64]) -> Vec<f64> {
    let n = lfsr.len();
    let mut ord: Vec<usize> = (0..n).collect();
    // sort(na.last = TRUE) and rank(ties.method = "first", na.last = TRUE): a stable sort
    // with NA last.
    ord.sort_by(|&a, &b| match (lfsr[a].is_nan(), lfsr[b].is_nan()) {
        (false, false) => lfsr[a].partial_cmp(&lfsr[b]).unwrap(),
        (true, false) => std::cmp::Ordering::Greater,
        (false, true) => std::cmp::Ordering::Less,
        (true, true) => std::cmp::Ordering::Equal,
    });
    let sorted: Vec<f64> = ord.iter().map(|&i| lfsr[i]).collect();
    let mut cs = r_cumsum(&sorted);
    if let Some(first_na) = sorted.iter().position(|v| v.is_nan()) {
        for v in cs.iter_mut().skip(first_na) {
            *v = f64::NAN;
        }
    }
    let mut out = vec![f64::NAN; n];
    for (pos, &i) in ord.iter().enumerate() {
        out[i] = cs[pos] / (pos + 1) as f64;
    }
    out
}

/// `lfcShrink(dds, coef, type = "apeglm")` from the unshrunken fit, as plain arrays:
///
/// - `counts`: `G x n` raw counts; `size_factors`: length `n`; `dispersions`: length `G`.
/// - `design`: the `n x p` model matrix; `coef`: 0-based column to shrink.
/// - `lfc_mle`, `lfc_se`: DESeq2's unshrunken `log2FoldChange` and `lfcSE` for `coef`.
#[allow(clippy::too_many_arguments)]
pub fn shrink_apeglm(
    counts: &Mat,
    size_factors: &[f64],
    dispersions: &[f64],
    design: &Mat,
    coef: usize,
    lfc_mle: &[f64],
    lfc_se: &[f64],
) -> Result<ApeglmFit, ShrinkError> {
    let g = counts.nrow;
    let n = counts.ncol;
    let p = design.ncol;
    if design.nrow != n || size_factors.len() != n {
        return Err(ShrinkError::InvalidInput(
            "design / size factors do not match the counts".into(),
        ));
    }
    if dispersions.len() != g || lfc_mle.len() != g || lfc_se.len() != g {
        return Err(ShrinkError::InvalidInput(
            "per-gene inputs do not match the counts".into(),
        ));
    }
    if coef >= p {
        return Err(ShrinkError::InvalidInput(format!(
            "coef {coef} out of range for {p} columns"
        )));
    }
    let ln2 = std::f64::consts::LN_2;
    let mle: Vec<f64> = lfc_mle.iter().map(|v| ln2 * v).collect();
    let se: Vec<f64> = lfc_se.iter().map(|v| ln2 * v).collect();
    let prior = prior_var(&mle, &se).ok_or_else(|| {
        ShrinkError::Numerical("priorVar: f() values at end points not of opposite sign".into())
    })?;
    let s = prior.prior_scale;
    let no_shrink: Vec<usize> = (0..p).filter(|&k| k != coef).collect();
    let shrink = vec![coef];
    let offset: Vec<f64> = size_factors.iter().map(|&v| glibm::ln(v)).collect();

    let intercept_idx: Vec<usize> = (0..n)
        .filter(|&i| (0..p).filter(|&k| design.at(i, k) == 0.0).count() == p - 1)
        .collect();
    let cols: Vec<usize> = if intercept_idx.is_empty() {
        (0..n).collect()
    } else {
        intercept_idx
    };
    let mut ys: Vec<Vec<f64>> = Vec::with_capacity(g);
    let mut basemean = vec![0.0; g];
    for i in 0..g {
        let y: Vec<f64> = (0..n).map(|j| counts.at(i, j)).collect();
        basemean[i] = r_sum(cols.iter().map(|&j| y[j])) / cols.len() as f64;
        ys.push(y);
    }

    let init1: Vec<f64> = (0..p)
        .map(|k| if k % 2 == 0 { 0.1 } else { -0.1 })
        .collect();
    let init2: Vec<f64> = (0..p)
        .map(|k| if k % 2 == 0 { -0.1 } else { 0.1 })
        .collect();
    let mut prefit: Vec<Option<PrefitRow>> = vec![None; g];
    let mut rows: Vec<Option<RowPass>> = vec![None; g];
    let mut map = Mat::zeros(g, p);
    let mut sd = Mat::zeros(g, p);
    map.data.iter_mut().for_each(|v| *v = f64::NAN);
    sd.data.iter_mut().for_each(|v| *v = f64::NAN);
    let mut interval_lo = vec![f64::NAN; g];
    let mut interval_hi = vec![f64::NAN; g];
    let mut fsr = vec![f64::NAN; g];
    let mut diag_conv = vec![f64::NAN; g];
    let mut diag_count = vec![f64::NAN; g];
    let qn = qnorm((1.0 - 0.95) / 2.0, 0.0, 1.0, false, false);
    let ctl = OptimControl::default();

    for i in 0..g {
        let y = &ys[i];
        if r_sum(y.iter().copied()) <= 0.0 {
            continue;
        }
        let row = NbRow {
            x: design,
            y,
            size: 1.0 / dispersions[i],
            offset: &offset,
            s,
            no_shrink: &no_shrink,
            shrink: &shrink,
        };
        // nbinomCppRoutine
        let cnst_raw = row.nbinom_fn(&vec![0.0; p], 0.0);
        let cnst = if cnst_raw > 1.0 { cnst_raw } else { 1.0 };
        let fit =
            |init: &[f64]| optim_lbfgs(|b, gr| row.f_grad_cpp(b, cnst, gr), init, 300, 1e-8, 1e-8);
        let fit1 = fit(&init1);
        let fit2 = fit(&init2);
        let mut delta = f64::NEG_INFINITY;
        for k in 0..p {
            let d = (fit1.x[k] - fit2.x[k]).abs();
            if d.is_nan() || delta.is_nan() {
                delta = f64::NAN;
            } else if d > delta {
                delta = d;
            }
        }
        let conv = if delta > 0.01 { -1 } else { fit1.status };
        let prefit_beta = fit1.x.clone();
        prefit[i] = Some(PrefitRow {
            cnst_raw,
            cnst,
            fit1,
            fit2,
            delta,
            conv,
        });

        // optimNbinomHess
        let nan_prefit = prefit_beta.iter().any(|v| v.is_nan());
        let init: Vec<f64> = if nan_prefit {
            let mut v: Vec<f64> = (0..p)
                .map(|k| if k % 2 == 0 { 1.0 } else { -1.0 })
                .collect();
            v[0] = if basemean[i] == 0.0 {
                0.0
            } else {
                glibm::ln(basemean[i])
            };
            v
        } else {
            prefit_beta
        };
        let cnst2 = -row.nbinom_fn(&init, 0.0) - 1.0;
        let gr = |b: &[f64]| row.nbinom_gr(b);
        let mut hess = None;
        let mut var_est: Option<Vec<f64>> = None;
        if conv == 0 {
            let h = neg(&optimhess(&init, gr, &ctl).map_err(optim_err)?);
            let inv = r_solve(&Mat::from_col_major(p, p, h.clone()))?;
            var_est = Some((0..p).map(|k| -inv.at(k, k)).collect());
            hess = Some(h);
        }
        let ve_bad = match &var_est {
            Some(v) => {
                if v.iter().any(|x| x.is_nan()) {
                    return Err(ShrinkError::Numerical(
                        "missing value where TRUE/FALSE needed (var.est)".into(),
                    ));
                }
                v.iter().any(|&x| x <= 0.0)
            }
            None => false,
        };
        let fallback = conv != 0 || ve_bad;
        let mut rp = RowPass {
            nan_prefit,
            init: init.clone(),
            cnst2,
            hess: hess.clone(),
            var_est,
            fallback,
            fb_fncount: None,
            fb_grcount: None,
            fb_conv: None,
            fb_value: None,
            final_hess: vec![],
        };
        let (par, final_hess, dconv, dcount) = if fallback {
            let o = optim_bfgs(&init, |b| row.nbinom_fn(b, cnst2), gr, &ctl).map_err(optim_err)?;
            let h = neg(&optimhess(&o.par, gr, &ctl).map_err(optim_err)?);
            rp.fb_fncount = Some(o.fncount);
            rp.fb_grcount = Some(o.grcount);
            rp.fb_conv = Some(o.convergence);
            rp.fb_value = Some(o.value);
            (o.par, h, o.convergence as f64, o.fncount as f64)
        } else {
            (init, hess.unwrap(), 0.0, f64::NAN)
        };
        rp.final_hess = final_hess.clone();
        for k in 0..p {
            map.set(i, k, par[k]);
        }
        let inv = r_solve(&Mat::from_col_major(p, p, final_hess))?;
        let cov_diag: Vec<f64> = (0..p).map(|k| -inv.at(k, k)).collect();
        rows[i] = Some(rp);
        if cov_diag.iter().any(|x| x.is_nan()) {
            return Err(ShrinkError::Numerical(
                "missing value where TRUE/FALSE needed (cov.mat)".into(),
            ));
        }
        if cov_diag.iter().any(|&x| x <= 0.0) {
            continue;
        }
        let sdv: Vec<f64> = cov_diag.iter().map(|x| x.sqrt()).collect();
        for k in 0..p {
            sd.set(i, k, sdv[k]);
        }
        interval_lo[i] = par[coef] - qn * sdv[coef];
        interval_hi[i] = par[coef] + qn * sdv[coef];
        fsr[i] = pnorm(-par[coef].abs(), 0.0, sdv[coef], true, false);
        diag_conv[i] = dconv;
        diag_count[i] = dcount;
    }
    let sval = svalue(&fsr);
    let l2e = std::f64::consts::LOG2_E;
    let log2_fold_change: Vec<f64> = (0..g).map(|i| l2e * map.at(i, coef)).collect();
    let lfc_se_out: Vec<f64> = (0..g).map(|i| l2e * sd.at(i, coef)).collect();
    let cri_left: Vec<f64> = interval_lo.iter().map(|v| l2e * v).collect();
    let cri_right: Vec<f64> = interval_hi.iter().map(|v| l2e * v).collect();
    Ok(ApeglmFit {
        prior,
        no_shrink,
        basemean,
        prefit,
        rows,
        map,
        sd,
        interval_lo,
        interval_hi,
        fsr,
        svalue: sval,
        diag_conv,
        diag_count,
        log2_fold_change,
        lfc_se: lfc_se_out,
        cri_left,
        cri_right,
    })
}
