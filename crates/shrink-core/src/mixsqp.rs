//! Port of mixsqp 0.3-54 (`mixsqp()` wrapper + `mixem.cpp` + `mixsqp.cpp`, full-L path) with
//! the control values ashr 2.2-63 passes (`eps = 1e-6`, `numiter.em = 20`, the rest default).
//!
//! The arithmetic follows the package's Armadillo expressions in the same order (see
//! `dense.rs` for the BLAS / LAPACK kernels). The one deliberate departure is the
//! ill-conditioned branch of `solve()`, see `linalg::approx_min_norm`.

use crate::dense::{arma_accu, arma_dot, chol_upper_ok, gemv_n, gemv_t, syrk_t, Mat};
use crate::linalg::{arma_solve, jacobi_eigen, SolveRoute};
use crate::xld::r_sum;
use crate::ShrinkError;

/// Fixed control values (mixsqp defaults, overridden by ashr where noted).
pub const CONVTOL_SQP: f64 = 1e-8;
pub const CONVTOL_ACTIVESET: f64 = 1e-10;
pub const ZERO_THRESHOLD_SOLUTION: f64 = 1e-8;
pub const ZERO_THRESHOLD_SEARCHDIR: f64 = 1e-14;
pub const SUFFDECR: f64 = 0.01;
pub const STEPSIZEREDUCE: f64 = 0.75;
pub const MINSTEPSIZE: f64 = 1e-8;
pub const IDENTITY_CONTRIB_INCREASE: f64 = 10.0;
pub const MAXITER_SQP: usize = 1000;
/// ashr's `control$eps`.
pub const EPS_ASHR: f64 = 1e-6;
/// ashr's `control$numiter.em`.
pub const NUMITER_EM: usize = 20;
/// mixsqp `tol.svd` default.
pub const TOL_SVD: f64 = 1e-6;

/// One active-set iteration, for comparison with the R trace (`_ashr_trace_qp.csv`).
#[derive(Clone, Debug)]
pub struct QpStep {
    pub sqp_iter: usize,
    pub qp_iter: usize,
    pub n_ws: usize,
    pub a_corr: f64,
    pub route: SolveRoute,
    pub rcond: f64,
    pub pnorm_inf: f64,
    /// 0 zero direction + add constraint, 1 step, 2/3 converged.
    pub kind: u8,
    /// Index added (kind 0) or blocking (kind 1), -1 if none.
    pub k: i64,
    /// Step length (1 unless blocked).
    pub step: f64,
    /// The system solved, `B p = rhs` (B column-major), its solution, and y after the step
    /// (`_ashr_trace_solve.bin`).
    pub b: Vec<f64>,
    pub rhs: Vec<f64>,
    pub p: Vec<f64>,
    pub y: Vec<f64>,
}

/// One SQP iteration (`_ashr_trace_sqp.csv`).
#[derive(Clone, Debug)]
pub struct SqpStep {
    pub x_em: Vec<f64>,
    pub obj: f64,
    pub gmin: f64,
    pub y: Option<Vec<f64>>,
    pub step: Option<f64>,
    pub nqp: usize,
    pub nls: usize,
}

#[derive(Clone, Debug)]
pub struct MixsqpResult {
    /// `x / sum(x)` as returned by `mixsqp()`.
    pub x: Vec<f64>,
    /// Iterates of the 20 initial EM updates.
    pub em_iterates: Vec<Vec<f64>>,
    pub sqp: Vec<SqpStep>,
    pub qp: Vec<QpStep>,
    pub converged: bool,
    /// Gradient of the objective at the returned `x` (before the final renormalisation),
    /// and its `1 + min_{j: x_j > zts} g_j` convergence statistic.
    pub grad: Vec<f64>,
    pub gmin: f64,
}

fn mixem_update(l: &Mat, w: &[f64], x: &mut Vec<f64>) {
    let e = 1e-15;
    let (n, m) = (l.nrow, l.ncol);
    let mut p = l.clone();
    for c in 0..m {
        let s = x[c] + e;
        for v in p.col_mut(c) {
            *v *= s;
        }
    }
    // normalizerowsbymax
    let mut rmax = p.col(0).to_vec();
    for c in 1..m {
        for (r, v) in p.col(c).iter().enumerate() {
            if *v > rmax[r] {
                rmax[r] = *v;
            }
        }
    }
    for c in 0..m {
        for (r, v) in p.col_mut(c).iter_mut().enumerate() {
            *v /= rmax[r];
            *v += e;
        }
    }
    // normalizerows: sum(A,1) column-sequential
    let mut rs = p.col(0).to_vec();
    for c in 1..m {
        for (r, v) in p.col(c).iter().enumerate() {
            rs[r] += *v;
        }
    }
    for c in 0..m {
        for (r, v) in p.col_mut(c).iter_mut().enumerate() {
            *v /= rs[r];
        }
    }
    let _ = n;
    *x = gemv_t(&p, w, 1.0);
}

fn obj(l: &Mat, w: &[f64], x: &[f64], z: &[f64], e: &[f64]) -> Result<f64, ShrinkError> {
    let mut u = gemv_n(l, x);
    for (ui, ei) in u.iter_mut().zip(e) {
        *ui += ei;
    }
    let umin = u.iter().cloned().fold(f64::INFINITY, f64::min);
    if umin <= 0.0 {
        return Err(ShrinkError::Numerical("mixsqp: objective is -Inf".into()));
    }
    Ok(-arma_accu((0..u.len()).map(|i| w[i] * (z[i] + rnum::glibm::ln(u[i])))))
}

fn compute_grad(l: &Mat, w: &[f64], x: &[f64], e: &[f64]) -> (Vec<f64>, Mat) {
    let mut u = gemv_n(l, x);
    for (ui, ei) in u.iter_mut().zip(e) {
        *ui += ei;
    }
    let wu: Vec<f64> = w.iter().zip(&u).map(|(a, b)| a / b).collect();
    let g = gemv_t(l, &wu, -1.0);
    let mut z = l.clone();
    let sc: Vec<f64> = w.iter().zip(&u).map(|(a, b)| a.sqrt() / b).collect();
    for c in 0..z.ncol {
        for (r, v) in z.col_mut(c).iter_mut().enumerate() {
            *v *= sc[r];
        }
    }
    (g, syrk_t(&z))
}

fn feasible_stepsize(x: &[f64], p: &[f64]) -> (Option<usize>, f64) {
    let idx: Vec<usize> = (0..p.len()).filter(|&i| p[i] < 0.0).collect();
    if idx.is_empty() {
        return (None, 1.0);
    }
    let t: Vec<f64> = idx.iter().map(|&i| -x[i] / p[i]).collect();
    let mut jm = 0;
    for k in 1..t.len() {
        if t[k] < t[jm] {
            jm = k;
        }
    }
    let a = if t[jm] < 1.0 { t[jm] } else { 1.0 };
    (Some(idx[jm]), a)
}

fn searchdir(h: &Mat, y: &[f64], ainc: f64) -> (Vec<f64>, f64, SolveRoute, f64, Mat, Vec<f64>) {
    let (a0, amax) = (1e-15, 1e15);
    let n = y.len();
    let d = (0..n).map(|i| h.at(i, i)).fold(f64::INFINITY, f64::min);
    let mut a = if d > a0 { 0.0 } else { a0 - d };
    let mut b;
    loop {
        b = h.clone();
        for i in 0..n {
            let v = b.at(i, i) + a;
            b.set(i, i, v);
        }
        if a * ainc > amax {
            break;
        } else if chol_upper_ok(&b) {
            break;
        } else if a <= 0.0 {
            a = a0;
        } else {
            a *= ainc;
        }
    }
    let rhs: Vec<f64> = y.iter().map(|v| -v).collect();
    let (p, route, rc) = arma_solve(&b, &rhs);
    (p, a, route, rc, b, rhs)
}

fn activesetqp(
    h: &Mat,
    g: &[f64],
    y: &mut [f64],
    maxiter: usize,
    sqp_iter: usize,
    trace: &mut Vec<QpStep>,
) -> usize {
    let m = g.len();
    let mut t: Vec<bool> = y.iter().map(|v| *v > 0.0).collect();
    let mut iter = 0;
    while iter < maxiter {
        let i: Vec<usize> = (0..m).filter(|&k| t[k]).collect();
        let j: Vec<usize> = (0..m).filter(|&k| !t[k]).collect();
        for &k in &j {
            y[k] = 0.0;
        }
        let hs = h.submat(&i);
        let yi: Vec<f64> = i.iter().map(|&k| y[k]).collect();
        let hy = gemv_n(&hs, &yi);
        let mut b = g.to_vec();
        for (c, &k) in i.iter().enumerate() {
            b[k] += hy[c];
        }
        let bs: Vec<f64> = i.iter().map(|&k| b[k]).collect();
        let (ps, a_corr, route, rcond, bmat, rhs) = searchdir(&hs, &bs, IDENTITY_CONTRIB_INCREASE);
        let dbg = |y: &[f64]| (bmat.data.clone(), rhs.clone(), ps.clone(), y.to_vec());
        let mut p = vec![0.0; m];
        for (c, &k) in i.iter().enumerate() {
            p[k] = ps[c];
        }
        let pn = p.iter().fold(0.0_f64, |acc, v| acc.max(v.abs()));
        let kind;
        let k_rec: i64;
        let mut a_rec = 1.0;
        if pn <= ZERO_THRESHOLD_SEARCHDIR {
            kind = 0;
            let hyf = gemv_n(h, y);
            let bb: Vec<f64> = g.iter().zip(&hyf).map(|(a, b)| a + b).collect();
            let kk = if j.is_empty() {
                None
            } else {
                let mut jm = 0;
                for q in 1..j.len() {
                    if bb[j[q]] < bb[j[jm]] {
                        jm = q;
                    }
                }
                Some(j[jm])
            };
            match kk {
                None => {
                    let (b, rhs, p, y) = dbg(y);
                    trace.push(QpStep { sqp_iter, qp_iter: iter, n_ws: i.len(), a_corr, route, rcond, pnorm_inf: pn, kind: 2, k: -1, step: 1.0, b, rhs, p, y });
                    iter += 1;
                    break;
                }
                Some(k) if bb[k] >= -CONVTOL_ACTIVESET => {
                    let (b, rhs, p, y) = dbg(y);
                    trace.push(QpStep { sqp_iter, qp_iter: iter, n_ws: i.len(), a_corr, route, rcond, pnorm_inf: pn, kind: 3, k: kk.map(|v| v as i64).unwrap_or(-1), step: 1.0, b, rhs, p, y });
                    iter += 1;
                    break;
                }
                Some(k) => {
                    t[k] = true;
                    k_rec = k as i64;
                }
            }
        } else {
            kind = 1;
            let (k, a) = feasible_stepsize(y, &p);
            k_rec = k.map(|v| v as i64).unwrap_or(-1);
            a_rec = a;
            let add = matches!(k, Some(_)) && a < 1.0 && i.len() > 1;
            for q in 0..m {
                y[q] += a * p[q];
            }
            for q in 0..m {
                if y[q] < 0.0 {
                    y[q] = 0.0;
                }
            }
            if add {
                let k = k.unwrap();
                t[k] = false;
                y[k] = 0.0;
            }
        }
        let (b, rhs, p, y) = dbg(y);
        trace.push(QpStep { sqp_iter, qp_iter: iter, n_ws: i.len(), a_corr, route, rcond, pnorm_inf: pn, kind, k: k_rec, step: a_rec, b, rhs, p, y });
        iter += 1;
    }
    iter
}

#[allow(clippy::too_many_arguments)]
fn linesearch(
    f: f64,
    l: &Mat,
    w: &[f64],
    z: &[f64],
    g: &[f64],
    x: &[f64],
    y: &[f64],
    e: &[f64],
) -> Result<(usize, f64, Vec<f64>), ShrinkError> {
    let m = x.len();
    let p: Vec<f64> = (0..m).map(|i| y[i] - x[i]).collect();
    let (_, afeas) = feasible_stepsize(x, &p);
    let comb = |a: f64| -> Vec<f64> { (0..m).map(|i| a * y[i] + (1.0 - a) * x[i]).collect() };
    let mut nls = 0;
    if afeas <= MINSTEPSIZE {
        return Ok((0, afeas, comb(afeas)));
    }
    let mut a = if 1.0 < afeas { 1.0 } else { afeas };
    let sx = arma_accu(x.iter().cloned());
    let ymx: Vec<f64> = (0..m).map(|i| y[i] - x[i]).collect();
    let g1: Vec<f64> = g.iter().map(|v| v + 1.0).collect();
    let dd = arma_dot(&ymx, &g1);
    loop {
        let xnew = comb(a);
        let fnew = obj(l, w, &xnew, z, e)?;
        nls += 1;
        let xmin = xnew.iter().cloned().fold(f64::INFINITY, f64::min);
        if xmin >= 0.0 && fnew + arma_accu(xnew.iter().cloned()) <= f + sx + SUFFDECR * a * dd {
            return Ok((nls, a, xnew));
        } else if a * STEPSIZEREDUCE < MINSTEPSIZE {
            a = MINSTEPSIZE;
            let mut xnew = comb(a);
            if xnew.iter().cloned().fold(f64::INFINITY, f64::min) < 0.0 {
                a = 0.0;
                xnew = x.to_vec();
            }
            return Ok((nls, a, xnew));
        }
        a *= STEPSIZEREDUCE;
    }
}

/// Outcome of mixsqp's `tsvd()` probe, which decides whether the full L is used.
#[derive(Clone, Debug, PartialEq)]
pub enum TsvdDecision {
    /// m <= 4: no SVD attempted.
    Skipped,
    /// irlba errored or warned at rank `k` (tsvd returns NULL): full L, the ported path.
    FullMatrix { k: usize },
    /// tsvd would return a truncated SVD (or the decision is within a safety margin of
    /// doing so). Not ported.
    LowRank { k: usize, sigma_k: f64 },
}

/// Walk `tsvd()`'s rank-doubling loop on exact singular values. irlba is run with
/// `svtol = 0.01 * tol.svd`, so its estimates agree with the exact values far below the
/// margin used here (a probed singular value under `1e-4`, 100x the 1e-6 cut, is treated
/// as low-rank and refused).
pub fn tsvd_decision(l: &Mat) -> TsvdDecision {
    let (n, m) = (l.nrow, l.ncol);
    if m <= 4 {
        return TsvdDecision::Skipped;
    }
    let r = n.min(m);
    let sv = singular_values(l);
    let mut k = 2usize;
    loop {
        if k > (n - 1).min(m - 1) || (k as f64) >= 0.5 * (r as f64) {
            return TsvdDecision::FullMatrix { k };
        }
        let sk = sv[k - 1];
        if k == r || sk < 1e-4 {
            return TsvdDecision::LowRank { k, sigma_k: sk };
        }
        k = (2 * k).min(r);
    }
}

/// Singular values of L, descending (eigenvalues of `L^T L` by Jacobi).
pub fn singular_values(l: &Mat) -> Vec<f64> {
    let g = syrk_t(l);
    let (vals, _) = jacobi_eigen(&g);
    let mut s: Vec<f64> = vals.into_iter().map(|v| v.max(0.0).sqrt()).collect();
    s.sort_by(|a, b| b.partial_cmp(a).unwrap());
    s
}

/// `mixsqp::mixsqp(L, w = rep(1, n), x0 = rep(1, m), control = list(eps = 1e-6, numiter.em = 20))`
/// as called by `ashr:::mixSQP` with uniform prior. `L` must be non-negative with every
/// row max equal to 1 (ashr's `exp(llik - rowmax)`), so mixsqp's row normalisation is the
/// identity and `z = 0`.
pub fn mixsqp_ashr(l: &Mat) -> Result<MixsqpResult, ShrinkError> {
    let (n, m) = (l.nrow, l.ncol);
    if n == 0 || m == 0 {
        return Err(ShrinkError::InvalidInput("mixsqp: empty likelihood matrix".into()));
    }
    for r in 0..n {
        let mut mx = 0.0_f64;
        for c in 0..m {
            let v = l.at(r, c);
            if v < 0.0 || !v.is_finite() {
                return Err(ShrinkError::InvalidInput("mixsqp: L must be finite and non-negative".into()));
            }
            mx = mx.max(v);
        }
        if mx != 1.0 {
            return Err(ShrinkError::InvalidInput("mixsqp: L rows must have max 1 (ashr scaling)".into()));
        }
    }
    if let TsvdDecision::LowRank { k, sigma_k } = tsvd_decision(l) {
        return Err(ShrinkError::Unsupported(format!(
            "mixsqp: tsvd would take the low-rank path (k = {k}, sigma_k = {sigma_k:e})"
        )));
    }
    // w <- w/sum(w); x0 <- x0/sum(x0) (both sums exact).
    let w = vec![1.0 / (n as f64); n];
    let mut x = vec![1.0 / (m as f64); m];
    let z = vec![0.0; n];
    // eps <- eps - min(0, min(L)) = eps (L >= 0)
    let e = vec![EPS_ASHR; n];

    let mut em_iterates = Vec::with_capacity(NUMITER_EM);
    for _ in 0..NUMITER_EM {
        mixem_update(l, &w, &mut x);
        em_iterates.push(x.clone());
    }

    let maxiter_as = 20usize.min(m + 1);
    let mut sqp = Vec::new();
    let mut qp = Vec::new();
    let mut converged = false;
    let mut last_g = Vec::new();
    let mut last_gmin = f64::NAN;
    for it in 0..MAXITER_SQP {
        mixem_update(l, &w, &mut x);
        let x_em = x.clone();
        let j1: Vec<usize> = (0..m).filter(|&k| x[k] > ZERO_THRESHOLD_SOLUTION).collect();
        for k in 0..m {
            if x[k] <= ZERO_THRESHOLD_SOLUTION {
                x[k] = 0.0;
            }
        }
        let f = obj(l, &w, &x, &z, &e)?;
        let (g, h) = compute_grad(l, &w, &x, &e);
        let gmin = 1.0 + j1.iter().map(|&k| g[k]).fold(f64::INFINITY, f64::min);
        last_g = g.clone();
        last_gmin = gmin;
        if gmin >= -CONVTOL_SQP {
            converged = true;
            sqp.push(SqpStep { x_em, obj: f, gmin, y: None, step: None, nqp: 0, nls: 0 });
            break;
        }
        let hx = gemv_n(&h, &x);
        let ghat: Vec<f64> = (0..m).map(|k| g[k] - hx[k] + 1.0).collect();
        let mut y = x.clone();
        let nqp = activesetqp(&h, &ghat, &mut y, maxiter_as, it, &mut qp);
        let (nls, a, xnew) = linesearch(f, l, &w, &z, &g, &x, &y, &e)?;
        sqp.push(SqpStep { x_em, obj: f, gmin, y: Some(y), step: Some(a), nqp, nls });
        x = xnew;
    }
    // x <- x/sum(x) (R sum: long double)
    let s = r_sum(x.iter().cloned());
    let xn: Vec<f64> = x.iter().map(|v| v / s).collect();
    Ok(MixsqpResult { x: xn, em_iterates, sqp, qp, converged, grad: last_g, gmin: last_gmin })
}

/// The mixsqp objective `f(x) = -sum_i w_i log((L x)_i + eps)` with `w = 1/n`, exposed for
/// the KKT certificate in the tests.
pub fn objective(l: &Mat, x: &[f64]) -> f64 {
    let n = l.nrow;
    let w = vec![1.0 / (n as f64); n];
    let z = vec![0.0; n];
    let e = vec![EPS_ASHR; n];
    obj(l, &w, x, &z, &e).unwrap_or(f64::INFINITY)
}

/// Gradient of [`objective`] and the KKT dual residual `max_j |min(x_j, g_j + 1)|`-style
/// statistics: returns `(g, 1 + min_j g_j, max_{x_j > zts} |g_j + 1|)` (dual feasibility, stationarity on the support).
pub fn kkt(l: &Mat, x: &[f64]) -> (Vec<f64>, f64, f64) {
    let n = l.nrow;
    let w = vec![1.0 / (n as f64); n];
    let e = vec![EPS_ASHR; n];
    let (g, _) = compute_grad(l, &w, x, &e);
    let sup: Vec<usize> = (0..x.len()).filter(|&k| x[k] > ZERO_THRESHOLD_SOLUTION).collect();
    let gmin = 1.0 + g.iter().cloned().fold(f64::INFINITY, f64::min);
    let comp = sup.iter().map(|&k| (g[k] + 1.0).abs()).fold(0.0, f64::max);
    (g, gmin, comp)
}
