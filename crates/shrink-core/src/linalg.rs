//! Armadillo 15 `solve(B, rhs)` dispatch as mixsqp reaches it, plus the symmetric Jacobi
//! eigensolver used for the singular values of L (and as the min-norm fallback above the
//! dgelsd branch ported in [`crate::lapack`]).

use crate::dense::{getrf, getrs, potrf_lower, potrs_lower, Mat};
use crate::lapack;

/// Which branch `solve()` took.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SolveRoute {
    /// Cholesky (dpotrf 'L' + dpotrs), taken when `guess_sympd(B, 16)` holds.
    Sympd,
    /// LU (dgetrf + dgetrs).
    Square,
    /// rcond below eps (or a failed LU): Armadillo's `solve_approx_svd`, LAPACK dgelsd
    /// ([`lapack::dgelsd_square`]).
    Approx,
}

/// `sym_helper::guess_sympd(A, min_n_rows)` (Armadillo 15, real case), verbatim.
pub fn guess_sympd(a: &Mat, min_n_rows: usize) -> bool {
    let n = a.nrow;
    if a.ncol != n || n < min_n_rows {
        return false;
    }
    let tol = 100.0 * f64::EPSILON;
    let mut diag_below_tol = true;
    let mut max_diag: f64 = 0.0;
    for j in 0..n {
        let d = a.at(j, j);
        if d <= 0.0 || !d.is_finite() {
            return false;
        }
        if d >= tol {
            diag_below_tol = false;
        }
        if d > max_diag {
            max_diag = d;
        }
    }
    if diag_below_tol {
        return false;
    }
    for j in 0..n.saturating_sub(1) {
        let ajj = a.at(j, j);
        for i in (j + 1)..n {
            let aij = a.at(i, j);
            let aji = a.at(j, i);
            let (aij_abs, aji_abs) = (aij.abs(), aji.abs());
            if aij_abs >= max_diag {
                return false;
            }
            let delta = (aij - aji).abs();
            let abs_max = aij_abs.max(aji_abs);
            if delta > tol && delta > abs_max * tol {
                return false;
            }
            let aii = a.at(i, i);
            if aij_abs + aij_abs >= aii + ajj {
                return false;
            }
        }
    }
    true
}

/// `solve(B, rhs)` with default options for a square dense `B` of size < 32 (so no band
/// detection) that is not triangular.
///
/// The flow is Armadillo 15.6's: if `guess_sympd(B, 16)`, `solve_sympd_rcond` (dlansy '1'
/// on B, dpotrf 'L', dpotrs, dpocon); a failed dpotrf falls through to `solve_square_rcond`
/// (dlange '1', dgetrf, dgetrs, dgecon), where a failed dgetrf goes straight to the
/// approximation. An rcond below eps or NaN sends the system to `solve_approx_svd`: dgelsd
/// with rcond = max(rows, cols) * eps. The returned f64 is the rcond that decided the route.
pub fn arma_solve(b: &Mat, rhs: &[f64]) -> (Vec<f64>, SolveRoute, f64) {
    let eps = f64::EPSILON;
    if guess_sympd(b, 16) {
        if let Some((f, rc)) = sympd_factor_rcond(b) {
            if !(rc < eps || rc.is_nan()) {
                let mut x = rhs.to_vec();
                potrs_lower(&f, &mut x);
                return (x, SolveRoute::Sympd, rc);
            }
            return (solve_approx_svd(b, rhs), SolveRoute::Approx, rc);
        }
    }
    let Some((f, ipiv, rc)) = lu_factor_rcond(b) else {
        return (solve_approx_svd(b, rhs), SolveRoute::Approx, 0.0);
    };
    if rc < eps || rc.is_nan() {
        return (solve_approx_svd(b, rhs), SolveRoute::Approx, rc);
    }
    let mut x = rhs.to_vec();
    getrs(&f, &ipiv, &mut x);
    (x, SolveRoute::Square, rc)
}

/// dlansy('1','L') then dpotrf('L') and dpocon('L'); None when dpotrf fails.
fn sympd_factor_rcond(b: &Mat) -> Option<(Mat, f64)> {
    let n = b.nrow;
    let anorm = lapack::dlansy_1l(&b.data, n, n);
    let mut f = b.clone();
    if !potrf_lower(&mut f) {
        return None;
    }
    let rc = lapack::dpocon_l(&f.data, n, n, anorm);
    Some((f, rc))
}

/// dlange('1') then dgetrf and dgecon('1'); None when dgetrf reports info != 0.
fn lu_factor_rcond(b: &Mat) -> Option<(Mat, Vec<usize>, f64)> {
    let n = b.nrow;
    let anorm = lapack::dlange_1(&b.data, n, n, n);
    let (f, ipiv, info) = getrf(b);
    if info != 0 {
        return None;
    }
    let rc = lapack::dgecon_1(&f.data, n, n, anorm);
    Some((f, ipiv, rc))
}

/// The rcond LAPACK's dpocon reports on Armadillo's sympd route (None if dpotrf fails).
pub fn dpocon_l(b: &Mat) -> Option<f64> {
    sympd_factor_rcond(b).map(|(_, rc)| rc)
}

/// The rcond LAPACK's dgecon reports on Armadillo's square route (None if dgetrf fails).
pub fn dgecon_1(b: &Mat) -> Option<f64> {
    lu_factor_rcond(b).map(|(_, _, rc)| rc)
}

/// Armadillo `solve_approx_svd` for a square system: dgelsd with rcond = n * eps. dgelsd is
/// ported for n <= 25 (every system mixsqp builds on the corpus); above that, or if dgelsd
/// fails, the same minimum-norm solution comes from [`approx_min_norm`].
fn solve_approx_svd(b: &Mat, rhs: &[f64]) -> Vec<f64> {
    let n = b.nrow;
    if (1..=lapack::DGELSD_SMLSIZ).contains(&n) {
        if let Some(x) = lapack::dgelsd_square(&b.data, n, rhs, n as f64 * f64::EPSILON) {
            return x;
        }
    }
    approx_min_norm(b, rhs)
}

/// Minimum-norm least-squares solution of a symmetric system, truncating singular values
/// at `max(m,n) * eps * s_max` (the threshold Armadillo hands to dgelsd), by a Jacobi
/// eigendecomposition. Used only outside the ported dgelsd branch (n > 25), where it gives
/// dgelsd's solution up to rounding.
pub fn approx_min_norm(b: &Mat, rhs: &[f64]) -> Vec<f64> {
    let n = b.nrow;
    let (vals, vecs) = jacobi_eigen(b);
    let smax = vals.iter().fold(0.0_f64, |m, v| m.max(v.abs()));
    let thr = (n as f64) * f64::EPSILON * smax;
    let mut x = vec![0.0; n];
    for k in 0..n {
        if vals[k].abs() <= thr {
            continue;
        }
        let vk = vecs.col(k);
        let c: f64 = vk.iter().zip(rhs).map(|(v, r)| v * r).sum::<f64>() / vals[k];
        for i in 0..n {
            x[i] += c * vk[i];
        }
    }
    x
}

/// Cyclic Jacobi eigendecomposition of a symmetric matrix. Returns (eigenvalues, eigenvectors
/// as columns), unsorted.
pub fn jacobi_eigen(a: &Mat) -> (Vec<f64>, Mat) {
    let n = a.nrow;
    let mut m = a.clone();
    // symmetrise defensively (inputs are exactly symmetric in practice)
    for j in 0..n {
        for i in (j + 1)..n {
            let v = 0.5 * (m.at(i, j) + m.at(j, i));
            m.set(i, j, v);
            m.set(j, i, v);
        }
    }
    let mut v = Mat::zeros(n, n);
    for i in 0..n {
        v.set(i, i, 1.0);
    }
    for _sweep in 0..100 {
        let mut off = 0.0;
        for j in 0..n {
            for i in (j + 1)..n {
                off += m.at(i, j) * m.at(i, j);
            }
        }
        let diag: f64 = (0..n).map(|i| m.at(i, i) * m.at(i, i)).sum();
        if off <= 1e-36 * diag || off == 0.0 {
            break;
        }
        for p in 0..n {
            for q in (p + 1)..n {
                let apq = m.at(p, q);
                if apq == 0.0 {
                    continue;
                }
                let app = m.at(p, p);
                let aqq = m.at(q, q);
                let theta = (aqq - app) / (2.0 * apq);
                let t = theta.signum() / (theta.abs() + (theta * theta + 1.0).sqrt());
                let t = if theta == 0.0 { 1.0 } else { t };
                let c = 1.0 / (t * t + 1.0).sqrt();
                let s = t * c;
                for k in 0..n {
                    let mkp = m.at(k, p);
                    let mkq = m.at(k, q);
                    m.set(k, p, c * mkp - s * mkq);
                    m.set(k, q, s * mkp + c * mkq);
                }
                for k in 0..n {
                    let mpk = m.at(p, k);
                    let mqk = m.at(q, k);
                    m.set(p, k, c * mpk - s * mqk);
                    m.set(q, k, s * mpk + c * mqk);
                }
                for k in 0..n {
                    let vkp = v.at(k, p);
                    let vkq = v.at(k, q);
                    v.set(k, p, c * vkp - s * vkq);
                    v.set(k, q, s * vkp + c * vkq);
                }
            }
        }
    }
    ((0..n).map(|i| m.at(i, i)).collect(), v)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn solve_matches_on_well_conditioned() {
        let b = Mat::from_col_major(3, 3, vec![4.0, 1.0, 0.5, 1.0, 3.0, 0.2, 0.5, 0.2, 2.0]);
        let rhs = [1.0, 2.0, 3.0];
        let (x, route, _) = arma_solve(&b, &rhs);
        assert_eq!(route, SolveRoute::Square);
        let y = approx_min_norm(&b, &rhs);
        for i in 0..3 {
            assert!((x[i] - y[i]).abs() < 1e-13);
            let r: f64 = (0..3).map(|j| b.at(i, j) * x[j]).sum();
            assert!((r - rhs[i]).abs() < 1e-13);
        }
    }

    #[test]
    fn min_norm_on_singular() {
        // rank-1: [[1,1],[1,1]] x = [2,2] -> min-norm x = [1,1]
        let b = Mat::from_col_major(2, 2, vec![1.0, 1.0, 1.0, 1.0]);
        let (x, route, _) = arma_solve(&b, &[2.0, 2.0]);
        assert_eq!(route, SolveRoute::Approx);
        assert!((x[0] - 1.0).abs() < 1e-14 && (x[1] - 1.0).abs() < 1e-14);
    }
}
