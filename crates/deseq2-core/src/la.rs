//! Small dense linear algebra exactly as DESeq2's C++ gets it from Armadillo 15.6 on R 4.5's
//! reference BLAS and LAPACK 3.12.1 (no FMA, sequential sums).
//!
//! Every routine mirrors the operation order of the code path Armadillo takes for the matrix
//! sizes DESeq2 uses (n samples by p <= ~8 coefficients), so the results are bit-identical:
//! `x.t() * B` is reference `dgemm('T','N')` (or `dgemv('T')` when one side is a vector),
//! `det()` / `inv()` take Armadillo's closed forms for N <= 3 and LAPACK `dgetrf2` /
//! `dgetri` otherwise, `qr_econ()` is `dgeqr2` + `dorg2r` with LAPACK 3.12's
//! `DLARF1F`, and `solve()` on the triangular `R` is `dtrtrs`.

/// Column-major dense matrix.
#[derive(Clone, Debug, PartialEq)]
pub struct Mat {
    /// Number of rows.
    pub nrow: usize,
    /// Number of columns.
    pub ncol: usize,
    /// Column-major values, `nrow * ncol` long.
    pub data: Vec<f64>,
}

impl Mat {
    /// An `nrow` by `ncol` matrix of zeros.
    pub fn zeros(nrow: usize, ncol: usize) -> Self {
        Mat {
            nrow,
            ncol,
            data: vec![0.0; nrow * ncol],
        }
    }
    /// Build from column-major values.
    pub fn from_col_major(nrow: usize, ncol: usize, data: Vec<f64>) -> Self {
        assert_eq!(data.len(), nrow * ncol);
        Mat { nrow, ncol, data }
    }
    /// Element `(i, j)`.
    #[inline]
    pub fn at(&self, i: usize, j: usize) -> f64 {
        self.data[i + j * self.nrow]
    }
    /// Mutable element `(i, j)`.
    #[inline]
    pub fn at_mut(&mut self, i: usize, j: usize) -> &mut f64 {
        &mut self.data[i + j * self.nrow]
    }
    /// Column `j` as a slice.
    pub fn col(&self, j: usize) -> &[f64] {
        &self.data[j * self.nrow..(j + 1) * self.nrow]
    }
    /// Row `i` as a vector.
    pub fn row(&self, i: usize) -> Vec<f64> {
        (0..self.ncol).map(|j| self.at(i, j)).collect()
    }
    fn is_diag(&self) -> bool {
        for j in 0..self.ncol {
            for i in 0..self.nrow {
                if i != j && self.at(i, j) != 0.0 {
                    return false;
                }
            }
        }
        true
    }
    fn is_triu(&self) -> bool {
        let n = self.nrow;
        if n < 2 {
            return false;
        }
        for j in 0..n {
            for i in (j + 1)..n {
                if self.at(i, j) != 0.0 {
                    return false;
                }
            }
        }
        true
    }
    fn is_tril(&self) -> bool {
        let n = self.nrow;
        if n < 2 {
            return false;
        }
        for j in 1..n {
            for i in 0..j {
                if self.at(i, j) != 0.0 {
                    return false;
                }
            }
        }
        true
    }
}

/// `A' * B` as reference `dgemm('T','N')`: `out(i,j) = sum_l A(l,i) * B(l,j)`, summed in order.
pub fn tmul(a: &Mat, b: &Mat) -> Mat {
    assert_eq!(a.nrow, b.nrow);
    let mut out = Mat::zeros(a.ncol, b.ncol);
    for j in 0..b.ncol {
        let bj = b.col(j);
        for i in 0..a.ncol {
            let ai = a.col(i);
            let mut t = 0.0;
            for l in 0..a.nrow {
                t += ai[l] * bj[l];
            }
            *out.at_mut(i, j) = t;
        }
    }
    out
}

/// `x' * (x.each_col() % w)` (DESeq2's `X'WX`), via [`tmul`].
pub fn xtwx(x: &Mat, w: &[f64]) -> Mat {
    let mut xw = x.clone();
    for j in 0..x.ncol {
        for i in 0..x.nrow {
            *xw.at_mut(i, j) = x.at(i, j) * w[i];
        }
    }
    tmul(x, &xw)
}

/// `A * B` as reference `dgemm('N','N')` (and Armadillo's tiny-square emulation, which sums
/// in the same order): `out(i,j) = sum_l A(i,l) * B(l,j)`.
pub fn mul(a: &Mat, b: &Mat) -> Mat {
    assert_eq!(a.ncol, b.nrow);
    let mut out = Mat::zeros(a.nrow, b.ncol);
    for j in 0..b.ncol {
        for i in 0..a.nrow {
            let mut t = 0.0;
            for l in 0..a.ncol {
                t += a.at(i, l) * b.at(l, j);
            }
            *out.at_mut(i, j) = t;
        }
    }
    out
}

/// `A * v` as reference `dgemv('N')`: `y(i) = sum_j A(i,j) * v(j)` in column order.
pub fn mul_vec(a: &Mat, v: &[f64]) -> Vec<f64> {
    assert_eq!(a.ncol, v.len());
    let mut y = vec![0.0; a.nrow];
    for (j, &vj) in v.iter().enumerate() {
        let c = a.col(j);
        for i in 0..a.nrow {
            y[i] += vj * c[i];
        }
    }
    y
}

/// `A' * v` as reference `dgemv('T')`: `y(j) = sum_i A(i,j) * v(i)`.
pub fn tmul_vec(a: &Mat, v: &[f64]) -> Vec<f64> {
    assert_eq!(a.nrow, v.len());
    (0..a.ncol)
        .map(|j| {
            let c = a.col(j);
            let mut t = 0.0;
            for i in 0..a.nrow {
                t += c[i] * v[i];
            }
            t
        })
        .collect()
}

/// Armadillo's `op_dot::direct_dot` for short vectors: two interleaved accumulators.
pub fn direct_dot(a: &[f64], b: &[f64]) -> f64 {
    let n = a.len();
    let (mut v1, mut v2) = (0.0, 0.0);
    let mut i = 0;
    let mut j = 1;
    while j < n {
        v1 += a[i] * b[i];
        v2 += a[j] * b[j];
        i += 2;
        j += 2;
    }
    if i < n {
        v1 += a[i] * b[i];
    }
    v1 + v2
}

/// Plain sequential dot product (reference `ddot`, whose unrolled loop still adds left to
/// right).
pub fn ddot(a: &[f64], b: &[f64]) -> f64 {
    let mut t = 0.0;
    for i in 0..a.len() {
        t += a[i] * b[i];
    }
    t
}

/// Armadillo's `trace(A * B)` (`fn_trace.hpp`): two accumulators alternating over the inner
/// index, carried across all diagonal positions.
pub fn trace_mul(a: &Mat, b: &Mat) -> f64 {
    let n = a.nrow.min(b.ncol);
    let (mut acc1, mut acc2) = (0.0, 0.0);
    for k in 0..n {
        let bc = b.col(k);
        let mut j = 1;
        while j < a.ncol {
            let i = j - 1;
            acc1 += a.at(k, i) * bc[i];
            acc2 += a.at(k, j) * bc[j];
            j += 2;
        }
        let i = j - 1;
        if i < a.ncol {
            acc1 += a.at(k, i) * bc[i];
        }
    }
    acc1 + acc2
}

fn idamax(x: impl Iterator<Item = f64>) -> usize {
    let mut best = 0usize;
    let mut dmax = -1.0f64;
    for (i, v) in x.enumerate() {
        let a = v.abs();
        if i == 0 {
            dmax = a;
        } else if a > dmax {
            best = i;
            dmax = a;
        }
    }
    best
}

/// LAPACK `dgetrf2` (recursive LU with partial pivoting) on the `m x n` block of `a` starting
/// at `(r0, c0)`. Pivots are 0-based, relative to the block. Returns the LAPACK `info`.
pub(crate) fn dgetrf2(
    a: &mut Mat,
    r0: usize,
    c0: usize,
    m: usize,
    n: usize,
    ipiv: &mut [usize],
) -> usize {
    if m == 0 || n == 0 {
        return 0;
    }
    let mut info = 0;
    if m == 1 {
        ipiv[0] = 0;
        if a.at(r0, c0) == 0.0 {
            info = 1;
        }
    } else if n == 1 {
        let sfmin = f64::MIN_POSITIVE;
        let i = idamax((0..m).map(|k| a.at(r0 + k, c0)));
        ipiv[0] = i;
        if a.at(r0 + i, c0) != 0.0 {
            if i != 0 {
                let t = a.at(r0, c0);
                *a.at_mut(r0, c0) = a.at(r0 + i, c0);
                *a.at_mut(r0 + i, c0) = t;
            }
            let p = a.at(r0, c0);
            if p.abs() >= sfmin {
                let s = 1.0 / p;
                for k in 1..m {
                    *a.at_mut(r0 + k, c0) *= s;
                }
            } else {
                for k in 1..m {
                    *a.at_mut(r0 + k, c0) /= p;
                }
            }
        } else {
            info = 1;
        }
    } else {
        let n1 = m.min(n) / 2;
        let n2 = n - n1;
        let iinfo = dgetrf2(a, r0, c0, m, n1, &mut ipiv[..n1]);
        if info == 0 && iinfo > 0 {
            info = iinfo;
        }
        // dlaswp on A12
        for i in 0..n1 {
            let ip = ipiv[i];
            if ip != i {
                for j in n1..n {
                    let t = a.at(r0 + i, c0 + j);
                    *a.at_mut(r0 + i, c0 + j) = a.at(r0 + ip, c0 + j);
                    *a.at_mut(r0 + ip, c0 + j) = t;
                }
            }
        }
        // dtrsm L,L,N,U: A12 = L11^-1 A12
        for j in n1..n {
            for k in 0..n1 {
                let bkj = a.at(r0 + k, c0 + j);
                if bkj != 0.0 {
                    for i in (k + 1)..n1 {
                        *a.at_mut(r0 + i, c0 + j) -= bkj * a.at(r0 + i, c0 + k);
                    }
                }
            }
        }
        // dgemm N,N alpha -1 beta 1: A22 -= A21 A12
        for j in n1..n {
            for l in 0..n1 {
                let t = -a.at(r0 + l, c0 + j);
                for i in n1..m {
                    *a.at_mut(r0 + i, c0 + j) += t * a.at(r0 + i, c0 + l);
                }
            }
        }
        let iinfo = dgetrf2(a, r0 + n1, c0 + n1, m - n1, n2, &mut ipiv[n1..]);
        if info == 0 && iinfo > 0 {
            info = iinfo + n1;
        }
        for p in ipiv.iter_mut().take(m.min(n)).skip(n1) {
            *p += n1;
        }
        // dlaswp on A21 (columns 0..n1) for pivots n1..min(m,n)
        for i in n1..m.min(n) {
            let ip = ipiv[i];
            if ip != i {
                for j in 0..n1 {
                    let t = a.at(r0 + i, c0 + j);
                    *a.at_mut(r0 + i, c0 + j) = a.at(r0 + ip, c0 + j);
                    *a.at_mut(r0 + ip, c0 + j) = t;
                }
            }
        }
    }
    info
}

/// Armadillo `det()` for a square matrix.
pub fn det(a: &Mat) -> f64 {
    let n = a.nrow;
    assert_eq!(n, a.ncol);
    if n == 0 {
        return 1.0;
    }
    if n == 1 {
        return a.data[0];
    }
    if n <= 3 {
        let v = if n == 2 {
            a.at(0, 0) * a.at(1, 1) - a.at(0, 1) * a.at(1, 0)
        } else {
            det3(a)
        };
        let av = v.abs();
        if av > f64::EPSILON && av < 1.0 / f64::EPSILON {
            return v;
        }
    }
    if a.is_diag() || a.is_triu() || a.is_tril() {
        let mut v = 1.0;
        for i in 0..n {
            v *= a.at(i, i);
        }
        return v;
    }
    let mut lu = a.clone();
    let mut ipiv = vec![0usize; n];
    dgetrf2(&mut lu, 0, 0, n, n, &mut ipiv);
    let mut v = lu.at(0, 0);
    for i in 1..n {
        v *= lu.at(i, i);
    }
    let mut neg = false;
    for (i, &p) in ipiv.iter().enumerate() {
        if p != i {
            neg = !neg;
        }
    }
    if neg {
        -v
    } else {
        v
    }
}

fn det3(x: &Mat) -> f64 {
    let g = |i: usize, j: usize| x.at(i, j);
    let val1 = g(0, 0) * (g(2, 2) * g(1, 1) - g(2, 1) * g(1, 2));
    let val2 = g(1, 0) * (g(2, 2) * g(0, 1) - g(2, 1) * g(0, 2));
    let val3 = g(2, 0) * (g(1, 2) * g(0, 1) - g(1, 1) * g(0, 2));
    val1 - val2 + val3
}

/// Error from [`inv`]: the matrix is singular or takes a path that is not ported.
#[derive(Debug, Clone, PartialEq)]
pub struct InvError(pub &'static str);

/// Armadillo `inv()` (`op_inv_gen_full` with default flags).
pub fn inv(a: &Mat) -> Result<Mat, InvError> {
    let n = a.nrow;
    assert_eq!(n, a.ncol);
    let mut out = a.clone();
    if n == 0 {
        return Ok(out);
    }
    if n == 1 {
        let v = out.data[0];
        out.data[0] = 1.0 / v;
        return if v != 0.0 {
            Ok(out)
        } else {
            Err(InvError("inv(): matrix is singular"))
        };
    }
    if n == 2 {
        let (a_, b, c, d) = (a.at(0, 0), a.at(0, 1), a.at(1, 0), a.at(1, 1));
        let det_val = a_ * d - b * c;
        let ad = det_val.abs();
        if !(ad < f64::EPSILON || ad > 1.0 / f64::EPSILON || det_val.is_nan()) {
            *out.at_mut(0, 0) = d / det_val;
            *out.at_mut(0, 1) = -b / det_val;
            *out.at_mut(1, 0) = -c / det_val;
            *out.at_mut(1, 1) = a_ / det_val;
            return Ok(out);
        }
    }
    if n == 3 {
        if let Some(y) = inv3(a) {
            return Ok(y);
        }
    }
    if out.is_diag() {
        for i in 0..n {
            let v = out.at(i, i);
            if v == 0.0 {
                return Err(InvError("inv(): matrix is singular"));
            }
            *out.at_mut(i, i) = 1.0 / v;
        }
        return Ok(out);
    }
    if out.is_triu() || out.is_tril() {
        return Err(InvError("inv(): triangular path not ported"));
    }
    // `sym_helper::is_approx_sym(out, 100)`: the second argument is a minimum size, so the
    // symmetric path only applies from 100 x 100 up.
    if n >= 100 && is_approx_sym(&out) {
        return inv_sym(out);
    }
    inv_gen(out)
}

/// Armadillo's `auxlib::inv`: LAPACK `dgetrf` (`dgetrf2` below the block size 64) then the
/// unblocked `dgetri` (`dtrti2`, the `dgemv` column sweep, then the column interchanges).
fn inv_gen(mut a: Mat) -> Result<Mat, InvError> {
    let n = a.nrow;
    if n >= 64 {
        return Err(InvError("inv(): blocked dgetrf/dgetri path not ported"));
    }
    let mut ipiv = vec![0usize; n];
    if dgetrf2(&mut a, 0, 0, n, n, &mut ipiv) != 0 {
        return Err(InvError("inv(): matrix is singular"));
    }
    // dtrtri('Upper', 'Non-unit'): singularity check, then dtrti2.
    if (0..n).any(|i| a.at(i, i) == 0.0) {
        return Err(InvError("inv(): matrix is singular"));
    }
    for j in 0..n {
        let ajj_inv = 1.0 / a.at(j, j);
        *a.at_mut(j, j) = ajj_inv;
        let ajj = -ajj_inv;
        // dtrmv('Upper', 'No transpose', 'Non-unit', j, A, A(0..j, j))
        for jj in 0..j {
            let temp = a.at(jj, j);
            if temp != 0.0 {
                for i in 0..jj {
                    let v = a.at(i, j) + temp * a.at(i, jj);
                    *a.at_mut(i, j) = v;
                }
                let v = a.at(jj, j) * a.at(jj, jj);
                *a.at_mut(jj, j) = v;
            }
        }
        // dscal(j, ajj, A(0..j, j))
        if ajj != 1.0 {
            for i in 0..j {
                let v = ajj * a.at(i, j);
                *a.at_mut(i, j) = v;
            }
        }
    }
    // Solve inv(A) * L = inv(U), column by column from the right.
    let mut work = vec![0.0; n];
    for j in (0..n).rev() {
        for i in (j + 1)..n {
            work[i] = a.at(i, j);
            *a.at_mut(i, j) = 0.0;
        }
        // dgemv('N', n, n-j-1, -1, A(:, j+1..), work(j+1..), 1, A(:, j))
        for jj in (j + 1)..n {
            let temp = -work[jj];
            for i in 0..n {
                let v = a.at(i, j) + temp * a.at(i, jj);
                *a.at_mut(i, j) = v;
            }
        }
    }
    for j in (0..n.saturating_sub(1)).rev() {
        let jp = ipiv[j];
        if jp != j {
            for i in 0..n {
                let t = a.at(i, j);
                *a.at_mut(i, j) = a.at(i, jp);
                *a.at_mut(i, jp) = t;
            }
        }
    }
    Ok(a)
}

fn inv3(x: &Mat) -> Option<Mat> {
    let det_val = det3(x);
    let ad = det_val.abs();
    if ad < f64::EPSILON || ad > 1.0 / f64::EPSILON || det_val.is_nan() {
        return None;
    }
    let g = |i: usize, j: usize| x.at(i, j);
    let mut y = Mat::zeros(3, 3);
    *y.at_mut(0, 0) = (g(2, 2) * g(1, 1) - g(2, 1) * g(1, 2)) / det_val;
    *y.at_mut(1, 0) = -(g(2, 2) * g(1, 0) - g(2, 0) * g(1, 2)) / det_val;
    *y.at_mut(2, 0) = (g(2, 1) * g(1, 0) - g(2, 0) * g(1, 1)) / det_val;
    *y.at_mut(0, 1) = -(g(2, 2) * g(0, 1) - g(2, 1) * g(0, 2)) / det_val;
    *y.at_mut(1, 1) = (g(2, 2) * g(0, 0) - g(2, 0) * g(0, 2)) / det_val;
    *y.at_mut(2, 1) = -(g(2, 1) * g(0, 0) - g(2, 0) * g(0, 1)) / det_val;
    *y.at_mut(0, 2) = (g(1, 2) * g(0, 1) - g(1, 1) * g(0, 2)) / det_val;
    *y.at_mut(1, 2) = -(g(1, 2) * g(0, 0) - g(1, 0) * g(0, 2)) / det_val;
    *y.at_mut(2, 2) = (g(1, 1) * g(0, 0) - g(1, 0) * g(0, 1)) / det_val;
    let check = g(0, 0) * y.at(0, 0) + g(0, 1) * y.at(1, 0) + g(0, 2) * y.at(2, 0);
    if (1.0 - check).abs() >= 1e-10 {
        return None;
    }
    Some(y)
}

fn is_approx_sym(a: &Mat) -> bool {
    let tol = 100.0 * f64::EPSILON;
    let n = a.nrow;
    let mut diag_below = true;
    for j in 0..n {
        let v = a.at(j, j);
        if !v.is_finite() {
            return false;
        }
        if v.abs() >= tol {
            diag_below = false;
        }
    }
    if diag_below {
        return false;
    }
    for j in 0..n.saturating_sub(1) {
        for i in (j + 1)..n {
            let aij = a.at(i, j);
            let aji = a.at(j, i);
            let delta = (aij - aji).abs();
            let amax = aij.abs().max(aji.abs());
            if delta > tol && delta > amax * tol {
                return false;
            }
        }
    }
    true
}

/// `dsytf2('L')` + `dsytri('L')` + `symmatl`, as Armadillo's `auxlib::inv_sym`.
fn inv_sym(mut a: Mat) -> Result<Mat, InvError> {
    let n = a.nrow;
    // ---- dsytf2, lower (0-based k; ipiv stores 1-based LAPACK values, negative for 2x2) ----
    let alpha = (1.0 + 17f64.sqrt()) / 8.0;
    let mut ipiv = vec![0i64; n];
    let mut info = 0usize;
    let mut k = 0usize;
    while k < n {
        let mut kstep = 1;
        let absakk = a.at(k, k).abs();
        let (imax, colmax) = if k + 1 < n {
            let im = k + 1 + idamax((k + 1..n).map(|i| a.at(i, k)));
            (im, a.at(im, k).abs())
        } else {
            (k, 0.0)
        };
        let kp;
        if absakk.max(colmax) == 0.0 || absakk.is_nan() {
            if info == 0 {
                info = k + 1;
            }
            kp = k;
        } else {
            if absakk >= alpha * colmax {
                kp = k;
            } else {
                let jmax = k + idamax((k..imax).map(|j| a.at(imax, j)));
                let mut rowmax = a.at(imax, jmax).abs();
                if imax + 1 < n {
                    let jm = imax + 1 + idamax((imax + 1..n).map(|i| a.at(i, imax)));
                    rowmax = rowmax.max(a.at(jm, imax).abs());
                }
                if absakk >= alpha * colmax * (colmax / rowmax) {
                    kp = k;
                } else if a.at(imax, imax).abs() >= alpha * rowmax {
                    kp = imax;
                } else {
                    kp = imax;
                    kstep = 2;
                }
            }
            let kk = k + kstep - 1;
            if kp != kk {
                for i in (kp + 1)..n {
                    let t = a.at(i, kk);
                    *a.at_mut(i, kk) = a.at(i, kp);
                    *a.at_mut(i, kp) = t;
                }
                for off in 0..(kp - kk - 1) {
                    let (r1, c2) = (kk + 1 + off, kk + 1 + off);
                    let t = a.at(r1, kk);
                    *a.at_mut(r1, kk) = a.at(kp, c2);
                    *a.at_mut(kp, c2) = t;
                }
                let t = a.at(kk, kk);
                *a.at_mut(kk, kk) = a.at(kp, kp);
                *a.at_mut(kp, kp) = t;
                if kstep == 2 {
                    let t = a.at(k + 1, k);
                    *a.at_mut(k + 1, k) = a.at(kp, k);
                    *a.at_mut(kp, k) = t;
                }
            }
            if kstep == 1 {
                if k + 1 < n {
                    let d11 = 1.0 / a.at(k, k);
                    // dsyr lower, alpha = -d11, x = A(k+1:n, k), on A(k+1:n, k+1:n)
                    let x: Vec<f64> = (k + 1..n).map(|i| a.at(i, k)).collect();
                    let m = n - k - 1;
                    for j in 0..m {
                        if x[j] != 0.0 {
                            let temp = -d11 * x[j];
                            for i in j..m {
                                *a.at_mut(k + 1 + i, k + 1 + j) += x[i] * temp;
                            }
                        }
                    }
                    for i in (k + 1)..n {
                        *a.at_mut(i, k) *= d11;
                    }
                }
            } else if k + 2 < n {
                let mut d21 = a.at(k + 1, k);
                let d11 = a.at(k + 1, k + 1) / d21;
                let d22 = a.at(k, k) / d21;
                let t = 1.0 / (d11 * d22 - 1.0);
                d21 = t / d21;
                for j in (k + 2)..n {
                    let wk = d21 * (d11 * a.at(j, k) - a.at(j, k + 1));
                    let wkp1 = d21 * (d22 * a.at(j, k + 1) - a.at(j, k));
                    for i in j..n {
                        *a.at_mut(i, j) = a.at(i, j) - a.at(i, k) * wk - a.at(i, k + 1) * wkp1;
                    }
                    *a.at_mut(j, k) = wk;
                    *a.at_mut(j, k + 1) = wkp1;
                }
            }
        }
        if kstep == 1 {
            ipiv[k] = kp as i64 + 1;
        } else {
            ipiv[k] = -(kp as i64 + 1);
            ipiv[k + 1] = -(kp as i64 + 1);
        }
        k += kstep;
    }
    if info != 0 {
        return Err(InvError("inv(): matrix is singular"));
    }
    // ---- dsytri, lower ----
    for kk in (0..n).rev() {
        if ipiv[kk] > 0 && a.at(kk, kk) == 0.0 {
            return Err(InvError("inv(): matrix is singular"));
        }
    }
    // dsymv lower with alpha -1, beta 0 on the trailing block starting at s.
    fn symv_neg(a: &Mat, s: usize, x: &[f64]) -> Vec<f64> {
        let m = x.len();
        let mut y = vec![0.0; m];
        for j in 0..m {
            let temp1 = -x[j];
            let mut temp2 = 0.0;
            y[j] += temp1 * a.at(s + j, s + j);
            for i in (j + 1)..m {
                y[i] += temp1 * a.at(s + i, s + j);
                temp2 += a.at(s + i, s + j) * x[i];
            }
            y[j] += -temp2;
        }
        y
    }
    let mut k = n as i64 - 1;
    while k >= 0 {
        let ku = k as usize;
        let kstep;
        if ipiv[ku] > 0 {
            *a.at_mut(ku, ku) = 1.0 / a.at(ku, ku);
            if ku + 1 < n {
                let work: Vec<f64> = (ku + 1..n).map(|i| a.at(i, ku)).collect();
                let y = symv_neg(&a, ku + 1, &work);
                for (o, v) in y.iter().enumerate() {
                    *a.at_mut(ku + 1 + o, ku) = *v;
                }
                let d = ddot(&work, &y);
                *a.at_mut(ku, ku) -= d;
            }
            kstep = 1;
        } else {
            let t = a.at(ku, ku - 1).abs();
            let ak = a.at(ku - 1, ku - 1) / t;
            let akp1 = a.at(ku, ku) / t;
            let akkp1 = a.at(ku, ku - 1) / t;
            let d = t * (ak * akp1 - 1.0);
            *a.at_mut(ku - 1, ku - 1) = akp1 / d;
            *a.at_mut(ku, ku) = ak / d;
            *a.at_mut(ku, ku - 1) = -akkp1 / d;
            if ku + 1 < n {
                let work: Vec<f64> = (ku + 1..n).map(|i| a.at(i, ku)).collect();
                let y = symv_neg(&a, ku + 1, &work);
                for (o, v) in y.iter().enumerate() {
                    *a.at_mut(ku + 1 + o, ku) = *v;
                }
                let d1 = ddot(&work, &y);
                *a.at_mut(ku, ku) -= d1;
                let colk: Vec<f64> = (ku + 1..n).map(|i| a.at(i, ku)).collect();
                let colk1: Vec<f64> = (ku + 1..n).map(|i| a.at(i, ku - 1)).collect();
                let d2 = ddot(&colk, &colk1);
                *a.at_mut(ku, ku - 1) -= d2;
                let work2 = colk1;
                let y2 = symv_neg(&a, ku + 1, &work2);
                for (o, v) in y2.iter().enumerate() {
                    *a.at_mut(ku + 1 + o, ku - 1) = *v;
                }
                let d3 = ddot(&work2, &y2);
                *a.at_mut(ku - 1, ku - 1) -= d3;
            }
            kstep = 2;
        }
        let kp = (ipiv[ku].unsigned_abs() - 1) as usize;
        if kp != ku {
            for i in (kp + 1)..n {
                let t = a.at(i, ku);
                *a.at_mut(i, ku) = a.at(i, kp);
                *a.at_mut(i, kp) = t;
            }
            for off in 0..(kp - ku - 1) {
                let t = a.at(ku + 1 + off, ku);
                *a.at_mut(ku + 1 + off, ku) = a.at(kp, ku + 1 + off);
                *a.at_mut(kp, ku + 1 + off) = t;
            }
            let t = a.at(ku, ku);
            *a.at_mut(ku, ku) = a.at(kp, kp);
            *a.at_mut(kp, kp) = t;
            if kstep == 2 {
                let t = a.at(ku, ku - 1);
                *a.at_mut(ku, ku - 1) = a.at(kp, ku - 1);
                *a.at_mut(kp, ku - 1) = t;
            }
        }
        k -= kstep as i64;
    }
    // symmatl: copy the lower triangle to the upper.
    for j in 0..n {
        for i in (j + 1)..n {
            *a.at_mut(j, i) = a.at(i, j);
        }
    }
    Ok(a)
}

/// R 4.5 `blas2.f90` `dnrm2` (Blue's algorithm).
pub fn dnrm2(x: &[f64]) -> f64 {
    let n = x.len();
    if n == 0 {
        return 0.0;
    }
    let tsml = 2f64.powi(-511);
    let tbig = 2f64.powi(486);
    let ssml = 2f64.powi(537);
    let sbig = 2f64.powi(-538);
    let maxn = f64::MAX;
    let mut notbig = true;
    let (mut asml, mut amed, mut abig) = (0.0f64, 0.0f64, 0.0f64);
    for &v in x {
        let ax = v.abs();
        if ax > tbig {
            let t = ax * sbig;
            abig += t * t;
            notbig = false;
        } else if ax < tsml {
            if notbig {
                let t = ax * ssml;
                asml += t * t;
            }
        } else {
            amed += ax * ax;
        }
    }
    let (scl, sumsq);
    if abig > 0.0 {
        if amed > 0.0 || amed > maxn || amed.is_nan() {
            abig += (amed * sbig) * sbig;
        }
        scl = 1.0 / sbig;
        sumsq = abig;
    } else if asml > 0.0 {
        if amed > 0.0 || amed > maxn || amed.is_nan() {
            let amed2 = amed.sqrt();
            let asml2 = asml.sqrt() / ssml;
            let (ymin, ymax) = if asml2 > amed2 {
                (amed2, asml2)
            } else {
                (asml2, amed2)
            };
            scl = 1.0;
            let r = ymin / ymax;
            sumsq = (ymax * ymax) * (1.0 + r * r);
        } else {
            scl = 1.0 / ssml;
            sumsq = asml;
        }
    } else {
        scl = 1.0;
        sumsq = amed;
    }
    scl * sumsq.sqrt()
}

fn dlapy2(x: f64, y: f64) -> f64 {
    if x.is_nan() {
        return x;
    }
    if y.is_nan() {
        return y;
    }
    let (xa, ya) = (x.abs(), y.abs());
    let w = xa.max(ya);
    let z = xa.min(ya);
    if z == 0.0 || w > f64::MAX {
        w
    } else {
        w * (1.0 + (z / w) * (z / w)).sqrt()
    }
}

/// `dlarfg` on column `c` from row `r` (alpha = A(r,c), x = A(r+1.., c)). Returns tau.
fn dlarfg(a: &mut Mat, r: usize, c: usize) -> f64 {
    let m = a.nrow;
    if m - r <= 1 {
        return 0.0;
    }
    let xnorm = dnrm2(&a.col(c)[r + 1..]);
    if xnorm == 0.0 {
        return 0.0;
    }
    let mut alpha = a.at(r, c);
    let mut beta = -dlapy2(alpha, xnorm).copysign(alpha);
    let safmin = f64::MIN_POSITIVE / f64::EPSILON;
    let mut knt = 0;
    if beta.abs() < safmin {
        let rsafmn = 1.0 / safmin;
        loop {
            knt += 1;
            for i in r + 1..m {
                *a.at_mut(i, c) *= rsafmn;
            }
            beta *= rsafmn;
            alpha *= rsafmn;
            if !(beta.abs() < safmin && knt < 20) {
                break;
            }
        }
        let xnorm = dnrm2(&a.col(c)[r + 1..]);
        beta = -dlapy2(alpha, xnorm).copysign(alpha);
    }
    let tau = (beta - alpha) / beta;
    let s = 1.0 / (alpha - beta);
    for i in r + 1..m {
        *a.at_mut(i, c) *= s;
    }
    for _ in 0..knt {
        beta *= safmin;
    }
    *a.at_mut(r, c) = beta;
    tau
}

/// `DLARF1F('Left')`: apply `H = I - tau v v'` (v(1) = 1 implicit, v = A(r+1.., vc)) to the
/// block `A(r.., c0..ncol)`.
fn dlarf1f_left(a: &mut Mat, r: usize, vc: usize, tau: f64, c0: usize) {
    let m = a.nrow - r;
    let n = a.ncol - c0;
    if tau == 0.0 || n == 0 {
        return;
    }
    let mut lastv = m;
    while lastv > 1 && a.at(r + lastv - 1, vc) == 0.0 {
        lastv -= 1;
    }
    // iladlc on C(1:lastv, 1:n)
    let lastc = {
        let cc = |i: usize, j: usize| a.at(r + i, c0 + j);
        if cc(0, n - 1) != 0.0 || cc(lastv - 1, n - 1) != 0.0 {
            n
        } else {
            let mut lc = 0;
            'outer: for j in (0..n).rev() {
                for i in 0..lastv {
                    if cc(i, j) != 0.0 {
                        lc = j + 1;
                        break 'outer;
                    }
                }
            }
            lc
        }
    };
    if lastc == 0 {
        return;
    }
    if lastv == 1 {
        let s = 1.0 - tau;
        for j in 0..lastc {
            *a.at_mut(r, c0 + j) *= s;
        }
        return;
    }
    let mut work = vec![0.0; lastc];
    for (j, w) in work.iter_mut().enumerate() {
        let mut t = 0.0;
        for i in 1..lastv {
            t += a.at(r + i, c0 + j) * a.at(r + i, vc);
        }
        *w = t;
    }
    for (j, w) in work.iter_mut().enumerate() {
        *w += a.at(r, c0 + j);
    }
    let mtau = -tau;
    for (j, w) in work.iter().enumerate() {
        *a.at_mut(r, c0 + j) += mtau * *w;
    }
    for (j, w) in work.iter().enumerate() {
        if *w != 0.0 {
            let temp = mtau * *w;
            for i in 1..lastv {
                let vi = a.at(r + i, vc);
                *a.at_mut(r + i, c0 + j) += vi * temp;
            }
        }
    }
}

/// Armadillo `qr_econ(Q, R, X)` for `nrow >= ncol`: `dgeqr2` then `dorg2r`; `R` has its
/// strict lower triangle zeroed.
pub fn qr_econ(x: &Mat) -> (Mat, Mat) {
    let (m, n) = (x.nrow, x.ncol);
    assert!(m >= n);
    let mut a = x.clone();
    let k = m.min(n);
    let mut tau = vec![0.0; k];
    for i in 0..k {
        tau[i] = dlarfg(&mut a, i, i);
        if i + 1 < n {
            dlarf1f_left(&mut a, i, i, tau[i], i + 1);
        }
    }
    let mut r = Mat::zeros(n, n);
    for j in 0..n {
        for i in 0..=j.min(m - 1) {
            *r.at_mut(i, j) = a.at(i, j);
        }
    }
    // dorg2r on the first n columns (here n == k).
    let mut q = Mat::zeros(m, n);
    for j in 0..n {
        for i in 0..m {
            *q.at_mut(i, j) = a.at(i, j);
        }
    }
    for i in (0..k).rev() {
        if i + 1 < n {
            dlarf1f_left(&mut q, i, i, tau[i], i + 1);
        }
        if i + 1 < m {
            let s = -tau[i];
            for l in i + 1..m {
                *q.at_mut(l, i) *= s;
            }
        }
        *q.at_mut(i, i) = 1.0 - tau[i];
        for l in 0..i {
            *q.at_mut(l, i) = 0.0;
        }
    }
    (q, r)
}

/// `solve(R, b)` for upper-triangular `R` as LAPACK `dtrtrs` (`dtrsm` L,U,N,N).
pub fn solve_upper(r: &Mat, b: &[f64]) -> Vec<f64> {
    let n = r.nrow;
    let mut x = b.to_vec();
    for k in (0..n).rev() {
        if x[k] != 0.0 {
            x[k] /= r.at(k, k);
            let xk = x[k];
            for i in 0..k {
                x[i] -= xk * r.at(i, k);
            }
        }
    }
    x
}

/// Armadillo 15.6 `solve(x, R, b)` with default options for the upper-triangular `R` that
/// `qr_econ` returns (`glue_solve_gen_full`, `solve_trimat_rcond`): `dtrtrs` then
/// `dtrcon('1','U','N')`. A zero diagonal, or an rcond below eps or NaN, sends the system to
/// `solve_approx_svd`, LAPACK `dgelsd` with rcond = n * eps, which refuses non-finite input
/// ("solve(): solution not found"; production's message differs, because DESeq2.cpp:356 uses the
/// bool form, which leaves beta empty, and `x * beta_hat` then throws "matrix multiplication:
/// incompatible matrix dimensions: {m}x{p} and 0x1"). A 1 x 1 `R` is not triangular to Armadillo
/// and takes the LU route, which on one element gives the same quotient; its rcond is taken as 1,
/// exact for normal-range values (Armadillo's dgecon gives 0 at subnormals, 1e-308 or f64::MAX).
/// Not ported: the band dispatch Armadillo checks first for 32 or more coefficients, and the
/// dgelsd branch above 25 (refused explicitly).
pub fn arma_solve_upper(r: &Mat, b: &[f64]) -> Result<Vec<f64>, String> {
    let n = r.nrow;
    let singular = (0..n).any(|k| r.at(k, k) == 0.0);
    if !singular {
        let rcond = if n < 2 {
            if r.data[0].is_finite() {
                1.0
            } else {
                f64::NAN
            }
        } else {
            shrink_core::lapack::dtrcon_1u(&r.data, n, n)
        };
        if !(rcond < f64::EPSILON || rcond.is_nan()) {
            return Ok(solve_upper(r, b));
        }
    }
    let not_found = || "solve(): solution not found".to_string();
    if r.data.iter().chain(b).any(|v| !v.is_finite()) {
        return Err(not_found());
    }
    if n > shrink_core::lapack::DGELSD_SMLSIZ {
        return Err(format!(
            "solve(): approximate solution for {n} coefficients needs the dgelsd divide and \
             conquer branch, which is not ported"
        ));
    }
    shrink_core::lapack::dgelsd_square(&r.data, n, b, n as f64 * f64::EPSILON).ok_or_else(not_found)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn spd(n: usize, seed: u64) -> Mat {
        let mut s = seed;
        let mut rnd = || {
            s = s
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((s >> 11) as f64) / ((1u64 << 53) as f64) - 0.5
        };
        let x = Mat::from_col_major(n + 3, n, (0..(n + 3) * n).map(|_| rnd()).collect());
        tmul(&x, &x)
    }

    #[test]
    fn inv_and_det_are_consistent() {
        for n in 1..7 {
            let a = spd(n, n as u64 + 7);
            let ai = inv(&a).unwrap();
            let id = mul(&a, &ai);
            for i in 0..n {
                for j in 0..n {
                    let want = if i == j { 1.0 } else { 0.0 };
                    assert!((id.at(i, j) - want).abs() < 1e-9, "n={n} ({i},{j})");
                }
            }
            let d = det(&a);
            let di = det(&ai);
            assert!((d * di - 1.0).abs() < 1e-9, "n={n}");
        }
    }

    #[test]
    fn qr_reconstructs() {
        let x = Mat::from_col_major(5, 2, vec![1., 1., 1., 1., 1., 0., 0., 1., 1., 3.]);
        let (q, r) = qr_econ(&x);
        let qr = mul(&q, &r);
        for (a, b) in qr.data.iter().zip(x.data.iter()) {
            assert!((a - b).abs() < 1e-14);
        }
        let b = [1.0, -2.0];
        let rb = mul_vec(&r, &solve_upper(&r, &b));
        assert!((rb[0] - 1.0).abs() < 1e-14 && (rb[1] + 2.0).abs() < 1e-14);
    }

    /// Reference values from R 4.5 / RcppArmadillo 15.6 in md-flexi-r45-local: `rcond(R, norm
    /// = "O", triangular = TRUE)` (LAPACK dtrcon) and `arma::solve(x, R, b)` via cppFunction,
    /// which warns "close to singular; rcond: 5.07029e-33" and takes solve_approx_svd.
    #[test]
    fn arma_solve_upper_matches_armadillo() {
        let mut r = Mat::from_col_major(
            4,
            4,
            vec![
                -4.0, 0.0, 0.0, 0.0, -1.3, 2.1, 0.0, 0.0, -1.2, -0.7, 1.9, 0.0, -3e15, 1e15, 5e14,
                0.03,
            ],
        );
        let b = [1.5, -2.25, 0.75, 3.125];
        let rcond = shrink_core::lapack::dtrcon_1u(&r.data, 4, 4);
        assert_eq!(rcond, 5.070288301167497e-33);
        let want = [
            -8.292682926829272e-31,
            -2.9582283945787943e-31,
            -2.4878048780487813e-31,
            -6.219512195121952e-16,
        ];
        assert_eq!(arma_solve_upper(&r, &b).unwrap(), want);

        *r.at_mut(0, 3) = -3.0;
        *r.at_mut(1, 3) = 1.0;
        *r.at_mut(2, 3) = 0.5;
        let rcond = shrink_core::lapack::dtrcon_1u(&r.data, 4, 4);
        assert_eq!(rcond, 0.0028608841315038726);
        let want = [
            -50.99859022556391,
            -59.680451127819545,
            -27.017543859649127,
            104.16666666666667,
        ];
        assert_eq!(arma_solve_upper(&r, &b).unwrap(), want);
        assert_eq!(solve_upper(&r, &b), want);

        *r.at_mut(3, 3) = f64::NAN;
        assert_eq!(
            arma_solve_upper(&r, &b).unwrap_err(),
            "solve(): solution not found"
        );
    }

    /// Boundary: diag(1, eps) has dtrcon rcond exactly eps, which Armadillo accepts (`rcond < eps`
    /// is false), so the plain back substitution must come back: 1 / eps = 2^52. Catches `<=` and
    /// any threshold above eps; diag(1, 1e-17) (rcond 1e-17 < eps) must fall back and zero the
    /// second coordinate, which catches any threshold below 1e-17.
    #[test]
    fn arma_solve_upper_threshold_is_strictly_below_eps() {
        let at_eps = Mat::from_col_major(2, 2, vec![1.0, 0.0, 0.0, f64::EPSILON]);
        assert_eq!(
            shrink_core::lapack::dtrcon_1u(&at_eps.data, 2, 2),
            f64::EPSILON
        );
        assert_eq!(
            arma_solve_upper(&at_eps, &[1.0, 1.0]).unwrap(),
            [1.0, 4503599627370496.0]
        );

        let below = Mat::from_col_major(2, 2, vec![1.0, 0.0, 0.0, 1e-17]);
        assert_eq!(shrink_core::lapack::dtrcon_1u(&below.data, 2, 2), 1e-17);
        assert_eq!(arma_solve_upper(&below, &[1.0, 1.0]).unwrap(), [1.0, 0.0]);
    }

    /// The non-finite refusal must not depend on dgelsd happening to fail: with an Inf above the
    /// diagonal (or in b) dgelsd returns Some(NaN, ...), so only the explicit check refuses.
    #[test]
    fn arma_solve_upper_refuses_infinite_input_on_the_fallback() {
        let b = [1.0, 2.0, 3.0];
        let r = Mat::from_col_major(
            3,
            3,
            vec![1.0, 0.0, 0.0, f64::INFINITY, 1.0, 0.0, 0.5, 0.25, 1e-30],
        );
        assert_eq!(
            arma_solve_upper(&r, &b).unwrap_err(),
            "solve(): solution not found"
        );
        let r = Mat::from_col_major(3, 3, vec![1.0, 0.0, 0.0, 2.0, 1.0, 0.0, 0.5, 0.25, 1e-30]);
        assert_eq!(
            arma_solve_upper(&r, &[1.0, f64::INFINITY, 3.0]).unwrap_err(),
            "solve(): solution not found"
        );
    }

    /// Above 25 coefficients the fallback is not ported: it must be the explicit error, not the
    /// `dgelsd_square` assert (a panic, which reaches Python as RuntimeError "internal error").
    #[test]
    fn arma_solve_upper_refuses_the_unported_large_fallback() {
        for n in [25usize, 26] {
            let mut r = Mat::zeros(n, n);
            for k in 0..n {
                *r.at_mut(k, k) = 1.0;
            }
            *r.at_mut(n - 1, n - 1) = 1e-30;
            let b = vec![1.0; n];
            let got = arma_solve_upper(&r, &b);
            if n <= shrink_core::lapack::DGELSD_SMLSIZ {
                let x = got.unwrap();
                assert_eq!(x[n - 1], 0.0);
                assert!(x[..n - 1].iter().all(|v| *v == 1.0));
            } else {
                let e = got.unwrap_err();
                assert!(
                    e.contains("26 coefficients") && e.contains("not ported"),
                    "{e}"
                );
            }
        }
    }

    /// 1 x 1: the plain quotient for a finite non-zero entry, the dgelsd minimum-norm 0 for a zero
    /// entry, and the refusal for a non-finite one (a NaN rcond must not be read as "fine").
    #[test]
    fn arma_solve_upper_one_by_one() {
        let one = |v: f64| Mat::from_col_major(1, 1, vec![v]);
        assert_eq!(arma_solve_upper(&one(-4.0), &[2.0]).unwrap(), [-0.5]);
        // Armadillo's LU route (dgecon rcond 1) gives the quotient; dgelsd would give
        // 2.5000000000000005e21 here, so this tells the two routes apart.
        assert_eq!(
            arma_solve_upper(&one(1e-21), &[2.5]).unwrap(),
            [2.5 / 1e-21]
        );
        assert_eq!(arma_solve_upper(&one(0.0), &[2.0]).unwrap(), [0.0]);
        for v in [f64::NAN, f64::INFINITY] {
            assert_eq!(
                arma_solve_upper(&one(v), &[2.0]).unwrap_err(),
                "solve(): solution not found"
            );
        }
    }
}
