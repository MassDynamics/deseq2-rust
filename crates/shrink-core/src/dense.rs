//! Dense kernels in the exact operation order of the reference image: R's reference BLAS
//! (`src/extra/blas/blas.f`), R's internal LAPACK 3.12.1 (`dpotrf2`, `dgetrf2`, `dgetrs`,
//! `dpotrs`) and the Armadillo 15 element loops that mixsqp's C++ compiles to.
//!
//! Matrices are column-major `n_rows x n_cols` slices.

/// Column-major dense matrix.
#[derive(Clone, Debug, PartialEq)]
pub struct Mat {
    pub nrow: usize,
    pub ncol: usize,
    pub data: Vec<f64>,
}

impl Mat {
    pub fn zeros(nrow: usize, ncol: usize) -> Mat {
        Mat { nrow, ncol, data: vec![0.0; nrow * ncol] }
    }
    pub fn from_col_major(nrow: usize, ncol: usize, data: Vec<f64>) -> Mat {
        assert_eq!(data.len(), nrow * ncol);
        Mat { nrow, ncol, data }
    }
    #[inline]
    pub fn at(&self, i: usize, j: usize) -> f64 {
        self.data[i + j * self.nrow]
    }
    #[inline]
    pub fn set(&mut self, i: usize, j: usize, v: f64) {
        self.data[i + j * self.nrow] = v;
    }
    pub fn col(&self, j: usize) -> &[f64] {
        &self.data[j * self.nrow..(j + 1) * self.nrow]
    }
    pub fn col_mut(&mut self, j: usize) -> &mut [f64] {
        let n = self.nrow;
        &mut self.data[j * n..(j + 1) * n]
    }
    /// `A(idx, idx)`.
    pub fn submat(&self, idx: &[usize]) -> Mat {
        let k = idx.len();
        let mut out = Mat::zeros(k, k);
        for (c, &j) in idx.iter().enumerate() {
            for (r, &i) in idx.iter().enumerate() {
                out.data[r + c * k] = self.at(i, j);
            }
        }
        out
    }
}

/// Reference DGEMV 'N' with alpha = 1, beta = 0: `y = A x`, column sweep.
pub fn gemv_n(a: &Mat, x: &[f64]) -> Vec<f64> {
    let mut y = vec![0.0; a.nrow];
    for j in 0..a.ncol {
        let t = x[j];
        let col = a.col(j);
        for i in 0..a.nrow {
            y[i] += t * col[i];
        }
    }
    y
}

/// Reference DGEMV 'T' with beta = 0: `y_j = alpha * sum_i A(i,j) x_i`, sequential sum.
pub fn gemv_t(a: &Mat, x: &[f64], alpha: f64) -> Vec<f64> {
    (0..a.ncol)
        .map(|j| {
            let col = a.col(j);
            let mut t = 0.0;
            for i in 0..a.nrow {
                t += col[i] * x[i];
            }
            alpha * t
        })
        .collect()
}

/// Armadillo `trans(Z) * Z` for `n_elem > 48`: reference DSYRK('U','T', alpha 1, beta 0)
/// then the upper triangle copied to the lower one.
pub fn syrk_t(z: &Mat) -> Mat {
    let n = z.ncol;
    let mut c = Mat::zeros(n, n);
    for j in 0..n {
        for i in 0..=j {
            let (ci, cj) = (z.col(i), z.col(j));
            let mut t = 0.0;
            for l in 0..z.nrow {
                t += ci[l] * cj[l];
            }
            c.set(i, j, t);
        }
    }
    for j in 0..n {
        for i in (j + 1)..n {
            let v = c.at(j, i);
            c.set(i, j, v);
        }
    }
    c
}

/// Armadillo `arrayops::accumulate` / `accu` for non-fast-math builds: two interleaved
/// accumulators, added at the end.
pub fn arma_accu<I: IntoIterator<Item = f64>>(xs: I) -> f64 {
    let (mut a1, mut a2) = (0.0, 0.0);
    let mut odd = false;
    for x in xs {
        if odd {
            a2 += x;
        } else {
            a1 += x;
        }
        odd = !odd;
    }
    a1 + a2
}

/// Armadillo `op_dot::direct_dot_generic`.
pub fn arma_dot(a: &[f64], b: &[f64]) -> f64 {
    arma_accu(a.iter().zip(b).map(|(x, y)| x * y))
}

// ---------------------------------------------------------------------------------------
// LAPACK 3.12.1 reference routines (n <= 64 so every blocked driver calls the recursive
// unblocked kernel). `lda` is the leading dimension; `off` the offset of A(1,1).

#[inline]
pub(crate) fn ix(lda: usize, i: usize, j: usize) -> usize {
    i + j * lda
}

/// DTRSM('L','U','T','N', m, n, 1, A, B): B := inv(A^T) B.
fn trsm_lutn(a: &[f64], ao: usize, b: &mut [f64], bo: usize, lda: usize, m: usize, n: usize) {
    for j in 0..n {
        for i in 0..m {
            let mut t = b[bo + ix(lda, i, j)];
            for k in 0..i {
                t -= a[ao + ix(lda, k, i)] * b[bo + ix(lda, k, j)];
            }
            t /= a[ao + ix(lda, i, i)];
            b[bo + ix(lda, i, j)] = t;
        }
    }
}

/// DSYRK('U','T', n, k, -1, A, 1, C): C := C - A^T A (upper).
fn syrk_ut_minus(a: &[f64], ao: usize, c: &mut [f64], co: usize, lda: usize, n: usize, k: usize) {
    for j in 0..n {
        for i in 0..=j {
            let mut t = 0.0;
            for l in 0..k {
                t += a[ao + ix(lda, l, i)] * a[ao + ix(lda, l, j)];
            }
            let p = co + ix(lda, i, j);
            c[p] += -t;
        }
    }
}

/// DTRSM('R','L','T','N', m, n, 1, A, B): B := B inv(A^T), A lower n x n.
fn trsm_rltn(a: &[f64], ao: usize, b: &mut [f64], bo: usize, lda: usize, m: usize, n: usize) {
    for k in 0..n {
        let t = 1.0 / a[ao + ix(lda, k, k)];
        for i in 0..m {
            b[bo + ix(lda, i, k)] *= t;
        }
        for j in (k + 1)..n {
            let ajk = a[ao + ix(lda, j, k)];
            if ajk != 0.0 {
                for i in 0..m {
                    let v = b[bo + ix(lda, i, k)];
                    b[bo + ix(lda, i, j)] -= ajk * v;
                }
            }
        }
    }
}

/// DSYRK('L','N', n, k, -1, A, 1, C): C := C - A A^T (lower).
fn syrk_ln_minus(a: &[f64], ao: usize, c: &mut [f64], co: usize, lda: usize, n: usize, k: usize) {
    for j in 0..n {
        for l in 0..k {
            let ajl = a[ao + ix(lda, j, l)];
            if ajl != 0.0 {
                let t = -ajl;
                for i in j..n {
                    c[co + ix(lda, i, j)] += t * a[ao + ix(lda, i, l)];
                }
            }
        }
    }
}

/// DPOTRF2 (recursive Cholesky). Returns `true` when `info == 0`.
fn potrf2(upper: bool, a: &mut [f64], off: usize, lda: usize, n: usize) -> bool {
    if n == 0 {
        return true;
    }
    if n == 1 {
        let v = a[off];
        if v <= 0.0 || v.is_nan() {
            return false;
        }
        a[off] = v.sqrt();
        return true;
    }
    let n1 = n / 2;
    let n2 = n - n1;
    if !potrf2(upper, a, off, lda, n1) {
        return false;
    }
    let o22 = off + ix(lda, n1, n1);
    if upper {
        let o12 = off + ix(lda, 0, n1);
        let a11 = a[..].to_vec();
        trsm_lutn(&a11, off, a, o12, lda, n1, n2);
        let a12 = a[..].to_vec();
        syrk_ut_minus(&a12, o12, a, o22, lda, n2, n1);
    } else {
        let o21 = off + ix(lda, n1, 0);
        let a11 = a[..].to_vec();
        trsm_rltn(&a11, off, a, o21, lda, n2, n1);
        let a21 = a[..].to_vec();
        syrk_ln_minus(&a21, o21, a, o22, lda, n2, n1);
    }
    potrf2(upper, a, o22, lda, n2)
}

/// Armadillo `chol(R, B)` success test: DPOTRF('U') on a copy.
pub fn chol_upper_ok(b: &Mat) -> bool {
    let mut a = b.data.clone();
    potrf2(true, &mut a, 0, b.nrow, b.nrow)
}

/// DPOTRF('L') in place; `true` on success.
pub fn potrf_lower(a: &mut Mat) -> bool {
    let n = a.nrow;
    potrf2(false, &mut a.data, 0, n, n)
}

/// DPOTRS('L') for one right-hand side, given the DPOTRF('L') factor.
pub fn potrs_lower(f: &Mat, b: &mut [f64]) {
    let n = f.nrow;
    // DTRSM('L','L','N','N')
    for k in 0..n {
        if b[k] != 0.0 {
            b[k] /= f.at(k, k);
            let bk = b[k];
            for i in (k + 1)..n {
                b[i] -= bk * f.at(i, k);
            }
        }
    }
    // DTRSM('L','L','T','N')
    for i in (0..n).rev() {
        let mut t = b[i];
        for k in (i + 1)..n {
            t -= f.at(k, i) * b[k];
        }
        t /= f.at(i, i);
        b[i] = t;
    }
}

/// DGETRF2 (recursive LU with partial pivoting) on the `m x n` block at `off`.
/// `ipiv` receives 1-based-equivalent pivots as 0-based row indices relative to the block.
fn getrf2(a: &mut [f64], off: usize, lda: usize, m: usize, n: usize, ipiv: &mut [usize]) -> usize {
    if m == 0 || n == 0 {
        return 0;
    }
    if m == 1 {
        ipiv[0] = 0;
        return if a[off] == 0.0 { 1 } else { 0 };
    }
    if n == 1 {
        let sfmin = f64::MIN_POSITIVE; // DLAMCH('S') for IEEE double
        let mut imax = 0;
        let mut dmax = a[off].abs();
        for i in 1..m {
            let v = a[off + i].abs();
            if v > dmax {
                dmax = v;
                imax = i;
            }
        }
        ipiv[0] = imax;
        if a[off + imax] != 0.0 {
            if imax != 0 {
                a.swap(off, off + imax);
            }
            let p = a[off];
            if p.abs() >= sfmin {
                let r = 1.0 / p;
                for i in 1..m {
                    a[off + i] *= r;
                }
            } else {
                for i in 1..m {
                    a[off + i] /= p;
                }
            }
            return 0;
        }
        return 1;
    }
    let n1 = m.min(n) / 2;
    let n2 = n - n1;
    let mut info = 0;
    let iinfo = getrf2(a, off, lda, m, n1, &mut ipiv[..n1]);
    if info == 0 && iinfo > 0 {
        info = iinfo;
    }
    // DLASWP(N2, A(1,N1+1), LDA, 1, N1, IPIV, 1)
    laswp(a, off + ix(lda, 0, n1), lda, n2, &ipiv[..n1], 0);
    // DTRSM('L','L','N','U', N1, N2, 1, A, A(1,N1+1))
    for j in 0..n2 {
        let bo = off + ix(lda, 0, n1 + j);
        for k in 0..n1 {
            let bk = a[bo + k];
            if bk != 0.0 {
                for i in (k + 1)..n1 {
                    a[bo + i] -= bk * a[off + ix(lda, i, k)];
                }
            }
        }
    }
    // DGEMM('N','N', M-N1, N2, N1, -1, A(N1+1,1), A(1,N1+1), 1, A(N1+1,N1+1))
    for j in 0..n2 {
        for l in 0..n1 {
            let t = -a[off + ix(lda, l, n1 + j)];
            for i in 0..(m - n1) {
                let v = a[off + ix(lda, n1 + i, l)];
                a[off + ix(lda, n1 + i, n1 + j)] += t * v;
            }
        }
    }
    let iinfo = getrf2(a, off + ix(lda, n1, n1), lda, m - n1, n2, &mut ipiv[n1..m.min(n)]);
    if info == 0 && iinfo > 0 {
        info = iinfo + n1;
    }
    for p in ipiv[n1..m.min(n)].iter_mut() {
        *p += n1;
    }
    // DLASWP(N1, A(1,1), LDA, N1+1, MIN(M,N), IPIV, 1)
    laswp(a, off, lda, n1, &ipiv[..m.min(n)], n1);
    info
}

/// DLASWP with INCX = 1 over pivots `ipiv[k1..]` (0-based rows), applied to `ncols` columns.
fn laswp(a: &mut [f64], off: usize, lda: usize, ncols: usize, ipiv: &[usize], k1: usize) {
    for i in k1..ipiv.len() {
        let ip = ipiv[i];
        if ip != i {
            for j in 0..ncols {
                a.swap(off + ix(lda, i, j), off + ix(lda, ip, j));
            }
        }
    }
}

/// LU factorisation (DGETRF -> DGETRF2). Returns (factor, pivots, info).
pub fn getrf(a: &Mat) -> (Mat, Vec<usize>, usize) {
    let n = a.nrow;
    let mut f = a.clone();
    let mut ipiv = vec![0usize; n.min(a.ncol)];
    let info = getrf2(&mut f.data, 0, n, n, a.ncol, &mut ipiv);
    (f, ipiv, info)
}

/// DGETRS('N') for one right-hand side.
pub fn getrs(f: &Mat, ipiv: &[usize], b: &mut [f64]) {
    let n = f.nrow;
    for (i, &ip) in ipiv.iter().enumerate() {
        if ip != i {
            b.swap(i, ip);
        }
    }
    for k in 0..n {
        let bk = b[k];
        if bk != 0.0 {
            for i in (k + 1)..n {
                b[i] -= bk * f.at(i, k);
            }
        }
    }
    for k in (0..n).rev() {
        if b[k] != 0.0 {
            b[k] /= f.at(k, k);
            let bk = b[k];
            for i in 0..k {
                b[i] -= bk * f.at(i, k);
            }
        }
    }
}

/// Exact reciprocal 1-norm condition number from an LU factor (the quantity DGECON
/// estimates): `1 / (||A||_1 ||A^-1||_1)`.
pub fn rcond_from_lu(a: &Mat, f: &Mat, ipiv: &[usize]) -> f64 {
    let n = a.nrow;
    let anorm = (0..n).map(|j| a.col(j).iter().map(|v| v.abs()).sum::<f64>()).fold(0.0, f64::max);
    let mut inorm: f64 = 0.0;
    for j in 0..n {
        let mut e = vec![0.0; n];
        e[j] = 1.0;
        getrs(f, ipiv, &mut e);
        inorm = inorm.max(e.iter().map(|v| v.abs()).sum());
    }
    if anorm == 0.0 || !inorm.is_finite() {
        return 0.0;
    }
    1.0 / (anorm * inorm)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn spd(n: usize) -> Mat {
        let mut a = Mat::zeros(n, n);
        for j in 0..n {
            for i in 0..n {
                a.set(i, j, 1.0 / (1.0 + (i as f64 - j as f64).abs()) + if i == j { n as f64 } else { 0.0 });
            }
        }
        a
    }

    #[test]
    fn cholesky_and_lu_solve() {
        for n in [1, 2, 3, 5, 8, 13, 20] {
            let a = spd(n);
            assert!(chol_upper_ok(&a));
            let mut f = a.clone();
            assert!(potrf_lower(&mut f));
            let rhs: Vec<f64> = (0..n).map(|i| i as f64 + 1.0).collect();
            let mut x1 = rhs.clone();
            potrs_lower(&f, &mut x1);
            let (lu, ipiv, info) = getrf(&a);
            assert_eq!(info, 0);
            let mut x2 = rhs.clone();
            getrs(&lu, &ipiv, &mut x2);
            for i in 0..n {
                let r: f64 = (0..n).map(|j| a.at(i, j) * x1[j]).sum();
                assert!((r - rhs[i]).abs() < 1e-12, "chol n={n}");
                assert!((x1[i] - x2[i]).abs() < 1e-12, "lu n={n}");
            }
        }
        let mut neg = spd(4);
        neg.set(2, 2, -1.0);
        assert!(!chol_upper_ok(&neg));
    }

    #[test]
    fn lu_pivots() {
        let a = Mat::from_col_major(3, 3, vec![0.0, 2.0, 1.0, 1.0, 1.0, 0.0, 3.0, 1.0, 2.0]);
        let (lu, ipiv, info) = getrf(&a);
        assert_eq!(info, 0);
        let rhs = [1.0, 2.0, 3.0];
        let mut x = rhs.to_vec();
        getrs(&lu, &ipiv, &mut x);
        for i in 0..3 {
            let r: f64 = (0..3).map(|j| a.at(i, j) * x[j]).sum();
            assert!((r - rhs[i]).abs() < 1e-13);
        }
    }
}
