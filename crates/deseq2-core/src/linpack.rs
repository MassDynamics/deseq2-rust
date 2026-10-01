//! LINPACK `dqrdc2` / `dqrsl` as R 4.5 runs them (`qr()`, `qr.qy`, `qr.qty`, and the
//! `Cdqrls` core of `glm.fit`). Copied from `rnum::linpack` with one change: column norms
//! use R 4.5's `blas2.f90` `dnrm2` ([`crate::la::dnrm2`]), which is what the reference image's
//! BLAS ships; `rnum::linpack` keeps the old `dnrm2` and is shared with other crates.
//!
//! R decides which design columns are *non-estimable* by `dqrdc2`'s pivot
//! rule (a column whose remaining norm falls below `tol` times its original
//! norm is moved to the end). limma's `lm.series` relies on that exact rule
//! per gene when NAs mask different rows, so the Fortran is ported
//! line-for-line rather than replaced by a Householder QR from a crate.
//!
//! Matrices are column-major `Vec<f64>` with explicit `(n, p)`.

use crate::la::dnrm2;

fn ddot(x: &[f64], y: &[f64]) -> f64 {
    let mut s = 0.0;
    for i in 0..x.len() {
        s += x[i] * y[i];
    }
    s
}

/// `y += a * x` (reference `daxpy` skips `a == 0`).
fn daxpy(a: f64, x: &[f64], y: &mut [f64]) {
    if a == 0.0 {
        return;
    }
    for i in 0..x.len() {
        y[i] += a * x[i];
    }
}

/// Result of `dqrdc2`: the packed QR in `qr` (column-major `n x p`),
/// `qraux`, the 0-based column `pivot`, and the `rank`.
#[derive(Debug, Clone)]
pub struct Qr {
    pub qr: Vec<f64>,
    pub n: usize,
    pub p: usize,
    pub qraux: Vec<f64>,
    pub pivot: Vec<usize>,
    pub rank: usize,
    pub tol: f64,
}

/// `dqrdc2`: Householder QR with limited column pivoting. `x` is
/// column-major `n x p` and is overwritten with the packed decomposition.
/// Returns the rank `k`.
pub fn dqrdc2(
    x: &mut [f64],
    n: usize,
    p: usize,
    tol: f64,
    qraux: &mut [f64],
    jpvt: &mut [usize],
    work: &mut [f64],
) -> usize {
    // work is p x 2, column-major: work[j] = work(j,1), work[p + j] = work(j,2)
    let col = |j: usize| j * n;
    if n > 0 {
        for j in 0..p {
            let nrm = dnrm2(&x[col(j)..col(j) + n]);
            qraux[j] = nrm;
            work[j] = nrm;
            work[p + j] = if nrm == 0.0 { 1.0 } else { nrm };
        }
    }
    let lup = n.min(p);
    let mut k = p + 1; // 1-based as in Fortran
    for l in 1..=lup {
        // l is 1-based here; li = l - 1 is the 0-based index.
        loop {
            let li = l - 1;
            if l >= k || qraux[li] >= work[p + li] * tol {
                break;
            }
            // Move column l to the end, shifting the others left.
            for i in 0..n {
                let t = x[col(li) + i];
                for j in (l + 1)..=p {
                    x[col(j - 2) + i] = x[col(j - 1) + i];
                }
                x[col(p - 1) + i] = t;
            }
            let i = jpvt[li];
            let t = qraux[li];
            let tt = work[li];
            let ttt = work[p + li];
            for j in (l + 1)..=p {
                jpvt[j - 2] = jpvt[j - 1];
                qraux[j - 2] = qraux[j - 1];
                work[j - 2] = work[j - 1];
                work[p + j - 2] = work[p + j - 1];
            }
            jpvt[p - 1] = i;
            qraux[p - 1] = t;
            work[p - 1] = tt;
            work[p + p - 1] = ttt;
            k -= 1;
        }
        let li = l - 1;
        if l != n {
            // Householder transformation for column l.
            let mut nrmxl = dnrm2(&x[col(li) + li..col(li) + n]);
            if nrmxl != 0.0 {
                let xll = x[col(li) + li];
                if xll != 0.0 {
                    nrmxl = nrmxl.abs() * if xll < 0.0 { -1.0 } else { 1.0 };
                }
                let s = 1.0 / nrmxl;
                for i in li..n {
                    x[col(li) + i] *= s;
                }
                x[col(li) + li] += 1.0;
                // Apply the transformation to the remaining columns,
                // updating the norms.
                for j in (l + 1)..=p {
                    let ji = j - 1;
                    let (head, tail) = x.split_at_mut(col(ji));
                    let xl = &head[col(li) + li..col(li) + n];
                    let xj = &mut tail[li..n];
                    let t = -ddot(xl, xj) / xl[0];
                    daxpy(t, xl, xj);
                    if qraux[ji] != 0.0 {
                        let r = xj[0].abs() / qraux[ji];
                        let mut tt = 1.0 - r * r;
                        tt = tt.max(0.0);
                        let t = tt;
                        if t.abs() >= 1e-6 {
                            qraux[ji] *= t.sqrt();
                        } else {
                            qraux[ji] = dnrm2(&xj[1..]);
                            work[ji] = qraux[ji];
                        }
                    }
                }
                // Save the transformation.
                qraux[li] = x[col(li) + li];
                x[col(li) + li] = -nrmxl;
            }
        }
    }
    (k - 1).min(n)
}

/// R's `qr(x, tol)` (LINPACK, the default `LAPACK = FALSE`).
pub fn qr_decompose(x: &[f64], n: usize, p: usize, tol: f64) -> Qr {
    assert_eq!(x.len(), n * p, "qr_decompose: x is not n x p");
    let mut qr = x.to_vec();
    let mut qraux = vec![0.0; p];
    let mut pivot: Vec<usize> = (0..p).collect();
    let mut work = vec![0.0; 2 * p];
    let rank = if p == 0 {
        0
    } else {
        dqrdc2(&mut qr, n, p, tol, &mut qraux, &mut pivot, &mut work)
    };
    Qr {
        qr,
        n,
        p,
        qraux,
        pivot,
        rank,
        tol,
    }
}

impl Qr {
    /// Column `j` of the packed matrix, rows `from..n`.
    fn col(&self, j: usize, from: usize) -> &[f64] {
        &self.qr[j * self.n + from..j * self.n + self.n]
    }

    /// Apply the j-th Householder reflector (with the diagonal temporarily
    /// replaced by `qraux[j]`, as `dqrsl` does) to `v[j..n]`.
    fn apply_reflector(&self, j: usize, v: &mut [f64]) {
        if self.qraux[j] == 0.0 {
            return;
        }
        let xj = self.col(j, j);
        let d = self.qraux[j];
        // ddot with the virtual column (d, xj[1..])
        let mut dot = d * v[j];
        for i in 1..xj.len() {
            dot += xj[i] * v[j + i];
        }
        let t = -dot / d;
        if t == 0.0 {
            return;
        }
        v[j] += t * d;
        for i in 1..xj.len() {
            v[j + i] += t * xj[i];
        }
    }

    /// `qr.qty`: returns `Q' y` for one column `y` of length `n`.
    pub fn qty(&self, y: &[f64]) -> Vec<f64> {
        assert_eq!(y.len(), self.n);
        let mut v = y.to_vec();
        let ju = self.rank.min(self.n.saturating_sub(1));
        for j in 0..ju {
            self.apply_reflector(j, &mut v);
        }
        v
    }

    /// `qr.qy`: returns `Q y`.
    pub fn qy(&self, y: &[f64]) -> Vec<f64> {
        assert_eq!(y.len(), self.n);
        let mut v = y.to_vec();
        let ju = self.rank.min(self.n.saturating_sub(1));
        for j in (0..ju).rev() {
            self.apply_reflector(j, &mut v);
        }
        v
    }

    /// Back-solve `R b = qty[0..rank]` (the `cb` branch of `dqrsl`).
    /// Coefficients are in *pivoted* order, length `rank`.
    pub fn coef_pivoted(&self, qty: &[f64]) -> Result<Vec<f64>, String> {
        let k = self.rank;
        let mut b: Vec<f64> = qty[..k].to_vec();
        for j in (0..k).rev() {
            let rjj = self.qr[j * self.n + j];
            if rjj == 0.0 {
                return Err(format!("dqrsl: zero diagonal in R at column {}", j + 1));
            }
            b[j] /= rjj;
            let t = -b[j];
            if j > 0 {
                let (head, _) = b.split_at_mut(j);
                daxpy(t, &self.qr[j * self.n..j * self.n + j], head);
            }
        }
        Ok(b)
    }
}
