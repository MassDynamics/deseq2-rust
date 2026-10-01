//! The Eigen 3.4 (RcppEigen 0.3.4) kernels apeglm's C++ objective runs through, reproduced
//! operation for operation so the LBFGSpp path is the one R takes.
//!
//! The image compiles packages for x86-64 with SSE2 and no FMA, so a `Packet2d` is two
//! doubles, `pmadd(a, b, c)` is `a * b + c` with two roundings, and the vectorised
//! transcendental functions are Eigen's Cephes-derived `pexp_double` / `plog_double`
//! (`arch/Default/GenericPacketMathFunctions.h`), not the C library. Elements left over
//! after the last full packet go through the scalar functor, i.e. the C library.

const PACKET: usize = 2;

/// `pexp_double` on one lane (SSE2 `pldexp` and the non-SSE4.1 `pfloor`).
pub fn pexp(x0: f64) -> f64 {
    const LOG2EF: f64 = 1.4426950408889634073599;
    const P0: f64 = 1.26177193074810590878e-4;
    const P1: f64 = 3.02994407707441961300e-2;
    const P2: f64 = 9.99999999999999999910e-1;
    const Q0: f64 = 3.00198505138664455042e-6;
    const Q1: f64 = 2.52448340349684104192e-3;
    const Q2: f64 = 2.27265548208155028766e-1;
    const Q3: f64 = 2.00000000000000000009e0;
    const C1: f64 = 0.693145751953125;
    const C2: f64 = 1.42860682030941723212e-6;
    if x0.is_nan() {
        return x0;
    }
    let mut x = x0.min(709.784).max(-709.784);
    let fx = (LOG2EF * x + 0.5).floor();
    let tmp = fx * C1;
    let z = fx * C2;
    x = x - tmp;
    x = x - z;
    let x2 = x * x;
    let mut px = P0;
    px = px * x2 + P1;
    px = px * x2 + P2;
    px = px * x;
    let mut qx = Q0;
    qx = qx * x2 + Q1;
    qx = qx * x2 + Q2;
    qx = qx * x2 + Q3;
    x = px / (qx - px);
    x = 2.0 * x + 1.0;
    let r = pldexp_sse(x, fx);
    if r > x0 {
        r
    } else {
        x0
    }
}

fn pow2i(b: i32) -> f64 {
    f64::from_bits(((b + 1023) as u32 as u64 & 0xFFF) << 52)
}

/// SSE `pldexp<Packet2d>`: `a * 2^e` as four exact power-of-two factors.
fn pldexp_sse(a: f64, exponent: f64) -> f64 {
    let e = exponent.max(-2099.0).min(2099.0);
    let ei = e.round_ties_even() as i32;
    let b = ei >> 2;
    let c = pow2i(b);
    let out = ((a * c) * c) * c;
    let b2 = ei - b - b - b;
    out * pow2i(b2)
}

/// `plog_double` on one lane.
pub fn plog(x0: f64) -> f64 {
    const SQRTHF: f64 = 0.70710678118654752440E0;
    const P0: f64 = 1.01875663804580931796E-4;
    const P1: f64 = 4.97494994976747001425E-1;
    const P2: f64 = 4.70579119878881725854E0;
    const P3: f64 = 1.44989225341610930846E1;
    const P4: f64 = 1.79368678507819816313E1;
    const P5: f64 = 7.70838733755885391666E0;
    const Q0: f64 = 1.0;
    const Q1: f64 = 1.12873587189167450590E1;
    const Q2: f64 = 4.52279145837532221105E1;
    const Q3: f64 = 8.29875266912776603211E1;
    const Q4: f64 = 7.11544750618563894466E1;
    const Q5: f64 = 2.31251620126765340583E1;
    if x0.is_nan() || x0 < 0.0 {
        return f64::NAN;
    }
    if x0 == 0.0 {
        return f64::NEG_INFINITY;
    }
    if x0 == f64::INFINITY {
        return f64::INFINITY;
    }
    let mut x = x0.max(f64::MIN_POSITIVE);
    let bits = x.to_bits();
    let biased = ((bits >> 52) & 0x7FF) as i64;
    let mut e = (biased - 1022) as f64;
    x = f64::from_bits((bits & 0x800F_FFFF_FFFF_FFFF) | 0x3FE0_0000_0000_0000);
    let mask = x < SQRTHF;
    let tmp = if mask { x } else { 0.0 };
    x = x - 1.0;
    e = e - if mask { 1.0 } else { 0.0 };
    x = x + tmp;
    let x2 = x * x;
    let x3 = x2 * x;
    let mut y = P0 * x + P1;
    let mut y1 = P3 * x + P4;
    y = y * x + P2;
    y1 = y1 * x + P5;
    let mut y_ = y * x3 + y1;
    y = Q0 * x + Q1;
    y1 = Q3 * x + Q4;
    y = y * x + Q2;
    y1 = y1 * x + Q5;
    y = y * x3 + y1;
    y_ = y_ * x3;
    y = y_ / y;
    y = -0.5 * x2 + y;
    x = x + y;
    e * std::f64::consts::LN_2 + x
}

/// `ArrayXd::exp()` assigned into an aligned array: packets, then the scalar tail.
pub fn array_exp(v: &[f64]) -> Vec<f64> {
    let full = v.len() / PACKET * PACKET;
    v.iter()
        .enumerate()
        .map(|(i, &x)| {
            if i < full {
                pexp(x)
            } else {
                crate::glibm::exp(x)
            }
        })
        .collect()
}

/// `ArrayXd::log()`, as [`array_exp`].
pub fn array_log(v: &[f64]) -> Vec<f64> {
    let full = v.len() / PACKET * PACKET;
    v.iter()
        .enumerate()
        .map(|(i, &x)| {
            if i < full {
                plog(x)
            } else {
                crate::glibm::log(x)
            }
        })
        .collect()
}

/// Eigen's `redux_impl<LinearVectorizedTraversal, NoUnrolling>` with `scalar_sum_op`
/// over `term(0..n)`, aligned start 0 (true of every operand here).
fn redux_sum_by<F: Fn(usize) -> f64>(n: usize, term: F) -> f64 {
    let aligned_size2 = n / (2 * PACKET) * (2 * PACKET);
    let aligned_size = n / PACKET * PACKET;
    if aligned_size == 0 {
        if n == 0 {
            return 0.0;
        }
        let mut res = term(0);
        for i in 1..n {
            res += term(i);
        }
        return res;
    }
    let mut r0 = [term(0), term(1)];
    if aligned_size > PACKET {
        let mut r1 = [term(2), term(3)];
        let mut idx = 2 * PACKET;
        while idx < aligned_size2 {
            r0[0] += term(idx);
            r0[1] += term(idx + 1);
            r1[0] += term(idx + 2);
            r1[1] += term(idx + 3);
            idx += 2 * PACKET;
        }
        r0[0] += r1[0];
        r0[1] += r1[1];
        if aligned_size > aligned_size2 {
            r0[0] += term(aligned_size2);
            r0[1] += term(aligned_size2 + 1);
        }
    }
    let mut res = r0[0] + r0[1];
    for i in aligned_size..n {
        res += term(i);
    }
    res
}

/// `v.sum()`.
pub fn sum(v: &[f64]) -> f64 {
    redux_sum_by(v.len(), |i| v[i])
}

/// `a.dot(b)`.
pub fn dot(a: &[f64], b: &[f64]) -> f64 {
    redux_sum_by(a.len(), |i| a[i] * b[i])
}

/// `v.squaredNorm()`.
pub fn squared_norm(v: &[f64]) -> f64 {
    redux_sum_by(v.len(), |i| v[i] * v[i])
}

/// `v.norm()`.
pub fn norm(v: &[f64]) -> f64 {
    squared_norm(v).sqrt()
}

/// Column-major GEMV `x * beta` (`general_matrix_vector_product<ColMajor>`, one column
/// block): per row a sequential `c = a * b + c` from zero, then `res = c * 1 + 0`.
/// `x` is `n x p` column-major.
pub fn gemv_colmajor(x: &[f64], n: usize, p: usize, beta: &[f64]) -> Vec<f64> {
    (0..n)
        .map(|i| {
            let mut c = 0.0;
            for j in 0..p {
                c = x[i + j * n] * beta[j] + c;
            }
            c * 1.0 + 0.0
        })
        .collect()
}

/// Row-major GEMV `alpha * x^T * v` (`general_matrix_vector_product<RowMajor>` on the
/// transposed column-major `x`): per output the two packet lanes accumulate even and odd
/// samples, `predux`, the odd tail sample is added, then `res = 0 + alpha * cc`.
pub fn gemv_t_rowmajor(x: &[f64], n: usize, p: usize, v: &[f64], alpha: f64) -> Vec<f64> {
    let full = n / PACKET * PACKET;
    (0..p)
        .map(|k| {
            let col = &x[k * n..(k + 1) * n];
            let mut c = [0.0, 0.0];
            let mut j = 0;
            while j < full {
                c[0] = col[j] * v[j] + c[0];
                c[1] = col[j + 1] * v[j + 1] + c[1];
                j += PACKET;
            }
            let mut cc = c[0] + c[1];
            while j < n {
                cc += col[j] * v[j];
                j += 1;
            }
            0.0 + alpha * cc
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pexp_and_plog_are_close_to_libm() {
        for &x in &[
            -700.0, -20.5, -1.0, -1e-9, 0.0, 1e-12, 0.3, 1.0, 2.5, 30.0, 700.0,
        ] {
            let r = pexp(x) / x.exp() - 1.0;
            assert!(r.abs() < 4e-16, "exp {x}: {r}");
        }
        for &x in &[
            1e-300, 1e-5, 0.5, 0.70710678, 0.9, 1.0, 1.5, 2.0, 1e5, 1e300,
        ] {
            let d = (plog(x) - x.ln()).abs() / x.ln().abs().max(1.0);
            assert!(d < 4e-16, "log {x}: {d}");
        }
        assert_eq!(plog(1.0), 0.0);
    }

    #[test]
    fn redux_order_matches_eigen() {
        let v = [1e16, 1.0, -1e16, 1.0, 3.0];
        assert_eq!(sum(&v), ((1e16 + -1e16) + (1.0 + 1.0)) + 3.0);
        assert_eq!(sum(&v[..3]), (1e16 + 1.0) + -1e16);
    }
}
