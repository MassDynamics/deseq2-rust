//! The arithmetic helpers R's nmath sources reach through `nmath.h`: `fmax2`/`fmin2`
//! (`src/nmath/fmax2.c`, `fmin2.c`) and `R_pow`/`R_pow_di` (`src/main/arithmetic.c`).
//!
//! `fmax2`/`fmin2` differ from `f64::max`/`f64::min` exactly where it matters to a port: a NaN
//! on either side comes back as NaN (`x + y`), where Rust's would return the other operand.

use super::G;
/// `fmax2(x, y)`: the larger of the two, NaN if either is NaN.
pub(crate) fn fmax2(x: f64, y: f64) -> f64 {
    if x.is_nan() || y.is_nan() {
        return x + y;
    }
    if x < y {
        y
    } else {
        x
    }
}

/// `fmin2(x, y)`: the smaller of the two, NaN if either is NaN.
pub(crate) fn fmin2(x: f64, y: f64) -> f64 {
    if x.is_nan() || y.is_nan() {
        return x + y;
    }
    if x < y {
        x
    } else {
        y
    }
}

/// `R_pow(x, y)`: `pow` with R's handling of the non-finite corners.
pub(crate) fn r_pow(x: f64, y: f64) -> f64 {
    if x == 1.0 || y == 0.0 {
        return 1.0;
    }
    if x == 0.0 {
        if y > 0.0 {
            return 0.0;
        } else if y < 0.0 {
            return f64::INFINITY;
        } else {
            return y; // NA or NaN
        }
    }
    if x.is_finite() && y.is_finite() {
        if y == 2.0 {
            return x * x;
        }
        return x.gpow(y);
    }
    if x.is_nan() || y.is_nan() {
        return x + y;
    }
    if !x.is_finite() {
        if x > 0.0 {
            // Inf ^ y
            return if y < 0.0 { 0.0 } else { f64::INFINITY };
        } else {
            // (-Inf) ^ y
            if y.is_finite() && y == y.floor() {
                // (-Inf) ^ n
                return if y < 0.0 {
                    0.0
                } else if y % 2.0 != 0.0 {
                    x
                } else {
                    -x
                };
            }
        }
    }
    if !y.is_finite() && x >= 0.0 {
        if y > 0.0 {
            // y == +Inf
            return if x >= 1.0 { f64::INFINITY } else { 0.0 };
        } else {
            // y == -Inf
            return if x < 1.0 { f64::INFINITY } else { 0.0 };
        }
    }
    f64::NAN
}

/// `R_pow_di(x, n)`: `x` to an integer power by repeated squaring, as R computes it (the
/// rounding of the result differs from `powf`, and the ports that use it are matched to R).
pub(crate) fn r_pow_di(x: f64, n: i32) -> f64 {
    let mut x = x;
    let mut n = n;
    let mut xn = 1.0;
    if x.is_nan() {
        return x;
    }
    if n != 0 {
        if !x.is_finite() {
            return r_pow(x, n as f64);
        }
        let is_neg = n < 0;
        if is_neg {
            n = -n;
        }
        loop {
            if n & 1 != 0 {
                xn *= x;
            }
            n >>= 1;
            if n != 0 {
                x *= x;
            } else {
                break;
            }
        }
        if is_neg {
            xn = 1.0 / xn;
        }
    }
    xn
}
