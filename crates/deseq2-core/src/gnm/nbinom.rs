//! Port of R's `src/nmath/dnbinom.c` (`dnbinom_mu`), R 4.5.0. DESeq2's C++ `fitBeta` and its
//! R `logLike` both evaluate the negative binomial density in the `(size, mu)`
//! parametrisation through this function.

use super::binom::dbinom_raw;
use super::dpq::{r_d_0, r_d_1, r_d_exp, r_forceint};
use super::gamma::{dpois_raw, lgamma1p};
use super::G;

/// `dnbinom_mu` in `src/nmath/dnbinom.c`: the negative binomial density with `size` and mean
/// `mu`, or its log when `give_log`. Non-integer `x` (beyond R's `1e-9` relative slack) gives
/// density 0, as R does (R also warns). `mu < 0` or `size < 0` gives NaN.
pub fn dnbinom_mu(x: f64, size: f64, mu: f64, give_log: bool) -> f64 {
    if x.is_nan() || size.is_nan() || mu.is_nan() {
        return x + size + mu;
    }
    if mu < 0.0 || size < 0.0 {
        return f64::NAN;
    }
    if (x - r_forceint(x)).abs() > 1e-9 * x.abs().max(1.0) {
        return r_d_0(give_log);
    }
    if x < 0.0 || !x.is_finite() {
        return r_d_0(give_log);
    }
    if x == 0.0 && size == 0.0 {
        return r_d_1(give_log);
    }
    let x = r_forceint(x);
    if !size.is_finite() {
        return dpois_raw(x, mu, give_log);
    }
    if x == 0.0 {
        return r_d_exp(
            size * (if size < mu {
                (size / (size + mu)).gln()
            } else {
                (-mu / (size + mu)).gln_1p()
            }),
            give_log,
        );
    }
    if x < 1e-10 * size {
        let p = if size < mu {
            (size / (1.0 + size / mu)).gln()
        } else {
            (mu / (1.0 + mu / size)).gln()
        };
        r_d_exp(
            x * p - mu - lgamma1p(x) + (x * (x - 1.0) / (2.0 * size)).gln_1p(),
            give_log,
        )
    } else {
        let p = if give_log {
            if x < size {
                (-x / (size + x)).gln_1p()
            } else {
                (size / (size + x)).gln()
            }
        } else {
            size / (size + x)
        };
        let ans = dbinom_raw(
            size,
            x + size,
            size / (size + mu),
            mu / (size + mu),
            give_log,
        );
        if give_log {
            p + ans
        } else {
            p * ans
        }
    }
}
