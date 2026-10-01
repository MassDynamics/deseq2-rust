//! `dbinom_raw` and `pow1p` from R 4.5.0 `src/nmath/dbinom.c`, copied from `rnum::nmath::f`
//! with glibc `log`/`exp`/`log1p`/`pow` (see `gnm`).

use super::consts::M_LN_2PI;
use super::dpq::{r_d_0, r_d_1, r_d_exp};
use super::gamma::{bd0, stirlerr};
use super::G;

/// `pow1p` in `src/nmath/dbinom.c`: compute `(1+x)^y` accurately also for `|x| << 1`.
fn pow1p(x: f64, y: f64) -> f64 {
    if y.is_nan() {
        return if x == 0.0 { 1.0 } else { y }; // (0+1)^NaN := 1  by standards
    }
    if 0.0 <= y && y == y.trunc() && y <= 4.0 {
        match y as i32 {
            0 => return 1.0,
            1 => return x + 1.0,
            2 => return x * (x + 2.0) + 1.0,
            3 => return x * (x * (x + 3.0) + 3.0) + 1.0,
            4 => return x * (x * (x * (x + 4.0) + 6.0) + 4.0) + 1.0,
            _ => {}
        }
    }
    // naive algorithm in two cases: (1) when 1+x is exact (compiler should not over-optimize !),
    // and (2) when |x| > 1/2 and we have no better algorithm.
    if (x + 1.0) - 1.0 == x || x.abs() > 0.5 || x.is_nan() {
        (1.0 + x).gpow(y)
    } else {
        // not perfect, e.g., for small |x|, non-huge y, use
        // binom expansion 1 + y*x + y(y-1)/2 x^2 + ..
        (y * x.gln_1p()).gexp()
    }
}

/// `dbinom_raw` in `src/nmath/dbinom.c`: the binomial probability of `x` successes in `n`
/// trials with success probability `p` and failure probability `q`, without argument checks.
///
/// `p` and `q` are both passed because one may be represented more accurately than the
/// other (in particular, in `df`). `x` and `n` are not checked to be integers, nor
/// `0 <= p, q <= 1`; the caller does that where necessary.
pub(crate) fn dbinom_raw(x: f64, n: f64, p: f64, q: f64, give_log: bool) -> f64 {
    if p == 0.0 {
        return if x == 0.0 {
            r_d_1(give_log)
        } else {
            r_d_0(give_log)
        };
    }
    if q == 0.0 {
        return if x == n {
            r_d_1(give_log)
        } else {
            r_d_0(give_log)
        };
    }

    // NB: The smaller of p and q is the most accurate
    if x == 0.0 {
        if n == 0.0 {
            return r_d_1(give_log);
        }
        if p > q {
            return if give_log { n * q.gln() } else { q.gpow(n) };
        } else {
            // 0 < p <= 1/2
            return if give_log {
                n * (-p).gln_1p()
            } else {
                pow1p(-p, n)
            };
        }
    }
    if x == n {
        // r = p^x = p^n  -- accurately
        if p > q {
            return if give_log {
                n * (-q).gln_1p()
            } else {
                pow1p(-q, n)
            };
        } else {
            return if give_log { n * p.gln() } else { p.gpow(n) };
        }
    }
    if x < 0.0 || x > n {
        return r_d_0(give_log);
    }

    // n*p or n*q can underflow to zero if n and p or q are small.  This
    // used to occur in dbeta, and gives NaN as from R 2.3.0.
    let lc = stirlerr(n) - stirlerr(x) - stirlerr(n - x) - bd0(x, n * p) - bd0(n - x, n * q);

    // f = (M_2PI*x*(n-x))/n; could overflow or underflow
    // Upto R 2.7.1:
    //  lf = log(M_2PI) + log(x) + log(n-x) - log(n);
    //  -- following is much better for  x << n :
    let lf = M_LN_2PI + x.gln() + (-x / n).gln_1p();

    r_d_exp(lc - 0.5 * lf, give_log)
}
