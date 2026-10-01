//! A private copy of the R 4.5.0 nmath functions DESeq2 reaches (`lgammafn`, `digamma`,
//! `trigamma`, `dnbinom_mu` and their helpers), taken from `rnum::nmath` with one change:
//! every `log`, `exp`, `log1p` and `pow` goes through the bit-exact glibc ports in `rnum`
//! (`glibm`, `glibm_log1p`, `glibm_pow`) instead of the host libm, and `stirlerr` calls the
//! C library's `lgamma` (`rnum::glibm_lgamma`) where R's C does. The reference image runs R on
//! glibc, so these agree with it to the bit where the host libm can differ in the last ulp.
//! `rnum::nmath` itself is shared with other crates and is left untouched.

#![allow(dead_code, clippy::all)]

pub(crate) mod arith;
mod binom;
pub(crate) mod consts;
pub(crate) mod dpq;
pub(crate) mod gamma;
pub(crate) mod nbinom;

pub(crate) use gamma::{digamma, lgammafn, trigamma};
pub(crate) use nbinom::dnbinom_mu;

/// glibc `log`, `exp`, `log1p`, `pow` as methods, so the copied code reads like the original.
pub(crate) trait G {
    fn gln(self) -> f64;
    fn gexp(self) -> f64;
    fn gln_1p(self) -> f64;
    fn gpow(self, y: f64) -> f64;
}

impl G for f64 {
    #[inline]
    fn gln(self) -> f64 {
        rnum::glibm::ln(self)
    }
    #[inline]
    fn gexp(self) -> f64 {
        rnum::glibm::exp(self)
    }
    #[inline]
    fn gln_1p(self) -> f64 {
        rnum::glibm_log1p::log1p(self)
    }
    #[inline]
    fn gpow(self, y: f64) -> f64 {
        rnum::glibm_pow::pow(self, y)
    }
}
