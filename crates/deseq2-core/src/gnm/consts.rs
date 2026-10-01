//! The `M_*` constants R's `src/nmath` sources use, reached there via `nmath.h` ->
//! `Rmath.h`.
//!
//! Only the constants the in-scope ports reference are here. Names keep R's spelling so a
//! port can be read against the C line for line; `M_LN_SQRT_PId2` is the one exception,
//! respelled `M_LN_SQRT_PI_D2` to be a legal Rust constant name.
//!
//! Where `std::f64::consts` already holds the same correctly-rounded double, the constant
//! aliases it instead of restating R's decimal literal.

// R writes these to ~30 significant digits; they are transcribed verbatim so they can be
// diffed against Rmath.h, and every one is the correctly-rounded double (see the tests).
#![allow(clippy::excessive_precision)]

/// `M_LN2` — `log(2)`.
pub(crate) const M_LN2: f64 = std::f64::consts::LN_2;

/// `M_PI` — pi.
pub(crate) const M_PI: f64 = std::f64::consts::PI;

/// `M_PI_2` — pi/2.
pub(crate) const M_PI_2: f64 = std::f64::consts::FRAC_PI_2;

/// `M_2PI` — 2*pi.
pub(crate) const M_2PI: f64 = std::f64::consts::TAU;

/// `M_1_PI` — 1/pi.
pub(crate) const M_1_PI: f64 = std::f64::consts::FRAC_1_PI;

/// `M_SQRT2` — `sqrt(2)`.
pub(crate) const M_SQRT2: f64 = std::f64::consts::SQRT_2;

/// `M_LOG10_2` — `log10(2)`.
pub(crate) const M_LOG10_2: f64 = std::f64::consts::LOG10_2;

/// `M_SQRT_32` — `sqrt(32)`.
pub(crate) const M_SQRT_32: f64 = 5.656854249492380195206754896838;

/// `M_SQRT_PI` — `sqrt(pi)`.
pub(crate) const M_SQRT_PI: f64 = 1.772453850905516027298167483341;

/// `M_1_SQRT_2PI` — `1/sqrt(2*pi)`.
pub(crate) const M_1_SQRT_2PI: f64 = 0.398942280401432677939946059934;

/// `M_LN_2PI` — `log(2*pi)`.
pub(crate) const M_LN_2PI: f64 = 1.837877066409345483560659472811;

/// `M_LN_SQRT_2PI` — `log(sqrt(2*pi))`.
pub(crate) const M_LN_SQRT_2PI: f64 = 0.918938533204672741780329736406;

/// `M_LN_SQRT_PId2` — `log(sqrt(pi/2))`.
pub(crate) const M_LN_SQRT_PI_D2: f64 = 0.225791352644727432363097614947;
