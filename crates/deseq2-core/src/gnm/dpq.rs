//! Port of R's `src/nmath/dpq.h` — the `p`/`q` half of its density/probability/quantile
//! helpers — plus `nmath.h`'s `R_forceint` (which `dpq.h` used to hold).
//!
//! The C macros read `lower_tail` and `log_p` out of the enclosing function's scope. Here
//! they are ordinary functions and the flags are passed explicitly, always last, so a call
//! site still lines up with the C one argument for argument.
//!
//! The three macros that `return` from their caller — `R_Q_P01_check`,
//! `R_Q_P01_boundaries`, `R_P_bounds_01` — stay macros, because the early return is the
//! whole point of them.
//!
//! Only the helpers with a caller in this crate are ported; an unused one is a finding.
//!
//! `dpq.h`'s density-only section (`give_log`, `R_D_fexp`, `R_D_rtxp`, `R_D_negInonint`,
//! `R_D_nonint_check`) is not ported: no `d*` function is in scope for this crate.
//!
//! R returns `NaN` from `ML_WARN_return_NAN` after raising an R-level warning. There is no
//! warning channel here, so the ports return `f64::NAN` and nothing else; matching R's
//! return value is the spec (see the crate `README.md`).

use super::consts::M_LN2;
use super::G;

/// `R_D__0` — 0 on the `log_p` scale. (One underscore here: `r_d__0` is not snake case.)
pub(crate) fn r_d_0(log_p: bool) -> f64 {
    if log_p {
        f64::NEG_INFINITY
    } else {
        0.0
    }
}

/// `R_D__1` — 1 on the `log_p` scale.
pub(crate) fn r_d_1(log_p: bool) -> f64 {
    if log_p {
        0.0
    } else {
        1.0
    }
}

/// `R_DT_0` — 0 on the `lower_tail`/`log_p` scale.
pub(crate) fn r_dt_0(lower_tail: bool, log_p: bool) -> f64 {
    if lower_tail {
        r_d_0(log_p)
    } else {
        r_d_1(log_p)
    }
}

/// `R_DT_1` — 1 on the `lower_tail`/`log_p` scale.
pub(crate) fn r_dt_1(lower_tail: bool, log_p: bool) -> f64 {
    if lower_tail {
        r_d_1(log_p)
    } else {
        r_d_0(log_p)
    }
}

/// `R_D_half` — 1/2, lower or upper tail alike.
pub(crate) fn r_d_half(log_p: bool) -> f64 {
    if log_p {
        -M_LN2
    } else {
        0.5
    }
}

/// `R_D_Lval` — `p`. The `0.5 - p + 0.5` is R's, to perhaps gain a bit of accuracy.
pub(crate) fn r_d_lval(p: f64, lower_tail: bool) -> f64 {
    if lower_tail {
        p
    } else {
        0.5 - p + 0.5
    }
}

/// `R_D_Cval` — `1 - p`.
pub(crate) fn r_d_cval(p: f64, lower_tail: bool) -> f64 {
    if lower_tail {
        0.5 - p + 0.5
    } else {
        p
    }
}

/// `R_D_qIv` — `p` in `qF(p, ..)`.
pub(crate) fn r_d_qiv(p: f64, log_p: bool) -> f64 {
    if log_p {
        p.gexp()
    } else {
        p
    }
}

/// `R_D_exp` — `exp(x)`.
pub(crate) fn r_d_exp(x: f64, log_p: bool) -> f64 {
    if log_p {
        x
    } else {
        x.gexp()
    }
}

/// `R_D_log` — `log(p)`.
pub(crate) fn r_d_log(p: f64, log_p: bool) -> f64 {
    if log_p {
        p
    } else {
        p.gln()
    }
}

/// `R_Log1_Exp` — `log(1 - exp(x))`, in a more stable form than `log1p(-exp(x))`.
pub(crate) fn r_log1_exp(x: f64) -> f64 {
    if x > -M_LN2 {
        (-x.exp_m1()).gln()
    } else {
        (-x.gexp()).gln_1p()
    }
}

/// `R_D_LExp` — `log(1 - exp(x))`, more stable still than `log1p(-R_D_qIv(x))`.
pub(crate) fn r_d_lexp(x: f64, log_p: bool) -> f64 {
    if log_p {
        r_log1_exp(x)
    } else {
        (-x).gln_1p()
    }
}

/// `R_DT_qIv` — `p` in `qF`.
pub(crate) fn r_dt_qiv(p: f64, lower_tail: bool, log_p: bool) -> f64 {
    if log_p {
        if lower_tail {
            p.gexp()
        } else {
            -p.exp_m1()
        }
    } else {
        r_d_lval(p, lower_tail)
    }
}

/// `R_DT_CIv` — `1 - p` in `qF`.
pub(crate) fn r_dt_civ(p: f64, lower_tail: bool, log_p: bool) -> f64 {
    if log_p {
        if lower_tail {
            -p.exp_m1()
        } else {
            p.gexp()
        }
    } else {
        r_d_cval(p, lower_tail)
    }
}

/// `R_DT_log` — `log(p)` in `qF`.
pub(crate) fn r_dt_log(p: f64, lower_tail: bool, log_p: bool) -> f64 {
    if lower_tail {
        r_d_log(p, log_p)
    } else {
        r_d_lexp(p, log_p)
    }
}

/// `R_DT_Clog` — `log(1 - p)` in `qF`.
pub(crate) fn r_dt_clog(p: f64, lower_tail: bool, log_p: bool) -> f64 {
    if lower_tail {
        r_d_lexp(p, log_p)
    } else {
        r_d_log(p, log_p)
    }
}

/// `R_forceint` — `nearbyint`, i.e. round half to even under the default rounding mode.
/// `nmath.h` holds this one; the comment there records that it came out of `dpq.h`.
pub(crate) fn r_forceint(x: f64) -> f64 {
    x.round_ties_even()
}
