//! Port of R's gamma family: `chebyshev.c`, `lgammacor.c`, `stirlerr.c`, `gamma.c`,
//! `lgamma.c`, `lbeta.c`, `polygamma.c` (`digamma`, `trigamma`, `psigamma`/`dpsifn`),
//! `bd0.c` (`bd0`, `ebd0`), `dpois.c` (`dpois_raw`), `dgamma.c`, and the helpers `pgamma.c`
//! exports (`log1pmx`, `lgamma1p`, `logspace_add`, `logspace_sub`, `dpois_wrap`), plus limma's
//! `trigammaInverse` and statmod's `logmdigamma`. See `tests/scalar_gamma.rs`.
//!
//! Two places where R's C calls the C library's `lgamma` rather than its own `lgammafn`
//! (`stirlerr.c` for `1 <= n < 5.25` off the half-integer table, and `lbeta.c` for
//! `p < 1e-306`) call `lgammafn` here: Rust's std has no `lgamma`. Both are accurate to a
//! few ulp, so the ports agree with R to well inside the corpus tolerance.
//!
//! R's `frexp`/`ldexp` (used by `ebd0`) are not in Rust's std either and are hand-written
//! below, exactly, from the bit layout.

// The Chebyshev and Bernoulli tables are transcribed from R to 30+ digits so they can be
// diffed against the C, and the Stirling `S*` constants likewise.
#![allow(clippy::excessive_precision)]

use crate::gnm::G;
use crate::gnm::arith::{fmax2, fmin2, r_pow_di};
use crate::gnm::consts::{
    M_2PI, M_LN2, M_LN_2PI, M_LN_SQRT_2PI, M_LN_SQRT_PI_D2, M_LOG10_2, M_PI,
};
use crate::gnm::dpq::{r_d_0, r_d_1, r_d_exp, r_forceint, r_log1_exp};

// ---------------------------------------------------------------------------------------
// chebyshev.c
// ---------------------------------------------------------------------------------------

/// `chebyshev.c: chebyshev_eval` — evaluate the `n`-term Chebyshev series `a` at `x`.
pub(crate) fn chebyshev_eval(x: f64, a: &[f64], n: usize) -> f64 {
    if !(1..=1000).contains(&n) {
        return f64::NAN;
    }
    // Not `contains`: a NaN `x` must fall through as it does in C (it comes out NaN anyway).
    #[allow(clippy::manual_range_contains)]
    if x < -1.1 || x > 1.1 {
        return f64::NAN;
    }

    let twox = x * 2.0;
    let mut b2 = 0.0;
    let mut b1 = 0.0;
    let mut b0 = 0.0;
    for i in 1..=n {
        b2 = b1;
        b1 = b0;
        b0 = twox * b1 - b2 + a[n - i];
    }
    (b0 - b2) * 0.5
}

// ---------------------------------------------------------------------------------------
// lgammacor.c
// ---------------------------------------------------------------------------------------

/// `lgammacor.c: lgammacor` — the log gamma correction factor for `x >= 10`:
/// `log(gamma(x)) = .5*log(2*pi) + (x-.5)*log(x) - x + lgammacor(x)`.
pub(crate) fn lgammacor(x: f64) -> f64 {
    // below, nalgm = 5 ==> only the first 5 are used!
    const ALGMCS: [f64; 15] = [
        0.1666389480451863247205729650822e+0,
        -0.1384948176067563840732986059135e-4,
        0.9810825646924729426157171547487e-8,
        -0.1809129475572494194263306266719e-10,
        0.6221098041892605227126015543416e-13,
        -0.3399615005417721944303330599666e-15,
        0.2683181998482698748957538846666e-17,
        -0.2868042435334643284144622399999e-19,
        0.3962837061046434803679306666666e-21,
        -0.6831888753985766870111999999999e-23,
        0.1429227355942498147573333333333e-24,
        -0.3547598158101070547199999999999e-26,
        0.1025680058010470912000000000000e-27,
        -0.3401102254316748799999999999999e-29,
        0.1276642195630062933333333333333e-30,
    ];
    // For IEEE double precision DBL_EPSILON = 2^-52 = 2.220446049250313e-16 :
    //   xbig = 2 ^ 26.5
    //   xmax = DBL_MAX / 48 =  2^1020 / 3
    const NALGM: usize = 5;
    const XBIG: f64 = 94906265.62425156;

    if x < 10.0 {
        // possibly consider stirlerr()
        return f64::NAN;
    } else if x < XBIG {
        let tmp = 10.0 / x;
        return chebyshev_eval(tmp * tmp * 2.0 - 1.0, &ALGMCS, NALGM) / x;
    }
    // x >= xbig
    1.0 / (x * 12.0)
}

// ---------------------------------------------------------------------------------------
// stirlerr.c
// ---------------------------------------------------------------------------------------

/// `stirlerr.c: stirlerr` — the log of the error term in Stirling's formula,
/// `stirlerr(n) = log(n!) - log( sqrt(2*pi*n)*(n/e)^n )`.
///
/// For `n > 15` uses the series `1/12n - 1/360n^3 + ...`; for `n <= 15` integers or
/// half-integers, stored values; for other `n < 15`, `lgamma` (`n >= 1`) or `lgamma1p`
/// (`n < 1`) directly. R 4.5.0's version.
pub(crate) fn stirlerr(n: f64) -> f64 {
    const S0: f64 = 0.083333333333333333333; /* 1/12 */
    const S1: f64 = 0.00277777777777777777778; /* 1/360 */
    const S2: f64 = 0.00079365079365079365079365; /* 1/1260 */
    const S3: f64 = 0.000595238095238095238095238; /* 1/1680 */
    const S4: f64 = 0.0008417508417508417508417508; /* 1/1188 */
    const S5: f64 = 0.0019175269175269175269175262; // 691/360360
    const S6: f64 = 0.0064102564102564102564102561; // 1/156
    const S7: f64 = 0.029550653594771241830065352; // 3617/122400
    const S8: f64 = 0.17964437236883057316493850; // 43867/244188
    const S9: f64 = 1.3924322169059011164274315; // 174611/125400
    const S10: f64 = 13.402864044168391994478957; // 77683/5796
    const S11: f64 = 156.84828462600201730636509; // 236364091/1506960
    const S12: f64 = 2193.1033333333333333333333; // 657931/300
    const S13: f64 = 36108.771253724989357173269; // 3392780147/93960
    const S14: f64 = 691472.26885131306710839498; // 1723168255201/2492028
    const S15: f64 = 15238221.539407416192283370; // 7709321041217/505920
    const S16: f64 = 382900751.39141414141414141; // 151628697551/396

    // exact values for 0, 0.5, 1.0, 1.5, ..., 14.5, 15.0.
    const SFERR_HALVES: [f64; 31] = [
        0.0,                           /* n=0 - wrong, place holder only */
        0.1534264097200273452913848,   /* 0.5 */
        0.0810614667953272582196702,   /* 1.0 */
        0.0548141210519176538961390,   /* 1.5 */
        0.0413406959554092940938221,   /* 2.0 */
        0.03316287351993628748511048,  /* 2.5 */
        0.02767792568499833914878929,  /* 3.0 */
        0.02374616365629749597132920,  /* 3.5 */
        0.02079067210376509311152277,  /* 4.0 */
        0.01848845053267318523077934,  /* 4.5 */
        0.01664469118982119216319487,  /* 5.0 */
        0.01513497322191737887351255,  /* 5.5 */
        0.01387612882307074799874573,  /* 6.0 */
        0.01281046524292022692424986,  /* 6.5 */
        0.01189670994589177009505572,  /* 7.0 */
        0.01110455975820691732662991,  /* 7.5 */
        0.010411265261972096497478567, /* 8.0 */
        0.009799416126158803298389475, /* 8.5 */
        0.009255462182712732917728637, /* 9.0 */
        0.008768700134139385462952823, /* 9.5 */
        0.008330563433362871256469318, /* 10.0 */
        0.007934114564314020547248100, /* 10.5 */
        0.007573675487951840794972024, /* 11.0 */
        0.007244554301320383179543912, /* 11.5 */
        0.006942840107209529865664152, /* 12.0 */
        0.006665247032707682442354394, /* 12.5 */
        0.006408994188004207068439631, /* 13.0 */
        0.006171712263039457647532867, /* 13.5 */
        0.005951370112758847735624416, /* 14.0 */
        0.005746216513010115682023589, /* 14.5 */
        0.005554733551962801371038690, /* 15.0 */
    ];

    let mut nn;

    if n <= 23.5 {
        nn = n + n;
        // C indexes the table with `(int)nn`; a negative `n` would be undefined behaviour
        // there, so the `>= 0` guard only closes a path C never takes.
        if n <= 15.0 && nn >= 0.0 && nn == (nn as i32) as f64 {
            return SFERR_HALVES[nn as usize];
        }
        // else:
        if n <= 5.25 {
            if n >= 1.0 {
                // "MM2"; slightly more accurate than direct form
                let l_n = n.gln(); // ldexp(u, -1) == u/2
                return rnum::glibm_lgamma::lgamma(n) + n * (1.0 - l_n) + (l_n - M_LN_2PI) * 0.5;
            } else {
                // n < 1
                return lgamma1p(n) - (n + 0.5) * n.gln() + n - M_LN_SQRT_2PI;
            }
        }
        // else  5.25 < n <= 23.5
        nn = n * n;
        if n > 12.8 {
            return (S0 - (S1 - (S2 - (S3 - (S4 - (S5 - S6 / nn) / nn) / nn) / nn) / nn) / nn) / n;
            // k = 7
        }
        if n > 12.3 {
            return (S0
                - (S1 - (S2 - (S3 - (S4 - (S5 - (S6 - S7 / nn) / nn) / nn) / nn) / nn) / nn) / nn)
                / n; // k = 8
        }
        if n > 8.9 {
            return (S0
                - (S1
                    - (S2 - (S3 - (S4 - (S5 - (S6 - (S7 - S8 / nn) / nn) / nn) / nn) / nn) / nn)
                        / nn)
                    / nn)
                / n; // k = 9
        }
        /* skip k = 10 */
        if n > 7.3 {
            return (S0
                - (S1
                    - (S2
                        - (S3
                            - (S4
                                - (S5
                                    - (S6 - (S7 - (S8 - (S9 - S10 / nn) / nn) / nn) / nn)
                                        / nn)
                                    / nn)
                                / nn)
                            / nn)
                        / nn)
                    / nn)
                / n; // 11
        }
        /* skip k=12 */
        if n > 6.6 {
            return (S0
                - (S1
                    - (S2
                        - (S3
                            - (S4
                                - (S5
                                    - (S6
                                        - (S7
                                            - (S8
                                                - (S9
                                                    - (S10 - (S11 - S12 / nn) / nn) / nn)
                                                    / nn)
                                                / nn)
                                            / nn)
                                        / nn)
                                    / nn)
                                / nn)
                            / nn)
                        / nn)
                    / nn)
                / n;
        }
        /* skip k= 14 */
        if n > 6.1 {
            return (S0
                - (S1
                    - (S2
                        - (S3
                            - (S4
                                - (S5
                                    - (S6
                                        - (S7
                                            - (S8
                                                - (S9
                                                    - (S10
                                                        - (S11
                                                            - (S12
                                                                - (S13 - S14 / nn)
                                                                    / nn)
                                                                / nn)
                                                            / nn)
                                                        / nn)
                                                    / nn)
                                                / nn)
                                            / nn)
                                        / nn)
                                    / nn)
                                / nn)
                            / nn)
                        / nn)
                    / nn)
                / n; // k = 15
        }
        /* skip order k=16 : never "good" for double prec */
        /* 6.1 >= n > 5.25 */
        (S0 - (S1
            - (S2
                - (S3
                    - (S4
                        - (S5
                            - (S6
                                - (S7
                                    - (S8
                                        - (S9
                                            - (S10
                                                - (S11
                                                    - (S12
                                                        - (S13
                                                            - (S14
                                                                - (S15 - S16 / nn)
                                                                    / nn)
                                                                / nn)
                                                            / nn)
                                                        / nn)
                                                    / nn)
                                                / nn)
                                            / nn)
                                        / nn)
                                    / nn)
                                / nn)
                            / nn)
                        / nn)
                    / nn)
                / nn)
            / nn)
            / n
    } else {
        // n > 23.5
        nn = n * n;
        if n > 15.7e6 {
            return S0 / n;
        }
        if n > 6180.0 {
            return (S0 - S1 / nn) / n;
        }
        if n > 205.0 {
            return (S0 - (S1 - S2 / nn) / nn) / n;
        }
        if n > 86.0 {
            return (S0 - (S1 - (S2 - S3 / nn) / nn) / nn) / n;
        }
        if n > 27.0 {
            return (S0 - (S1 - (S2 - (S3 - S4 / nn) / nn) / nn) / nn) / n;
        }
        /* 23.5 < n <= 27 */
        (S0 - (S1 - (S2 - (S3 - (S4 - S5 / nn) / nn) / nn) / nn) / nn) / n
    }
}

// ---------------------------------------------------------------------------------------
// cospi.c: sinpi (only what gammafn / lgammafn_sign need)
// ---------------------------------------------------------------------------------------

/// `cospi.c: sinpi` — `sin(pi * x)`, exact when `x = k/2` for all integer `k`.
fn sinpi(x: f64) -> f64 {
    if x.is_nan() {
        return x;
    }
    if !x.is_finite() {
        return f64::NAN;
    }
    let mut x = x % 2.0; // sin(pi(x + 2k)) == sin(pi x)  for all integer k
                         // map (-2,2) --> (-1,1] :
    if x <= -1.0 {
        x += 2.0;
    } else if x > 1.0 {
        x -= 2.0;
    }
    if x == 0.0 || x == 1.0 {
        return 0.0;
    }
    if x == 0.5 {
        return 1.0;
    }
    if x == -0.5 {
        return -1.0;
    }
    // otherwise
    (M_PI * x).sin()
}

// ---------------------------------------------------------------------------------------
// gamma.c
// ---------------------------------------------------------------------------------------

/// `gamma.c: gammafn` — the gamma function. Fullerton's Chebyshev series on `|x| <= 10`,
/// Stirling beyond; `n!` for integer `n < 50` computed exactly as a product.
pub fn gammafn(x: f64) -> f64 {
    const GAMCS: [f64; 42] = [
        0.8571195590989331421920062399942e-2,
        0.4415381324841006757191315771652e-2,
        0.5685043681599363378632664588789e-1,
        -0.4219835396418560501012500186624e-2,
        0.1326808181212460220584006796352e-2,
        -0.1893024529798880432523947023886e-3,
        0.3606925327441245256578082217225e-4,
        -0.6056761904460864218485548290365e-5,
        0.1055829546302283344731823509093e-5,
        -0.1811967365542384048291855891166e-6,
        0.3117724964715322277790254593169e-7,
        -0.5354219639019687140874081024347e-8,
        0.9193275519859588946887786825940e-9,
        -0.1577941280288339761767423273953e-9,
        0.2707980622934954543266540433089e-10,
        -0.4646818653825730144081661058933e-11,
        0.7973350192007419656460767175359e-12,
        -0.1368078209830916025799499172309e-12,
        0.2347319486563800657233471771688e-13,
        -0.4027432614949066932766570534699e-14,
        0.6910051747372100912138336975257e-15,
        -0.1185584500221992907052387126192e-15,
        0.2034148542496373955201026051932e-16,
        -0.3490054341717405849274012949108e-17,
        0.5987993856485305567135051066026e-18,
        -0.1027378057872228074490069778431e-18,
        0.1762702816060529824942759660748e-19,
        -0.3024320653735306260958772112042e-20,
        0.5188914660218397839717833550506e-21,
        -0.8902770842456576692449251601066e-22,
        0.1527474068493342602274596891306e-22,
        -0.2620731256187362900257328332799e-23,
        0.4496464047830538670331046570666e-24,
        -0.7714712731336877911703901525333e-25,
        0.1323635453126044036486572714666e-25,
        -0.2270999412942928816702313813333e-26,
        0.3896418998003991449320816639999e-27,
        -0.6685198115125953327792127999999e-28,
        0.1146998663140024384347613866666e-28,
        -0.1967938586345134677295103999999e-29,
        0.3376448816585338090334890666666e-30,
        -0.5793070335782135784625493333333e-31,
    ];

    // For IEEE double precision DBL_EPSILON = 2^-52 = 2.220446049250313e-16 :
    // (xmin, xmax) are non-trivial, see ./gammalims.c
    // xsml = exp(.01)*DBL_MIN
    // dxrel = sqrt(DBL_EPSILON) = 2 ^ -26
    const NGAM: usize = 22;
    const XMIN: f64 = -170.5674972726612;
    const XMAX: f64 = 171.61447887182298;
    const XSML: f64 = 2.2474362225598545e-308;
    // `dxrel` only feeds the "less than half precision" ML_WARNINGs, which are dropped.

    if x.is_nan() {
        return x;
    }

    // If the argument is exactly zero or a negative integer
    // then return NaN.
    if x == 0.0 || (x < 0.0 && x == x.round()) {
        return f64::NAN;
    }

    let mut y = x.abs();
    let mut value;

    if y <= 10.0 {
        // Compute gamma(x) for -10 <= x <= 10
        // Reduce the interval and find gamma(1 + y) for 0 <= y < 1
        // first of all.

        let mut n = x as i32;
        if x < 0.0 {
            n -= 1;
        }
        y = x - n as f64; /* n = floor(x)  ==>	y in [ 0, 1 ) */
        n -= 1;
        value = chebyshev_eval(y * 2.0 - 1.0, &GAMCS, NGAM) + 0.9375;
        if n == 0 {
            return value; /* x = 1.dddd = 1+y */
        }

        if n < 0 {
            // compute gamma(x) for -10 <= x < 1

            // exact 0 or "-n" checked already above

            // (The "answer is less than half precision because x too near a negative
            // integer" ML_WARNING is dropped.)

            // The argument is so close to 0 that the result would overflow.
            if y < XSML {
                if x > 0.0 {
                    return f64::INFINITY;
                } else {
                    return f64::NEG_INFINITY;
                }
            }

            n = -n;

            for i in 0..n {
                value /= x + i as f64;
            }
            value
        } else {
            // gamma(x) for 2 <= x <= 10

            for i in 1..=n {
                value *= y + i as f64;
            }
            value
        }
    } else {
        // gamma(x) for	 y = |x| > 10.

        if x > XMAX {
            /* Overflow */
            // No warning: +Inf is the best answer
            return f64::INFINITY;
        }

        if x < XMIN {
            /* Underflow */
            // No warning: 0 is the best answer
            return 0.0;
        }

        if y <= 50.0 && y == (y as i32) as f64 {
            /* compute (n - 1)! */
            value = 1.0;
            let mut i = 2;
            while (i as f64) < y {
                value *= i as f64;
                i += 1;
            }
        } else {
            /* normal case */
            // C: `(2*y == (int)2*y) ? stirlerr(y) : lgammacor(y)`. The cast binds tighter
            // than `*`, so `(int)2*y` is `2*y` and the test is a tautology for the finite
            // `y` that reaches here: R always takes the `stirlerr` branch, and so does this.
            value = ((y - 0.5) * y.gln() - y + M_LN_SQRT_2PI + stirlerr(y)).gexp();
        }

        if x > 0.0 {
            return value;
        }
        // else:  x < 0, not an integer :

        // (The "too near a negative integer" ML_WARNING is dropped.)

        let sinpiy = sinpi(y);
        if sinpiy == 0.0 {
            /* Negative integer arg - overflow */
            return f64::INFINITY;
        }

        -M_PI / (y * sinpiy * value)
    }
}

// ---------------------------------------------------------------------------------------
// lgamma.c
// ---------------------------------------------------------------------------------------

/// `lgamma.c: lgammafn_sign` — `log|gamma(x)|`, and when `sgn` is given, the sign of
/// `gamma(x)` written to it.
pub(crate) fn lgammafn_sign(x: f64, sgn: Option<&mut i32>) -> f64 {
    // For IEEE double precision DBL_EPSILON = 2^-52 = 2.220446049250313e-16 :
    //   xmax  = DBL_MAX / log(DBL_MAX) = 2^1024 / (1024 * log(2)) = 2^1014 / log(2)
    //   dxrel = sqrt(DBL_EPSILON) = 2^-26 = 5^26 * 1e-26 (is *exact* below !)
    const XMAX: f64 = 2.5327372760800758e+305;
    // `dxrel` only feeds the "less than half precision" ML_WARNING, which is dropped.

    if let Some(sgn) = sgn {
        *sgn = 1;
        if x < 0.0 && (-x).floor() % 2.0 == 0.0 {
            *sgn = -1;
        }
    }

    if x.is_nan() {
        return x;
    }

    if x <= 0.0 && x == x.trunc() {
        /* Negative integer argument */
        // No warning: this is the best answer; was  ML_WARNING(ME_RANGE, "lgamma");
        return f64::INFINITY; /* +Inf, since lgamma(x) = log|gamma(x)| */
    }

    let y = x.abs();

    if y < 1e-306 {
        return -y.gln(); // denormalized range, R change
    }
    if y <= 10.0 {
        return gammafn(x).abs().gln();
    }
    // ELSE  y = |x| > 10 ----------------------

    if y > XMAX {
        // No warning: +Inf is the best answer
        return f64::INFINITY;
    }

    if x > 0.0 {
        /* i.e. y = x > 10 */
        if x > 1e17 {
            return x * (x.gln() - 1.0);
        } else if x > 4934720.0 {
            return M_LN_SQRT_2PI + (x - 0.5) * x.gln() - x;
        } else {
            return M_LN_SQRT_2PI + (x - 0.5) * x.gln() - x + lgammacor(x);
        }
    }
    /* else: x < -10; y = -x */
    let sinpiy = sinpi(y).abs();

    if sinpiy == 0.0 {
        /* Negative integer argument === Now UNNECESSARY: caught above */
        return f64::NAN;
    }

    // (The "too near a negative integer" ML_WARNING is dropped.)
    M_LN_SQRT_PI_D2 + (x - 0.5) * y.gln() - x - sinpiy.gln() - lgammacor(y)
}

/// `lgamma.c: lgammafn` — `log|gamma(x)|`.
pub fn lgammafn(x: f64) -> f64 {
    lgammafn_sign(x, None)
}

// ---------------------------------------------------------------------------------------
// lbeta.c
// ---------------------------------------------------------------------------------------

/// `lbeta.c: lbeta` — `log B(a,b) = log G(a) + log G(b) - log G(a+b)`.
pub fn lbeta(a: f64, b: f64) -> f64 {
    if a.is_nan() || b.is_nan() {
        return a + b;
    }
    let mut p = a;
    let mut q = a;
    if b < p {
        p = b; /* := min(a,b) */
    }
    if b > q {
        q = b; /* := max(a,b) */
    }

    /* both arguments must be >= 0 */
    if p < 0.0 {
        return f64::NAN;
    } else if p == 0.0 {
        return f64::INFINITY;
    } else if !q.is_finite() {
        /* q == +Inf */
        return f64::NEG_INFINITY;
    }

    if p >= 10.0 {
        /* p and q are big. */
        let corr = lgammacor(p) + lgammacor(q) - lgammacor(p + q);
        q.gln() * -0.5
            + M_LN_SQRT_2PI
            + corr
            + (p - 0.5) * (p / (p + q)).gln()
            + q * (-p / (p + q)).gln_1p()
    } else if q >= 10.0 {
        /* p is small, but q is big. */
        let corr = lgammacor(q) - lgammacor(p + q);
        lgammafn(p) + corr + p - p * (p + q).gln() + (q - 0.5) * (-p / (p + q)).gln_1p()
    } else {
        /* p and q are small: p <= q < 10. */
        /* R change for very small args */
        if p < 1e-306 {
            // C calls libm `lgamma` here; see the module note.
            lgammafn(p) + (lgammafn(q) - lgammafn(p + q))
        } else {
            (gammafn(p) * (gammafn(q) / gammafn(p + q))).gln()
        }
    }
}

// ---------------------------------------------------------------------------------------
// polygamma.c
// ---------------------------------------------------------------------------------------

/// `polygamma.c: n_max`.
const N_MAX: i32 = 100;

/// `polygamma.c: d_n_cot` — `d_n(x) = (d/dx)^n cot(x)`; `cot(x) := cos(x) / sin(x)`.
fn d_n_cot(x: f64, n: i32) -> f64 {
    if n == 0 {
        x.cos() / x.sin()
    } else if n == 1 {
        // -1/sin^2
        -1.0 / r_pow_di(x.sin(), 2)
    } else if n == 2 {
        // 2 cos / sin^3
        2.0 * x.cos() / r_pow_di(x.sin(), 3)
    } else if n == 3 {
        // (-4 cos^2 - 2) / sin^4 ; num.= -2(2 cos^2 +1) = -2(3 - 2 sin^2)
        let sin2 = r_pow_di(x.sin(), 2); // = sin^2
        -2.0 * (3.0 - 2.0 * sin2) / r_pow_di(sin2, 2)
    } else if n == 4 {
        // 8 cos (cos^2  + 2) / sin^5
        let co = x.cos();
        8.0 * co * (r_pow_di(co, 2) + 2.0) / r_pow_di(x.sin(), 5)
    } else if n == 5 {
        // (-16 cos^4 - 88 cos^2 - 16)/ sin^6
        let co2 = r_pow_di(x.cos(), 2); // cos^2
        -8.0 * (2.0 * r_pow_di(co2, 2) + 11.0 * co2 + 2.0) / r_pow_di(x.sin(), 6)
    } else {
        f64::NAN
    }
}

/// `polygamma.c: dpsifn` — Amos's scaled derivatives of the psi function: for fixed `x`
/// and `m`, `ans[j] = (-1)^(k+1) / gamma(k+1) * psi(k, x)` for `k = n, ..., n+m-1`
/// (`kode = 2` returns `-psi(x) + ln(x)` in place of `-psi(x)` when `k = 0`). `nz` counts
/// underflowed trailing members; `ierr` is 0 on success, 1 for bad input, 2 for
/// overflow, 3 when `n` is too large for the recursion table, 4 for `x < 0` with `n > 5`.
///
/// The C indexes its `trm`/`trmr` work arrays from 1 and drives them from the loop
/// counter, which is why the range loops stay.
#[allow(clippy::needless_range_loop)]
pub(crate) fn dpsifn(
    x: f64,
    n: i32,
    kode: i32,
    m: i32,
    ans: &mut [f64],
    nz: &mut i32,
    ierr: &mut i32,
) {
    /* Bernoulli Numbers */
    const BVALUES: [f64; 22] = [
        1.00000000000000000e+00,
        -5.00000000000000000e-01,
        1.66666666666666667e-01,
        -3.33333333333333333e-02,
        2.38095238095238095e-02,
        -3.33333333333333333e-02,
        7.57575757575757576e-02,
        -2.53113553113553114e-01,
        1.16666666666666667e+00,
        -7.09215686274509804e+00,
        5.49711779448621554e+01,
        -5.29124242424242424e+02,
        6.19212318840579710e+03,
        -8.65802531135531136e+04,
        1.42551716666666667e+06,
        -2.72982310678160920e+07,
        6.01580873900642368e+08,
        -1.51163157670921569e+10,
        4.29614643061166667e+11,
        -1.37116552050883328e+13,
        4.88332318973593167e+14,
        -1.92965793419400681e+16,
    ];

    let mut trm = [0.0f64; 23];
    let mut trmr = [0.0f64; N_MAX as usize + 1];

    *ierr = 0;
    *nz = 0;
    if n < 0 || !(1..=2).contains(&kode) || m < 1 {
        *ierr = 1;
        return;
    }
    if x <= 0.0 {
        /* use	Abramowitz & Stegun 6.4.7 "Reflection Formula", p.260
         *  psi(n, 1-x) + (-1)^(n+1) psi(n, x) = (-1)^n pi (d/dx)^n cot(pi*x)
         *  psi(n, x) = (-1)^n psi(n, 1-x)   -  pi^{n+1} d_n(pi*x),
         *               where    d_n(x) := (d/dx)^n cot(x)
         */
        if x == x.round() {
            /* non-positive integer : +Inf or NaN depends on n */
            for (j, a) in ans.iter_mut().enumerate().take(m as usize) {
                /* k = j + n : */
                *a = if (j as i32 + n) % 2 != 0 {
                    f64::INFINITY
                } else {
                    f64::NAN
                };
            }
            return;
        }
        /* This could cancel badly */
        dpsifn(1.0 - x, n, /*kode = */ 1, m, ans, nz, ierr);
        /* ans[j] == (-1)^(k+1) / gamma(k+1) * psi(k, 1 - x)
         *	     for j = 0:(m-1) ,	k = n + j
         */

        /* For now: only work for  n in {0,1,..,5} : */
        if n > 5 {
            /* not yet implemented for x < 0 and n >= 6 */
            *ierr = 4;
            return;
        }
        // tt := d_n(pi * x)
        let x = x * M_PI; /* pi * x */

        // t := pi^(n+1) * d_n(x) / gamma(n+1)
        let mut t1 = 1.0;
        let mut t2 = 1.0;
        let mut s = 1.0;
        let mut k = 0;
        let mut j = k - n;
        while j < m {
            /* k == n+j , s = (-1)^k */
            t1 *= M_PI; /* t1 == pi^(k+1) */
            if k >= 2 {
                t2 *= k as f64; /* t2 == k! == gamma(k+1) */
            }
            if j >= 0 {
                /* now using d_k(x) */
                ans[j as usize] = s * (ans[j as usize] + t1 / t2 * d_n_cot(x, k));
            }
            k += 1;
            j += 1;
            s = -s;
        }
        /* if (n == 0 && kode == 2)  -- nonsense for x < 0 !
         *     ans[0] += log(x); */
        return;
    } /* x <= 0 */

    /* else :  x > 0 */
    let xln = x.gln();
    if kode == 1 && m == 1 {
        /* the R case  ---  for very large x: */
        let lrg = 1.0 / (2.0 * f64::EPSILON);
        if n == 0 && x * xln > lrg {
            ans[0] = -xln;
            return;
        } else if n >= 1 && x > n as f64 * lrg {
            ans[0] = (-(n as f64) * xln).gexp() / n as f64; /* == x^-n / n  ==  1/(n * x^n) */
            return;
        }
    }
    let mut nx: i32 = i32::min(-(f64::MIN_EXP), f64::MAX_EXP); /* = 1021 */

    let r1m5 = M_LOG10_2; // = log10(2) = 0.30103..
    let r1m4 = f64::EPSILON * 0.5; // = DBL_EPSILON * 0.5 = 2^-53 = 1.110223e-16
    let wdtol = fmax2(r1m4, 0.5e-18); /* = 2^-53 = 1.11e-16 */

    /* elim = approximate exponential over and underflow limit */
    let elim = 2.302 * (nx as f64 * r1m5 - 3.0); /* = 700.6174... */
    let rln = fmin2(r1m5 * f64::MANTISSA_DIGITS as f64, 18.06); // = 0.30103 * 53 = 15.95.. ~= #{decimals}
    let mut fln = fmax2(rln, 3.0) - 3.0; // = 12.95..
    let yint = 3.50 + 0.40 * fln; // = 8.6818..
    let slope = 0.21 + fln * (0.0006038 * fln + 0.008677); // = 0.4237..
    let mut mm = m;

    // The C's `for(;;)` either `break`s to the series (false), `goto L10`s to the
    // asymptotic expansion (true), or returns.
    let mut nn;
    let mut fn_;
    let mut t;
    let mut t1;
    let mut t2;
    let mut tk;
    let mut xm;
    let mut xmin;
    let mut xdmy = x;
    let mut xdmln = xln;
    let mut xinc = 0.0;
    let asymptotic = loop {
        nn = n + mm - 1;
        fn_ = nn;
        t = (fn_ + 1) as f64 * xln;

        /* overflow and underflow test for small and large x */
        if t.abs() > elim {
            if t <= 0.0 {
                *ierr = 2;
                return;
            }
        } else {
            if x < wdtol {
                ans[0] = r_pow_di(x, -n - 1);
                if mm != 1 {
                    for k in 1..mm as usize {
                        ans[k] = ans[k - 1] / x;
                    }
                }
                if n == 0 && kode == 2 {
                    ans[0] += xln;
                }
                return;
            }

            /* compute xmin and the number of terms of the series, fln+1 */
            xm = yint + slope * fn_ as f64;
            let mx = xm as i32 + 1;
            xmin = mx as f64;
            if n != 0 {
                xm = -2.302 * rln - fmin2(0.0, xln);
                let arg = fmin2(0.0, xm / n as f64);
                let eps = arg.gexp();
                xm = if arg.abs() < 1.0e-3 { -arg } else { 1.0 - eps };
                fln = x * xm / eps;
                xm = xmin - x;
                if xm > 7.0 && fln < 15.0 {
                    break false;
                }
            }
            xdmy = x;
            xdmln = xln;
            xinc = 0.0;
            if x < xmin {
                nx = x as i32;
                xinc = xmin - nx as f64;
                xdmy = x + xinc;
                xdmln = xdmy.gln();
            }

            /* generate w(n+mm-1, x) by the asymptotic expansion */

            t = fn_ as f64 * xdmln;
            t1 = xdmln + xdmln;
            t2 = t + xdmln;
            tk = fmax2(t.abs(), fmax2(t1.abs(), t2.abs()));
            if tk <= elim {
                /* for all but large x */
                break true;
            }
        }
        *nz += 1; /* nz := #{underflows} */
        mm -= 1;
        ans[mm as usize] = 0.0;
        if mm == 0 {
            return;
        }
    }; /* end{for()} */

    let mut s;
    if !asymptotic {
        nn = fln as i32 + 1;
        let np = n + 1;
        t1 = (n + 1) as f64 * xln;
        t = (-t1).gexp();
        s = t;
        let mut den = x;
        for i in 1..=nn as usize {
            den += 1.0;
            trm[i] = den.gpow(-np as f64);
            s += trm[i];
        }
        ans[0] = s;
        if n == 0 && kode == 2 {
            ans[0] = s + xln;
        }

        if mm != 1 {
            /* generate higher derivatives, j > n */

            let tol = wdtol / 5.0;
            for j in 1..mm as usize {
                t /= x;
                s = t;
                let tols = t * tol;
                den = x;
                for i in 1..=nn as usize {
                    den += 1.0;
                    trm[i] /= den;
                    s += trm[i];
                    if trm[i] < tols {
                        break;
                    }
                }
                ans[j] = s;
            }
        }
        return;
    }

    // L10:
    let mut tss = (-t).gexp();
    let tt = 0.5 / xdmy;
    t1 = tt;
    let tst = wdtol * tt;
    if nn != 0 {
        t1 = tt + 1.0 / fn_ as f64;
    }
    let rxsq = 1.0 / (xdmy * xdmy);
    let ta = 0.5 * rxsq;
    t = (fn_ + 1) as f64 * ta;
    s = t * BVALUES[2];
    if s.abs() >= tst {
        tk = 2.0;
        for k in 4..=22usize {
            t = t
                * ((tk + fn_ as f64 + 1.0) / (tk + 1.0))
                * ((tk + fn_ as f64) / (tk + 2.0))
                * rxsq;
            trm[k] = t * BVALUES[k - 1];
            if trm[k].abs() < tst {
                break;
            }
            s += trm[k];
            tk += 2.0;
        }
    }
    s = (s + t1) * tss;

    // The C from here `goto`s L20 (the PR#13714 correction, then L30) or L30 (the final
    // `ans[0]`), or returns; `finish` carries which.
    #[derive(PartialEq)]
    enum Finish {
        L20,
        L30,
    }
    let finish = 'body: {
        if xinc != 0.0 {
            /* backward recur from xdmy to x */

            nx = xinc as i32;
            let np = nn + 1;
            if nx > N_MAX {
                *ierr = 3;
                return;
            } else {
                if nn == 0 {
                    break 'body Finish::L20;
                }
                xm = xinc - 1.0;
                let mut fx = x + xm;

                /* this loop should not be changed. fx is accurate when x is small */
                for i in 1..=nx as usize {
                    trmr[i] = fx.gpow(-np as f64);
                    s += trmr[i];
                    xm -= 1.0;
                    fx = x + xm;
                }
            }
        }
        ans[(mm - 1) as usize] = s;
        if fn_ == 0 {
            break 'body Finish::L30;
        }

        /* generate lower derivatives,  j < n+mm-1 */

        for j in 2..=mm {
            fn_ -= 1;
            tss *= xdmy;
            t1 = tt;
            if fn_ != 0 {
                t1 = tt + 1.0 / fn_ as f64;
            }
            t = (fn_ + 1) as f64 * ta;
            s = t * BVALUES[2];
            if s.abs() >= tst {
                tk = (4 + fn_) as f64;
                for k in 4..=22usize {
                    trm[k] = trm[k] * (fn_ + 1) as f64 / tk;
                    if trm[k].abs() < tst {
                        break;
                    }
                    s += trm[k];
                    tk += 2.0;
                }
            }
            s = (s + t1) * tss;
            if xinc != 0.0 {
                if fn_ == 0 {
                    break 'body Finish::L20;
                }
                xm = xinc - 1.0;
                let mut fx = x + xm;
                for i in 1..=nx as usize {
                    trmr[i] *= fx;
                    s += trmr[i];
                    xm -= 1.0;
                    fx = x + xm;
                }
            }
            ans[(mm - j) as usize] = s;
            if fn_ == 0 {
                break 'body Finish::L30;
            }
        }
        return;
    };

    if finish == Finish::L20 {
        for i in 1..=nx {
            s += 1.0 / (x + (nx - i) as f64); /* avoid disastrous cancellation, PR#13714 */
        }
    }

    // L30:
    if kode != 2 {
        /* always */
        ans[0] = s - xdmln;
    } else if xdmy != x {
        let xq = xdmy / x;
        ans[0] = s - xq.gln();
    }
} /* dpsifn() */

/// `polygamma.c: psigamma` — the `deriv`-th derivative of `psi(x)`;
/// `psigamma(x, 0) == digamma(x)`.
pub(crate) fn psigamma(x: f64, deriv: f64) -> f64 {
    /* n-th derivative of psi(x);  e.g., psigamma(x,0) == digamma(x) */
    let mut ans = [0.0f64; 1];
    let mut nz = 0;
    let mut ierr = 0;

    if x.is_nan() {
        return x;
    }
    let deriv = r_forceint(deriv);
    let n = deriv as i32;
    if n > N_MAX {
        return f64::NAN;
    }
    dpsifn(x, n, 1, 1, &mut ans, &mut nz, &mut ierr);
    if ierr != 0 {
        return f64::NAN;
    }
    /* Now, ans ==  A := (-1)^(n+1) / gamma(n+1) * psi(n, x) */
    let mut ans = -ans[0]; /* = (-1)^(0+1) * gamma(0+1) * A */
    for k in 1..=n {
        ans *= -(k as f64); /* = (-1)^(k+1) * gamma(k+1) * A */
    }
    ans /* = psi(n, x) */
}

/// `polygamma.c: digamma` — `psi(x) = d/dx log(gamma(x))`.
pub fn digamma(x: f64) -> f64 {
    let mut ans = [0.0f64; 1];
    let mut nz = 0;
    let mut ierr = 0;
    if x.is_nan() {
        return x;
    }
    dpsifn(x, 0, 1, 1, &mut ans, &mut nz, &mut ierr);
    if ierr != 0 {
        return f64::NAN;
    }
    -ans[0]
}

/// `polygamma.c: trigamma` — `psi'(x)`.
pub fn trigamma(x: f64) -> f64 {
    let mut ans = [0.0f64; 1];
    let mut nz = 0;
    let mut ierr = 0;
    if x.is_nan() {
        return x;
    }
    dpsifn(x, 1, 1, 1, &mut ans, &mut nz, &mut ierr);
    if ierr != 0 {
        return f64::NAN;
    }
    ans[0]
}

// ---------------------------------------------------------------------------------------
// pgamma.c: the exported helpers (the rest of pgamma.c lives in `pgamma.rs`)
// ---------------------------------------------------------------------------------------

/// `pgamma.c: scalefactor` — `(2^32)^8 = 2^256 = 1.157921e+77`.
const SCALEFACTOR: f64 = {
    const fn sqr(x: f64) -> f64 {
        x * x
    }
    sqr(sqr(sqr(4294967296.0)))
};

/// `pgamma.c: M_cutoff` — if `|x| > |k| * M_cutoff`, then `log[ exp(-x) * k^x ] =~= -x`.
const M_CUTOFF: f64 = M_LN2 * f64::MAX_EXP as f64 / f64::EPSILON; /*=3.196577e18*/

/// `pgamma.c: logcf` — continued fraction for
/// `1/i + x/(i+d) + x^2/(i+2*d) + x^3/(i+3*d) + ... = sum_{k=0}^Inf x^k/(i+k*d)`;
/// auxiliary in `log1pmx` and `lgamma1p`. `eps` is ~ a relative tolerance.
fn logcf(x: f64, i: f64, d: f64, eps: f64) -> f64 {
    let mut c1 = 2.0 * d;
    let mut c2 = i + d;
    let mut c4 = c2 + d;
    let mut a1 = c2;
    let mut b1 = i * (c2 - i * x);
    let mut b2 = d * d * x;
    let mut a2 = c4 * c2 - b2;

    b2 = c4 * b1 - i * b2;

    while (a2 * b1 - a1 * b2).abs() > (eps * b1 * b2).abs() {
        let mut c3 = c2 * c2 * x;
        c2 += d;
        c4 += d;
        a1 = c4 * a2 - c3 * a1;
        b1 = c4 * b2 - c3 * b1;

        c3 = c1 * c1 * x;
        c1 += d;
        c4 += d;
        a2 = c4 * a1 - c3 * a2;
        b2 = c4 * b1 - c3 * b2;

        if b2.abs() > SCALEFACTOR {
            a1 /= SCALEFACTOR;
            b1 /= SCALEFACTOR;
            a2 /= SCALEFACTOR;
            b2 /= SCALEFACTOR;
        } else if b2.abs() < 1.0 / SCALEFACTOR {
            a1 *= SCALEFACTOR;
            b1 *= SCALEFACTOR;
            a2 *= SCALEFACTOR;
            b2 *= SCALEFACTOR;
        }
    }

    a2 / b2
}

/// `pgamma.c: log1pmx` — accurate calculation of `log(1+x)-x`, particularly for small `x`.
pub(crate) fn log1pmx(x: f64) -> f64 {
    const MIN_LOG1_VALUE: f64 = -0.79149064;

    // Kept as the C writes it: a NaN `x` must take the series branch as it does there.
    #[allow(clippy::manual_range_contains)]
    if x > 1.0 || x < MIN_LOG1_VALUE {
        x.gln_1p() - x
    } else {
        /* -.791 <=  x <= 1  -- expand in  [x/(2+x)]^2 =: y :
         * log(1+x) - x =  x/(2+x) * [ 2 * y * S(y) - x],  with
         * ---------------------------------------------
         * S(y) = 1/3 + y/5 + y^2/7 + ... = \sum_{k=0}^\infty  y^k / (2k + 3)
         */
        let r = x / (2.0 + x);
        let y = r * r;
        if x.abs() < 1e-2 {
            const TWO: f64 = 2.0;
            r * ((((TWO / 9.0 * y + TWO / 7.0) * y + TWO / 5.0) * y + TWO / 3.0) * y - x)
        } else {
            const TOL_LOGCF: f64 = 1e-14;
            r * (2.0 * y * logcf(y, 3.0, 2.0, TOL_LOGCF) - x)
        }
    }
}

/// `pgamma.c: lgamma1p` — `log(gamma(a+1))`, accurate also for small `a` (`0 < a < 0.5`).
pub(crate) fn lgamma1p(a: f64) -> f64 {
    if a.abs() >= 0.5 {
        return lgammafn(a + 1.0);
    }

    const EULERS_CONST: f64 = 0.5772156649015328606065120900824024;

    /* coeffs[i] holds (zeta(i+2)-1)/(i+2) , i = 0:(N-1), N = 40 : */
    const N: usize = 40;
    const COEFFS: [f64; 40] = [
        0.3224670334241132182362075833230126e-0, /* = (zeta(2)-1)/2 */
        0.6735230105319809513324605383715000e-1, /* = (zeta(3)-1)/3 */
        0.2058080842778454787900092413529198e-1,
        0.7385551028673985266273097291406834e-2,
        0.2890510330741523285752988298486755e-2,
        0.1192753911703260977113935692828109e-2,
        0.5096695247430424223356548135815582e-3,
        0.2231547584535793797614188036013401e-3,
        0.9945751278180853371459589003190170e-4,
        0.4492623673813314170020750240635786e-4,
        0.2050721277567069155316650397830591e-4,
        0.9439488275268395903987425104415055e-5,
        0.4374866789907487804181793223952411e-5,
        0.2039215753801366236781900709670839e-5,
        0.9551412130407419832857179772951265e-6,
        0.4492469198764566043294290331193655e-6,
        0.2120718480555466586923135901077628e-6,
        0.1004322482396809960872083050053344e-6,
        0.4769810169363980565760193417246730e-7,
        0.2271109460894316491031998116062124e-7,
        0.1083865921489695409107491757968159e-7,
        0.5183475041970046655121248647057669e-8,
        0.2483674543802478317185008663991718e-8,
        0.1192140140586091207442548202774640e-8,
        0.5731367241678862013330194857961011e-9,
        0.2759522885124233145178149692816341e-9,
        0.1330476437424448948149715720858008e-9,
        0.6422964563838100022082448087644648e-10,
        0.3104424774732227276239215783404066e-10,
        0.1502138408075414217093301048780668e-10,
        0.7275974480239079662504549924814047e-11,
        0.3527742476575915083615072228655483e-11,
        0.1711991790559617908601084114443031e-11,
        0.8315385841420284819798357793954418e-12,
        0.4042200525289440065536008957032895e-12,
        0.1966475631096616490411045679010286e-12,
        0.9573630387838555763782200936508615e-13,
        0.4664076026428374224576492565974577e-13,
        0.2273736960065972320633279596737272e-13,
        0.1109139947083452201658320007192334e-13, /* = (zeta(40+1)-1)/(40+1) */
    ];

    const C: f64 = 0.2273736845824652515226821577978691e-12; /* zeta(N+2)-1 */
    const TOL_LOGCF: f64 = 1e-14;

    /* Abramowitz & Stegun 6.1.33 : for |x| < 2,
     * <==> log(gamma(1+x)) = -(log(1+x) - x) - gamma*x + x^2 * \sum_{n=0}^\infty c_n (-x)^n
     * where c_n := (Zeta(n+2) - 1)/(n+2)  = coeffs[n]
     *
     * Here, another convergence acceleration trick is used to compute
     * lgam(x) :=  sum_{n=0..Inf} c_n (-x)^n
     */
    let mut lgam = C * logcf(-a / 2.0, (N + 2) as f64, 1.0, TOL_LOGCF);
    for i in (0..N).rev() {
        lgam = COEFFS[i] - a * lgam;
    }

    (a * lgam - EULERS_CONST) * a - log1pmx(a)
} /* lgamma1p */

/// `pgamma.c: logspace_add` — `log(exp(logx) + exp(logy))` without overflow.
///
/// No caller yet: `pgamma`'s log-scale tails are out of scope for the limma port. Kept
/// with its unit test so the port stays line-for-line with `pgamma.c`.
#[allow(dead_code)]
pub(crate) fn logspace_add(logx: f64, logy: f64) -> f64 {
    fmax2(logx, logy) + (-(logx - logy).abs()).gexp().gln_1p()
}

/// `pgamma.c: logspace_sub` — `log(exp(logx) - exp(logy))` without overflow. See
/// `logspace_add` for why it has no caller.
#[allow(dead_code)]
pub(crate) fn logspace_sub(logx: f64, logy: f64) -> f64 {
    logx + r_log1_exp(logy - logx)
}

// ---------------------------------------------------------------------------------------
// bd0.c
// ---------------------------------------------------------------------------------------

/// `bd0.c: bd0` — the "deviance part" `bd0(x,M) := M * D0(x/M) = x * log(x/M) + M - x`
/// for `x, M > 0`, evaluated stably for `x/M` close to 1 by the Taylor series of
/// `log((1+v)/(1-v))` with `v = (x-M)/(x+M)`.
pub(crate) fn bd0(x: f64, np: f64) -> f64 {
    if !x.is_finite() || !np.is_finite() || np == 0.0 {
        return f64::NAN;
    }

    if (x - np).abs() < 0.1 * (x + np) {
        let mut v = (x - np) / (x + np); // might underflow to 0
        let mut s = (x - np) * v;
        if s.abs() < f64::MIN_POSITIVE {
            return s;
        }
        let mut ej = 2.0 * x * v;
        v *= v; // "v = v^2"
        for j in 1..1000 {
            /* Taylor series; 1000: no infinite loop as |v| < .1,  v^2000 is "zero" */
            ej *= v; // = 2 x v^(2j+1)
            let s_ = s;
            s += ej / ((j << 1) + 1) as f64;
            if s == s_ {
                /* last term was effectively 0 */
                return s;
            }
        }
        // (MATHLIB_WARNING "T.series failed to converge in 1000 it." dropped.)
    }
    /* else:  | x - np |  is not too small */
    x * (x / np).gln() + np - x
}

/// `frexp(x)`: `(r, e)` with `x = r * 2^e` and `r` in `[0.5, 1)` (`x` itself, and `e = 0`,
/// for zero, infinities and NaN, as C).
fn frexp(x: f64) -> (f64, i32) {
    if x == 0.0 || !x.is_finite() {
        return (x, 0);
    }
    let bits = x.to_bits();
    let biased = ((bits >> 52) & 0x7ff) as i32;
    if biased == 0 {
        // subnormal: normalise first
        let (r, e) = frexp(x * f64::from_bits((1023 + 54) << 52));
        return (r, e - 54);
    }
    let e = biased - 1022;
    let r = f64::from_bits((bits & !(0x7ff << 52)) | (1022 << 52));
    (r, e)
}

/// `ldexp(x, e)`: `x * 2^e`, in steps so an `e` outside the normal exponent range still
/// scales exactly where the result is representable.
fn ldexp(x: f64, e: i32) -> f64 {
    fn pow2(e: i32) -> f64 {
        // e in [-1022, 1023]
        f64::from_bits(((e + 1023) as u64) << 52)
    }
    let mut x = x;
    let mut e = e;
    while e > 1023 {
        x *= pow2(1023);
        e -= 1023;
    }
    while e < -1022 {
        x *= pow2(-1022);
        e += 1022;
    }
    x * pow2(e)
}

/// `bd0.c: bd0_scale` — a table of logs for scaling purposes. Each value has four parts
/// with 23 bits in each, so each part can be multiplied by a double with at most 30 bits
/// set and not have any rounding error. The first entry is `log(2)`.
///
/// Entry `i` is associated with the value `r = 0.5 + i / 256.0`. The argument to log is
/// `p/q` where `q=1024` and `p=floor(q / r + 0.5)`. Thus `r*p/q` is close to 1.
///
/// The C holds these as `float` hex literals; each is written here as the shortest decimal
/// that reads back to exactly that single-precision value (checked when transcribing).
const BD0_SCALE: [[f64; 4]; 128 + 1] = [
    [
        0.6931471824645996,
        -1.9046542121259336e-09,
        -8.78318373858934e-17,
        3.061840738529369e-24,
    ], // 128: log(2048/1024.)
    [
        0.6853040456771851,
        -4.2578264469739224e-08,
        -1.1723105396588968e-15,
        6.20339263720161e-23,
    ], // 129: log(2032/1024.)
    [
        0.6773988008499146,
        2.274189014883632e-08,
        1.4411920431605914e-15,
        7.046384546650164e-23,
    ], // 130: log(2016/1024.)
    [
        0.6699305772781372,
        -4.8293856025338755e-08,
        -8.664795531738382e-16,
        7.049557660755325e-24,
    ], // 131: log(2001/1024.)
    [
        0.6624060869216919,
        -4.791602492559832e-08,
        -2.161508226230938e-15,
        7.092968360825275e-23,
    ], // 132: log(1986/1024.)
    [
        0.6548244953155518,
        6.237715233226027e-09,
        1.1806699607549086e-16,
        6.660323533535123e-25,
    ], // 133: log(1971/1024.)
    [
        0.6471850872039795,
        -4.220866856030625e-08,
        -1.3817589176253094e-15,
        -1.1159932395766606e-23,
    ], // 134: log(1956/1024.)
    [
        0.6400018930435181,
        -4.979170142860312e-08,
        4.887013763907625e-16,
        1.6847465694533583e-23,
    ], // 135: log(1942/1024.)
    [
        0.6327667236328125,
        -5.406177194799966e-08,
        -2.7224545250006775e-15,
        -2.9070780223955397e-23,
    ], // 136: log(1928/1024.)
    [
        0.6249561309814453,
        3.285935434860221e-08,
        1.201672936775467e-16,
        -8.668233185091897e-24,
    ], // 137: log(1913/1024.)
    [
        0.6181373596191406,
        -6.406189467789147e-11,
        3.550580962335878e-18,
        -3.623679406748965e-25,
    ], // 138: log(1900/1024.)
    [
        0.6107416152954102,
        4.229853800552519e-08,
        1.548143293520366e-15,
        -5.446353118719766e-23,
    ], // 139: log(1886/1024.)
    [
        0.6032907962799072,
        5.515817491641428e-08,
        2.1193636784238266e-15,
        1.247109826410663e-22,
    ], // 140: log(1872/1024.)
    [
        0.5963221788406372,
        3.28135385530004e-09,
        -1.4033469816541837e-16,
        5.544061982643074e-24,
    ], // 141: log(1859/1024.)
    [
        0.5893045663833618,
        4.307998580088679e-08,
        -3.2446651600595306e-15,
        -2.569782398567476e-23,
    ], // 142: log(1846/1024.)
    [
        0.5822374820709229,
        -3.983067387025585e-08,
        2.9387905513776885e-15,
        1.6124430498687833e-22,
    ], // 143: log(1833/1024.)
    [
        0.5751199722290039,
        2.242384056216906e-09,
        -2.204516403537114e-17,
        4.078543753751354e-25,
    ], // 144: log(1820/1024.)
    [
        0.5685046911239624,
        4.422870603093543e-08,
        2.7879957109977287e-16,
        -9.741708794029408e-24,
    ], // 145: log(1808/1024.)
    [
        0.5618454217910767,
        2.147161382026752e-08,
        1.3374919170106156e-15,
        -2.326922260336506e-23,
    ], // 146: log(1796/1024.)
    [
        0.554580807685852,
        4.57783499996367e-09,
        2.633145912141026e-16,
        2.070995903387998e-23,
    ], // 147: log(1783/1024.)
    [
        0.5478278398513794,
        -7.667999568639061e-09,
        6.199095327775322e-16,
        6.341979011056023e-24,
    ], // 148: log(1771/1024.)
    [
        0.5410289764404297,
        -3.7302321231891256e-08,
        -3.3801781496424835e-15,
        1.4469198371142414e-22,
    ], // 149: log(1759/1024.)
    [
        0.5347557067871094,
        4.382891916066001e-08,
        -8.601820749692393e-16,
        -8.775950563978016e-24,
    ], // 150: log(1748/1024.)
    [
        0.5278670787811279,
        1.0839714903454478e-08,
        -4.480281248130319e-16,
        3.8840996084516777e-23,
    ], // 151: log(1736/1024.)
    [
        0.5215104818344116,
        4.2031594205127476e-08,
        1.211009992711288e-15,
        -2.3206744819671555e-23,
    ], // 152: log(1725/1024.)
    [
        0.5145297050476074,
        -1.2322940889930578e-08,
        2.820084495413274e-16,
        1.880026562185923e-23,
    ], // 153: log(1713/1024.)
    [
        0.5080875158309937,
        -1.2297126872340414e-08,
        -4.295835588700959e-16,
        -1.8036626177499945e-23,
    ], // 154: log(1702/1024.)
    [
        0.5016034841537476,
        5.9145378372704727e-08,
        1.2728561033550608e-15,
        -5.824094430019416e-23,
    ], // 155: log(1691/1024.)
    [
        0.4950772523880005,
        1.4409851090135817e-08,
        -6.381910184083422e-17,
        -3.500509376717118e-25,
    ], // 156: log(1680/1024.)
    [
        0.48910707235336304,
        2.745798610703787e-08,
        -1.4470418644525664e-15,
        4.211866896949939e-23,
    ], // 157: log(1670/1024.)
    [
        0.4824984669685364,
        1.762245460668055e-08,
        4.128675224747099e-16,
        2.6082691866392177e-23,
    ], // 158: log(1659/1024.)
    [
        0.4764525294303894,
        -1.2470243504481004e-08,
        -9.162193924794196e-17,
        3.765782542240376e-24,
    ], // 159: log(1649/1024.)
    [
        0.469759464263916,
        -5.450353945946063e-09,
        -4.304846960468561e-16,
        2.074710343707083e-25,
    ], // 160: log(1638/1024.)
    [
        0.46363574266433716,
        -1.7013046527125653e-09,
        7.601622738878589e-18,
        -2.0415879231895994e-25,
    ], // 161: log(1628/1024.)
    [
        0.4574742913246155,
        6.943684516258486e-10,
        -2.5461310777374286e-17,
        5.5412533419976475e-25,
    ], // 162: log(1618/1024.)
    [
        0.4512746334075928,
        1.0731865174307131e-08,
        6.374000219845047e-16,
        3.901547318408994e-23,
    ], // 163: log(1608/1024.)
    [
        0.44503629207611084,
        2.8650656958006948e-08,
        -9.155352545558342e-16,
        -4.7365878254772817e-23,
    ], // 164: log(1598/1024.)
    [
        0.43938833475112915,
        2.6186132373595683e-08,
        1.3619601577579505e-15,
        -5.2672794844240613e-23,
    ], // 165: log(1589/1024.)
    [
        0.4330751895904541,
        1.9065733880552216e-08,
        1.014319201479946e-15,
        1.0145670737413337e-22,
    ], // 166: log(1579/1024.)
    [
        0.42735910415649414,
        -1.141359362577532e-08,
        1.3242044582274781e-16,
        -1.0240055893684678e-23,
    ], // 167: log(1570/1024.)
    [
        0.42096930742263794,
        -1.2778508917676845e-08,
        6.143525731270704e-16,
        1.419242178938019e-23,
    ], // 168: log(1560/1024.)
    [
        0.4151833653450012,
        -7.767916088141646e-09,
        5.955443124124071e-16,
        2.732668133771502e-23,
    ], // 169: log(1551/1024.)
    [
        0.4093637466430664,
        1.8807551072086426e-09,
        1.9153331349462894e-16,
        -5.6208063158075e-24,
    ], // 170: log(1542/1024.)
    [
        0.40351009368896484,
        -2.0416603518924603e-08,
        -2.93405013148838e-16,
        1.894346911970581e-24,
    ], // 171: log(1533/1024.)
    [
        0.39762192964553833,
        1.0016001361634608e-09,
        2.2863242352463633e-17,
        9.458133611770764e-25,
    ], // 172: log(1524/1024.)
    [
        0.3916988968849182,
        1.545909711353488e-08,
        1.0962823446210085e-15,
        3.108302239821126e-23,
    ], // 173: log(1515/1024.)
    [
        0.3864043951034546,
        -2.076412375373593e-09,
        1.5073464810114547e-16,
        7.412449454375094e-24,
    ], // 174: log(1507/1024.)
    [
        0.38041436672210693,
        -8.244395388601333e-09,
        1.4866224964678982e-16,
        -3.927292740683968e-24,
    ], // 175: log(1498/1024.)
    [
        0.3743882179260254,
        9.158529934438775e-09,
        5.656919067358501e-16,
        3.4213474905617904e-23,
    ], // 176: log(1489/1024.)
    [
        0.36900103092193604,
        -2.2253590969967263e-08,
        6.231405399548338e-16,
        -2.1564751355837555e-23,
    ], // 177: log(1481/1024.)
    [
        0.36358463764190674,
        -2.6778728567933285e-08,
        -9.943908455716573e-16,
        -4.704929732945495e-24,
    ], // 178: log(1473/1024.)
    [
        0.35745590925216675,
        -2.033036139437172e-08,
        -1.5794492077344114e-15,
        6.318678032068057e-23,
    ], // 179: log(1464/1024.)
    [
        0.3519763946533203,
        2.850385882879891e-08,
        -9.566434575519799e-16,
        -6.409595040255684e-24,
    ], // 180: log(1456/1024.)
    [
        0.3464667797088623,
        -1.236265312343221e-08,
        -6.003368248279719e-16,
        -2.860901497105794e-24,
    ], // 181: log(1448/1024.)
    [
        0.3409265875816345,
        -6.110413286464222e-10,
        1.7467136243918857e-17,
        1.9962587429804357e-25,
    ], // 182: log(1440/1024.)
    [
        0.3353555202484131,
        2.167272583619706e-08,
        -1.0918773497788125e-15,
        -3.047574780704126e-23,
    ], // 183: log(1432/1024.)
    [
        0.33045530319213867,
        -1.608884048209802e-08,
        -3.833435008838916e-16,
        -7.683741875124221e-24,
    ], // 184: log(1425/1024.)
    [
        0.3248254060745239,
        2.801670362373443e-08,
        -2.0725720961098414e-16,
        1.3160777896524739e-23,
    ], // 185: log(1417/1024.)
    [
        0.3191636800765991,
        2.6222629401218e-08,
        -1.3995222573204161e-15,
        8.599883909680833e-23,
    ], // 186: log(1409/1024.)
    [
        0.31418323516845703,
        2.6826626253750874e-08,
        -9.792556373536439e-16,
        2.2954960929108544e-23,
    ], // 187: log(1402/1024.)
    [
        0.3084607720375061,
        1.368350943664609e-08,
        5.591995050742643e-16,
        -1.1938701403427125e-23,
    ], // 188: log(1394/1024.)
    [
        0.3034266233444214,
        -8.629042369534545e-09,
        -5.222554219259392e-16,
        3.228770766379237e-23,
    ], // 189: log(1387/1024.)
    [
        0.2983669638633728,
        8.688424202318856e-09,
        2.764116793423446e-16,
        -1.0171858868428321e-23,
    ], // 190: log(1380/1024.)
    [
        0.29255300760269165,
        -4.91631446664087e-09,
        2.562284723960507e-16,
        -2.6341575505102177e-23,
    ], // 191: log(1372/1024.)
    [
        0.28743791580200195,
        -1.3782395669181824e-08,
        7.29039351877645e-16,
        -4.431977943282236e-24,
    ], // 192: log(1365/1024.)
    [
        0.28229647874832153,
        2.3770866164340987e-08,
        6.449228392264254e-16,
        3.417536976845211e-23,
    ], // 193: log(1358/1024.)
    [
        0.27712851762771606,
        1.4733029018998423e-08,
        6.793364114448283e-16,
        3.593898034032222e-24,
    ], // 194: log(1351/1024.)
    [
        0.27193373441696167,
        -1.8933320689029642e-08,
        7.77939410955583e-16,
        2.591972424742619e-23,
    ], // 195: log(1344/1024.)
    [
        0.2667117714881897,
        1.4300387263244119e-11,
        -7.458876800955772e-19,
        4.247418782257993e-26,
    ], // 196: log(1337/1024.)
    [
        0.2622140049934387,
        7.802219315067305e-09,
        5.022431038767539e-16,
        -2.4063174816868367e-23,
    ], // 197: log(1331/1024.)
    [
        0.25694090127944946,
        2.9618050234603288e-08,
        7.279528203645545e-16,
        2.638556081145549e-23,
    ], // 198: log(1324/1024.)
    [
        0.2516399025917053,
        -6.44787778725231e-09,
        4.122028133693521e-16,
        7.427559296153698e-24,
    ], // 199: log(1317/1024.)
    [
        0.2470736801624298,
        -1.9981829524340355e-09,
        -1.0909757890855787e-16,
        6.236552089808213e-24,
    ], // 200: log(1311/1024.)
    [
        0.24171993136405945,
        5.523085988556886e-09,
        -2.686547633696265e-16,
        -2.764495989170092e-24,
    ], // 201: log(1304/1024.)
    [
        0.23710808157920837,
        1.0085374313462125e-08,
        -4.775626813761317e-16,
        2.330620598032363e-23,
    ], // 202: log(1298/1024.)
    [
        0.231700599193573,
        -1.3946383603524737e-08,
        1.0970921279709183e-17,
        3.229909378627549e-25,
    ], // 203: log(1291/1024.)
    [
        0.22704219818115234,
        -6.451284839670279e-09,
        -4.2529947798112047e-16,
        -1.0260254677162862e-23,
    ], // 204: log(1285/1024.)
    [
        0.2223619818687439,
        1.4110645096820917e-08,
        6.025568981827254e-16,
        1.9385180739531644e-23,
    ], // 205: log(1279/1024.)
    [
        0.21765980124473572,
        -8.286782815503102e-09,
        5.232363919579781e-16,
        5.078443538654066e-23,
    ], // 206: log(1273/1024.)
    [
        0.21214580535888672,
        -8.254218641923217e-09,
        3.255553133321774e-16,
        1.571430013634162e-23,
    ], // 207: log(1266/1024.)
    [
        0.20739519596099854,
        -1.6149279691290985e-09,
        2.1131592679643073e-17,
        1.4275617427027514e-24,
    ], // 208: log(1260/1024.)
    [
        0.20262190699577332,
        8.597639933327628e-09,
        -3.3804619056798137e-16,
        2.5623235609364303e-24,
    ], // 209: log(1254/1024.)
    [
        0.19782572984695435,
        1.348296585490516e-08,
        -3.2024568730231384e-16,
        -2.5712252251631252e-23,
    ], // 210: log(1248/1024.)
    [
        0.1930064558982849,
        9.956859781112826e-10,
        9.001674563844606e-17,
        -3.754797654135173e-24,
    ], // 211: log(1242/1024.)
    [
        0.1889725625514984,
        4.2415360113068346e-09,
        3.8086815297465933e-16,
        -2.1147402916208568e-23,
    ], // 212: log(1237/1024.)
    [
        0.18411031365394592,
        6.931054841174955e-09,
        -3.4784858522920016e-16,
        2.4665943434547742e-23,
    ], // 213: log(1231/1024.)
    [
        0.17922431230545044,
        5.0739235035734964e-09,
        3.2221329189922637e-16,
        -1.0379009008973928e-23,
    ], // 214: log(1225/1024.)
    [
        0.17431432008743286,
        3.794385250444066e-09,
        3.190066898587176e-16,
        2.0292714723890484e-23,
    ], // 215: log(1219/1024.)
    [
        0.17020416259765625,
        3.4223344158590407e-09,
        -1.8846416901959178e-16,
        1.1415315069779235e-23,
    ], // 216: log(1214/1024.)
    [
        0.16524958610534668,
        -1.3210039284672348e-08,
        -2.3213954359040806e-16,
        3.043054213286757e-24,
    ], // 217: log(1208/1024.)
    [
        0.1602703034877777,
        6.007922159767531e-09,
        -7.521047737154288e-17,
        -1.2649106048711768e-25,
    ], // 218: log(1202/1024.)
    [
        0.15610191226005554,
        -1.2301535790015805e-08,
        3.0175617567361414e-16,
        -8.633806506327147e-24,
    ], // 219: log(1197/1024.)
    [
        0.15191605687141418,
        -1.4845571882915465e-08,
        -3.265830289949929e-16,
        -1.5268151962784823e-23,
    ], // 220: log(1192/1024.)
    [
        0.14686977863311768,
        -4.6748995785605985e-09,
        -4.294291341758996e-16,
        1.328295982896899e-23,
    ], // 221: log(1186/1024.)
    [
        0.142645001411438,
        9.186472027522541e-09,
        -8.049373155998929e-16,
        1.437998766909278e-23,
    ], // 222: log(1181/1024.)
    [
        0.1384023129940033,
        9.86511672351753e-09,
        -8.837306496940929e-16,
        7.295324909194215e-24,
    ], // 223: log(1176/1024.)
    [
        0.1332872211933136,
        9.990350768873668e-10,
        3.316946610734358e-17,
        2.7351440526086287e-24,
    ], // 224: log(1170/1024.)
    [
        0.129004567861557,
        -7.46120853989396e-09,
        -6.212113165383437e-16,
        1.855187264989731e-24,
    ], // 225: log(1165/1024.)
    [
        0.12470348179340363,
        -3.2924463155836747e-09,
        -7.404120132741752e-17,
        1.3246955625609024e-24,
    ], // 226: log(1160/1024.)
    [
        0.12038381397724152,
        3.3791991427278845e-09,
        1.6214981996371606e-16,
        -6.00070673940474e-24,
    ], // 227: log(1155/1024.)
    [
        0.11604541540145874,
        3.5638392237302696e-10,
        -7.354219635108878e-18,
        7.794312440645008e-26,
    ], // 228: log(1150/1024.)
    [
        0.11168810725212097,
        3.136765958089427e-09,
        -8.994406293238225e-19,
        -7.920937558182207e-26,
    ], // 229: log(1145/1024.)
    [
        0.10731174051761627,
        -4.728527791542092e-09,
        -4.2976339455270984e-16,
        5.351143287258107e-24,
    ], // 230: log(1140/1024.)
    [
        0.10291612148284912,
        2.8332007850906393e-09,
        4.925742757234052e-17,
        2.794436810397303e-24,
    ], // 231: log(1135/1024.)
    [
        0.09850110113620758,
        4.970725164810119e-09,
        4.130512756847049e-16,
        6.313449560595865e-25,
    ], // 232: log(1130/1024.)
    [
        0.09406651556491852,
        -6.525850970717784e-09,
        -1.3492816627243298e-16,
        -9.079650179574527e-24,
    ], // 233: log(1125/1024.)
    [
        0.08961215615272522,
        2.5369617517867482e-09,
        1.6110664594961319e-16,
        -5.189504504486964e-24,
    ], // 234: log(1120/1024.)
    [
        0.08603434264659882,
        -5.304795713811927e-09,
        5.127575488441481e-17,
        1.463615545692133e-24,
    ], // 235: log(1116/1024.)
    [
        0.08154398202896118,
        2.0112156384755053e-09,
        8.065769331577608e-17,
        -3.0150319017810437e-24,
    ], // 236: log(1111/1024.)
    [
        0.07703337073326111,
        5.7495661565099e-09,
        -2.503851097302985e-16,
        -1.846143093040508e-23,
    ], // 237: log(1106/1024.)
    [
        0.07250232994556427,
        1.177662634077592e-09,
        -3.525476857925506e-17,
        1.3164077898906505e-24,
    ], // 238: log(1101/1024.)
    [
        0.0688626617193222,
        -7.043545302565235e-09,
        2.49712406751501e-16,
        1.0686882487619963e-23,
    ], // 239: log(1097/1024.)
    [
        0.06429435312747955,
        -2.422082090447475e-09,
        -2.0555896149129847e-16,
        8.602907613530545e-24,
    ], // 240: log(1092/1024.)
    [
        0.0606246218085289,
        7.905943261166115e-12,
        -8.270443345471096e-19,
        -2.3533820836754142e-26,
    ], // 241: log(1088/1024.)
    [
        0.056018441915512085,
        -5.139745296034448e-10,
        -3.811651571366802e-17,
        1.9072195442219776e-24,
    ], // 242: log(1083/1024.)
    [
        0.052318163216114044,
        -3.5574325707443677e-09,
        9.191155834145393e-17,
        -5.321463973420977e-24,
    ], // 243: log(1079/1024.)
    [
        0.04767347127199173,
        -1.8026349302147082e-09,
        1.0329634289704177e-16,
        -2.2283569301283993e-24,
    ], // 244: log(1074/1024.)
    [
        0.04394212365150452,
        -1.795005699634089e-09,
        -5.3817402974104447e-17,
        -1.3996196977941442e-24,
    ], // 245: log(1070/1024.)
    [
        0.040196798741817474,
        3.8451930528538014e-10,
        -2.4485452721520977e-17,
        -7.386769024377949e-26,
    ], // 246: log(1066/1024.)
    [
        0.0354953333735466,
        -3.5901653872016936e-10,
        -2.073207767866976e-17,
        -2.412097216555168e-26,
    ], // 247: log(1061/1024.)
    [
        0.0317181795835495,
        6.87235046648027e-10,
        -6.430478093749473e-18,
        1.3508692031871337e-25,
    ], // 248: log(1057/1024.)
    [
        0.02792670577764511,
        7.568773385813188e-10,
        -4.22031585165944e-17,
        2.534760563692782e-24,
    ], // 249: log(1053/1024.)
    [
        0.02412080392241478,
        -1.1255707477175747e-09,
        4.89700584100947e-17,
        1.4172214525647275e-24,
    ], // 250: log(1049/1024.)
    [
        0.01934296265244484,
        1.9068610579431322e-10,
        -1.0635946218849709e-17,
        -5.300489542457734e-25,
    ], // 251: log(1044/1024.)
    [
        0.015504186972975731,
        -4.3701048335620385e-10,
        6.61106154763715e-18,
        2.5398086818405174e-25,
    ], // 252: log(1040/1024.)
    [
        0.01165061630308628,
        9.168890091615367e-10,
        -1.5848697818454755e-17,
        -1.350491609984469e-24,
    ], // 253: log(1036/1024.)
    [
        0.007782140746712685,
        -3.046577434773212e-10,
        7.793435996762285e-18,
        4.660100148208369e-25,
    ], // 254: log(1032/1024.)
    [
        0.0038986406289041042,
        -2.1324678134426733e-10,
        1.2541658163801307e-19,
        8.745035431740123e-27,
    ], // 255: log(1028/1024.)
    [0.0, 0.0, 0.0, 0.0], // log(1024/1024) = log(1) = 0
];

/// `bd0.c: ebd0` — `x * log(x/M) + (M - x)` (aka `-x * log1pmx((M - x)/x)`) delivered in
/// two parts `yh + yl` (Welinder's accuracy improvement, PR#15628). `yh` collects the
/// integer-rounded contributions, `yl` the fractional remainders.
///
/// The loop index `j` walks two rows of `bd0_scale` in lock step, as in the C.
#[allow(clippy::needless_range_loop)]
pub(crate) fn ebd0(x: f64, m: f64, yh: &mut f64, yl: &mut f64) {
    const SB: i32 = 10;
    const S: f64 = (1u32 << SB) as f64; // = 2^10 = 1024
    const N: i32 = 128; // == bd0_scale.len() - 1

    *yl = 0.0;
    *yh = 0.0;

    if x == m {
        return;
    }
    if x == 0.0 {
        *yh = m;
        return;
    }
    if m == 0.0 {
        *yh = f64::INFINITY;
        return;
    }

    if m / x == f64::INFINITY {
        *yh = m; //  as when (x == 0)
        return;
    }

    // NB: M/x overflow handled above; underflow should be handled by fg = Inf
    let (r, e) = frexp(m / x); // => r in  [0.5, 1) and 'e' (int) such that  M/x = r * 2^e

    // prevent later overflow
    if M_LN2 * (-e as f64) > 1.0 + f64::MAX / x {
        *yh = f64::INFINITY;
        return;
    }

    let i = ((r - 0.5) * (2 * N) as f64 + 0.5).floor() as i32;
    // now,  0 <= i <= N
    let f = (S / (0.5 + i as f64 / (2.0 * N as f64)) + 0.5).floor();
    let fg = ldexp(f, -(e + SB)); // ldexp(f, E) := f * 2^E
    if fg == f64::INFINITY {
        *yh = fg;
        return;
    }
    /* We now have (M * fg / x) close to 1.  */

    /*
     * We need to compute this:
     * (x/M)^x * exp(M-x) =
     * (M/x)^-x * exp(M-x) =
     * (M*fg/x)^-x * (fg)^x * exp(M-x) =
     * (M*fg/x)^-x * (fg)^x * exp(M*fg-x) * exp(M-M*fg)
     *
     * In log terms:
     * log((x/M)^x * exp(M-x)) =
     * log((M*fg/x)^-x * (fg)^x * exp(M*fg-x) * exp(M-M*fg)) =
     * log((M*fg/x)^-x * exp(M*fg-x)) + x*log(fg) + (M-M*fg) =
     * -x*log1pmx((M*fg-x)/x) + x*log(fg) + M - M*fg =
     *
     * Note, that fg has at most 10 bits.  If M and x are suitably
     * "nice" -- such as being integers or half-integers -- then
     * we can compute M*fg as well as x * bd0_scale[.][.] without
     * rounding errors.
     */

    // The C's ADD1 macro.
    fn add1(d: f64, yh: &mut f64, yl: &mut f64) {
        let d1 = (d + 0.5).floor();
        let d2 = d - d1; /* in [-.5,.5) */
        *yh += d1;
        *yl += d2;
    }

    add1(-x * log1pmx((m * fg - x) / x), yh, yl);
    if fg == 1.0 {
        return;
    }
    // else (fg != 1) :
    let i = i as usize;
    for j in 0..4 {
        add1(x * BD0_SCALE[i][j], yh, yl); // handles  x*log(fg*2^e)
        add1(-x * BD0_SCALE[0][j] * e as f64, yh, yl); // handles  x*log(1/ 2^e)
                                                       //                            ^^^ at end prevents overflow in  ebd0(1e307, 1e300)
        if !yh.is_finite() {
            *yh = f64::INFINITY;
            *yl = 0.0;
            return;
        }
    }

    add1(m, yh, yl);
    add1(-m * fg, yh, yl);
}

// ---------------------------------------------------------------------------------------
// dpois.c
// ---------------------------------------------------------------------------------------

/// `dpois.c: dpois_raw` — the Poisson probability `lambda^x exp(-lambda) / x!` for real
/// `x >= 0` (integer for `dpois()`, but not e.g. for `pgamma()`), via `ebd0` and
/// `stirlerr` (R >= 4.1).
pub(crate) fn dpois_raw(x: f64, lambda: f64, give_log: bool) -> f64 {
    const M_SQRT_2PI: f64 = 2.50662827463100050241576528481104525301; /* sqrt(2*pi) */
    const X_LRG: f64 = 2.86111748575702815380240589208115399625e+307; /* = 2^1023 / pi */

    if lambda == 0.0 {
        return if x == 0.0 {
            r_d_1(give_log)
        } else {
            r_d_0(give_log)
        };
    }
    if !lambda.is_finite() {
        return r_d_0(give_log); // including for the case where  x = lambda = +Inf
    }
    if x < 0.0 {
        return r_d_0(give_log);
    }
    if x <= lambda * f64::MIN_POSITIVE {
        return r_d_exp(-lambda, give_log);
    }
    if lambda < x * f64::MIN_POSITIVE {
        if !x.is_finite() {
            // lambda < x = +Inf
            return r_d_0(give_log);
        }
        // else
        return r_d_exp(-lambda + x * lambda.gln() - lgammafn(x + 1.0), give_log);
    }
    // R <= 4.0.x  had   return(R_D_fexp( M_2PI*x, -stirlerr(x)-bd0(x,lambda) ));
    let mut yh = 0.0;
    let mut yl = 0.0;
    ebd0(x, lambda, &mut yh, &mut yl);
    yl += stirlerr(x);
    let lrg_x = x >= X_LRG; //really large x  <==>  2*pi*x  overflows
    let r = if lrg_x {
        M_SQRT_2PI * x.sqrt() // sqrt(.): avoid overflow for very large x
    } else {
        M_2PI * x
    };
    if give_log {
        -yl - yh - (if lrg_x { r.gln() } else { 0.5 * r.gln() })
    } else {
        (-yl).gexp() * (-yh).gexp() / (if lrg_x { r } else { r.sqrt() })
    }
}

/// `pgamma.c: dpois_wrap` — `dpois(x_plus_1 - 1, lambda)`, where
/// `dpois(k, L) := exp(-L) L^k / gamma(k+1)` (the usual Poisson probabilities), stable
/// for `x_plus_1 <= 1`.
pub(crate) fn dpois_wrap(x_plus_1: f64, lambda: f64, give_log: bool) -> f64 {
    if !lambda.is_finite() {
        return r_d_0(give_log);
    }
    if x_plus_1 > 1.0 {
        return dpois_raw(x_plus_1 - 1.0, lambda, give_log);
    }
    if lambda > (x_plus_1 - 1.0).abs() * M_CUTOFF {
        r_d_exp(-lambda - lgammafn(x_plus_1), give_log)
    } else {
        let d = dpois_raw(x_plus_1, lambda, give_log);
        if give_log {
            d + (x_plus_1 / lambda).gln()
        } else {
            d * (x_plus_1 / lambda)
        }
    }
}

// ---------------------------------------------------------------------------------------
// dgamma.c
// ---------------------------------------------------------------------------------------

/// `dgamma.c: dgamma` — the density of the gamma distribution with `shape` and `scale`,
/// computed through `dpois_raw` (Loader's method).
pub fn dgamma(x: f64, shape: f64, scale: f64, give_log: bool) -> f64 {
    if x.is_nan() || shape.is_nan() || scale.is_nan() {
        return x + shape + scale;
    }
    if shape < 0.0 || scale <= 0.0 {
        return f64::NAN;
    }
    if x < 0.0 {
        return r_d_0(give_log);
    }
    if shape == 0.0 {
        /* point mass at 0 */
        return if x == 0.0 {
            f64::INFINITY
        } else {
            r_d_0(give_log)
        };
    }
    if x == 0.0 {
        if shape < 1.0 {
            return f64::INFINITY;
        }
        if shape > 1.0 {
            return r_d_0(give_log);
        }
        /* else */
        return if give_log { -scale.gln() } else { 1.0 / scale };
    }

    let pr;
    if shape < 1.0 {
        pr = dpois_raw(shape, x / scale, give_log);
        return if give_log {
            /* NB: currently *always*  shape/x > 0  if shape < 1:
             * -- overflow to Inf happens, but underflow to 0 does NOT : */
            pr + (if (shape / x).is_finite() {
                (shape / x).gln()
            } else {
                /* shape/x overflows to +Inf */
                shape.gln() - x.gln()
            })
        } else {
            pr * shape / x
        };
    }
    /* else  shape >= 1 */
    pr = dpois_raw(shape - 1.0, x / scale, give_log);
    if give_log {
        pr - scale.gln()
    } else {
        pr / scale
    }
}

// ---------------------------------------------------------------------------------------
// limma: fitFDist.R trigammaInverse; statmod: logmdigamma.R
// ---------------------------------------------------------------------------------------

/// limma `fitFDist.R: trigammaInverse` — solve `trigamma(y) = x` for `y` by Newton's
/// method (Gordon Smyth, 2002/2004), scalar form. `x > 1e7` returns `1/sqrt(x)`,
/// `x < 1e-6` returns `1/x`, `x < 0` is NaN (R warns).
pub fn trigamma_inverse(x: f64) -> f64 {
    //	Treat out-of-range values as special cases
    if x.is_nan() {
        return x;
    }
    if x < 0.0 {
        return f64::NAN;
    }
    if x > 1e7 {
        return 1.0 / x.sqrt();
    }
    if x < 1e-6 {
        return 1.0 / x;
    }

    //	Newton's method
    //	1/trigamma(y) is convex, nearly linear and strictly > y-0.5,
    //	so iteration to solve 1/x = 1/trigamma is monotonically convergent
    let mut y = 0.5 + 1.0 / x;
    let mut iter = 0;
    loop {
        iter += 1;
        let tri = trigamma(y);
        let dif = tri * (1.0 - tri / x) / psigamma(y, 2.0);
        y += dif;
        if -dif / y < 1e-8 {
            break;
        }
        if iter > 50 {
            // (R warns "Iteration limit exceeded".)
            break;
        }
    }
    y
}

/// statmod `logmdigamma.R: logmdigamma` — `log(x) - digamma(x)`, accurate for large `x`
/// where the two terms cancel; recursion `x -> x + 5` below `|x| < 5`, then the
/// asymptotic series. Scalar, real-argument form; `x <= 0` or NaN gives NaN (R's NA).
pub fn logmdigamma(x: f64) -> f64 {
    if x.is_nan() || x <= 0.0 {
        return f64::NAN;
    }
    if x.abs() < 5.0 {
        return (x / (x + 5.0)).gln()
            + logmdigamma(x + 5.0)
            + 1.0 / x
            + 1.0 / (x + 1.0)
            + 1.0 / (x + 2.0)
            + 1.0 / (x + 3.0)
            + 1.0 / (x + 4.0);
    }
    let z = x;
    let x = 1.0 / (z * z); // R: 1/z^2, and R_pow squares by multiplication
    let tail = x
        * (-1.0 / 12.0
            + (x * (1.0 / 120.0
                + (x * (-1.0 / 252.0
                    + (x * (1.0 / 240.0
                        + (x * (-1.0 / 132.0
                            + (x * (691.0 / 32760.0
                                + (x * (-1.0 / 12.0 + (3617.0 * x) / 8160.0)))))))))))));
    1.0 / (2.0 * z) - tail
}
