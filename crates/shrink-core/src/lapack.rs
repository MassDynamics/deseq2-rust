//! The LAPACK 3.12.1 routines behind Armadillo's rcond test and its `solve_approx_svd`
//! fallback, in the operation order of R's internal `libRlapack` / `libRblas` (reference
//! BLAS, `dnrm2` and `dlartg` in their f90 versions). Ported routine by routine:
//!
//! - rcond: `dlansy` ('1','L'), `dlange` ('1','M'), `dlacn2`, `dlatrs`, `dtrsv`, `drscl`,
//!   `dpocon` ('L') and `dgecon` ('1').
//! - minimum-norm least squares: `dgelsd` for square `A`, one right-hand side and
//!   `n <= SMLSIZ = 25`, which is the only path mixsqp reaches: `dgebd2` (via `dgebrd`, whose
//!   block size 32 exceeds n), `dormbr` -> `dorm2r` / `dorml2` -> `dlarf1f`, and `dlalsd`'s
//!   small-n branch (`dlasdq` -> `dbdsqr` with `dlartg`, `dlas2`, `dlasv2`, `dlasr`).
//!
//! Matrices are column-major with an explicit leading dimension; indices are 0-based except
//! inside `dbdsqr_u`, which keeps Fortran's 1-based `d` and `e` so it reads like the source.
#![allow(clippy::too_many_arguments)]

/// dlamch('E'): relative machine epsilon, 2^-53.
const EPS_E: f64 = f64::EPSILON * 0.5;
/// dlamch('P'): eps * base, 2^-52.
const EPS_P: f64 = f64::EPSILON;
/// dlamch('S'): safe minimum, 2^-1022.
const SFMIN: f64 = f64::MIN_POSITIVE;

/// Fortran SIGN(a, b).
#[inline]
fn sign(a: f64, b: f64) -> f64 {
    a.abs().copysign(b)
}

// ---------------------------------------------------------------- BLAS level 1

/// dasum (the unrolled loop is left-associative, so it is a sequential sum).
fn dasum(n: usize, x: &[f64], off: usize, inc: usize) -> f64 {
    let mut s = 0.0;
    for i in 0..n {
        s += x[off + i * inc].abs();
    }
    s
}

/// idamax, 0-based (first maximum of |x|).
fn idamax(n: usize, x: &[f64], off: usize, inc: usize) -> usize {
    if n == 0 {
        return 0;
    }
    let mut imax = 0;
    let mut dmax = x[off].abs();
    for i in 1..n {
        let v = x[off + i * inc].abs();
        if v > dmax {
            imax = i;
            dmax = v;
        }
    }
    imax
}

/// ddot (sequential, as the unrolled reference loop is left-associative).
fn ddot(n: usize, x: &[f64], xo: usize, y: &[f64], yo: usize) -> f64 {
    let mut s = 0.0;
    for i in 0..n {
        s += x[xo + i] * y[yo + i];
    }
    s
}

/// dscal (3.12 returns early when da == 1).
fn dscal(n: usize, da: f64, x: &mut [f64], off: usize, inc: usize) {
    if da == 1.0 {
        return;
    }
    for i in 0..n {
        x[off + i * inc] *= da;
    }
}

/// dnrm2, the LAPACK 3.10+ f90 version (Blue's scaling with three accumulators).
fn dnrm2(n: usize, x: &[f64], off: usize, inc: usize) -> f64 {
    let tsml = 2f64.powi(-511);
    let tbig = 2f64.powi(486);
    let ssml = 2f64.powi(537);
    let sbig = 2f64.powi(-538);
    if n == 0 {
        return 0.0;
    }
    let mut notbig = true;
    let (mut asml, mut amed, mut abig) = (0.0, 0.0, 0.0);
    for i in 0..n {
        let ax = x[off + i * inc].abs();
        if ax > tbig {
            let t = ax * sbig;
            abig += t * t;
            notbig = false;
        } else if ax < tsml {
            if notbig {
                let t = ax * ssml;
                asml += t * t;
            }
        } else {
            amed += ax * ax;
        }
    }
    let (scl, sumsq);
    if abig > 0.0 {
        if amed > 0.0 || amed > f64::MAX || amed.is_nan() {
            abig += (amed * sbig) * sbig;
        }
        scl = 1.0 / sbig;
        sumsq = abig;
    } else if asml > 0.0 {
        if amed > 0.0 || amed > f64::MAX || amed.is_nan() {
            let amed = amed.sqrt();
            let asml = asml.sqrt() / ssml;
            let (ymin, ymax) = if asml > amed { (amed, asml) } else { (asml, amed) };
            scl = 1.0;
            let r = ymin / ymax;
            sumsq = ymax * ymax * (1.0 + r * r);
        } else {
            scl = 1.0 / ssml;
            sumsq = asml;
        }
    } else {
        scl = 1.0;
        sumsq = amed;
    }
    scl * sumsq.sqrt()
}

/// max |v| with dlange's NaN propagation (`value < temp .or. isnan(temp)`).
fn max_abs(vals: impl Iterator<Item = f64>) -> f64 {
    let mut value: f64 = 0.0;
    for v in vals {
        let t = v.abs();
        if value < t || t.is_nan() {
            value = t;
        }
    }
    value
}

// ---------------------------------------------------------------- norms

/// dlansy('1', 'L') of a symmetric matrix held in its lower triangle.
pub fn dlansy_1l(a: &[f64], n: usize, lda: usize) -> f64 {
    let mut work = vec![0.0; n];
    let mut value: f64 = 0.0;
    for j in 0..n {
        let mut sum = work[j] + a[j + j * lda].abs();
        for i in (j + 1)..n {
            let absa = a[i + j * lda].abs();
            sum += absa;
            work[i] += absa;
        }
        if value < sum || sum.is_nan() {
            value = sum;
        }
    }
    value
}

/// dlange('1'): maximum absolute column sum.
pub fn dlange_1(a: &[f64], m: usize, n: usize, lda: usize) -> f64 {
    let mut value: f64 = 0.0;
    for j in 0..n {
        let mut sum = 0.0;
        for i in 0..m {
            sum += a[i + j * lda].abs();
        }
        if value < sum || sum.is_nan() {
            value = sum;
        }
    }
    value
}

/// dlange('M'): maximum absolute element.
fn dlange_m(a: &[f64], m: usize, n: usize, lda: usize) -> f64 {
    max_abs((0..n).flat_map(|j| (0..m).map(move |i| a[i + j * lda])))
}

// ---------------------------------------------------------------- scaling

/// The sequence of multipliers `dlascl` applies to every element to scale by cto/cfrom
/// without over/underflow. Empty when nothing is applied.
fn dlascl_muls(cfrom: f64, cto: f64) -> Vec<f64> {
    let mut out = Vec::new();
    if cfrom == 0.0 || cfrom.is_nan() || cto.is_nan() {
        return out;
    }
    let smlnum = SFMIN;
    let bignum = 1.0 / smlnum;
    let (mut cfromc, mut ctoc) = (cfrom, cto);
    loop {
        let cfrom1 = cfromc * smlnum;
        let (mul, done);
        if cfrom1 == cfromc {
            mul = ctoc / cfromc;
            done = true;
        } else {
            let cto1 = ctoc / bignum;
            if cto1 == ctoc {
                mul = ctoc;
                done = true;
                cfromc = 1.0;
            } else if cfrom1.abs() > ctoc.abs() && ctoc != 0.0 {
                mul = smlnum;
                done = false;
                cfromc = cfrom1;
            } else if cto1.abs() > cfromc.abs() {
                mul = bignum;
                done = false;
                ctoc = cto1;
            } else {
                mul = ctoc / cfromc;
                done = true;
                if mul == 1.0 {
                    return out;
                }
            }
        }
        out.push(mul);
        if done {
            return out;
        }
    }
}

/// dlascl('G') on a strided set of elements `x[off + k*inc]`, k < n.
fn dlascl(cfrom: f64, cto: f64, n: usize, x: &mut [f64], off: usize, inc: usize) {
    for mul in dlascl_muls(cfrom, cto) {
        for k in 0..n {
            x[off + k * inc] *= mul;
        }
    }
}

/// drscl: x <- x / sa, scaled to avoid over/underflow.
fn drscl(n: usize, sa: f64, x: &mut [f64]) {
    let smlnum = SFMIN;
    let bignum = 1.0 / smlnum;
    let mut cden = sa;
    let mut cnum = 1.0;
    loop {
        let cden1 = cden * smlnum;
        let cnum1 = cnum / bignum;
        let (mul, done);
        if cden1.abs() > cnum.abs() && cnum != 0.0 {
            mul = smlnum;
            done = false;
            cden = cden1;
        } else if cnum1.abs() > cden.abs() {
            mul = bignum;
            done = false;
            cnum = cnum1;
        } else {
            mul = cnum / cden;
            done = true;
        }
        dscal(n, mul, x, 0, 1);
        if done {
            return;
        }
    }
}

// ---------------------------------------------------------------- triangular solves

/// dtrsv with incx = 1.
fn dtrsv(upper: bool, notran: bool, nounit: bool, n: usize, a: &[f64], lda: usize, x: &mut [f64]) {
    let at = |i: usize, j: usize| a[i + j * lda];
    if notran {
        if upper {
            for j in (0..n).rev() {
                if x[j] != 0.0 {
                    if nounit {
                        x[j] /= at(j, j);
                    }
                    let temp = x[j];
                    for i in (0..j).rev() {
                        x[i] -= temp * at(i, j);
                    }
                }
            }
        } else {
            for j in 0..n {
                if x[j] != 0.0 {
                    if nounit {
                        x[j] /= at(j, j);
                    }
                    let temp = x[j];
                    for i in (j + 1)..n {
                        x[i] -= temp * at(i, j);
                    }
                }
            }
        }
    } else if upper {
        for j in 0..n {
            let mut temp = x[j];
            for i in 0..j {
                temp -= at(i, j) * x[i];
            }
            if nounit {
                temp /= at(j, j);
            }
            x[j] = temp;
        }
    } else {
        for j in (0..n).rev() {
            let mut temp = x[j];
            for i in ((j + 1)..n).rev() {
                temp -= at(i, j) * x[i];
            }
            if nounit {
                temp /= at(j, j);
            }
            x[j] = temp;
        }
    }
}

/// dlatrs: solve a triangular system with scaling to prevent overflow. Returns `scale`.
/// `normin` is true for NORMIN = 'Y' (cnorm already holds the column norms).
fn dlatrs(
    upper: bool,
    notran: bool,
    nounit: bool,
    normin: bool,
    n: usize,
    a: &[f64],
    lda: usize,
    x: &mut [f64],
    cnorm: &mut [f64],
) -> f64 {
    let at = |i: usize, j: usize| a[i + j * lda];
    let mut scale = 1.0;
    if n == 0 {
        return scale;
    }
    let smlnum = SFMIN / EPS_P;
    let bignum = 1.0 / smlnum;
    if !normin {
        if upper {
            for j in 0..n {
                cnorm[j] = dasum(j, a, j * lda, 1);
            }
        } else {
            for j in 0..n - 1 {
                cnorm[j] = dasum(n - j - 1, a, j + 1 + j * lda, 1);
            }
            cnorm[n - 1] = 0.0;
        }
    }
    let imax = idamax(n, cnorm, 0, 1);
    let mut tmax = cnorm[imax];
    let tscal;
    if tmax <= bignum {
        tscal = 1.0;
    } else if tmax <= f64::MAX {
        tscal = 1.0 / (smlnum * tmax);
        dscal(n, tscal, cnorm, 0, 1);
    } else {
        tmax = 0.0;
        if upper {
            for j in 1..n {
                tmax = max_abs((0..j).map(|i| at(i, j))).max(tmax);
            }
        } else {
            for j in 0..n - 1 {
                tmax = max_abs(((j + 1)..n).map(|i| at(i, j))).max(tmax);
            }
        }
        if tmax <= f64::MAX {
            tscal = 1.0 / (smlnum * tmax);
            for j in 0..n {
                if cnorm[j] <= f64::MAX {
                    cnorm[j] *= tscal;
                } else {
                    cnorm[j] = 0.0;
                    let rows = if upper { 0..j } else { (j + 1)..n };
                    for i in rows {
                        cnorm[j] += tscal * at(i, j).abs();
                    }
                }
            }
        } else {
            dtrsv(upper, notran, nounit, n, a, lda, x);
            return scale;
        }
    }

    let j = idamax(n, x, 0, 1);
    let mut xmax = x[j].abs();
    let mut xbnd = xmax;
    let grow;
    // Column order of the solve: backward for N/Upper and T/Lower.
    let forward = if notran { !upper } else { upper };
    let order: Vec<usize> = if forward { (0..n).collect() } else { (0..n).rev().collect() };
    if notran {
        if tscal != 1.0 {
            grow = 0.0;
        } else if nounit {
            let mut g = 1.0 / xbnd.max(smlnum);
            xbnd = g;
            let mut broke = false;
            for &j in &order {
                if g <= smlnum {
                    broke = true;
                    break;
                }
                let tjj = at(j, j).abs();
                xbnd = xbnd.min(tjj.min(1.0) * g);
                if tjj + cnorm[j] >= smlnum {
                    g *= tjj / (tjj + cnorm[j]);
                } else {
                    g = 0.0;
                }
            }
            grow = if broke { g } else { xbnd };
        } else {
            let mut g = (1.0 / xbnd.max(smlnum)).min(1.0);
            for &j in &order {
                if g <= smlnum {
                    break;
                }
                g *= 1.0 / (1.0 + cnorm[j]);
            }
            grow = g;
        }
    } else if tscal != 1.0 {
        grow = 0.0;
    } else if nounit {
        let mut g = 1.0 / xbnd.max(smlnum);
        xbnd = g;
        let mut broke = false;
        for &j in &order {
            if g <= smlnum {
                broke = true;
                break;
            }
            let xj = 1.0 + cnorm[j];
            g = g.min(xbnd / xj);
            let tjj = at(j, j).abs();
            if xj > tjj {
                xbnd *= tjj / xj;
            }
        }
        grow = if broke { g } else { g.min(xbnd) };
    } else {
        let mut g = (1.0 / xbnd.max(smlnum)).min(1.0);
        for &j in &order {
            if g <= smlnum {
                break;
            }
            g /= 1.0 + cnorm[j];
        }
        grow = g;
    }

    if grow * tscal > smlnum {
        dtrsv(upper, notran, nounit, n, a, lda, x);
    } else {
        if xmax > bignum {
            scale = bignum / xmax;
            dscal(n, scale, x, 0, 1);
            xmax = bignum;
        }
        if notran {
            for &j in &order {
                let mut xj = x[j].abs();
                let tjjs = if nounit { at(j, j) * tscal } else { tscal };
                if nounit || tscal != 1.0 {
                    let tjj = tjjs.abs();
                    if tjj > smlnum {
                        if tjj < 1.0 && xj > tjj * bignum {
                            let rec = 1.0 / xj;
                            dscal(n, rec, x, 0, 1);
                            scale *= rec;
                            xmax *= rec;
                        }
                        x[j] /= tjjs;
                        xj = x[j].abs();
                    } else if tjj > 0.0 {
                        if xj > tjj * bignum {
                            let mut rec = (tjj * bignum) / xj;
                            if cnorm[j] > 1.0 {
                                rec /= cnorm[j];
                            }
                            dscal(n, rec, x, 0, 1);
                            scale *= rec;
                            xmax *= rec;
                        }
                        x[j] /= tjjs;
                        xj = x[j].abs();
                    } else {
                        x[..n].iter_mut().for_each(|v| *v = 0.0);
                        x[j] = 1.0;
                        xj = 1.0;
                        scale = 0.0;
                        xmax = 0.0;
                    }
                }
                // label 100
                if xj > 1.0 {
                    let mut rec = 1.0 / xj;
                    if cnorm[j] > (bignum - xmax) * rec {
                        rec *= 0.5;
                        dscal(n, rec, x, 0, 1);
                        scale *= rec;
                    }
                } else if xj * cnorm[j] > bignum - xmax {
                    dscal(n, 0.5, x, 0, 1);
                    scale *= 0.5;
                }
                if upper {
                    if j > 0 {
                        let da = -x[j] * tscal;
                        if da != 0.0 {
                            for i in 0..j {
                                x[i] += da * at(i, j);
                            }
                        }
                        let i = idamax(j, x, 0, 1);
                        xmax = x[i].abs();
                    }
                } else if j < n - 1 {
                    let da = -x[j] * tscal;
                    if da != 0.0 {
                        for i in (j + 1)..n {
                            x[i] += da * at(i, j);
                        }
                    }
                    let i = j + 1 + idamax(n - j - 1, x, j + 1, 1);
                    xmax = x[i].abs();
                }
            }
        } else {
            for &j in &order {
                let mut xj = x[j].abs();
                let mut uscal = tscal;
                let mut rec = 1.0 / xmax.max(1.0);
                let mut tjjs = 0.0;
                if cnorm[j] > (bignum - xj) * rec {
                    rec *= 0.5;
                    tjjs = if nounit { at(j, j) * tscal } else { tscal };
                    let tjj = tjjs.abs();
                    if tjj > 1.0 {
                        rec = (rec * tjj).min(1.0);
                        uscal /= tjjs;
                    }
                    if rec < 1.0 {
                        dscal(n, rec, x, 0, 1);
                        scale *= rec;
                        xmax *= rec;
                    }
                }
                let mut sumj = 0.0;
                if uscal == 1.0 {
                    if upper {
                        sumj = ddot(j, a, j * lda, x, 0);
                    } else if j < n - 1 {
                        sumj = ddot(n - j - 1, a, j + 1 + j * lda, x, j + 1);
                    }
                } else if upper {
                    for i in 0..j {
                        sumj += (at(i, j) * uscal) * x[i];
                    }
                } else if j < n - 1 {
                    for i in (j + 1)..n {
                        sumj += (at(i, j) * uscal) * x[i];
                    }
                }
                if uscal == tscal {
                    x[j] -= sumj;
                    xj = x[j].abs();
                    let tjjs = if nounit { at(j, j) * tscal } else { tscal };
                    if nounit || tscal != 1.0 {
                        let tjj = tjjs.abs();
                        if tjj > smlnum {
                            if tjj < 1.0 && xj > tjj * bignum {
                                let rec = 1.0 / xj;
                                dscal(n, rec, x, 0, 1);
                                scale *= rec;
                                xmax *= rec;
                            }
                            x[j] /= tjjs;
                        } else if tjj > 0.0 {
                            if xj > tjj * bignum {
                                let rec = (tjj * bignum) / xj;
                                dscal(n, rec, x, 0, 1);
                                scale *= rec;
                                xmax *= rec;
                            }
                            x[j] /= tjjs;
                        } else {
                            x[..n].iter_mut().for_each(|v| *v = 0.0);
                            x[j] = 1.0;
                            scale = 0.0;
                            xmax = 0.0;
                        }
                    }
                } else {
                    x[j] = x[j] / tjjs - sumj;
                }
                xmax = xmax.max(x[j].abs());
            }
        }
        scale /= tscal;
    }
    if tscal != 1.0 {
        dscal(n, 1.0 / tscal, cnorm, 0, 1);
    }
    scale
}

// ---------------------------------------------------------------- condition estimates

/// dlacn2 reverse-communication state (`isave`, `isgn`), 0-based indices.
struct Lacn2 {
    isave: [usize; 3],
    isgn: Vec<i64>,
}

/// One dlacn2 call. `kase` is 0 on entry to start and 0 on return when done.
fn dlacn2(n: usize, v: &mut [f64], x: &mut [f64], st: &mut Lacn2, est: &mut f64, kase: &mut i32) {
    const ITMAX: usize = 5;
    if *kase == 0 {
        for xi in x.iter_mut().take(n) {
            *xi = 1.0 / n as f64;
        }
        *kase = 1;
        st.isave[0] = 1;
        return;
    }
    let sgn = |v: f64| if v >= 0.0 { 1.0 } else { -1.0 };
    // Fortran labels: isave(1) = 1 -> 20, 2 -> 40, 3 -> 70, 4 -> 110, 5 -> 140.
    let mut label = match st.isave[0] {
        1 => 20,
        2 => 40,
        3 => 70,
        4 => 110,
        _ => 140,
    };
    loop {
        match label {
            20 => {
                if n == 1 {
                    v[0] = x[0];
                    *est = v[0].abs();
                    *kase = 0;
                    return;
                }
                *est = dasum(n, x, 0, 1);
                for i in 0..n {
                    x[i] = sgn(x[i]);
                    st.isgn[i] = x[i] as i64;
                }
                *kase = 2;
                st.isave[0] = 2;
                return;
            }
            40 => {
                st.isave[1] = idamax(n, x, 0, 1);
                st.isave[2] = 2;
                label = 50;
            }
            50 => {
                for xi in x.iter_mut().take(n) {
                    *xi = 0.0;
                }
                x[st.isave[1]] = 1.0;
                *kase = 1;
                st.isave[0] = 3;
                return;
            }
            70 => {
                v[..n].copy_from_slice(&x[..n]);
                let estold = *est;
                *est = dasum(n, v, 0, 1);
                let differ = (0..n).any(|i| sgn(x[i]) as i64 != st.isgn[i]);
                if !differ || *est <= estold {
                    label = 120;
                    continue;
                }
                for i in 0..n {
                    x[i] = sgn(x[i]);
                    st.isgn[i] = x[i] as i64;
                }
                *kase = 2;
                st.isave[0] = 4;
                return;
            }
            110 => {
                let jlast = st.isave[1];
                st.isave[1] = idamax(n, x, 0, 1);
                if x[jlast] != x[st.isave[1]].abs() && st.isave[2] < ITMAX {
                    st.isave[2] += 1;
                    label = 50;
                    continue;
                }
                label = 120;
            }
            120 => {
                let mut altsgn = 1.0;
                for (i, xi) in x.iter_mut().take(n).enumerate() {
                    *xi = altsgn * (1.0 + i as f64 / (n - 1) as f64);
                    altsgn = -altsgn;
                }
                *kase = 1;
                st.isave[0] = 5;
                return;
            }
            _ => {
                let temp = 2.0 * (dasum(n, x, 0, 1) / (3 * n) as f64);
                if temp > *est {
                    v[..n].copy_from_slice(&x[..n]);
                    *est = temp;
                }
                *kase = 0;
                return;
            }
        }
    }
}

/// dpocon('L'): reciprocal 1-norm condition estimate from the Cholesky factor `L` (lower
/// triangle of `a`) and `anorm` = dlansy('1','L') of the original matrix.
pub fn dpocon_l(a: &[f64], n: usize, lda: usize, anorm: f64) -> f64 {
    if n == 0 {
        return 1.0;
    }
    if anorm == 0.0 {
        return 0.0;
    }
    let smlnum = SFMIN;
    let mut x = vec![0.0; n];
    let mut v = vec![0.0; n];
    let mut cnorm = vec![0.0; n];
    let mut st = Lacn2 { isave: [0; 3], isgn: vec![0; n] };
    let mut ainvnm = 0.0;
    let mut kase = 0;
    let mut normin = false;
    loop {
        dlacn2(n, &mut v, &mut x, &mut st, &mut ainvnm, &mut kase);
        if kase == 0 {
            break;
        }
        let scalel = dlatrs(false, true, true, normin, n, a, lda, &mut x, &mut cnorm);
        normin = true;
        let scaleu = dlatrs(false, false, true, normin, n, a, lda, &mut x, &mut cnorm);
        let scale = scalel * scaleu;
        if scale != 1.0 {
            let ix = idamax(n, &x, 0, 1);
            if scale < x[ix].abs() * smlnum || scale == 0.0 {
                return 0.0;
            }
            drscl(n, scale, &mut x);
        }
    }
    if ainvnm != 0.0 {
        (1.0 / ainvnm) / anorm
    } else {
        0.0
    }
}

/// dgecon('1'): reciprocal 1-norm condition estimate from the dgetrf factors `L\U` and
/// `anorm` = dlange('1') of the original matrix. Returns the raw `rcond`: 0 also when dgecon
/// sets info = 1 (Armadillo's `lu_rcond` maps that case to 0 as well), NaN for a NaN anorm.
pub fn dgecon_1(a: &[f64], n: usize, lda: usize, anorm: f64) -> f64 {
    if n == 0 {
        return 1.0;
    }
    if anorm == 0.0 {
        return 0.0;
    }
    if anorm.is_nan() {
        return anorm;
    }
    if anorm > f64::MAX {
        return 0.0;
    }
    let smlnum = SFMIN;
    let mut x = vec![0.0; n];
    let mut v = vec![0.0; n];
    let mut cnorm_l = vec![0.0; n];
    let mut cnorm_u = vec![0.0; n];
    let mut st = Lacn2 { isave: [0; 3], isgn: vec![0; n] };
    let mut ainvnm = 0.0;
    let mut kase = 0;
    let mut normin = false;
    loop {
        dlacn2(n, &mut v, &mut x, &mut st, &mut ainvnm, &mut kase);
        if kase == 0 {
            break;
        }
        let (sl, su);
        if kase == 1 {
            sl = dlatrs(false, true, false, normin, n, a, lda, &mut x, &mut cnorm_l);
            su = dlatrs(true, true, true, normin, n, a, lda, &mut x, &mut cnorm_u);
        } else {
            su = dlatrs(true, false, true, normin, n, a, lda, &mut x, &mut cnorm_u);
            sl = dlatrs(false, false, false, normin, n, a, lda, &mut x, &mut cnorm_l);
        }
        let scale = sl * su;
        normin = true;
        if scale != 1.0 {
            let ix = idamax(n, &x, 0, 1);
            if scale < x[ix].abs() * smlnum || scale == 0.0 {
                return 0.0;
            }
            drscl(n, scale, &mut x);
        }
    }
    if ainvnm != 0.0 {
        let rc = (1.0 / ainvnm) / anorm;
        if rc.is_nan() || rc > f64::MAX {
            return 0.0;
        }
        rc
    } else {
        0.0
    }
}

// ---------------------------------------------------------------- Householder

/// dlapy2: sqrt(x^2 + y^2) without destructive underflow or overflow.
fn dlapy2(x: f64, y: f64) -> f64 {
    let (xn, yn) = (x.is_nan(), y.is_nan());
    let mut out = 0.0;
    if xn {
        out = x;
    }
    if yn {
        out = y;
    }
    if !(xn || yn) {
        let (xa, ya) = (x.abs(), y.abs());
        let w = xa.max(ya);
        let z = xa.min(ya);
        if z == 0.0 || w > f64::MAX {
            out = w;
        } else {
            let r = z / w;
            out = w * (1.0 + r * r).sqrt();
        }
    }
    out
}

/// dlarfg on alpha = a[ia] and x = a[ix + k*incx], k < n-1. Returns tau.
fn dlarfg(n: usize, a: &mut [f64], ia: usize, ix: usize, incx: usize) -> f64 {
    if n <= 1 {
        return 0.0;
    }
    let mut xnorm = dnrm2(n - 1, a, ix, incx);
    if xnorm == 0.0 {
        return 0.0;
    }
    let mut alpha = a[ia];
    let mut beta = -sign(dlapy2(alpha, xnorm), alpha);
    let safmin = SFMIN / EPS_E;
    let mut knt = 0;
    if beta.abs() < safmin {
        let rsafmn = 1.0 / safmin;
        loop {
            knt += 1;
            dscal(n - 1, rsafmn, a, ix, incx);
            beta *= rsafmn;
            alpha *= rsafmn;
            if !(beta.abs() < safmin && knt < 20) {
                break;
            }
        }
        xnorm = dnrm2(n - 1, a, ix, incx);
        beta = -sign(dlapy2(alpha, xnorm), alpha);
    }
    let tau = (beta - alpha) / beta;
    dscal(n - 1, 1.0 / (alpha - beta), a, ix, incx);
    for _ in 0..knt {
        beta *= safmin;
    }
    a[ia] = beta;
    tau
}

/// iladlc: number of the last non-zero column of the m x n block at `co` (0 if none).
fn iladlc(m: usize, n: usize, c: &[f64], co: usize, ldc: usize) -> usize {
    if n == 0 {
        return 0;
    }
    let at = |i: usize, j: usize| c[co + i + j * ldc];
    if at(0, n - 1) != 0.0 || at(m - 1, n - 1) != 0.0 {
        return n;
    }
    for j in (0..n).rev() {
        if (0..m).any(|i| at(i, j) != 0.0) {
            return j + 1;
        }
    }
    0
}

/// iladlr: number of the last non-zero row of the m x n block at `co` (0 if none).
fn iladlr(m: usize, n: usize, c: &[f64], co: usize, ldc: usize) -> usize {
    if m == 0 {
        return 0;
    }
    let at = |i: usize, j: usize| c[co + i + j * ldc];
    if at(m - 1, 0) != 0.0 || at(m - 1, n - 1) != 0.0 {
        return m;
    }
    let mut out = 0;
    for j in 0..n {
        let mut i = m; // 1-based row count
        while i >= 1 && at(i - 1, j) == 0.0 {
            i -= 1;
        }
        out = out.max(i);
    }
    out
}

/// dlarf1f: apply H = I - tau v v^T (v[0] taken as 1, `v` contiguous) to the m x n block of
/// `c` at `co` from the left or the right.
fn dlarf1f(left: bool, m: usize, n: usize, v: &[f64], tau: f64, c: &mut [f64], co: usize, ldc: usize) {
    let mut lastv = 1;
    let mut lastc = 0;
    if tau != 0.0 {
        lastv = if left { m } else { n };
        while lastv > 1 && v[lastv - 1] == 0.0 {
            lastv -= 1;
        }
        lastc = if left { iladlc(lastv, n, c, co, ldc) } else { iladlr(m, lastv, c, co, ldc) };
    }
    if lastc == 0 {
        return;
    }
    let ix = |i: usize, j: usize| co + i + j * ldc;
    if left {
        if lastv == 1 {
            dscal(lastc, 1.0 - tau, c, co, ldc);
        } else {
            // dgemv('T', lastv-1, lastc, 1, C(2,1), v(2), 0, work)
            let mut work = vec![0.0; lastc];
            for (j, w) in work.iter_mut().enumerate() {
                let mut temp = 0.0;
                for i in 1..lastv {
                    temp += c[ix(i, j)] * v[i];
                }
                *w += temp;
            }
            // daxpy(lastc, 1, C(1,1) row, work)
            for (j, w) in work.iter_mut().enumerate() {
                *w += c[ix(0, j)];
            }
            // daxpy(lastc, -tau, work, C(1,1) row)
            for (j, w) in work.iter().enumerate() {
                c[ix(0, j)] += -tau * w;
            }
            // dger(lastv-1, lastc, -tau, v(2), work, C(2,1))
            for (j, &wj) in work.iter().enumerate() {
                if wj != 0.0 {
                    let temp = -tau * wj;
                    for i in 1..lastv {
                        c[ix(i, j)] += v[i] * temp;
                    }
                }
            }
        }
    } else if lastv == 1 {
        dscal(lastc, 1.0 - tau, c, co, 1);
    } else {
        // dgemv('N', lastc, lastv-1, 1, C(1,2), v(2), 0, work)
        let mut work = vec![0.0; lastc];
        for j in 1..lastv {
            let temp = v[j];
            for (i, w) in work.iter_mut().enumerate() {
                *w += temp * c[ix(i, j)];
            }
        }
        for (i, w) in work.iter_mut().enumerate() {
            *w += c[ix(i, 0)];
        }
        for (i, w) in work.iter().enumerate() {
            c[ix(i, 0)] += -tau * w;
        }
        // dger(lastc, lastv-1, -tau, work, v(2), C(1,2))
        for j in 1..lastv {
            if v[j] != 0.0 {
                let temp = -tau * v[j];
                for (i, w) in work.iter().enumerate() {
                    c[ix(i, j)] += w * temp;
                }
            }
        }
    }
}

/// dgebd2 for m >= n: reduce A to upper bidiagonal form. Returns (d, e, tauq, taup).
type Bidiag = (Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>);
fn dgebd2(m: usize, n: usize, a: &mut [f64], lda: usize) -> Bidiag {
    let ix = |i: usize, j: usize| i + j * lda;
    let mut d = vec![0.0; n];
    let mut e = vec![0.0; n.saturating_sub(1)];
    let mut tauq = vec![0.0; n];
    let mut taup = vec![0.0; n];
    for i in 0..n {
        tauq[i] = dlarfg(m - i, a, ix(i, i), ix((i + 1).min(m - 1), i), 1);
        d[i] = a[ix(i, i)];
        if i < n - 1 {
            let v: Vec<f64> = (0..m - i).map(|k| a[ix(i + k, i)]).collect();
            dlarf1f(true, m - i, n - i - 1, &v, tauq[i], a, ix(i, i + 1), lda);
            taup[i] = dlarfg(n - i - 1, a, ix(i, i + 1), ix(i, (i + 2).min(n - 1)), lda);
            e[i] = a[ix(i, i + 1)];
            let v: Vec<f64> = (0..n - i - 1).map(|k| a[ix(i, i + 1 + k)]).collect();
            dlarf1f(false, m - i - 1, n - i - 1, &v, taup[i], a, ix(i + 1, i + 1), lda);
        } else {
            taup[i] = 0.0;
        }
    }
    (d, e, tauq, taup)
}

// ---------------------------------------------------------------- bidiagonal SVD

/// dlartg (LAPACK 3.10+ f90 version): plane rotation with c*f + s*g = r, -s*f + c*g = 0.
fn dlartg(f: f64, g: f64) -> (f64, f64, f64) {
    let safmin = SFMIN;
    let safmax = 1.0 / safmin;
    let rtmin = safmin.sqrt();
    let rtmax = (safmax / 2.0).sqrt();
    let f1 = f.abs();
    let g1 = g.abs();
    if g == 0.0 {
        (1.0, 0.0, f)
    } else if f == 0.0 {
        (0.0, sign(1.0, g), g1)
    } else if f1 > rtmin && f1 < rtmax && g1 > rtmin && g1 < rtmax {
        let d = (f * f + g * g).sqrt();
        let c = f1 / d;
        let r = sign(d, f);
        (c, g / r, r)
    } else {
        let u = safmax.min(safmin.max(f1).max(g1));
        let fs = f / u;
        let gs = g / u;
        let d = (fs * fs + gs * gs).sqrt();
        let c = fs.abs() / d;
        let r = sign(d, f);
        let s = gs / r;
        (c, s, r * u)
    }
}

/// dlas2: singular values (ssmin, ssmax) of [[f, g], [0, h]].
fn dlas2(f: f64, g: f64, h: f64) -> (f64, f64) {
    let fa = f.abs();
    let ga = g.abs();
    let ha = h.abs();
    let fhmn = fa.min(ha);
    let fhmx = fa.max(ha);
    if fhmn == 0.0 {
        let ssmax = if fhmx == 0.0 {
            ga
        } else {
            let r = fhmx.min(ga) / fhmx.max(ga);
            fhmx.max(ga) * (1.0 + r * r).sqrt()
        };
        (0.0, ssmax)
    } else if ga < fhmx {
        let as_ = 1.0 + fhmn / fhmx;
        let at = (fhmx - fhmn) / fhmx;
        let au = (ga / fhmx) * (ga / fhmx);
        let c = 2.0 / ((as_ * as_ + au).sqrt() + (at * at + au).sqrt());
        (fhmn * c, fhmx / c)
    } else {
        let au = fhmx / ga;
        if au == 0.0 {
            ((fhmn * fhmx) / ga, ga)
        } else {
            let as_ = 1.0 + fhmn / fhmx;
            let at = (fhmx - fhmn) / fhmx;
            let p = as_ * au;
            let q = at * au;
            let c = 1.0 / ((1.0 + p * p).sqrt() + (1.0 + q * q).sqrt());
            let mut ssmin = (fhmn * c) * au;
            ssmin += ssmin;
            (ssmin, ga / (c + c))
        }
    }
}

/// dlasv2: SVD of [[f, g], [0, h]]. Returns (ssmin, ssmax, snr, csr, snl, csl).
fn dlasv2(f: f64, g: f64, h: f64) -> (f64, f64, f64, f64, f64, f64) {
    let mut ft = f;
    let mut fa = ft.abs();
    let mut ht = h;
    let mut ha = h.abs();
    let mut pmax = 1;
    let swap = ha > fa;
    if swap {
        pmax = 3;
        std::mem::swap(&mut ft, &mut ht);
        std::mem::swap(&mut fa, &mut ha);
    }
    let gt = g;
    let ga = gt.abs();
    let (mut ssmin, mut ssmax, clt, crt, slt, srt);
    if ga == 0.0 {
        ssmin = ha;
        ssmax = fa;
        clt = 1.0;
        crt = 1.0;
        slt = 0.0;
        srt = 0.0;
    } else {
        let mut gasmal = true;
        let mut r = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0); // ssmin, ssmax, clt, crt, slt, srt
        if ga > fa {
            pmax = 2;
            if fa / ga < EPS_E {
                gasmal = false;
                let smin = if ha > 1.0 { fa / (ga / ha) } else { (fa / ga) * ha };
                r = (smin, ga, 1.0, ft / gt, ht / gt, 1.0);
            }
        }
        if gasmal {
            let d = fa - ha;
            let mut l = if d == fa { 1.0 } else { d / fa };
            let m = gt / ft;
            let mut t = 2.0 - l;
            let mm = m * m;
            let tt = t * t;
            let s = (tt + mm).sqrt();
            let rr = if l == 0.0 { m.abs() } else { (l * l + mm).sqrt() };
            let a = 0.5 * (s + rr);
            let smin = ha / a;
            let smax = fa * a;
            if mm == 0.0 {
                if l == 0.0 {
                    t = sign(2.0, ft) * sign(1.0, gt);
                } else {
                    t = gt / sign(d, ft) + m / t;
                }
            } else {
                t = (m / (s + t) + m / (rr + l)) * (1.0 + a);
            }
            l = (t * t + 4.0).sqrt();
            let c_rt = 2.0 / l;
            let s_rt = t / l;
            let c_lt = (c_rt + s_rt * m) / a;
            let s_lt = (ht / ft) * s_rt / a;
            r = (smin, smax, c_lt, c_rt, s_lt, s_rt);
        }
        ssmin = r.0;
        ssmax = r.1;
        clt = r.2;
        crt = r.3;
        slt = r.4;
        srt = r.5;
    }
    let (csl, snl, csr, snr) = if swap { (srt, crt, slt, clt) } else { (clt, slt, crt, srt) };
    let tsign = match pmax {
        1 => sign(1.0, csr) * sign(1.0, csl) * sign(1.0, f),
        2 => sign(1.0, snr) * sign(1.0, csl) * sign(1.0, g),
        _ => sign(1.0, snr) * sign(1.0, snl) * sign(1.0, h),
    };
    ssmax = sign(ssmax, tsign);
    ssmin = sign(ssmin, tsign * sign(1.0, f) * sign(1.0, h));
    (ssmin, ssmax, snr, csr, snl, csl)
}

/// dlasr('L', 'V', direct) on rows `r0 .. r0+mm` of the `ncol`-column matrix `a`.
fn dlasr_lv(forward: bool, mm: usize, ncol: usize, c: &[f64], s: &[f64], a: &mut [f64], r0: usize, lda: usize) {
    if mm <= 1 || ncol == 0 {
        return;
    }
    let mut step = |j: usize| {
        let (ctemp, stemp) = (c[j], s[j]);
        if ctemp != 1.0 || stemp != 0.0 {
            for i in 0..ncol {
                let p = r0 + j + i * lda;
                let temp = a[p + 1];
                a[p + 1] = ctemp * temp - stemp * a[p];
                a[p] = stemp * temp + ctemp * a[p];
            }
        }
    };
    if forward {
        (0..mm - 1).for_each(&mut step);
    } else {
        (0..mm - 1).rev().for_each(&mut step);
    }
}

/// drot on rows r1 and r2 of the `ncol`-column matrix `a`.
fn drot_rows(ncol: usize, a: &mut [f64], r1: usize, r2: usize, lda: usize, c: f64, s: f64) {
    for k in 0..ncol {
        let (p, q) = (r1 + k * lda, r2 + k * lda);
        let dtemp = c * a[p] + s * a[q];
        a[q] = c * a[q] - s * a[p];
        a[p] = dtemp;
    }
}

fn swap_rows(ncol: usize, a: &mut [f64], r1: usize, r2: usize, lda: usize) {
    for k in 0..ncol {
        a.swap(r1 + k * lda, r2 + k * lda);
    }
}

/// dbdsqr('U') with nru = 0: singular values of the upper bidiagonal (d, e), applying the
/// right rotations to the rows of `vt` (n x ncvt) and the left rotations to the rows of `c`
/// (n x ncc). Returns info.
fn dbdsqr_u(
    n: usize,
    d0: &mut [f64],
    e0: &mut [f64],
    vt: &mut [f64],
    ldvt: usize,
    ncvt: usize,
    c: &mut [f64],
    ldc: usize,
    ncc: usize,
) -> usize {
    const MAXITR: usize = 6;
    if n == 0 {
        return 0;
    }
    // 1-based working copies of d and e; row i (1-based) of VT / C is row i-1.
    let mut d = vec![0.0; n + 1];
    d[1..].copy_from_slice(&d0[..n]);
    let mut e = vec![0.0; n + 1];
    e[1..n].copy_from_slice(&e0[..n - 1]);
    let row = |i: usize| i - 1;
    let mut info = 0;
    let mut failed = false;

    if n > 1 {
        let nm1 = n - 1;
        let nm12 = nm1 + nm1;
        let nm13 = nm12 + nm1;
        let mut work = vec![0.0; 4 * n + 1];
        let mut idir = 0;
        let eps = EPS_E;
        let unfl = SFMIN;
        let tolmul = 10f64.max(100f64.min(eps.powf(-0.125)));
        let tol = tolmul * eps;
        let nf = n as f64;
        // tol >= 0: relative accuracy; the smax computed here only feeds the tol < 0 branch.
        let mut sminoa = d[1].abs();
        if sminoa != 0.0 {
            let mut mu = sminoa;
            for i in 2..=n {
                mu = d[i].abs() * (mu / (mu + e[i - 1].abs()));
                sminoa = sminoa.min(mu);
                if sminoa == 0.0 {
                    break;
                }
            }
        }
        sminoa /= nf.sqrt();
        let thresh = (tol * sminoa).max(MAXITR as f64 * (nf * (nf * unfl)));
        let maxitdivn = MAXITR * n;
        let mut iterdivn = 0;
        let mut iter: i64 = -1;
        let mut oldll: i64 = -1;
        let mut oldm: i64 = -1;
        let mut m = n;

        'main: loop {
            // label 60
            if m <= 1 {
                break;
            }
            if iter >= n as i64 {
                iter -= n as i64;
                iterdivn += 1;
                if iterdivn >= maxitdivn {
                    failed = true;
                    break;
                }
            }
            let mut smax = d[m].abs();
            let mut found = None;
            for lll in 1..m {
                let ll = m - lll;
                let abss = d[ll].abs();
                let abse = e[ll].abs();
                if abse <= thresh {
                    found = Some(ll);
                    break;
                }
                smax = smax.max(abss).max(abse);
            }
            let mut ll = match found {
                Some(l) => {
                    e[l] = 0.0;
                    if l == m - 1 {
                        m -= 1;
                        continue 'main;
                    }
                    l
                }
                None => 0,
            };
            // label 90
            ll += 1;
            if ll == m - 1 {
                let (sigmn, sigmx, sinr, cosr, sinl, cosl) = dlasv2(d[m - 1], e[m - 1], d[m]);
                d[m - 1] = sigmx;
                e[m - 1] = 0.0;
                d[m] = sigmn;
                if ncvt > 0 {
                    drot_rows(ncvt, vt, row(m - 1), row(m), ldvt, cosr, sinr);
                }
                if ncc > 0 {
                    drot_rows(ncc, c, row(m - 1), row(m), ldc, cosl, sinl);
                }
                m -= 2;
                continue 'main;
            }
            if ll as i64 > oldm || (m as i64) < oldll {
                idir = if d[ll].abs() >= d[m].abs() { 1 } else { 2 };
            }
            let mut smin;
            if idir == 1 {
                if e[m - 1].abs() <= tol.abs() * d[m].abs() {
                    e[m - 1] = 0.0;
                    continue 'main;
                }
                let mut mu = d[ll].abs();
                smin = mu;
                for lll in ll..m {
                    if e[lll].abs() <= tol * mu {
                        e[lll] = 0.0;
                        continue 'main;
                    }
                    mu = d[lll + 1].abs() * (mu / (mu + e[lll].abs()));
                    smin = smin.min(mu);
                }
            } else {
                if e[ll].abs() <= tol.abs() * d[ll].abs() {
                    e[ll] = 0.0;
                    continue 'main;
                }
                let mut mu = d[m].abs();
                smin = mu;
                for lll in (ll..m).rev() {
                    if e[lll].abs() <= tol * mu {
                        e[lll] = 0.0;
                        continue 'main;
                    }
                    mu = d[lll].abs() * (mu / (mu + e[lll].abs()));
                    smin = smin.min(mu);
                }
            }
            oldll = ll as i64;
            oldm = m as i64;
            let mut shift;
            if nf * tol * (smin / smax) <= eps.max(0.01 * tol) {
                shift = 0.0;
            } else {
                let sll;
                if idir == 1 {
                    sll = d[ll].abs();
                    shift = dlas2(d[m - 1], e[m - 1], d[m]).0;
                } else {
                    sll = d[m].abs();
                    shift = dlas2(d[ll], e[ll], d[ll + 1]).0;
                }
                if sll > 0.0 {
                    let q = shift / sll;
                    if q * q < eps {
                        shift = 0.0;
                    }
                }
            }
            iter += (m - ll) as i64;
            let mlen = m - ll + 1;
            // work(1), work(n) = work(nm1+1), work(nm12+1), work(nm13+1) as slices
            let w1 = 1;
            let wn = n;
            let w2 = nm12 + 1;
            let w3 = nm13 + 1;
            if shift == 0.0 {
                if idir == 1 {
                    let mut cs = 1.0;
                    let mut oldcs = 1.0;
                    let mut oldsn = 0.0;
                    for i in ll..m {
                        let (c1, sn, r) = dlartg(d[i] * cs, e[i]);
                        cs = c1;
                        if i > ll {
                            e[i - 1] = oldsn * r;
                        }
                        let (oc, os, di) = dlartg(oldcs * r, d[i + 1] * sn);
                        oldcs = oc;
                        oldsn = os;
                        d[i] = di;
                        let k = i - ll + 1;
                        work[k] = cs;
                        work[k + nm1] = sn;
                        work[k + nm12] = oldcs;
                        work[k + nm13] = oldsn;
                    }
                    let h = d[m] * cs;
                    d[m] = h * oldcs;
                    e[m - 1] = h * oldsn;
                    if ncvt > 0 {
                        dlasr_lv(true, mlen, ncvt, &work[w1..], &work[wn..], vt, row(ll), ldvt);
                    }
                    if ncc > 0 {
                        dlasr_lv(true, mlen, ncc, &work[w2..], &work[w3..], c, row(ll), ldc);
                    }
                    if e[m - 1].abs() <= thresh {
                        e[m - 1] = 0.0;
                    }
                } else {
                    let mut cs = 1.0;
                    let mut oldcs = 1.0;
                    let mut oldsn = 0.0;
                    for i in ((ll + 1)..=m).rev() {
                        let (c1, sn, r) = dlartg(d[i] * cs, e[i - 1]);
                        cs = c1;
                        if i < m {
                            e[i] = oldsn * r;
                        }
                        let (oc, os, di) = dlartg(oldcs * r, d[i - 1] * sn);
                        oldcs = oc;
                        oldsn = os;
                        d[i] = di;
                        let k = i - ll;
                        work[k] = cs;
                        work[k + nm1] = -sn;
                        work[k + nm12] = oldcs;
                        work[k + nm13] = -oldsn;
                    }
                    let h = d[ll] * cs;
                    d[ll] = h * oldcs;
                    e[ll] = h * oldsn;
                    if ncvt > 0 {
                        dlasr_lv(false, mlen, ncvt, &work[w2..], &work[w3..], vt, row(ll), ldvt);
                    }
                    if ncc > 0 {
                        dlasr_lv(false, mlen, ncc, &work[w1..], &work[wn..], c, row(ll), ldc);
                    }
                    if e[ll].abs() <= thresh {
                        e[ll] = 0.0;
                    }
                }
            } else if idir == 1 {
                let mut f = (d[ll].abs() - shift) * (sign(1.0, d[ll]) + shift / d[ll]);
                let mut g = e[ll];
                for i in ll..m {
                    let (cosr, sinr, r) = dlartg(f, g);
                    if i > ll {
                        e[i - 1] = r;
                    }
                    f = cosr * d[i] + sinr * e[i];
                    e[i] = cosr * e[i] - sinr * d[i];
                    g = sinr * d[i + 1];
                    d[i + 1] *= cosr;
                    let (cosl, sinl, r) = dlartg(f, g);
                    d[i] = r;
                    f = cosl * e[i] + sinl * d[i + 1];
                    d[i + 1] = cosl * d[i + 1] - sinl * e[i];
                    if i < m - 1 {
                        g = sinl * e[i + 1];
                        e[i + 1] *= cosl;
                    }
                    let k = i - ll + 1;
                    work[k] = cosr;
                    work[k + nm1] = sinr;
                    work[k + nm12] = cosl;
                    work[k + nm13] = sinl;
                }
                e[m - 1] = f;
                if ncvt > 0 {
                    dlasr_lv(true, mlen, ncvt, &work[w1..], &work[wn..], vt, row(ll), ldvt);
                }
                if ncc > 0 {
                    dlasr_lv(true, mlen, ncc, &work[w2..], &work[w3..], c, row(ll), ldc);
                }
                if e[m - 1].abs() <= thresh {
                    e[m - 1] = 0.0;
                }
            } else {
                let mut f = (d[m].abs() - shift) * (sign(1.0, d[m]) + shift / d[m]);
                let mut g = e[m - 1];
                for i in ((ll + 1)..=m).rev() {
                    let (cosr, sinr, r) = dlartg(f, g);
                    if i < m {
                        e[i] = r;
                    }
                    f = cosr * d[i] + sinr * e[i - 1];
                    e[i - 1] = cosr * e[i - 1] - sinr * d[i];
                    g = sinr * d[i - 1];
                    d[i - 1] *= cosr;
                    let (cosl, sinl, r) = dlartg(f, g);
                    d[i] = r;
                    f = cosl * e[i - 1] + sinl * d[i - 1];
                    d[i - 1] = cosl * d[i - 1] - sinl * e[i - 1];
                    if i > ll + 1 {
                        g = sinl * e[i - 2];
                        e[i - 2] *= cosl;
                    }
                    let k = i - ll;
                    work[k] = cosr;
                    work[k + nm1] = -sinr;
                    work[k + nm12] = cosl;
                    work[k + nm13] = -sinl;
                }
                e[ll] = f;
                if e[ll].abs() <= thresh {
                    e[ll] = 0.0;
                }
                if ncvt > 0 {
                    dlasr_lv(false, mlen, ncvt, &work[w2..], &work[w3..], vt, row(ll), ldvt);
                }
                if ncc > 0 {
                    dlasr_lv(false, mlen, ncc, &work[w1..], &work[wn..], c, row(ll), ldc);
                }
            }
        }
    }

    if failed {
        // label 200
        for i in 1..n {
            if e[i] != 0.0 {
                info += 1;
            }
        }
    } else {
        // label 160: make singular values non-negative, then sort decreasing
        for i in 1..=n {
            if d[i] == 0.0 {
                d[i] = 0.0;
            }
            if d[i] < 0.0 {
                d[i] = -d[i];
                if ncvt > 0 {
                    dscal(ncvt, -1.0, vt, row(i), ldvt);
                }
            }
        }
        for i in 1..n {
            let mut isub = 1;
            let mut smin = d[1];
            for j in 2..=(n + 1 - i) {
                if d[j] <= smin {
                    isub = j;
                    smin = d[j];
                }
            }
            let last = n + 1 - i;
            if isub != last {
                d[isub] = d[last];
                d[last] = smin;
                if ncvt > 0 {
                    swap_rows(ncvt, vt, row(isub), row(last), ldvt);
                }
                if ncc > 0 {
                    swap_rows(ncc, c, row(isub), row(last), ldc);
                }
            }
        }
    }
    d0[..n].copy_from_slice(&d[1..]);
    e0[..n - 1].copy_from_slice(&e[1..n]);
    info
}

/// dlasdq('U', sqre = 0, nru = 0): dbdsqr followed by dlasdq's own ascending selection sort.
fn dlasdq_u(n: usize, d: &mut [f64], e: &mut [f64], vt: &mut [f64], ldvt: usize, ncvt: usize, c: &mut [f64], ldc: usize, ncc: usize) -> usize {
    let info = dbdsqr_u(n, d, e, vt, ldvt, ncvt, c, ldc, ncc);
    if info != 0 {
        return info;
    }
    for i in 0..n {
        let mut isub = i;
        let mut smin = d[i];
        for j in (i + 1)..n {
            if d[j] < smin {
                isub = j;
                smin = d[j];
            }
        }
        if isub != i {
            d[isub] = d[i];
            d[i] = smin;
            if ncvt > 0 {
                swap_rows(ncvt, vt, isub, i, ldvt);
            }
            if ncc > 0 {
                swap_rows(ncc, c, isub, i, ldc);
            }
        }
    }
    0
}

/// dlalsd('U') for 1 <= n <= SMLSIZ with nrhs right-hand sides in `b` (n x nrhs, ldb).
/// Returns (rank, info).
fn dlalsd_small(n: usize, nrhs: usize, d: &mut [f64], e: &mut [f64], b: &mut [f64], ldb: usize, rcond: f64) -> (usize, usize) {
    let rcnd = if rcond <= 0.0 || rcond >= 1.0 { EPS_E } else { rcond };
    let mut rank = 0;
    if n == 0 {
        return (0, 0);
    }
    if n == 1 {
        if d[0] == 0.0 {
            for j in 0..nrhs {
                b[j * ldb] = 0.0;
            }
        } else {
            rank = 1;
            dlascl(d[0], 1.0, nrhs, b, 0, ldb);
            d[0] = d[0].abs();
        }
        return (rank, 0);
    }
    // dlanst('M')
    let mut orgnrm = d[n - 1].abs();
    for i in 0..n - 1 {
        let s = d[i].abs();
        if orgnrm < s || s.is_nan() {
            orgnrm = s;
        }
        let s = e[i].abs();
        if orgnrm < s || s.is_nan() {
            orgnrm = s;
        }
    }
    if orgnrm == 0.0 {
        for j in 0..nrhs {
            for i in 0..n {
                b[i + j * ldb] = 0.0;
            }
        }
        return (0, 0);
    }
    dlascl(orgnrm, 1.0, n, d, 0, 1);
    dlascl(orgnrm, 1.0, n - 1, e, 0, 1);
    let mut vt = vec![0.0; n * n];
    for i in 0..n {
        vt[i + i * n] = 1.0;
    }
    let info = dlasdq_u(n, d, e, &mut vt, n, n, b, ldb, nrhs);
    if info != 0 {
        return (0, info);
    }
    let tol = rcnd * d[idamax(n, d, 0, 1)].abs();
    for i in 0..n {
        if d[i] <= tol {
            for j in 0..nrhs {
                b[i + j * ldb] = 0.0;
            }
        } else {
            dlascl(d[i], 1.0, nrhs, b, i, ldb);
            rank += 1;
        }
    }
    // dgemm('T', 'N', n, nrhs, n, 1, VT, n, B, ldb, 0, work, n), then dlacpy back to B.
    let mut out = vec![0.0; n * nrhs];
    for j in 0..nrhs {
        for i in 0..n {
            let mut temp = 0.0;
            for l in 0..n {
                temp += vt[l + i * n] * b[l + j * ldb];
            }
            out[i + j * n] = temp;
        }
    }
    for j in 0..nrhs {
        for i in 0..n {
            b[i + j * ldb] = out[i + j * n];
        }
    }
    // (d is unscaled and sorted here; neither touches B.)
    for j in 0..nrhs {
        dlascl(orgnrm, 1.0, n, b, j * ldb, 1);
    }
    (rank, 0)
}

/// Largest n for which [`dgelsd_square`] reproduces dgelsd (its `n <= SMLSIZ` branch).
pub const DGELSD_SMLSIZ: usize = 25;

/// dgelsd for a square n x n `a` (column-major, lda = n) and one right-hand side,
/// 1 <= n <= 25. Returns None when dgelsd reports info != 0 (Armadillo then fails the solve).
pub fn dgelsd_square(a_in: &[f64], n: usize, rhs: &[f64], rcond: f64) -> Option<Vec<f64>> {
    assert!((1..=DGELSD_SMLSIZ).contains(&n), "dgelsd_square: n = {n} outside the ported branch");
    let (m, lda, nrhs, ldb) = (n, n, 1, n);
    let mut a = a_in[..n * n].to_vec();
    let mut b = rhs[..n].to_vec();
    let smlnum = SFMIN / EPS_P;
    let bignum = 1.0 / smlnum;
    let anrm = dlange_m(&a, m, n, lda);
    let mut iascl = 0;
    if anrm > 0.0 && anrm < smlnum {
        dlascl(anrm, smlnum, m * n, &mut a, 0, 1);
        iascl = 1;
    } else if anrm > bignum {
        dlascl(anrm, bignum, m * n, &mut a, 0, 1);
        iascl = 2;
    } else if anrm == 0.0 {
        return Some(vec![0.0; n]);
    }
    let bnrm = dlange_m(&b, m, nrhs, ldb);
    let mut ibscl = 0;
    if bnrm > 0.0 && bnrm < smlnum {
        dlascl(bnrm, smlnum, m, &mut b, 0, 1);
        ibscl = 1;
    } else if bnrm > bignum {
        dlascl(bnrm, bignum, m, &mut b, 0, 1);
        ibscl = 2;
    }
    // m == n: MNTHR = int(1.6 n) > n for n >= 2, so no QR pre-step; at n = 1 the QR step
    // has tau = 0 and leaves A and B unchanged, so both paths coincide.
    let (mut d, mut e, tauq, taup) = dgebd2(m, n, &mut a, lda);
    // dormbr('Q','L','T') -> dormqr -> dorm2r (block size 32 >= k): forward over i.
    for i in 0..n {
        let v: Vec<f64> = (0..m - i).map(|k| a[i + k + i * lda]).collect();
        dlarf1f(true, m - i, nrhs, &v, tauq[i], &mut b, i, ldb);
    }
    let (_rank, info) = dlalsd_small(n, nrhs, &mut d, &mut e, &mut b, ldb, rcond);
    if info != 0 {
        return None;
    }
    // dormbr('P','L','N') with nq = k = n -> dormlq('L','T', n-1, nrhs, n-1, A(1,2), .., B(2,1))
    // -> dorml2, which for 'L','T' runs backward over i.
    if n > 1 {
        let k = n - 1;
        for i in (0..k).rev() {
            let mi = k - i;
            let v: Vec<f64> = (0..mi).map(|t| a[i + (i + 1 + t) * lda]).collect();
            dlarf1f(true, mi, nrhs, &v, taup[i], &mut b, i + 1, ldb);
        }
    }
    match iascl {
        1 => dlascl(anrm, smlnum, n, &mut b, 0, 1),
        2 => dlascl(anrm, bignum, n, &mut b, 0, 1),
        _ => {}
    }
    match ibscl {
        1 => dlascl(smlnum, bnrm, n, &mut b, 0, 1),
        2 => dlascl(bignum, bnrm, n, &mut b, 0, 1),
        _ => {}
    }
    Some(b)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dgelsd_min_norm_rank_one() {
        let a = [1.0, 1.0, 1.0, 1.0];
        let x = dgelsd_square(&a, 2, &[2.0, 2.0], 2.0 * f64::EPSILON).unwrap();
        assert!((x[0] - 1.0).abs() < 1e-14 && (x[1] - 1.0).abs() < 1e-14, "{x:?}");
    }

    #[test]
    fn dgelsd_solves_full_rank() {
        let a = [4.0, 1.0, 0.5, 1.0, 3.0, 0.2, 0.5, 0.2, 2.0];
        let rhs = [1.0, 2.0, 3.0];
        let x = dgelsd_square(&a, 3, &rhs, 3.0 * f64::EPSILON).unwrap();
        for i in 0..3 {
            let r: f64 = (0..3).map(|j| a[i + 3 * j] * x[j]).sum();
            assert!((r - rhs[i]).abs() < 1e-13);
        }
    }

    #[test]
    fn rcond_estimates_on_diagonal() {
        // diag(2, 0.5): 1-norm condition 4
        let a = [2.0, 0.0, 0.0, 0.5];
        assert_eq!(dgecon_1(&a, 2, 2, 2.0), 0.25);
        let l = [2f64.sqrt(), 0.0, 0.0, 0.5f64.sqrt()];
        assert!((dpocon_l(&l, 2, 2, 2.0) - 0.25).abs() < 1e-15);
    }
}
