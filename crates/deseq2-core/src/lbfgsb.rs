//! R 4.5.0's `optim(method = "L-BFGS-B")` with numerical gradients, as DESeq2's
//! `fitNbinomGLMsOptim` calls it: the driver `lbfgsb()` (`src/appl/optim.c`), the bounded
//! finite-difference `fmingr` (`src/library/stats/src/optim.c`), and the f2c translation of
//! L-BFGS-B 2.x in `src/appl/lbfgsb.c` (`setulb`, `mainlb` and their subroutines), with the
//! reference BLAS (`ddot`, `daxpy`, `dscal`, `dcopy`) and LINPACK `dpofa` / `dtrsl` it calls.
//!
//! The port is line for line, so summation order and branch logic match R. Arrays keep the
//! Fortran 1-based, column-major layout of the C: a vector `v` is used as `v[1..=n]`, and an
//! `ld x ncol` matrix stores element `(i, j)` at `i + j * ld`, which is what f2c's
//! `a -= a_offset` makes of it. `rnum` (read-only for this crate) has only `vmmin`, so this lives
//! here.

/// The f2c `max` macro, `(a < b) ? b : a`.
fn cmax(a: f64, b: f64) -> f64 {
    if a < b {
        b
    } else {
        a
    }
}

/// The f2c `min` macro, `(a > b) ? b : a`.
fn cmin(a: f64, b: f64) -> f64 {
    if a > b {
        b
    } else {
        a
    }
}

/// A 1-based `ld x ncol` column-major buffer (element `(i, j)` at `i + j * ld`).
fn mat(ld: usize, ncol: usize) -> Vec<f64> {
    vec![0.0; ld * (ncol + 1) + 1]
}

/// Reference `daxpy` (`dy += da * dx`, skipped when `da == 0`) on 1-based slices.
fn daxpy(n: usize, da: f64, dx: &[f64], dy: &mut [f64]) {
    if da == 0.0 {
        return;
    }
    for i in 1..=n {
        dy[i] += da * dx[i];
    }
}

/// Reference `ddot` on 1-based slices (the unrolled loop still adds left to right).
fn ddot(n: usize, dx: &[f64], dy: &[f64]) -> f64 {
    let mut s = 0.0;
    for i in 1..=n {
        s += dx[i] * dy[i];
    }
    s
}

/// LINPACK `dpofa` on the `n x n` matrix whose element `(i, j)` is `a[base + i + j * lda]`.
/// Returns `info`.
fn dpofa(a: &mut [f64], base: usize, lda: usize, n: usize) -> i32 {
    let at = move |i: usize, j: usize| base + i + j * lda;
    for j in 1..=n {
        let mut s = 0.0;
        for k in 1..j {
            let mut dot = 0.0;
            for i in 1..k {
                dot += a[at(i, k)] * a[at(i, j)];
            }
            let mut t = a[at(k, j)] - dot;
            t /= a[at(k, k)];
            a[at(k, j)] = t;
            s += t * t;
        }
        s = a[at(j, j)] - s;
        if s <= 1e-14 * a[at(j, j)].abs() {
            return j as i32;
        }
        a[at(j, j)] = s.sqrt();
    }
    0
}

/// LINPACK `dtrsl` with `t(i, j) = t[base + i + j * ldt]` and a 1-based right-hand side `b`.
/// Returns `info`.
fn dtrsl(t: &[f64], base: usize, ldt: usize, n: usize, b: &mut [f64], job: i32) -> i32 {
    let at = move |i: usize, j: usize| base + i + j * ldt;
    for info in 1..=n {
        if t[at(info, info)] == 0.0 {
            return info as i32;
        }
    }
    if n == 0 {
        return 0;
    }
    let mut kase = 1;
    if job % 10 != 0 {
        kase = 2;
    }
    if (job % 100) / 10 != 0 {
        kase += 2;
    }
    match kase {
        1 => {
            b[1] /= t[at(1, 1)];
            for j in 2..=n {
                let temp = -b[j - 1];
                if temp != 0.0 {
                    for i in j..=n {
                        b[i] += temp * t[at(i, j - 1)];
                    }
                }
                b[j] /= t[at(j, j)];
            }
        }
        2 => {
            b[n] /= t[at(n, n)];
            for jj in 2..=n {
                let j = n - jj + 1;
                let temp = -b[j + 1];
                if temp != 0.0 {
                    for i in 1..=j {
                        b[i] += temp * t[at(i, j + 1)];
                    }
                }
                b[j] /= t[at(j, j)];
            }
        }
        3 => {
            b[n] /= t[at(n, n)];
            for jj in 2..=n {
                let j = n - jj + 1;
                let mut dot = 0.0;
                for i in 1..jj {
                    dot += t[at(j + i, j)] * b[j + i];
                }
                b[j] -= dot;
                b[j] /= t[at(j, j)];
            }
        }
        _ => {
            b[1] /= t[at(1, 1)];
            for j in 2..=n {
                let mut dot = 0.0;
                for i in 1..j {
                    dot += t[at(i, j)] * b[i];
                }
                b[j] -= dot;
                b[j] /= t[at(j, j)];
            }
        }
    }
    0
}

/// `bmv`: the product of the 2m x 2m middle matrix with `v` (`p` and `v` are 1-based).
fn bmv(m: usize, sy: &[f64], wt: &[f64], col: usize, v: &[f64], p: &mut [f64]) -> i32 {
    let q = move |i: usize, j: usize| i + j * m;
    if col == 0 {
        return 0;
    }
    p[col + 1] = v[col + 1];
    for i in 2..=col {
        let i2 = col + i;
        let mut sum = 0.0;
        for k in 1..i {
            sum += sy[q(i, k)] * v[k] / sy[q(k, k)];
        }
        p[i2] = v[i2] + sum;
    }
    let info = dtrsl(wt, 0, m, col, &mut p[col..], 11);
    if info != 0 {
        return info;
    }
    for i in 1..=col {
        p[i] = v[i] / sy[q(i, i)].sqrt();
    }
    let info = dtrsl(wt, 0, m, col, &mut p[col..], 1);
    if info != 0 {
        return info;
    }
    for i in 1..=col {
        p[i] = -p[i] / sy[q(i, i)].sqrt();
    }
    for i in 1..=col {
        let mut sum = 0.0;
        for k in (i + 1)..=col {
            sum += sy[q(k, i)] * p[col + k] / sy[q(i, i)];
        }
        p[i] += sum;
    }
    0
}

/// `hpsolb`: heap sort step on the breakpoints `t` (1-based) with their indices.
fn hpsolb(n: usize, t: &mut [f64], iorder: &mut [usize], iheap: usize) {
    if iheap == 0 {
        for k in 2..=n {
            let ddum = t[k];
            let indxin = iorder[k];
            let mut i = k;
            while i > 1 {
                let j = i / 2;
                if ddum < t[j] {
                    t[i] = t[j];
                    iorder[i] = iorder[j];
                    i = j;
                } else {
                    break;
                }
            }
            t[i] = ddum;
            iorder[i] = indxin;
        }
    }
    if n > 1 {
        let mut i = 1;
        let out = t[1];
        let indxou = iorder[1];
        let ddum = t[n];
        let indxin = iorder[n];
        loop {
            let mut j = i + i;
            if j < n {
                if t[j + 1] < t[j] {
                    j += 1;
                }
                if t[j] < ddum {
                    t[i] = t[j];
                    iorder[i] = iorder[j];
                    i = j;
                    continue;
                }
            }
            break;
        }
        t[i] = ddum;
        iorder[i] = indxin;
        t[n] = out;
        iorder[n] = indxou;
    }
}

/// `dcstep`: one safeguarded step of the More-Thuente line search.
#[allow(clippy::too_many_arguments)]
fn dcstep(
    stx: &mut f64,
    fx: &mut f64,
    dx: &mut f64,
    sty: &mut f64,
    fy: &mut f64,
    dy: &mut f64,
    stp: &mut f64,
    fp: f64,
    dp: f64,
    brackt: &mut bool,
    stpmin: f64,
    stpmax: f64,
) {
    let sgnd = dp * (*dx / dx.abs());
    let stpf;
    if fp > *fx {
        let theta = (*fx - fp) * 3. / (*stp - *stx) + *dx + dp;
        let s = cmax(cmax(theta.abs(), dx.abs()), dp.abs());
        let d1 = theta / s;
        let mut gamm = s * (d1 * d1 - *dx / s * (dp / s)).sqrt();
        if *stp < *stx {
            gamm = -gamm;
        }
        let p = gamm - *dx + theta;
        let q = gamm - *dx + gamm + dp;
        let r = p / q;
        let stpc = *stx + r * (*stp - *stx);
        let stpq = *stx + *dx / ((*fx - fp) / (*stp - *stx) + *dx) / 2. * (*stp - *stx);
        if (stpc - *stx).abs() < (stpq - *stx).abs() {
            stpf = stpc;
        } else {
            stpf = stpc + (stpq - stpc) / 2.;
        }
        *brackt = true;
    } else if sgnd < 0. {
        let theta = (*fx - fp) * 3. / (*stp - *stx) + *dx + dp;
        let s = cmax(cmax(theta.abs(), dx.abs()), dp.abs());
        let d1 = theta / s;
        let mut gamm = s * (d1 * d1 - *dx / s * (dp / s)).sqrt();
        if *stp > *stx {
            gamm = -gamm;
        }
        let p = gamm - dp + theta;
        let q = gamm - dp + gamm + *dx;
        let r = p / q;
        let stpc = *stp + r * (*stx - *stp);
        let stpq = *stp + dp / (dp - *dx) * (*stx - *stp);
        if (stpc - *stp).abs() > (stpq - *stp).abs() {
            stpf = stpc;
        } else {
            stpf = stpq;
        }
        *brackt = true;
    } else if dp.abs() < dx.abs() {
        let theta = (*fx - fp) * 3. / (*stp - *stx) + *dx + dp;
        let s = cmax(cmax(theta.abs(), dx.abs()), dp.abs());
        let mut d1 = theta / s;
        d1 = d1 * d1 - *dx / s * (dp / s);
        let mut gamm = if d1 < 0. { 0. } else { s * d1.sqrt() };
        if *stp > *stx {
            gamm = -gamm;
        }
        let p = gamm - dp + theta;
        let q = gamm + (*dx - dp) + gamm;
        let r = p / q;
        let stpc = if r < 0. && gamm != 0. {
            *stp + r * (*stx - *stp)
        } else if *stp > *stx {
            stpmax
        } else {
            stpmin
        };
        let stpq = *stp + dp / (dp - *dx) * (*stx - *stp);
        if *brackt {
            let mut f = if (stpc - *stp).abs() < (stpq - *stp).abs() {
                stpc
            } else {
                stpq
            };
            let d1 = *stp + (*sty - *stp) * 0.66;
            if *stp > *stx {
                f = cmin(d1, f);
            } else {
                f = cmax(d1, f);
            }
            stpf = f;
        } else {
            let mut f = if (stpc - *stp).abs() > (stpq - *stp).abs() {
                stpc
            } else {
                stpq
            };
            f = cmin(stpmax, f);
            f = cmax(stpmin, f);
            stpf = f;
        }
    } else if *brackt {
        let theta = (fp - *fy) * 3. / (*sty - *stp) + *dy + dp;
        let s = cmax(cmax(theta.abs(), dy.abs()), dp.abs());
        let d1 = theta / s;
        let mut gamm = s * (d1 * d1 - *dy / s * (dp / s)).sqrt();
        if *stp > *sty {
            gamm = -gamm;
        }
        let p = gamm - dp + theta;
        let q = gamm - dp + gamm + *dy;
        let r = p / q;
        stpf = *stp + r * (*sty - *stp);
    } else if *stp > *stx {
        stpf = stpmax;
    } else {
        stpf = stpmin;
    }
    if fp > *fx {
        *sty = *stp;
        *fy = fp;
        *dy = dp;
    } else {
        if sgnd < 0. {
            *sty = *stx;
            *fy = *fx;
            *dy = *dx;
        }
        *stx = *stp;
        *fx = fp;
        *dx = dp;
    }
    *stp = stpf;
}

/// The static locals of `dcsrch`.
#[derive(Default)]
struct Dcsrch {
    stage: i32,
    brackt: bool,
    ginit: f64,
    gtest: f64,
    gx: f64,
    gy: f64,
    finit: f64,
    fx: f64,
    fy: f64,
    stx: f64,
    sty: f64,
    stmin: f64,
    stmax: f64,
    width: f64,
    width1: f64,
}

impl Dcsrch {
    /// `dcsrch`: the More-Thuente line search driver (`task` is `csave`).
    #[allow(clippy::too_many_arguments)]
    // The restart test mirrors dcsrch's C expression term for term.
    #[allow(clippy::nonminimal_bool)]
    fn run(
        &mut self,
        f: f64,
        g: f64,
        stp: &mut f64,
        ftol: f64,
        gtol: f64,
        xtol: f64,
        stpmin: f64,
        stpmax: f64,
        task: &mut String,
    ) {
        if task.starts_with("START") {
            if *stp < stpmin {
                *task = "ERROR: STP .LT. STPMIN".into();
            }
            if *stp > stpmax {
                *task = "ERROR: STP .GT. STPMAX".into();
            }
            if g >= 0. {
                *task = "ERROR: INITIAL G .GE. ZERO".into();
            }
            if ftol < 0. {
                *task = "ERROR: FTOL .LT. ZERO".into();
            }
            if gtol < 0. {
                *task = "ERROR: GTOL .LT. ZERO".into();
            }
            if xtol < 0. {
                *task = "ERROR: XTOL .LT. ZERO".into();
            }
            if stpmin < 0. {
                *task = "ERROR: STPMIN .LT. ZERO".into();
            }
            if stpmax < stpmin {
                *task = "ERROR: STPMAX .LT. STPMIN".into();
            }
            if task.starts_with("ERROR") {
                return;
            }
            self.brackt = false;
            self.stage = 1;
            self.finit = f;
            self.ginit = g;
            self.gtest = ftol * self.ginit;
            self.width = stpmax - stpmin;
            self.width1 = self.width / 0.5;
            self.stx = 0.;
            self.fx = self.finit;
            self.gx = self.ginit;
            self.sty = 0.;
            self.fy = self.finit;
            self.gy = self.ginit;
            self.stmin = 0.;
            self.stmax = *stp + *stp * 4.;
            *task = "FG".into();
            return;
        }
        let ftest = self.finit + *stp * self.gtest;
        if self.stage == 1 && f <= ftest && g >= 0. {
            self.stage = 2;
        }
        if self.brackt && (*stp <= self.stmin || *stp >= self.stmax) {
            *task = "WARNING: ROUNDING ERRORS PREVENT PROGRESS".into();
        }
        if self.brackt && self.stmax - self.stmin <= xtol * self.stmax {
            *task = "WARNING: XTOL TEST SATISFIED".into();
        }
        if *stp == stpmax && f <= ftest && g <= self.gtest {
            *task = "WARNING: STP = STPMAX".into();
        }
        if *stp == stpmin && (f > ftest || g >= self.gtest) {
            *task = "WARNING: STP = STPMIN".into();
        }
        if f <= ftest && g.abs() <= gtol * (-self.ginit) {
            *task = "CONVERGENCE".into();
        }
        if task.starts_with("WARN") || task.starts_with("CONV") {
            return;
        }
        if self.stage == 1 && f <= self.fx && f > ftest {
            let fm = f - *stp * self.gtest;
            let mut fxm = self.fx - self.stx * self.gtest;
            let mut fym = self.fy - self.sty * self.gtest;
            let gm = g - self.gtest;
            let mut gxm = self.gx - self.gtest;
            let mut gym = self.gy - self.gtest;
            dcstep(
                &mut self.stx,
                &mut fxm,
                &mut gxm,
                &mut self.sty,
                &mut fym,
                &mut gym,
                stp,
                fm,
                gm,
                &mut self.brackt,
                self.stmin,
                self.stmax,
            );
            self.fx = fxm + self.stx * self.gtest;
            self.fy = fym + self.sty * self.gtest;
            self.gx = gxm + self.gtest;
            self.gy = gym + self.gtest;
        } else {
            dcstep(
                &mut self.stx,
                &mut self.fx,
                &mut self.gx,
                &mut self.sty,
                &mut self.fy,
                &mut self.gy,
                stp,
                f,
                g,
                &mut self.brackt,
                self.stmin,
                self.stmax,
            );
        }
        if self.brackt {
            if (self.sty - self.stx).abs() >= self.width1 * 0.66 {
                *stp = self.stx + (self.sty - self.stx) * 0.5;
            }
            self.width1 = self.width;
            self.width = (self.sty - self.stx).abs();
        }
        if self.brackt {
            self.stmin = cmin(self.stx, self.sty);
            self.stmax = cmax(self.stx, self.sty);
        } else {
            self.stmin = *stp + (*stp - self.stx) * 1.1;
            self.stmax = *stp + (*stp - self.stx) * 4.;
        }
        if *stp < stpmin {
            *stp = stpmin;
        }
        if *stp > stpmax {
            *stp = stpmax;
        }
        if (self.brackt && (*stp <= self.stmin || *stp >= self.stmax))
            || (self.brackt && (self.stmax - self.stmin <= xtol * self.stmax))
        {
            *stp = self.stx;
        }
        *task = "FG".into();
    }
}

/// `active`: project the start point onto the box and set `iwhere`. Returns
/// `(prjctd, cnstnd, boxed)`.
fn active(
    n: usize,
    l: &[f64],
    u: &[f64],
    nbd: &[i32],
    x: &mut [f64],
    iwhere: &mut [i32],
) -> (bool, bool, bool) {
    let mut prjctd = false;
    let mut cnstnd = false;
    let mut boxed = true;
    for i in 1..=n {
        if nbd[i] > 0 {
            if nbd[i] <= 2 && x[i] <= l[i] {
                if x[i] < l[i] {
                    prjctd = true;
                    x[i] = l[i];
                }
            } else if nbd[i] >= 2 && x[i] >= u[i] && x[i] > u[i] {
                prjctd = true;
                x[i] = u[i];
            }
        }
    }
    for i in 1..=n {
        if nbd[i] != 2 {
            boxed = false;
        }
        if nbd[i] == 0 {
            iwhere[i] = -1;
        } else {
            cnstnd = true;
            if nbd[i] == 2 && u[i] - l[i] <= 0. {
                iwhere[i] = 3;
            } else {
                iwhere[i] = 0;
            }
        }
    }
    (prjctd, cnstnd, boxed)
}

/// `projgr`: the infinity norm of the projected gradient.
fn projgr(n: usize, l: &[f64], u: &[f64], nbd: &[i32], x: &[f64], g: &[f64]) -> f64 {
    let mut sbgnrm = 0.;
    for i in 1..=n {
        let mut gi = g[i];
        if nbd[i] != 0 {
            if gi < 0. {
                if nbd[i] >= 2 {
                    let d1 = x[i] - u[i];
                    if gi < d1 {
                        gi = d1;
                    }
                }
            } else if nbd[i] <= 2 {
                let d1 = x[i] - l[i];
                if gi > d1 {
                    gi = d1;
                }
            }
        }
        let d1 = gi.abs();
        if sbgnrm < d1 {
            sbgnrm = d1;
        }
    }
    sbgnrm
}

/// `cauchy`: the generalized Cauchy point. Returns `info`.
#[allow(clippy::too_many_arguments)]
fn cauchy(
    n: usize,
    x: &[f64],
    l: &[f64],
    u: &[f64],
    nbd: &[i32],
    g: &[f64],
    iorder: &mut [usize],
    iwhere: &mut [i32],
    t: &mut [f64],
    d: &mut [f64],
    xcp: &mut [f64],
    m: usize,
    wy: &[f64],
    ws: &[f64],
    sy: &[f64],
    wt: &[f64],
    theta: f64,
    col: usize,
    head: usize,
    p: &mut [f64],
    c: &mut [f64],
    wbp: &mut [f64],
    v: &mut [f64],
    nint: &mut i64,
    sbgnrm: f64,
    epsmch: f64,
) -> i32 {
    let s = move |i: usize, j: usize| i + j * n;
    if sbgnrm <= 0. {
        xcp[1..=n].copy_from_slice(&x[1..=n]);
        return 0;
    }
    let mut bnded = true;
    let mut nfree = n + 1;
    let mut nbreak = 0usize;
    let mut ibkmin = 0usize;
    let mut bkmin = 0.;
    let col2 = col * 2;
    let mut f1 = 0.;
    let mut tl = 0.;
    let mut tu = 0.;
    for i in 1..=col2 {
        p[i] = 0.;
    }
    for i in 1..=n {
        let neggi = -g[i];
        if iwhere[i] != 3 && iwhere[i] != -1 {
            if nbd[i] <= 2 {
                tl = x[i] - l[i];
            }
            if nbd[i] >= 2 {
                tu = u[i] - x[i];
            }
            let xlower = nbd[i] <= 2 && tl <= 0.;
            let xupper = nbd[i] >= 2 && tu <= 0.;
            iwhere[i] = 0;
            if xlower {
                if neggi <= 0. {
                    iwhere[i] = 1;
                }
            } else if xupper {
                if neggi >= 0. {
                    iwhere[i] = 2;
                }
            } else if neggi.abs() <= 0. {
                iwhere[i] = -3;
            }
        }
        let mut pointr = head;
        if iwhere[i] != 0 && iwhere[i] != -1 {
            d[i] = 0.;
        } else {
            d[i] = neggi;
            f1 -= neggi * neggi;
            for j in 1..=col {
                p[j] += wy[s(i, pointr)] * neggi;
                p[col + j] += ws[s(i, pointr)] * neggi;
                pointr = pointr % m + 1;
            }
            if nbd[i] <= 2 && nbd[i] != 0 && neggi < 0. {
                nbreak += 1;
                iorder[nbreak] = i;
                t[nbreak] = tl / (-neggi);
                if nbreak == 1 || t[nbreak] < bkmin {
                    bkmin = t[nbreak];
                    ibkmin = nbreak;
                }
            } else if nbd[i] >= 2 && neggi > 0. {
                nbreak += 1;
                iorder[nbreak] = i;
                t[nbreak] = tu / neggi;
                if nbreak == 1 || t[nbreak] < bkmin {
                    bkmin = t[nbreak];
                    ibkmin = nbreak;
                }
            } else {
                nfree -= 1;
                iorder[nfree] = i;
                if neggi.abs() > 0. {
                    bnded = false;
                }
            }
        }
    }
    if theta != 1. {
        for j in 1..=col {
            p[col + j] *= theta;
        }
    }
    xcp[1..=n].copy_from_slice(&x[1..=n]);
    if nbreak == 0 && nfree == n + 1 {
        return 0;
    }
    for j in 1..=col2 {
        c[j] = 0.;
    }
    let mut f2 = -theta * f1;
    let f2_org = f2;
    if col > 0 {
        let info = bmv(m, sy, wt, col, p, v);
        if info != 0 {
            return info;
        }
        f2 -= ddot(col2, v, p);
    }
    let mut dtm = -f1 / f2;
    let mut tsum = 0.;
    *nint = 1;
    // `true` when the loop ends at L999 (all variables fixed), skipping L888.
    let mut at999 = false;
    if nbreak > 0 {
        let mut nleft = nbreak;
        let mut iter = 1usize;
        let mut tj = 0.;
        loop {
            let tj0 = tj;
            let ibp;
            if iter == 1 {
                tj = bkmin;
                ibp = iorder[ibkmin];
            } else {
                if iter == 2 && ibkmin != nbreak {
                    t[ibkmin] = t[nbreak];
                    iorder[ibkmin] = iorder[nbreak];
                }
                hpsolb(nleft, t, iorder, iter - 2);
                tj = t[nleft];
                ibp = iorder[nleft];
            }
            let dt = tj - tj0;
            if dtm < dt {
                break;
            }
            tsum += dt;
            nleft -= 1;
            iter += 1;
            let dibp = d[ibp];
            d[ibp] = 0.;
            let zibp;
            if dibp > 0. {
                zibp = u[ibp] - x[ibp];
                xcp[ibp] = u[ibp];
                iwhere[ibp] = 2;
            } else {
                zibp = l[ibp] - x[ibp];
                xcp[ibp] = l[ibp];
                iwhere[ibp] = 1;
            }
            if nleft == 0 && nbreak == n {
                dtm = dt;
                at999 = true;
                break;
            }
            *nint += 1;
            let dibp2 = dibp * dibp;
            f1 += dt * f2 + dibp2 - theta * dibp * zibp;
            f2 -= theta * dibp2;
            if col > 0 {
                daxpy(col2, dt, p, c);
                let mut pointr = head;
                for j in 1..=col {
                    wbp[j] = wy[s(ibp, pointr)];
                    wbp[col + j] = theta * ws[s(ibp, pointr)];
                    pointr = pointr % m + 1;
                }
                let info = bmv(m, sy, wt, col, wbp, v);
                if info != 0 {
                    return info;
                }
                let wmc = ddot(col2, c, v);
                let wmp = ddot(col2, p, v);
                let wmw = ddot(col2, wbp, v);
                daxpy(col2, -dibp, wbp, p);
                f1 += dibp * wmc;
                f2 += 2. * dibp * wmp - dibp2 * wmw;
            }
            let d1 = epsmch * f2_org;
            if f2 < d1 {
                f2 = d1;
            }
            if nleft > 0 {
                dtm = -f1 / f2;
                continue;
            } else if bnded {
                f1 = 0.;
                f2 = 0.;
                dtm = 0.;
            } else {
                dtm = -f1 / f2;
            }
            break;
        }
    }
    if !at999 {
        if dtm <= 0. {
            dtm = 0.;
        }
        tsum += dtm;
        daxpy(n, tsum, d, xcp);
    }
    if col > 0 {
        daxpy(col2, dtm, p, c);
    }
    let _ = (f1, f2);
    0
}

/// `cmprlb`: the reduced gradient `r` of the quadratic model at the Cauchy point. `wa_p` is
/// `wa[1..2m]` (output) and `wa_c` is `wa[2m+1..4m]` (the `c` from `cauchy`). Returns `info`.
#[allow(clippy::too_many_arguments)]
fn cmprlb(
    n: usize,
    m: usize,
    x: &[f64],
    g: &[f64],
    ws: &[f64],
    wy: &[f64],
    sy: &[f64],
    wt: &[f64],
    z: &[f64],
    r: &mut [f64],
    wa_p: &mut [f64],
    wa_c: &[f64],
    indx: &[usize],
    theta: f64,
    col: usize,
    head: usize,
    nfree: usize,
    cnstnd: bool,
) -> i32 {
    let s = move |i: usize, j: usize| i + j * n;
    if !cnstnd && col > 0 {
        for i in 1..=n {
            r[i] = -g[i];
        }
    } else {
        for i in 1..=nfree {
            let k = indx[i];
            r[i] = -theta * (z[k] - x[k]) - g[k];
        }
        let info = bmv(m, sy, wt, col, wa_c, wa_p);
        if info != 0 {
            return -8;
        }
        let mut pointr = head;
        for j in 1..=col {
            let a1 = wa_p[j];
            let a2 = theta * wa_p[col + j];
            for i in 1..=nfree {
                let k = indx[i];
                r[i] += wy[s(k, pointr)] * a1 + ws[s(k, pointr)] * a2;
            }
            pointr = pointr % m + 1;
        }
    }
    0
}

/// `errclb`: input checks. Returns `(info, k)` and may set `task` to an error.
fn errclb(
    n: usize,
    m: usize,
    factr: f64,
    l: &[f64],
    u: &[f64],
    nbd: &[i32],
    task: &mut String,
) -> (i32, usize) {
    let mut info = 0;
    let mut k = 0;
    if n == 0 {
        *task = "ERROR: N .LE. 0".into();
    }
    if m == 0 {
        *task = "ERROR: M .LE. 0".into();
    }
    if factr < 0. {
        *task = "ERROR: FACTR .LT. 0".into();
    }
    for i in 1..=n {
        if nbd[i] < 0 || nbd[i] > 3 {
            *task = "ERROR: INVALID NBD".into();
            info = -6;
            k = i;
        }
        if nbd[i] == 2 && l[i] > u[i] {
            *task = "ERROR: NO FEASIBLE SOLUTION".into();
            info = -7;
            k = i;
        }
    }
    (info, k)
}

/// `formk`: the LEL^T factorization of the 2col x 2col middle matrix in `wn` (`wn1` is `snd`).
/// Returns `info`.
#[allow(clippy::too_many_arguments)]
fn formk(
    n: usize,
    nsub: usize,
    ind: &[usize],
    nenter: usize,
    ileave: usize,
    indx2: &[usize],
    iupdat: usize,
    updatd: bool,
    wn: &mut [f64],
    wn1: &mut [f64],
    m: usize,
    ws: &[f64],
    wy: &[f64],
    sy: &[f64],
    theta: f64,
    col: usize,
    head: usize,
) -> i32 {
    let m2 = 2 * m;
    let w = move |i: usize, j: usize| i + j * m2;
    let s = move |i: usize, j: usize| i + j * n;
    let q = move |i: usize, j: usize| i + j * m;
    let upcl;
    if updatd {
        if iupdat > m {
            for jy in 1..m {
                let js = m + jy;
                for k in 0..(m - jy) {
                    wn1[w(jy + k, jy)] = wn1[w(jy + 1 + k, jy + 1)];
                }
                for k in 0..(m - jy) {
                    wn1[w(js + k, js)] = wn1[w(js + 1 + k, js + 1)];
                }
                for k in 0..(m - 1) {
                    wn1[w(m + 1 + k, jy)] = wn1[w(m + 2 + k, jy + 1)];
                }
            }
        }
        let (pbegin, pend, dbegin, dend) = (1, nsub, nsub + 1, n);
        let iy = col;
        let is = m + col;
        let mut ipntr = head + col - 1;
        if ipntr > m {
            ipntr -= m;
        }
        let mut jpntr = head;
        for jy in 1..=col {
            let js = m + jy;
            let mut temp1 = 0.;
            let mut temp2 = 0.;
            let mut temp3 = 0.;
            for k in pbegin..=pend {
                let k1 = ind[k];
                temp1 += wy[s(k1, ipntr)] * wy[s(k1, jpntr)];
            }
            for k in dbegin..=dend {
                let k1 = ind[k];
                temp2 += ws[s(k1, ipntr)] * ws[s(k1, jpntr)];
                temp3 += ws[s(k1, ipntr)] * wy[s(k1, jpntr)];
            }
            wn1[w(iy, jy)] = temp1;
            wn1[w(is, js)] = temp2;
            wn1[w(is, jy)] = temp3;
            jpntr = jpntr % m + 1;
        }
        let jy = col;
        let mut jpntr = head + col - 1;
        if jpntr > m {
            jpntr -= m;
        }
        let mut ipntr = head;
        for i in 1..=col {
            let is = m + i;
            let mut temp3 = 0.;
            for k in pbegin..=pend {
                let k1 = ind[k];
                temp3 += ws[s(k1, ipntr)] * wy[s(k1, jpntr)];
            }
            ipntr = ipntr % m + 1;
            wn1[w(is, jy)] = temp3;
        }
        upcl = col - 1;
    } else {
        upcl = col;
    }
    let mut ipntr = head;
    for iy in 1..=upcl {
        let is = m + iy;
        let mut jpntr = head;
        for jy in 1..=iy {
            let js = m + jy;
            let mut temp1 = 0.;
            let mut temp2 = 0.;
            let mut temp3 = 0.;
            let mut temp4 = 0.;
            for k in 1..=nenter {
                let k1 = indx2[k];
                temp1 += wy[s(k1, ipntr)] * wy[s(k1, jpntr)];
                temp2 += ws[s(k1, ipntr)] * ws[s(k1, jpntr)];
            }
            for k in ileave..=n {
                let k1 = indx2[k];
                temp3 += wy[s(k1, ipntr)] * wy[s(k1, jpntr)];
                temp4 += ws[s(k1, ipntr)] * ws[s(k1, jpntr)];
            }
            wn1[w(iy, jy)] = wn1[w(iy, jy)] + temp1 - temp3;
            wn1[w(is, js)] = wn1[w(is, js)] - temp2 + temp4;
            jpntr = jpntr % m + 1;
        }
        ipntr = ipntr % m + 1;
    }
    let mut ipntr = head;
    for is in (m + 1)..=(m + upcl) {
        let mut jpntr = head;
        for jy in 1..=upcl {
            let mut temp1 = 0.;
            let mut temp3 = 0.;
            for k in 1..=nenter {
                let k1 = indx2[k];
                temp1 += ws[s(k1, ipntr)] * wy[s(k1, jpntr)];
            }
            for k in ileave..=n {
                let k1 = indx2[k];
                temp3 += ws[s(k1, ipntr)] * wy[s(k1, jpntr)];
            }
            if is <= jy + m {
                wn1[w(is, jy)] += temp1 - temp3;
            } else {
                wn1[w(is, jy)] += -temp1 + temp3;
            }
            jpntr = jpntr % m + 1;
        }
        ipntr = ipntr % m + 1;
    }
    for iy in 1..=col {
        let is = col + iy;
        let is1 = m + iy;
        for jy in 1..=iy {
            let js = col + jy;
            let js1 = m + jy;
            wn[w(jy, iy)] = wn1[w(iy, jy)] / theta;
            wn[w(js, is)] = wn1[w(is1, js1)] * theta;
        }
        for jy in 1..iy {
            wn[w(jy, is)] = -wn1[w(is1, jy)];
        }
        for jy in iy..=col {
            wn[w(jy, is)] = wn1[w(is1, jy)];
        }
        wn[w(iy, iy)] += sy[q(iy, iy)];
    }
    if dpofa(wn, 0, m2, col) != 0 {
        return -1;
    }
    let col2 = col * 2;
    let mut b = vec![0.0; col + 1];
    for js in (col + 1)..=col2 {
        for i in 1..=col {
            b[i] = wn[w(i, js)];
        }
        // R ignores this dtrsl's info (the dpofa below overwrites it).
        let _ = dtrsl(wn, 0, m2, col, &mut b, 11);
        for i in 1..=col {
            wn[w(i, js)] = b[i];
        }
    }
    for is in (col + 1)..=col2 {
        for js in is..=col2 {
            let mut dot = 0.;
            for i in 1..=col {
                dot += wn[w(i, is)] * wn[w(i, js)];
            }
            wn[w(is, js)] += dot;
        }
    }
    if dpofa(wn, col + col * m2, m2, col) != 0 {
        return -2;
    }
    0
}

/// `formt`: the Cholesky factor of `T = theta*S'S + L D^-1 L'` in `wt`. Returns `info`.
fn formt(m: usize, wt: &mut [f64], sy: &[f64], ss: &[f64], col: usize, theta: f64) -> i32 {
    let q = move |i: usize, j: usize| i + j * m;
    for j in 1..=col {
        wt[q(1, j)] = theta * ss[q(1, j)];
    }
    for i in 2..=col {
        for j in i..=col {
            let k1 = i.min(j) - 1;
            let mut ddum = 0.;
            for k in 1..=k1 {
                ddum += sy[q(i, k)] * sy[q(j, k)] / sy[q(k, k)];
            }
            wt[q(i, j)] = ddum + theta * ss[q(i, j)];
        }
    }
    if dpofa(wt, 0, m, col) != 0 {
        return -3;
    }
    0
}

/// `freev`: the entering and leaving free variables. Returns `wrk`.
#[allow(clippy::too_many_arguments)]
fn freev(
    n: usize,
    nfree: &mut usize,
    indx: &mut [usize],
    nenter: &mut usize,
    ileave: &mut usize,
    indx2: &mut [usize],
    iwhere: &[i32],
    updatd: bool,
    cnstnd: bool,
    iter: i64,
) -> bool {
    *nenter = 0;
    *ileave = n + 1;
    if iter > 0 && cnstnd {
        for i in 1..=*nfree {
            let k = indx[i];
            if iwhere[k] > 0 {
                *ileave -= 1;
                indx2[*ileave] = k;
            }
        }
        for i in (*nfree + 1)..=n {
            let k = indx[i];
            if iwhere[k] <= 0 {
                *nenter += 1;
                indx2[*nenter] = k;
            }
        }
    }
    let wrk = *ileave < n + 1 || *nenter > 0 || updatd;
    *nfree = 0;
    let mut iact = n + 1;
    for i in 1..=n {
        if iwhere[i] <= 0 {
            *nfree += 1;
            indx[*nfree] = i;
        } else {
            iact -= 1;
            indx[iact] = i;
        }
    }
    wrk
}

/// `matupd`: update the limited-memory matrices `WS`, `WY`, `SY` and `SS`.
#[allow(clippy::too_many_arguments)]
fn matupd(
    n: usize,
    m: usize,
    ws: &mut [f64],
    wy: &mut [f64],
    sy: &mut [f64],
    ss: &mut [f64],
    d: &[f64],
    r: &[f64],
    itail: &mut usize,
    iupdat: usize,
    col: &mut usize,
    head: &mut usize,
    theta: &mut f64,
    rr: f64,
    dr: f64,
    stp: f64,
    dtd: f64,
) {
    let s = move |i: usize, j: usize| i + j * n;
    let q = move |i: usize, j: usize| i + j * m;
    if iupdat <= m {
        *col = iupdat;
        *itail = (*head + iupdat - 2) % m + 1;
    } else {
        *itail = *itail % m + 1;
        *head = *head % m + 1;
    }
    for i in 1..=n {
        ws[s(i, *itail)] = d[i];
        wy[s(i, *itail)] = r[i];
    }
    *theta = rr / dr;
    if iupdat > m {
        for j in 1..*col {
            for k in 0..j {
                ss[q(1 + k, j)] = ss[q(2 + k, j + 1)];
            }
            for k in 0..(*col - j) {
                sy[q(j + k, j)] = sy[q(j + 1 + k, j + 1)];
            }
        }
    }
    let mut pointr = *head;
    for j in 1..*col {
        let mut a = 0.;
        let mut b = 0.;
        for i in 1..=n {
            a += d[i] * wy[s(i, pointr)];
            b += ws[s(i, pointr)] * d[i];
        }
        sy[q(*col, j)] = a;
        ss[q(j, *col)] = b;
        pointr = pointr % m + 1;
    }
    if stp == 1. {
        ss[q(*col, *col)] = dtd;
    } else {
        ss[q(*col, *col)] = stp * stp * dtd;
    }
    sy[q(*col, *col)] = dr;
}

/// `subsm`: subspace minimization over the free variables. `x` is `z` and `d` is `r` in the
/// caller; `wv` is `wa[1..2m]`. Returns `info`.
#[allow(clippy::too_many_arguments)]
fn subsm(
    n: usize,
    m: usize,
    nsub: usize,
    ind: &[usize],
    l: &[f64],
    u: &[f64],
    nbd: &[i32],
    x: &mut [f64],
    d: &mut [f64],
    ws: &[f64],
    wy: &[f64],
    theta: f64,
    col: usize,
    head: usize,
    iword: &mut i32,
    wv: &mut [f64],
    wn: &[f64],
) -> i32 {
    let s = move |i: usize, j: usize| i + j * n;
    let ns = nsub;
    if ns == 0 {
        return 0;
    }
    let mut pointr = head;
    for i in 1..=col {
        let mut temp1 = 0.;
        let mut temp2 = 0.;
        for j in 1..=ns {
            let k = ind[j];
            temp1 += wy[s(k, pointr)] * d[j];
            temp2 += ws[s(k, pointr)] * d[j];
        }
        wv[i] = temp1;
        wv[col + i] = theta * temp2;
        pointr = pointr % m + 1;
    }
    let m2 = 2 * m;
    let col2 = 2 * col;
    let info = dtrsl(wn, 0, m2, col2, wv, 11);
    if info != 0 {
        return info;
    }
    for i in 1..=col {
        wv[i] = -wv[i];
    }
    let info = dtrsl(wn, 0, m2, col2, wv, 1);
    if info != 0 {
        return info;
    }
    let mut pointr = head;
    for jy in 1..=col {
        let js = col + jy;
        for i in 1..=ns {
            let k = ind[i];
            d[i] += wy[s(k, pointr)] * wv[jy] / theta + ws[s(k, pointr)] * wv[js];
        }
        pointr = pointr % m + 1;
    }
    for i in 1..=ns {
        d[i] /= theta;
    }
    let mut alpha = 1.;
    let mut temp1 = alpha;
    let mut ibd = 0;
    for i in 1..=ns {
        let k = ind[i];
        let dk = d[i];
        if nbd[k] != 0 {
            if dk < 0. && nbd[k] <= 2 {
                let temp2 = l[k] - x[k];
                if temp2 >= 0. {
                    temp1 = 0.;
                } else if dk * alpha < temp2 {
                    temp1 = temp2 / dk;
                }
            } else if dk > 0. && nbd[k] >= 2 {
                let temp2 = u[k] - x[k];
                if temp2 <= 0. {
                    temp1 = 0.;
                } else if dk * alpha > temp2 {
                    temp1 = temp2 / dk;
                }
            }
            if temp1 < alpha {
                alpha = temp1;
                ibd = i;
            }
        }
    }
    if alpha < 1. {
        let dk = d[ibd];
        let k = ind[ibd];
        if dk > 0. {
            x[k] = u[k];
            d[ibd] = 0.;
        } else if dk < 0. {
            x[k] = l[k];
            d[ibd] = 0.;
        }
    }
    for i in 1..=ns {
        x[ind[i]] += alpha * d[i];
    }
    *iword = if alpha < 1. { 1 } else { 0 };
    0
}

/// Workspace and the static locals of `mainlb` / `dcsrch`, carried across `setulb` calls.
struct Lbfgsb {
    n: usize,
    m: usize,
    ws: Vec<f64>,
    wy: Vec<f64>,
    sy: Vec<f64>,
    ss: Vec<f64>,
    wt: Vec<f64>,
    wn: Vec<f64>,
    snd: Vec<f64>,
    z: Vec<f64>,
    r: Vec<f64>,
    d: Vec<f64>,
    t: Vec<f64>,
    // `wa` split into its four 2m blocks: p, c, wbp, v.
    wa_p: Vec<f64>,
    wa_c: Vec<f64>,
    wa_wbp: Vec<f64>,
    wa_v: Vec<f64>,
    indx: Vec<usize>,
    iwhere: Vec<i32>,
    indx2: Vec<usize>,
    cnstnd: bool,
    boxed: bool,
    updatd: bool,
    nintol: i64,
    iback: i64,
    nskip: i64,
    head: usize,
    col: usize,
    itail: usize,
    iter: i64,
    iupdat: usize,
    nint: i64,
    nfgv: i64,
    info: i32,
    ifun: i64,
    iword: i32,
    nfree: usize,
    nact: usize,
    ileave: usize,
    nenter: usize,
    theta: f64,
    fold: f64,
    tol: f64,
    dnorm: f64,
    epsmch: f64,
    gd: f64,
    stpmx: f64,
    sbgnrm: f64,
    stp: f64,
    gdold: f64,
    dtd: f64,
    /// `isave[12]` (0-based), the function evaluation count `optim` reports.
    isave12: i64,
    ls: Dcsrch,
}

/// Where `mainlb` resumes inside its `goto` graph.
#[derive(Clone, Copy)]
enum Label {
    L111,
    L222,
    L333,
    L555,
    L666,
    L777,
}

impl Lbfgsb {
    fn new(n: usize, m: usize) -> Lbfgsb {
        Lbfgsb {
            n,
            m,
            ws: mat(n, m),
            wy: mat(n, m),
            sy: mat(m, m),
            ss: mat(m, m),
            wt: mat(m, m),
            wn: mat(2 * m, 2 * m),
            snd: mat(2 * m, 2 * m),
            z: vec![0.0; n + 1],
            r: vec![0.0; n + 1],
            d: vec![0.0; n + 1],
            t: vec![0.0; n + 1],
            wa_p: vec![0.0; 2 * m + 1],
            wa_c: vec![0.0; 2 * m + 1],
            wa_wbp: vec![0.0; 2 * m + 1],
            wa_v: vec![0.0; 2 * m + 1],
            indx: vec![0; n + 1],
            iwhere: vec![0; n + 1],
            indx2: vec![0; n + 1],
            cnstnd: false,
            boxed: false,
            updatd: false,
            nintol: 0,
            iback: 0,
            nskip: 0,
            head: 1,
            col: 0,
            itail: 0,
            iter: 0,
            iupdat: 0,
            nint: 0,
            nfgv: 0,
            info: 0,
            ifun: 0,
            iword: 0,
            nfree: n,
            nact: 0,
            ileave: 0,
            nenter: 0,
            theta: 1.0,
            fold: 0.0,
            tol: 0.0,
            dnorm: 0.0,
            epsmch: 0.0,
            gd: 0.0,
            stpmx: 0.0,
            sbgnrm: 0.0,
            stp: 0.0,
            gdold: 0.0,
            dtd: 0.0,
            isave12: 0,
            ls: Dcsrch::default(),
        }
    }

    /// The "refresh the lbfgs memory and restart the iteration" reset.
    fn refresh(&mut self) {
        self.info = 0;
        self.col = 0;
        self.head = 1;
        self.theta = 1.;
        self.iupdat = 0;
        self.updatd = false;
    }

    /// `lnsrlb`: the line search, with `csave` as the line-search task.
    #[allow(clippy::too_many_arguments)]
    fn lnsrlb(
        &mut self,
        l: &[f64],
        u: &[f64],
        nbd: &[i32],
        x: &mut [f64],
        f: f64,
        g: &[f64],
        task: &mut String,
        csave: &mut String,
    ) {
        let n = self.n;
        let (ftol, gtol, xtol, stpmin) = (0.001, 0.9, 0.1, 0.);
        if !task.starts_with("FG_LN") {
            self.dtd = ddot(n, &self.d, &self.d);
            self.dnorm = self.dtd.sqrt();
            self.stpmx = 1e10;
            if self.cnstnd {
                if self.iter == 0 {
                    self.stpmx = 1.;
                } else {
                    for i in 1..=n {
                        let a1 = self.d[i];
                        if nbd[i] != 0 {
                            if a1 < 0. && nbd[i] <= 2 {
                                let a2 = l[i] - x[i];
                                if a2 >= 0. {
                                    self.stpmx = 0.;
                                } else if a1 * self.stpmx < a2 {
                                    self.stpmx = a2 / a1;
                                }
                            } else if a1 > 0. && nbd[i] >= 2 {
                                let a2 = u[i] - x[i];
                                if a2 <= 0. {
                                    self.stpmx = 0.;
                                } else if a1 * self.stpmx > a2 {
                                    self.stpmx = a2 / a1;
                                }
                            }
                        }
                    }
                }
            }
            if self.iter == 0 && !self.boxed {
                let d1 = 1. / self.dnorm;
                self.stp = cmin(d1, self.stpmx);
            } else {
                self.stp = 1.;
            }
            self.t[1..=n].copy_from_slice(&x[1..=n]);
            self.r[1..=n].copy_from_slice(&g[1..=n]);
            self.fold = f;
            self.ifun = 0;
            self.iback = 0;
            *csave = "START".into();
        }
        self.gd = ddot(n, g, &self.d);
        if self.ifun == 0 {
            self.gdold = self.gd;
            if self.gd >= 0. {
                self.info = -4;
                return;
            }
        }
        let mut stp = self.stp;
        self.ls.run(
            f, self.gd, &mut stp, ftol, gtol, xtol, stpmin, self.stpmx, csave,
        );
        self.stp = stp;
        if !csave.starts_with("CONV") && !csave.starts_with("WARN") {
            *task = "FG_LNSRCH".into();
            self.ifun += 1;
            self.nfgv += 1;
            self.iback = self.ifun - 1;
            if self.stp == 1. {
                x[1..=n].copy_from_slice(&self.z[1..=n]);
            } else {
                for i in 1..=n {
                    x[i] = self.stp * self.d[i] + self.t[i];
                }
            }
        } else {
            *task = "NEW_X".into();
        }
    }

    /// `setulb` + `mainlb` (reverse communication). `x`, `l`, `u`, `nbd`, `g` are 1-based.
    #[allow(clippy::too_many_arguments)]
    fn setulb(
        &mut self,
        x: &mut [f64],
        l: &[f64],
        u: &[f64],
        nbd: &[i32],
        f: &mut f64,
        g: &mut [f64],
        factr: f64,
        pgtol: f64,
        task: &mut String,
    ) {
        // `csave` is a local of `setulb`, emptied on every call.
        let mut csave = String::new();
        let n = self.n;
        let m = self.m;
        let mut wrk = false;
        let mut lab;
        if task.starts_with("START") {
            self.epsmch = f64::EPSILON;
            self.fold = 0.;
            self.dnorm = 0.;
            self.gd = 0.;
            self.sbgnrm = 0.;
            self.stp = 0.;
            self.stpmx = 0.;
            self.gdold = 0.;
            self.dtd = 0.;
            self.col = 0;
            self.head = 1;
            self.theta = 1.;
            self.iupdat = 0;
            self.updatd = false;
            self.iback = 0;
            self.itail = 0;
            self.ifun = 0;
            self.iword = 0;
            self.nact = 0;
            self.ileave = 0;
            self.nenter = 0;
            self.iter = 0;
            self.nfgv = 0;
            self.nint = 0;
            self.nintol = 0;
            self.nskip = 0;
            self.nfree = n;
            self.tol = factr * self.epsmch;
            self.info = 0;
            let (info, _k) = errclb(n, m, factr, l, u, nbd, task);
            self.info = info;
            if task.starts_with("ERROR") {
                return;
            }
            let (_prjctd, cnstnd, boxed) = active(n, l, u, nbd, x, &mut self.iwhere);
            self.cnstnd = cnstnd;
            self.boxed = boxed;
            *task = "FG_START".into();
            self.isave12 = self.nfgv;
            return;
        } else if task.starts_with("FG_LN") {
            lab = Label::L666;
        } else if task.starts_with("NEW_X") {
            lab = Label::L777;
        } else if task.starts_with("FG_ST") {
            lab = Label::L111;
        } else if task.starts_with("STOP") {
            if task.get(6..9) == Some("CPU") {
                x[1..=n].copy_from_slice(&self.t[1..=n]);
                g[1..=n].copy_from_slice(&self.r[1..=n]);
                *f = self.fold;
            }
            self.isave12 = self.nfgv;
            return;
        } else {
            *task = "FG_START".into();
            self.isave12 = self.nfgv;
            return;
        }
        loop {
            match lab {
                Label::L111 => {
                    self.nfgv = 1;
                    self.sbgnrm = projgr(n, l, u, nbd, x, g);
                    if self.sbgnrm <= pgtol {
                        *task = "CONVERGENCE: NORM OF PROJECTED GRADIENT <= PGTOL".into();
                        break;
                    }
                    lab = Label::L222;
                }
                Label::L222 => {
                    self.iword = -1;
                    if !self.cnstnd && self.col > 0 {
                        self.z[1..=n].copy_from_slice(&x[1..=n]);
                        wrk = self.updatd;
                        self.nint = 0;
                        lab = Label::L333;
                        continue;
                    }
                    self.info = cauchy(
                        n,
                        x,
                        l,
                        u,
                        nbd,
                        g,
                        &mut self.indx2,
                        &mut self.iwhere,
                        &mut self.t,
                        &mut self.d,
                        &mut self.z,
                        m,
                        &self.wy,
                        &self.ws,
                        &self.sy,
                        &self.wt,
                        self.theta,
                        self.col,
                        self.head,
                        &mut self.wa_p,
                        &mut self.wa_c,
                        &mut self.wa_wbp,
                        &mut self.wa_v,
                        &mut self.nint,
                        self.sbgnrm,
                        self.epsmch,
                    );
                    if self.info != 0 {
                        self.refresh();
                        lab = Label::L222;
                        continue;
                    }
                    self.nintol += self.nint;
                    wrk = freev(
                        n,
                        &mut self.nfree,
                        &mut self.indx,
                        &mut self.nenter,
                        &mut self.ileave,
                        &mut self.indx2,
                        &self.iwhere,
                        self.updatd,
                        self.cnstnd,
                        self.iter,
                    );
                    self.nact = n - self.nfree;
                    lab = Label::L333;
                }
                Label::L333 => {
                    if self.nfree == 0 || self.col == 0 {
                        lab = Label::L555;
                        continue;
                    }
                    if wrk {
                        self.info = formk(
                            n,
                            self.nfree,
                            &self.indx,
                            self.nenter,
                            self.ileave,
                            &self.indx2,
                            self.iupdat,
                            self.updatd,
                            &mut self.wn,
                            &mut self.snd,
                            m,
                            &self.ws,
                            &self.wy,
                            &self.sy,
                            self.theta,
                            self.col,
                            self.head,
                        );
                    }
                    if self.info != 0 {
                        self.refresh();
                        lab = Label::L222;
                        continue;
                    }
                    self.info = cmprlb(
                        n,
                        m,
                        x,
                        g,
                        &self.ws,
                        &self.wy,
                        &self.sy,
                        &self.wt,
                        &self.z,
                        &mut self.r,
                        &mut self.wa_p,
                        &self.wa_c,
                        &self.indx,
                        self.theta,
                        self.col,
                        self.head,
                        self.nfree,
                        self.cnstnd,
                    );
                    if self.info == 0 {
                        self.info = subsm(
                            n,
                            m,
                            self.nfree,
                            &self.indx,
                            l,
                            u,
                            nbd,
                            &mut self.z,
                            &mut self.r,
                            &self.ws,
                            &self.wy,
                            self.theta,
                            self.col,
                            self.head,
                            &mut self.iword,
                            &mut self.wa_p,
                            &self.wn,
                        );
                    }
                    if self.info != 0 {
                        self.refresh();
                        lab = Label::L222;
                        continue;
                    }
                    lab = Label::L555;
                }
                Label::L555 => {
                    for i in 1..=n {
                        self.d[i] = self.z[i] - x[i];
                    }
                    lab = Label::L666;
                }
                Label::L666 => {
                    self.lnsrlb(l, u, nbd, x, *f, g, task, &mut csave);
                    if self.info != 0 || self.iback >= 20 {
                        x[1..=n].copy_from_slice(&self.t[1..=n]);
                        g[1..=n].copy_from_slice(&self.r[1..=n]);
                        *f = self.fold;
                        if self.col == 0 {
                            if self.info == 0 {
                                self.info = -9;
                                self.nfgv -= 1;
                                self.ifun -= 1;
                                self.iback -= 1;
                            }
                            *task = "ERROR: ABNORMAL_TERMINATION_IN_LNSRCH".into();
                            self.iter += 1;
                            break;
                        } else {
                            if self.info == 0 {
                                self.nfgv -= 1;
                            }
                            self.refresh();
                            *task = "RESTART_FROM_LNSRCH".into();
                            lab = Label::L222;
                            continue;
                        }
                    } else if task.starts_with("FG_LN") {
                        break;
                    } else {
                        self.iter += 1;
                        self.sbgnrm = projgr(n, l, u, nbd, x, g);
                        break;
                    }
                }
                Label::L777 => {
                    if self.sbgnrm <= pgtol {
                        *task = "CONVERGENCE: NORM OF PROJECTED GRADIENT <= PGTOL".into();
                        break;
                    }
                    let d1 = cmax(self.fold.abs(), f.abs());
                    let ddum = cmax(d1, 1.);
                    if self.fold - *f <= self.tol * ddum {
                        *task = "CONVERGENCE: REL_REDUCTION_OF_F <= FACTR*EPSMCH".into();
                        if self.iback >= 10 {
                            self.info = -5;
                        }
                        break;
                    }
                    for i in 1..=n {
                        self.r[i] = g[i] - self.r[i];
                    }
                    let rr = ddot(n, &self.r, &self.r);
                    let dr;
                    let ddum;
                    if self.stp == 1. {
                        dr = self.gd - self.gdold;
                        ddum = -self.gdold;
                    } else {
                        dr = (self.gd - self.gdold) * self.stp;
                        for i in 1..=n {
                            self.d[i] *= self.stp;
                        }
                        ddum = -self.gdold * self.stp;
                    }
                    if dr <= self.epsmch * ddum {
                        self.nskip += 1;
                        self.updatd = false;
                        lab = Label::L222;
                        continue;
                    }
                    self.updatd = true;
                    self.iupdat += 1;
                    matupd(
                        n,
                        m,
                        &mut self.ws,
                        &mut self.wy,
                        &mut self.sy,
                        &mut self.ss,
                        &self.d,
                        &self.r,
                        &mut self.itail,
                        self.iupdat,
                        &mut self.col,
                        &mut self.head,
                        &mut self.theta,
                        rr,
                        dr,
                        self.stp,
                        self.dtd,
                    );
                    self.info = formt(m, &mut self.wt, &self.sy, &self.ss, self.col, self.theta);
                    if self.info != 0 {
                        self.refresh();
                    }
                    lab = Label::L222;
                }
            }
        }
        self.isave12 = self.nfgv;
    }
}

/// What [`optim_lbfgsb`] returns: R's `optim()` list (`par`, `value`, `counts`, `convergence`,
/// `message`).
#[derive(Clone, Debug)]
pub struct OptimResult {
    /// `o$par`.
    pub par: Vec<f64>,
    /// `o$value`.
    pub value: f64,
    /// `o$counts` (function and gradient evaluations; the same number for L-BFGS-B).
    pub counts: i64,
    /// `o$convergence`: 0, 1 (`maxit` reached), 51 (warning) or 52 (error).
    pub convergence: i32,
    /// `o$message`.
    pub message: String,
}

/// `optim(par, fn, method = "L-BFGS-B", lower, upper)` with R's defaults (`lmm = 5`,
/// `factr = 1e7`, `pgtol = 0`, `maxit = 100`, `ndeps = 1e-3`, unit `parscale` / `fnscale`) and
/// the numerical gradient of `fmingr` with bounds. `Err` carries the message R would stop with.
pub fn optim_lbfgsb(
    fun: &mut dyn FnMut(&[f64]) -> f64,
    par: &[f64],
    lower: &[f64],
    upper: &[f64],
) -> Result<OptimResult, String> {
    let n = par.len();
    let (m, factr, pgtol, maxit, ndeps) = (5usize, 1e7, 0.0, 100i64, 1e-3);
    if n == 0 {
        return Ok(OptimResult {
            par: vec![],
            value: fun(upper),
            counts: 1,
            convergence: 0,
            message: "NOTHING TO DO".into(),
        });
    }
    let mut x = vec![0.0; n + 1];
    let mut l = vec![0.0; n + 1];
    let mut u = vec![0.0; n + 1];
    let mut nbd = vec![0i32; n + 1];
    for i in 0..n {
        x[i + 1] = par[i];
        l[i + 1] = lower[i];
        u[i + 1] = upper[i];
        nbd[i + 1] = match (lower[i].is_finite(), upper[i].is_finite()) {
            (false, false) => 0,
            (false, true) => 3,
            (true, false) => 1,
            (true, true) => 2,
        };
    }
    let mut fminfn = |p: &[f64]| -> Result<f64, String> {
        if p.iter().any(|v| !v.is_finite()) {
            return Err("non-finite value supplied by optim".into());
        }
        Ok(fun(p))
    };
    let mut g = vec![0.0; n + 1];
    let mut f = 0.0;
    let mut st = Lbfgsb::new(n, m);
    let mut task = String::from("START");
    let mut iter = 0i64;
    let mut fail = 0;
    loop {
        st.setulb(
            &mut x, &l, &u, &nbd, &mut f, &mut g, factr, pgtol, &mut task,
        );
        if task.starts_with("FG") {
            f = fminfn(&x[1..])?;
            if !f.is_finite() {
                return Err("L-BFGS-B needs finite values of 'fn'".into());
            }
            // fmingr, numerical derivatives with bounds.
            let p: Vec<f64> = x[1..].to_vec();
            let mut xx = p.clone();
            for i in 0..n {
                let mut eps = ndeps;
                let mut epsused = ndeps;
                let mut tmp = p[i] + eps;
                if tmp > u[i + 1] {
                    tmp = u[i + 1];
                    epsused = tmp - p[i];
                }
                xx[i] = tmp;
                let val1 = fminfn(&xx)?;
                tmp = p[i] - eps;
                if tmp < l[i + 1] {
                    tmp = l[i + 1];
                    eps = p[i] - tmp;
                }
                xx[i] = tmp;
                let val2 = fminfn(&xx)?;
                g[i + 1] = (val1 - val2) / (epsused + eps);
                if !g[i + 1].is_finite() {
                    return Err(format!("non-finite finite-difference value [{}]", i + 1));
                }
                xx[i] = p[i];
            }
        } else if task.starts_with("NEW_X") {
            iter += 1;
            if iter > maxit {
                fail = 1;
                break;
            }
        } else if task.starts_with("WARN") {
            fail = 51;
            break;
        } else if task.starts_with("CONV") {
            break;
        } else {
            fail = 52;
            break;
        }
    }
    Ok(OptimResult {
        par: x[1..].to_vec(),
        value: f,
        counts: st.isave12,
        convergence: fail,
        message: task,
    })
}
