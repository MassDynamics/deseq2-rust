//! glibc 2.34 `exp` and `log` (x86_64 FMA variants `__exp_fma` / `__log_fma`), so the
//! R-side objective (`nbinomFn` / `nbinomGr`) reproduces the reference R's bits on any host.
//!
//! The code is the ARM optimized-routines algorithm glibc ships; the FMA placement follows
//! GCC's contraction of the C source under `-mfma` (a product whose only uses are additions
//! becomes a fused multiply-add). Checked against glibc on 4M arguments, see the tests.
#![allow(clippy::unreadable_literal)]

mod tables {
    include!("glibm_tables.rs");
}
use tables::*;

#[inline]
fn fma(a: f64, b: f64, c: f64) -> f64 {
    a.mul_add(b, c)
}

const N: u64 = 128;

fn exp_specialcase(tmp: f64, sbits: u64, ki: u64) -> f64 {
    if ki & 0x8000_0000 == 0 {
        let scale = f64::from_bits(sbits.wrapping_sub(1009u64 << 52));
        return f64::from_bits(0x7f00000000000000) * fma(scale, tmp, scale); // 0x1p1009
    }
    let scale = f64::from_bits(sbits.wrapping_add(1022u64 << 52));
    let mut y = fma(scale, tmp, scale);
    if y < 1.0 {
        let lo = fma(scale, tmp, scale - y);
        let hi = 1.0 + y;
        let lo = 1.0 - hi + y + lo;
        y = (hi + lo) - 1.0;
        if y == 0.0 {
            y = 0.0;
        }
    }
    f64::from_bits(0x0010000000000000) * y // 0x1p-1022
}

/// glibc `exp`.
pub fn exp(x: f64) -> f64 {
    let top12 = |v: f64| (v.to_bits() >> 52) as u32;
    let mut abstop = top12(x) & 0x7ff;
    if abstop.wrapping_sub(top12(f64::from_bits(0x3c90000000000000))) // 0x1p-54
        >= top12(512.0) - top12(f64::from_bits(0x3c90000000000000))
    {
        if abstop.wrapping_sub(top12(f64::from_bits(0x3c90000000000000))) >= 0x8000_0000 {
            return 1.0 + x;
        }
        if abstop >= top12(1024.0) {
            if x == f64::NEG_INFINITY {
                return 0.0;
            }
            if abstop >= top12(f64::INFINITY) {
                return 1.0 + x;
            }
            return if x.to_bits() >> 63 != 0 {
                0.0
            } else {
                f64::INFINITY
            };
        }
        abstop = 0;
    }
    let invln2n = f64::from_bits(0x3ff71547652b82fe) * N as f64;
    let negln2hin = -f64::from_bits(0x3f762e42fefa0000); // -0x1.62e42fefa0000p-8
    let negln2lon = -f64::from_bits(0x3d0cf79abc9e3b3a); // -0x1.cf79abc9e3b3ap-47
    let shift = f64::from_bits(0x4338000000000000); // 0x1.8p52
    let kd = fma(invln2n, x, shift);
    let ki = kd.to_bits();
    let kd = kd - shift;
    let r = fma(kd, negln2lon, fma(kd, negln2hin, x));
    let idx = (2 * (ki % N)) as usize;
    let top = ki << (52 - 7);
    let tail = f64::from_bits(EXP_TAB[idx]);
    let sbits = EXP_TAB[idx + 1].wrapping_add(top);
    let c2 = f64::from_bits(EXP_POLY[0]);
    let c3 = f64::from_bits(EXP_POLY[1]);
    let c4 = f64::from_bits(EXP_POLY[2]);
    let c5 = f64::from_bits(EXP_POLY[3]);
    let r2 = r * r;
    let tmp = fma(r2 * r2, fma(r, c5, c4), fma(r2, fma(r, c3, c2), tail + r));
    if abstop == 0 {
        return exp_specialcase(tmp, sbits, ki);
    }
    let scale = f64::from_bits(sbits);
    fma(scale, tmp, scale)
}

/// glibc `log`.
pub fn log(x: f64) -> f64 {
    let mut ix = x.to_bits();
    let top = (ix >> 48) as u32;
    let lo_b = (1.0f64 - f64::from_bits(0x3fb0000000000000)).to_bits(); // 1 - 0x1p-4
    let hi_b = (1.0f64 + f64::from_bits(0x3fb0900000000000)).to_bits(); // 1 + 0x1.09p-4
    let b = |k: usize| f64::from_bits(LOG_POLY1[k]);
    if ix.wrapping_sub(lo_b) < hi_b - lo_b {
        if ix == 1.0f64.to_bits() {
            return 0.0;
        }
        let r = x - 1.0;
        let r2 = r * r;
        let r3 = r * r2;
        let c = fma(r3, b(10), fma(r2, b(9), fma(r, b(8), b(7))));
        let bb = fma(r3, c, fma(r2, b(6), fma(r, b(5), b(4))));
        let a = fma(r3, bb, fma(r2, b(3), fma(r, b(2), b(1))));
        let w = r * f64::from_bits(0x41a0000000000000); // 0x1p27
        let rhi = r + w - w;
        let rlo = r - rhi;
        let w = rhi * rhi * b(0);
        let hi = r + w;
        let lo = r - hi + w;
        let lo = fma(b(0) * rlo, rhi + r, lo);
        let y = fma(r3, a, lo);
        return y + hi;
    }
    if top.wrapping_sub(0x0010) >= 0x7ff0 - 0x0010 {
        if ix.wrapping_mul(2) == 0 {
            return f64::NEG_INFINITY;
        }
        if x == f64::INFINITY {
            return x;
        }
        if (top & 0x8000) != 0 || (top & 0x7ff0) == 0x7ff0 {
            return f64::NAN;
        }
        ix = (x * f64::from_bits(0x4330000000000000)).to_bits(); // 0x1p52
        ix = ix.wrapping_sub(52u64 << 52);
    }
    const OFF: u64 = 0x3fe6000000000000;
    let tmp = ix.wrapping_sub(OFF);
    let i = ((tmp >> (52 - 7)) % N) as usize;
    let k = (tmp as i64) >> 52;
    let iz = ix.wrapping_sub(tmp & (0xfffu64 << 52));
    let invc = f64::from_bits(LOG_TAB[i].0);
    let logc = f64::from_bits(LOG_TAB[i].1);
    let z = f64::from_bits(iz);
    let r = fma(z, invc, -1.0);
    let kd = k as f64;
    let ln2hi = f64::from_bits(0x3fe62e42fefa3800);
    let ln2lo = f64::from_bits(0x3d2ef35793c76730);
    let w = fma(kd, ln2hi, logc);
    let hi = w + r;
    let lo = fma(kd, ln2lo, w - hi + r);
    let a = |k: usize| f64::from_bits(LOG_POLY[k]);
    let r2 = r * r;
    let u = fma(r, a(4), a(3));
    let v = fma(r2, u, fma(r, a(2), a(1)));
    let s1 = fma(r2, a(0), lo);
    let y = fma(r * r2, v, s1);
    y + hi
}

/// glibc `log1p` (`s_log1p.c`, fdlibm-derived; x86_64 has no FMA variant, so no fusing).
pub fn log1p(x: f64) -> f64 {
    const LN2_HI: f64 = 6.93147180369123816490e-01;
    const LN2_LO: f64 = 1.90821492927058770002e-10;
    const LP: [f64; 8] = [
        0.0,
        6.666666666666735130e-01,
        3.999999999940941908e-01,
        2.857142874366239149e-01,
        2.222219843214978396e-01,
        1.818357216161805012e-01,
        1.531383769920937332e-01,
        1.479819860511658591e-01,
    ];
    let high = |v: f64| (v.to_bits() >> 32) as u32 as i32;
    let set_high =
        |v: f64, h: i32| f64::from_bits(((h as u32 as u64) << 32) | (v.to_bits() & 0xffff_ffff));
    let hx = high(x);
    let ax = hx & 0x7fffffff;
    let mut k: i32 = 1;
    let mut f = 0.0;
    let mut hu: i32 = 0;
    let mut c = 0.0;
    if hx < 0x3FDA827A {
        if ax >= 0x3ff00000 {
            return if x == -1.0 {
                f64::NEG_INFINITY
            } else {
                f64::NAN
            };
        }
        if ax < 0x3e200000 {
            if ax < 0x3c900000 {
                return x;
            }
            return x - x * x * 0.5;
        }
        if hx > 0 || hx <= 0xbfd2bec3u32 as i32 {
            k = 0;
            f = x;
            hu = 1;
        }
    } else if hx >= 0x7ff00000 {
        return x + x;
    }
    if k != 0 {
        let mut u;
        if hx < 0x43400000 {
            u = 1.0 + x;
            hu = high(u);
            k = (hu >> 20) - 1023;
            c = if k > 0 { 1.0 - (u - x) } else { x - (u - 1.0) };
            c /= u;
        } else {
            u = x;
            hu = high(u);
            k = (hu >> 20) - 1023;
            c = 0.0;
        }
        hu &= 0x000fffff;
        if hu < 0x6a09e {
            u = set_high(u, hu | 0x3ff00000);
        } else {
            k += 1;
            u = set_high(u, hu | 0x3fe00000);
            hu = (0x00100000 - hu) >> 2;
        }
        f = u - 1.0;
    }
    let hfsq = 0.5 * f * f;
    let kf = k as f64;
    if hu == 0 {
        if f == 0.0 {
            if k == 0 {
                return 0.0;
            }
            c += kf * LN2_LO;
            return kf * LN2_HI + c;
        }
        let r = hfsq * (1.0 - 0.66666666666666666 * f);
        if k == 0 {
            return f - r;
        }
        return kf * LN2_HI - ((r - (kf * LN2_LO + c)) - f);
    }
    let s = f / (2.0 + f);
    let z = s * s;
    let r1 = z * LP[1];
    let z2 = z * z;
    let r2 = LP[2] + z * LP[3];
    let z4 = z2 * z2;
    let r3 = LP[4] + z * LP[5];
    let z6 = z4 * z2;
    let r4 = LP[6] + z * LP[7];
    let r = r1 + z2 * r2 + z4 * r3 + z6 * r4;
    if k == 0 {
        f - (hfsq - s * (hfsq + r))
    } else {
        kf * LN2_HI - ((hfsq - (s * (hfsq + r) + (kf * LN2_LO + c))) - f)
    }
}
