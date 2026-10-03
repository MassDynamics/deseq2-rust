//! R's x87 80-bit `long double` accumulation, emulated in software.
//!
//! R on x86_64 Linux accumulates `sum()`, `mean()`, `rowSums()` and `rowMeans()` in an
//! 80-bit extended register (64-bit mantissa) and rounds once to double at the end.
//! `median()` of an even-length vector is `mean()` of the two middle values, so it goes
//! through the same path. The DESeq2 stages branch on these values (size factors feed
//! every downstream stage), so plain `f64` sums are not bit-identical. [`F80`] holds a
//! sign, an unbiased exponent and a 64-bit mantissa with an explicit leading bit, and
//! implements round-to-nearest-even addition and division by an integer exactly as the
//! x87 FPU does in extended precision mode.

use std::cmp::Ordering;

/// An x87 extended-precision value (finite, or a carried non-finite `f64`).
#[derive(Clone, Copy, Debug)]
pub struct F80 {
    neg: bool,
    /// Value = `mant * 2^(exp - 63)`; `mant == 0` is zero.
    exp: i32,
    mant: u64,
    /// `Some(x)` for Inf/NaN, propagated with `f64` semantics.
    nonfinite: Option<f64>,
}

// `neg`, `add` and `sub` are explicit F80 operations (rounding to the 64-bit mantissa), named
// after R's long-double expressions; operator traits would hide where the rounding happens.
#[allow(clippy::should_implement_trait)]
impl F80 {
    pub const ZERO: F80 = F80 {
        neg: false,
        exp: 0,
        mant: 0,
        nonfinite: None,
    };

    /// Exact conversion from `f64`.
    pub fn from_f64(x: f64) -> F80 {
        if !x.is_finite() {
            return F80 {
                nonfinite: Some(x),
                ..F80::ZERO
            };
        }
        if x == 0.0 {
            return F80 {
                neg: x.is_sign_negative(),
                ..F80::ZERO
            };
        }
        let bits = x.to_bits();
        let neg = bits >> 63 == 1;
        let e = ((bits >> 52) & 0x7ff) as i32;
        let frac = bits & ((1u64 << 52) - 1);
        let (m, big_e) = if e == 0 {
            (frac, -1074)
        } else {
            (frac | (1u64 << 52), e - 1075)
        };
        let lz = m.leading_zeros() as i32;
        F80 {
            neg,
            exp: big_e - lz + 63,
            mant: m << lz,
            nonfinite: None,
        }
    }

    /// Exact conversion from an integer count.
    pub fn from_u64(n: u64) -> F80 {
        if n == 0 {
            return F80::ZERO;
        }
        let lz = n.leading_zeros() as i32;
        F80 {
            neg: false,
            exp: 63 - lz,
            mant: n << lz,
            nonfinite: None,
        }
    }

    /// Round to double (round to nearest, ties to even), as an x87 `fstp` to a double does.
    pub fn to_f64(self) -> f64 {
        if let Some(x) = self.nonfinite {
            return x;
        }
        let sign = if self.neg { 1u64 << 63 } else { 0 };
        if self.mant == 0 {
            return f64::from_bits(sign);
        }
        let lsb = (self.exp - 52).max(-1074);
        let shift = (lsb - (self.exp - 63)) as u32;
        let (mut q, rem, half) = if shift >= 64 {
            // Far below the smallest subnormal: rounds to zero (or to the smallest when > half).
            (
                0u64,
                if shift == 64 { self.mant as u128 } else { 1 },
                if shift == 64 { 1u128 << 63 } else { u128::MAX },
            )
        } else {
            let m = self.mant as u128;
            (
                (self.mant >> shift),
                m & ((1u128 << shift) - 1),
                1u128 << (shift - 1),
            )
        };
        if rem > half || (rem == half && q & 1 == 1) {
            q += 1;
        }
        let mut lsb = lsb;
        if q == 1u64 << 53 {
            q >>= 1;
            lsb += 1;
        }
        if q == 0 {
            return f64::from_bits(sign);
        }
        if q < (1u64 << 52) {
            // Subnormal (lsb == -1074).
            return f64::from_bits(sign | q);
        }
        let e = lsb + 52 + 1023;
        if e >= 0x7ff {
            return if self.neg {
                f64::NEG_INFINITY
            } else {
                f64::INFINITY
            };
        }
        f64::from_bits(sign | ((e as u64) << 52) | (q & ((1u64 << 52) - 1)))
    }

    pub fn is_zero(&self) -> bool {
        self.nonfinite.is_none() && self.mant == 0
    }

    pub fn neg(self) -> F80 {
        F80 {
            neg: !self.neg,
            nonfinite: self.nonfinite.map(|x| -x),
            ..self
        }
    }

    fn cmp_mag(&self, o: &F80) -> Ordering {
        self.exp.cmp(&o.exp).then(self.mant.cmp(&o.mant))
    }

    /// Normalise `s * 2^(e_lsb)` (with a sticky bit already folded into bit 0) to 64 bits.
    fn round_u128(neg: bool, s: u128, e_lsb: i32) -> F80 {
        if s == 0 {
            return F80::ZERO;
        }
        let top = 127 - s.leading_zeros() as i32;
        if top <= 63 {
            let m = (s as u64) << (63 - top);
            return F80 {
                neg,
                exp: e_lsb + top,
                mant: m,
                nonfinite: None,
            };
        }
        let shift = (top - 63) as u32;
        let rem = s & ((1u128 << shift) - 1);
        let half = 1u128 << (shift - 1);
        let mut m = (s >> shift) as u64;
        let mut exp = e_lsb + top;
        if rem > half || (rem == half && m & 1 == 1) {
            let (mm, carry) = m.overflowing_add(1);
            if carry {
                m = 1u64 << 63;
                exp += 1;
            } else {
                m = mm;
            }
        }
        F80 {
            neg,
            exp,
            mant: m,
            nonfinite: None,
        }
    }

    /// Extended-precision addition.
    pub fn add(self, o: F80) -> F80 {
        if self.nonfinite.is_some() || o.nonfinite.is_some() {
            let a = self.nonfinite.unwrap_or(0.0);
            let b = o.nonfinite.unwrap_or(0.0);
            return F80 {
                nonfinite: Some(a + b),
                ..F80::ZERO
            };
        }
        if o.mant == 0 {
            if self.mant == 0 {
                return F80 {
                    neg: self.neg && o.neg,
                    ..F80::ZERO
                };
            }
            return self;
        }
        if self.mant == 0 {
            return o;
        }
        let (a, b) = if self.cmp_mag(&o) == Ordering::Less {
            (o, self)
        } else {
            (self, o)
        };
        let a_big = (a.mant as u128) << 62;
        let d = (a.exp - b.exp) as u32;
        let b_full = (b.mant as u128) << 62;
        let (mut b_big, sticky) = if d == 0 {
            (b_full, false)
        } else if d >= 126 {
            (0u128, true)
        } else {
            (b_full >> d, b_full & ((1u128 << d) - 1) != 0)
        };
        if sticky {
            b_big |= 1;
        }
        let e_lsb = a.exp - 63 - 62;
        if a.neg == b.neg {
            F80::round_u128(a.neg, a_big + b_big, e_lsb)
        } else {
            let s = a_big - b_big;
            if s == 0 {
                return F80::ZERO;
            }
            F80::round_u128(a.neg, s, e_lsb)
        }
    }

    pub fn sub(self, o: F80) -> F80 {
        self.add(o.neg())
    }

    /// Extended-precision division by a positive integer (`s /= n` with `n` an `R_xlen_t`).
    pub fn div_u64(self, n: u64) -> F80 {
        if let Some(x) = self.nonfinite {
            return F80 {
                nonfinite: Some(x / n as f64),
                ..F80::ZERO
            };
        }
        if self.mant == 0 {
            return self;
        }
        let num = (self.mant as u128) << 64;
        let q = num / n as u128;
        let r = num % n as u128;
        let s = q | u128::from(r != 0);
        F80::round_u128(self.neg, s, self.exp - 63 - 64)
    }

    pub fn add_f64(self, x: f64) -> F80 {
        self.add(F80::from_f64(x))
    }
}

/// `sum(x)` (R `rsum`).
pub fn sum(x: &[f64]) -> f64 {
    x.iter().fold(F80::ZERO, |s, &v| s.add_f64(v)).to_f64()
}

/// `mean(x)` for a double vector (R `real_mean`): extended sum over `n`, then one
/// correction pass `s += sum(x - s) / n`, all in extended precision.
pub fn mean(x: &[f64]) -> f64 {
    let n = x.len() as u64;
    if n == 0 {
        return f64::NAN;
    }
    let mut s = x.iter().fold(F80::ZERO, |s, &v| s.add_f64(v));
    let finite_s = s.to_f64().is_finite();
    if finite_s {
        s = s.div_u64(n);
    } else {
        s = x.iter().fold(F80::ZERO, |s, &v| s.add_f64(v / n as f64));
    }
    if finite_s && s.to_f64().is_finite() {
        let mut t = F80::ZERO;
        for &v in x {
            t = t.add(F80::from_f64(v).sub(s));
        }
        s = s.add(t.div_u64(n));
    } else if s.to_f64().is_finite() {
        let mut t = F80::ZERO;
        for &v in x {
            t = t.add(F80::from_f64(v / n as f64).sub(s.div_u64(n)));
        }
        s = s.add(t);
    }
    s.to_f64()
}

/// `mean()` of a logical/integer vector: `(double)(s / n)` with an extended sum.
pub fn mean_count(count: u64, n: u64) -> f64 {
    F80::from_u64(count).div_u64(n).to_f64()
}

/// `rowSums(x)` for a row-major `nrow x ncol` matrix given as rows.
pub fn row_sum(row: &[f64]) -> f64 {
    sum(row)
}

/// One row of `rowMeans(x)`: extended sum, extended division, one rounding.
pub fn row_mean(row: &[f64]) -> f64 {
    row.iter()
        .fold(F80::ZERO, |s, &v| s.add_f64(v))
        .div_u64(row.len() as u64)
        .to_f64()
}

/// R's `rcmp` with NA last.
fn rcmp(x: f64, y: f64) -> Ordering {
    match (x.is_nan(), y.is_nan()) {
        (true, true) => Ordering::Equal,
        (true, false) => Ordering::Greater,
        (false, true) => Ordering::Less,
        _ => x.partial_cmp(&y).unwrap(),
    }
}

/// R's `rPsort2` (0-based `lo`, `hi`, `k`): partial quicksort placing `x[k]`.
fn psort2(x: &mut [f64], lo: isize, hi: isize, k: isize) {
    let (mut l, mut r) = (lo, hi);
    while l < r {
        let v = x[k as usize];
        let (mut i, mut j) = (l, r);
        while i <= j {
            while rcmp(x[i as usize], v) == Ordering::Less {
                i += 1;
            }
            while rcmp(v, x[j as usize]) == Ordering::Less {
                j -= 1;
            }
            if i <= j {
                x.swap(i as usize, j as usize);
                i += 1;
                j -= 1;
            }
        }
        if j < k {
            l = i;
        }
        if k < i {
            r = j;
        }
    }
}

/// R's `Psort0` with 1-based sorted indices `ind`.
fn psort0(x: &mut [f64], lo: isize, hi: isize, ind: &[isize]) {
    if ind.is_empty() || hi - lo < 1 {
        return;
    }
    if ind.len() == 1 {
        psort2(x, lo, hi, ind[0] - 1);
    } else {
        let mid = (lo + hi) / 2;
        let mut this = 0;
        for (i, &v) in ind.iter().enumerate() {
            if v - 1 <= mid {
                this = i;
            }
        }
        let z = ind[this] - 1;
        psort2(x, lo, hi, z);
        psort0(x, lo, z - 1, &ind[..this]);
        psort0(x, z + 1, hi, &ind[this + 1..]);
    }
}

/// `sort(x, partial = ind)` (1-based, any order; duplicates removed as `unique()` does).
pub fn psort(x: &[f64], ind: &[usize]) -> Vec<f64> {
    let mut y = x.to_vec();
    let mut p: Vec<isize> = Vec::new();
    for &i in ind {
        if !p.contains(&(i as isize)) {
            p.push(i as isize);
        }
    }
    p.sort();
    let hi = y.len() as isize - 1;
    psort0(&mut y, 0, hi, &p);
    y
}

/// `mean(x, trim = trim)` (R `mean.default`): partial sort, slice, `real_mean`.
pub fn trimmed_mean(x: &[f64], trim: f64) -> f64 {
    let n = x.len();
    if trim > 0.0 && n > 0 {
        if x.iter().any(|v| v.is_nan()) {
            return f64::NAN;
        }
        if trim >= 0.5 {
            return median(x);
        }
        let lo = (n as f64 * trim).floor() as usize + 1;
        let hi = n + 1 - lo;
        let y = psort(x, &[lo, hi]);
        return mean(&y[lo - 1..hi]);
    }
    mean(x)
}

/// `median(x)` (R `median.default`, NA if any NA).
pub fn median(x: &[f64]) -> f64 {
    let n = x.len();
    if n == 0 || x.iter().any(|v| v.is_nan()) {
        return f64::NAN;
    }
    let half = n.div_ceil(2);
    if n % 2 == 1 {
        psort(x, &[half])[half - 1]
    } else {
        let y = psort(x, &[half, half + 1]);
        mean(&y[half - 1..half + 1])
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn extended_sum_keeps_tiny_terms() {
        let t = 2f64.powi(-60);
        assert_eq!(sum(&[1.0, t, -1.0]), t);
        // In double this is 0.
        assert_eq!(1.0 + t - 1.0, 0.0);
    }

    #[test]
    fn roundtrip_and_division() {
        for &x in &[1.0, -3.5, 1e-310, 1e300, 0.1, 123456.789] {
            assert_eq!(F80::from_f64(x).to_f64(), x);
        }
        assert_eq!(F80::from_f64(1.0).div_u64(3).to_f64(), 1.0 / 3.0);
        assert_eq!(mean(&[0.1, 0.2, 0.3]), 0.2);
        assert_eq!(median(&[4.0, 1.0, 3.0, 2.0]), 2.5);
        assert_eq!(trimmed_mean(&[1.0, 2.0, 3.0, 100.0, 4.0], 0.2), 3.0);
    }
}

#[cfg(test)]
mod r_cases {
    /// 3000 cases dumped from R 4.5 in the production image by `tests/data/ext_cases.R`.
    #[test]
    fn matches_r_dump() {
        let path = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/data/ext_cases.txt");
        let txt = std::fs::read_to_string(path).expect("tests/data/ext_cases.txt");
        assert_eq!(txt.lines().count(), 3000);
        let mut bad = 0;
        for line in txt.lines() {
            let (a, b) = line.split_once(" | ").unwrap();
            let x: Vec<f64> = a.split(' ').map(|v| v.parse().unwrap()).collect();
            let w: Vec<f64> = b.split(' ').map(|v| v.parse().unwrap()).collect();
            let g = [
                super::sum(&x),
                super::mean(&x),
                super::median(&x),
                super::trimmed_mean(&x, 0.2),
                super::trimmed_mean(&x, 1.0 / 8.0),
                super::row_mean(&x),
            ];
            for k in 0..6 {
                if g[k].to_bits() != w[k].to_bits() {
                    bad += 1;
                    eprintln!("case k={k} got {:e} want {:e} x={x:?}", g[k], w[k]);
                }
            }
        }
        assert_eq!(bad, 0);
    }
}
