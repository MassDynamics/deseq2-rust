//! Software x87 extended-precision accumulator.
//!
//! The reference R runs on x86_64, where `LDOUBLE` is the x87 80-bit type (64-bit
//! mantissa). R's `sum`, `colSums`, `rowSums` and `cumsum` accumulate in it and round to
//! `double` only at the end. This type reproduces that: every addition is rounded to a
//! 64-bit mantissa (round to nearest, ties to even) and the final conversion rounds again
//! to 53 bits, so the double rounding of the real hardware is kept.
//!
//! Only finite values are expected; a non-finite input switches to plain `f64` addition,
//! which gives the same Inf / NaN result as the hardware.

/// An x87 extended value: `(-1)^neg * mant * 2^(exp - 63)`, `mant` normalised (top bit
/// set) unless the value is zero.
#[derive(Clone, Copy, Debug)]
pub struct Xld {
    neg: bool,
    exp: i32,
    mant: u64,
    /// Set once a non-finite value is added; the accumulator then degrades to `f64`.
    nonfinite: Option<f64>,
}

impl Default for Xld {
    fn default() -> Self {
        Self::ZERO
    }
}

impl Xld {
    pub const ZERO: Xld = Xld { neg: false, exp: 0, mant: 0, nonfinite: None };

    /// Exact conversion from `f64`.
    pub fn from_f64(x: f64) -> Xld {
        if !x.is_finite() {
            return Xld { nonfinite: Some(x), ..Xld::ZERO };
        }
        if x == 0.0 {
            return Xld { neg: x.is_sign_negative(), ..Xld::ZERO };
        }
        let bits = x.to_bits();
        let neg = bits >> 63 == 1;
        let e = ((bits >> 52) & 0x7ff) as i32;
        let f = bits & ((1u64 << 52) - 1);
        let (mut mant, mut exp) = if e == 0 { (f, -1022) } else { (f | (1u64 << 52), e - 1023) };
        // value = mant * 2^(exp - 52); normalise so the top bit is bit 63
        let lz = mant.leading_zeros() as i32;
        mant <<= lz;
        exp -= lz - 11;
        Xld { neg, exp, mant, nonfinite: None }
    }

    /// Round to `f64` (nearest, ties to even).
    pub fn to_f64(self) -> f64 {
        if let Some(v) = self.nonfinite {
            return v;
        }
        if self.mant == 0 {
            return if self.neg { -0.0 } else { 0.0 };
        }
        // 64 -> 53 bits
        let r = 11u32;
        let mut m = self.mant >> r;
        let rem = self.mant & ((1u64 << r) - 1);
        let half = 1u64 << (r - 1);
        if rem > half || (rem == half && (m & 1) == 1) {
            m += 1;
        }
        let mut exp = self.exp;
        if m == (1u64 << 53) {
            m >>= 1;
            exp += 1;
        }
        // value = m * 2^(exp - 52)
        // split the scaling so neither factor over- or underflows on its own
        let e = exp - 52;
        let (e1, e2) = (e / 2, e - e / 2);
        let v = (m as f64) * 2f64.powi(e1) * 2f64.powi(e2);
        if self.neg {
            -v
        } else {
            v
        }
    }

    /// `self + other`, rounded to a 64-bit mantissa.
    #[allow(clippy::should_implement_trait)]
    pub fn add(self, other: Xld) -> Xld {
        if self.nonfinite.is_some() || other.nonfinite.is_some() {
            return Xld { nonfinite: Some(self.to_f64() + other.to_f64()), ..Xld::ZERO };
        }
        if other.mant == 0 {
            if self.mant == 0 {
                return Xld { neg: self.neg && other.neg, ..Xld::ZERO };
            }
            return self;
        }
        if self.mant == 0 {
            return other;
        }
        let (a, b) = if (self.exp, self.mant) >= (other.exp, other.mant) { (self, other) } else { (other, self) };
        let big = (a.mant as u128) << 62;
        let mut small = (b.mant as u128) << 62;
        let d = (a.exp - b.exp) as u32;
        if d >= 126 {
            small = 1; // sticky only
        } else if d > 0 {
            let lost = small & ((1u128 << d) - 1);
            small >>= d;
            if lost != 0 {
                small |= 1;
            }
        }
        let (s, neg) = if a.neg == b.neg { (big + small, a.neg) } else { (big - small, a.neg) };
        if s == 0 {
            return Xld::ZERO;
        }
        let p = 127 - s.leading_zeros() as i32; // index of the leading bit
        let exp = a.exp - 125 + p;
        let mant = if p > 63 {
            let r = (p - 63) as u32;
            let mut m = (s >> r) as u64;
            let rem = s & ((1u128 << r) - 1);
            let half = 1u128 << (r - 1);
            if rem > half || (rem == half && (m & 1) == 1) {
                let (mm, carry) = m.overflowing_add(1);
                if carry {
                    return Xld { neg, exp: exp + 1, mant: 1u64 << 63, nonfinite: None };
                }
                m = mm;
            }
            m
        } else {
            (s << (63 - p) as u32) as u64
        };
        Xld { neg, exp, mant, nonfinite: None }
    }

    pub fn add_f64(self, x: f64) -> Xld {
        self.add(Xld::from_f64(x))
    }
}

/// R `sum(x)` for a double vector (no NA handling needed by the callers).
pub fn r_sum<I: IntoIterator<Item = f64>>(xs: I) -> f64 {
    xs.into_iter().fold(Xld::ZERO, |acc, x| acc.add_f64(x)).to_f64()
}

/// R `cumsum(x)`.
pub fn r_cumsum(xs: &[f64]) -> Vec<f64> {
    let mut acc = Xld::ZERO;
    xs.iter()
        .map(|&x| {
            acc = acc.add_f64(x);
            acc.to_f64()
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn roundtrip_and_simple_sums() {
        for &x in &[1.0, -2.5, 1e-300, 3.141592653589793, 1e300, 5e-324] {
            assert_eq!(Xld::from_f64(x).to_f64(), x);
        }
        assert_eq!(r_sum([0.1, 0.2, 0.3]), 0.6);
        // 1 + 2^-60 - 1 is exact in extended precision, zero in double
        let e = 2f64.powi(-60);
        assert_eq!(r_sum([1.0, e, -1.0]), e);
        assert_eq!(r_sum([1.0, 2f64.powi(-70), -1.0]), 0.0);
    }
}
