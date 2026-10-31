//! Scalars for amplitudes: the [`Amplitude`] trait over ncalg's coefficient
//! types, and [`Phasor`], the exact scalars `√2^p · ζ_{2^b}^k` (or zero) that
//! stabilizer computations produce.

use ncalg::cyclo::expsum::CliffordSum;
use ncalg::cyclo::Cyclo;
use ncalg::{BigInt, BigRational, Coeff, Complex, ComplexCoeff, Field};
use std::fmt;

/// `Complex<f64>`.
pub type C64 = Complex<f64>;

/// The scalars amplitudes are computed in.
///
/// Implemented for `Complex<f64>` (floating point) and for `Cyclo<N>` with
/// `4 | N` (exact arithmetic in `ℚ(ζ_N)`).  Every constructor that a type may
/// not hold exactly returns `None`; backends turn that into
/// [`SimError::NotRepresentable`](crate::SimError::NotRepresentable).
pub trait Amplitude: ComplexCoeff + Field {
    /// Exact arithmetic?
    const EXACT: bool;
    /// `ζ_{2^bits}^k = exp(2πi k / 2^bits)`.
    fn root_of_unity(k: i64, bits: u32) -> Option<Self>;
    /// `2^{k/2}` (`√2^k`, any integer `k`).
    fn sqrt2_pow(k: i64) -> Option<Self>;
    /// From a cyclotomic number, exactly when the type contains `ℚ(ζ_N)`.
    fn from_cyclo<const N: u32>(c: &Cyclo<N>) -> Option<Self>;
    /// From a floating value (`None` for exact types).
    fn from_c64(z: C64) -> Option<Self>;
    /// The closest floating value.
    fn to_c64(&self) -> C64 {
        C64::new(self.re_f64(), self.im_f64())
    }
    /// A real floating value.
    fn from_f64(x: f64) -> Option<Self> {
        Self::from_c64(C64::new(x, 0.0))
    }
    /// The exact scalar `p`.
    fn from_phasor(p: Phasor) -> Option<Self> {
        match p {
            Phasor::Zero => Some(Self::zero()),
            Phasor::Value {
                sqrt2_power,
                phase,
                bits,
            } => {
                let r = Self::root_of_unity(phase as i64, bits)?;
                let s = Self::sqrt2_pow(sqrt2_power as i64)?;
                Some(r.mul_ref(&s))
            }
        }
    }
    /// `i^k`.
    fn i_pow(k: i64) -> Self {
        match k.rem_euclid(4) {
            0 => Self::one(),
            1 => Self::imaginary_unit(),
            2 => Self::one().negated(),
            _ => Self::imaginary_unit().negated(),
        }
    }
    /// Is this exactly zero?  The test used to skip work on zero entries:
    /// unlike [`Coeff::is_zero`] it has no float
    /// tolerance, so skipping never changes a result, and it is a plain
    /// comparison (no thread-local lookup) for floats.
    #[inline]
    fn is_exactly_zero(&self) -> bool {
        self.is_zero()
    }
    /// `|z|²` as a scalar (`z · conj z`).
    fn abs_sq(&self) -> Self {
        self.mul_ref(&self.conj())
    }
    /// `|z|²` as a float.
    fn abs_sq_f64(&self) -> f64 {
        self.to_c64().norm_sqr()
    }
    /// Sum of many terms: pairwise, so floating-point errors grow like
    /// `O(ε log n)` instead of `O(ε n)` (free for exact types).
    fn sum(items: &[Self]) -> Self {
        fn rec<S: Amplitude>(items: &[S]) -> S {
            if items.len() <= 8 {
                let mut acc = S::zero();
                for x in items {
                    acc.add_assign_ref(x);
                }
                acc
            } else {
                let (a, b) = items.split_at(items.len() / 2);
                rec(a).add_ref(&rec(b))
            }
        }
        rec(items)
    }
    /// `|self − other| ≤ tol` as floats (for checks and tests).
    fn approx_eq(&self, other: &Self, tol: f64) -> bool {
        (self.to_c64() - other.to_c64()).norm() <= tol
    }
}

fn quarter_turn(j: i64) -> C64 {
    match j.rem_euclid(4) {
        0 => C64::new(1.0, 0.0),
        1 => C64::new(0.0, 1.0),
        2 => C64::new(-1.0, 0.0),
        _ => C64::new(0.0, -1.0),
    }
}

impl Amplitude for C64 {
    const EXACT: bool = false;
    fn root_of_unity(k: i64, bits: u32) -> Option<Self> {
        assert!(bits <= 62, "phase bits must be at most 62");
        let m = 1i64 << bits;
        let k = k.rem_euclid(m);
        if (4 * k) % m == 0 {
            return Some(quarter_turn(4 * k / m));
        }
        if (8 * k) % m == 0 {
            let h = std::f64::consts::FRAC_1_SQRT_2;
            return Some(match 8 * k / m {
                1 => C64::new(h, h),
                3 => C64::new(-h, h),
                5 => C64::new(-h, -h),
                _ => C64::new(h, -h),
            });
        }
        Some(C64::from_polar(
            1.0,
            2.0 * std::f64::consts::PI * (k as f64) / (m as f64),
        ))
    }
    fn sqrt2_pow(k: i64) -> Option<Self> {
        let half = k.div_euclid(2);
        let mut v = 2f64.powi(half as i32);
        if k.rem_euclid(2) == 1 {
            v *= std::f64::consts::SQRT_2;
        }
        Some(C64::new(v, 0.0))
    }
    fn from_cyclo<const N: u32>(c: &Cyclo<N>) -> Option<Self> {
        Some(c.to_complex())
    }
    fn from_c64(z: C64) -> Option<Self> {
        Some(z)
    }
    fn to_c64(&self) -> C64 {
        *self
    }
    #[inline]
    fn is_exactly_zero(&self) -> bool {
        self.re == 0.0 && self.im == 0.0
    }
}

impl<const N: u32> Amplitude for Cyclo<N> {
    const EXACT: bool = true;
    fn root_of_unity(k: i64, bits: u32) -> Option<Self> {
        assert!(bits <= 30, "phase bits must be at most 30 for cyclotomic scalars");
        let m = 1u32 << bits;
        if N % m != 0 {
            return None;
        }
        let step = (N / m) as i64;
        Some(Cyclo::zeta(k.rem_euclid(m as i64) * step))
    }
    fn sqrt2_pow(k: i64) -> Option<Self> {
        let half = k.div_euclid(2);
        let odd = k.rem_euclid(2) == 1;
        if odd && N % 8 != 0 {
            return None;
        }
        let pow2 = if half >= 0 {
            BigRational::from_integer(BigInt::from(1) << (half as usize))
        } else {
            BigRational::new(BigInt::from(1), BigInt::from(1) << ((-half) as usize))
        };
        let mut v = Cyclo::from_rational(pow2);
        if odd {
            v = v.mul_ref(&Cyclo::sqrt2());
        }
        Some(v)
    }
    fn from_cyclo<const M: u32>(c: &Cyclo<M>) -> Option<Self> {
        if M == N {
            return Some(Cyclo::from_coeffs(c.coeffs().to_vec()));
        }
        if N % M != 0 {
            return None;
        }
        let s = (N / M) as usize;
        let src = c.coeffs();
        let mut v = vec![BigRational::from_integer(BigInt::from(0)); src.len() * s];
        for (j, x) in src.iter().enumerate() {
            v[j * s] = x.clone();
        }
        Some(Cyclo::from_coeffs(v))
    }
    fn from_c64(_: C64) -> Option<Self> {
        None
    }
}

/// An exact scalar of the form `√2^p · ζ_{2^bits}^phase`, or zero: every
/// amplitude, inner product and expectation value of stabilizer states, and
/// every Clifford-class exponential sum (`ncalg`'s [`CliffordSum`]).
///
/// Normalized: `phase < 2^bits` and `bits` minimal (`phase` odd unless
/// `bits = 0`), so structural equality is equality.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Phasor {
    /// The scalar `0`.
    Zero,
    /// `√2^{sqrt2_power} · exp(2πi · phase / 2^bits)`.
    Value {
        /// `p` in `√2^p` (negative allowed).
        sqrt2_power: i32,
        /// The argument in units of `2π / 2^bits`.
        phase: u64,
        /// Phase resolution.
        bits: u32,
    },
}

impl Phasor {
    /// The scalar `1`.
    pub const ONE: Phasor = Phasor::Value {
        sqrt2_power: 0,
        phase: 0,
        bits: 0,
    };

    /// `√2^p · ζ_{2^bits}^phase`, normalized.
    pub fn new(sqrt2_power: i32, phase: i64, bits: u32) -> Phasor {
        assert!(bits <= 62, "phase bits must be at most 62");
        let m = 1u64 << bits;
        let mut phase = phase.rem_euclid(m as i64) as u64;
        let mut bits = bits;
        while bits > 0 && phase % 2 == 0 {
            phase /= 2;
            bits -= 1;
        }
        Phasor::Value {
            sqrt2_power,
            phase,
            bits,
        }
    }
    /// `ζ_{2^bits}^phase`.
    pub fn root_of_unity(phase: i64, bits: u32) -> Phasor {
        Phasor::new(0, phase, bits)
    }
    /// `i^k`.
    pub fn i_pow(k: i64) -> Phasor {
        Phasor::new(0, k, 2)
    }
    /// `(−1)^k`.
    pub fn sign(k: i64) -> Phasor {
        Phasor::new(0, k, 1)
    }
    /// `√2^k`.
    pub fn sqrt2_pow(k: i32) -> Phasor {
        Phasor::new(k, 0, 0)
    }
    /// Is this zero?
    pub fn is_zero(&self) -> bool {
        matches!(self, Phasor::Zero)
    }
    /// Is this one?
    pub fn is_one(&self) -> bool {
        *self == Phasor::ONE
    }
    /// The product (also `*`).
    #[allow(clippy::should_implement_trait)]
    pub fn mul(self, o: Phasor) -> Phasor {
        match (self, o) {
            (Phasor::Zero, _) | (_, Phasor::Zero) => Phasor::Zero,
            (
                Phasor::Value {
                    sqrt2_power: p1,
                    phase: f1,
                    bits: b1,
                },
                Phasor::Value {
                    sqrt2_power: p2,
                    phase: f2,
                    bits: b2,
                },
            ) => {
                let b = b1.max(b2);
                let f = (f1 << (b - b1)) + (f2 << (b - b2));
                Phasor::new(p1 + p2, f as i64, b)
            }
        }
    }
    /// The complex conjugate.
    pub fn conj(self) -> Phasor {
        match self {
            Phasor::Zero => Phasor::Zero,
            Phasor::Value {
                sqrt2_power,
                phase,
                bits,
            } => Phasor::new(sqrt2_power, -(phase as i64), bits),
        }
    }
    /// The inverse (`None` for zero).
    pub fn inverse(self) -> Option<Phasor> {
        match self {
            Phasor::Zero => None,
            Phasor::Value {
                sqrt2_power,
                phase,
                bits,
            } => Some(Phasor::new(-sqrt2_power, -(phase as i64), bits)),
        }
    }
    /// `−self`.
    pub fn negated(self) -> Phasor {
        self.mul(Phasor::sign(1))
    }
    /// `i · self`.
    pub fn times_i(self) -> Phasor {
        self.mul(Phasor::i_pow(1))
    }
    /// `self · √2^k`.
    pub fn scale_sqrt2(self, k: i32) -> Phasor {
        self.mul(Phasor::sqrt2_pow(k))
    }
    /// `log₂ |self|²`, `None` for zero.
    pub fn abs_sq_log2(&self) -> Option<i32> {
        match self {
            Phasor::Zero => None,
            Phasor::Value { sqrt2_power, .. } => Some(*sqrt2_power),
        }
    }
    /// The argument as `(phase, bits)`: `exp(2πi phase / 2^bits)`; `(0, 0)` for zero.
    pub fn argument(&self) -> (u64, u32) {
        match self {
            Phasor::Zero => (0, 0),
            Phasor::Value { phase, bits, .. } => (*phase, *bits),
        }
    }
    /// Real (phase `0` or `π`)?
    pub fn is_real(&self) -> bool {
        self.argument().1 <= 1
    }
    /// The floating value.
    pub fn to_c64(self) -> C64 {
        match self {
            Phasor::Zero => C64::new(0.0, 0.0),
            Phasor::Value {
                sqrt2_power,
                phase,
                bits,
            } => {
                C64::root_of_unity(phase as i64, bits).unwrap() * C64::sqrt2_pow(sqrt2_power as i64).unwrap()
            }
        }
    }
    /// The value in an [`Amplitude`] type (`None` if not representable).
    pub fn to_amplitude<S: Amplitude>(self) -> Option<S> {
        S::from_phasor(self)
    }
    /// From ncalg's closed form of a Clifford-class exponential sum.
    pub fn from_clifford_sum(cs: &CliffordSum) -> Phasor {
        match *cs {
            CliffordSum::Zero => Phasor::Zero,
            CliffordSum::Value {
                sqrt2_power,
                phase,
                phase_bits,
            } => Phasor::new(sqrt2_power as i32, phase as i64, phase_bits),
        }
    }
}

impl std::ops::Mul for Phasor {
    type Output = Phasor;
    fn mul(self, o: Phasor) -> Phasor {
        Phasor::mul(self, o)
    }
}

impl Default for Phasor {
    fn default() -> Self {
        Phasor::ONE
    }
}

impl fmt::Display for Phasor {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Phasor::Zero => write!(f, "0"),
            Phasor::Value {
                sqrt2_power,
                phase,
                bits,
            } => {
                let mut parts = Vec::new();
                if *sqrt2_power != 0 {
                    parts.push(format!("√2^{sqrt2_power}"));
                }
                match bits {
                    0 => {}
                    1 => parts.push("-1".to_string()),
                    2 => parts.push(if *phase == 1 {
                        "i".to_string()
                    } else {
                        "-i".to_string()
                    }),
                    _ => parts.push(format!("ζ{}^{}", 1u64 << bits, phase)),
                }
                if parts.is_empty() {
                    write!(f, "1")
                } else {
                    write!(f, "{}", parts.join("·"))
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn phasor_arithmetic() {
        let z8 = Phasor::root_of_unity(1, 3);
        let p = z8.mul(z8).mul(z8).mul(z8);
        assert_eq!(p, Phasor::sign(1));
        assert_eq!(z8.mul(z8.conj()), Phasor::ONE);
        assert_eq!(Phasor::new(0, 4, 3), Phasor::sign(1));
        assert_eq!(Phasor::new(0, 2, 3), Phasor::i_pow(1));
        let v = Phasor::new(-3, 5, 3);
        let c = v.to_c64();
        let e = C64::from_polar(2f64.powf(-1.5), 2.0 * std::f64::consts::PI * 5.0 / 8.0);
        assert!((c - e).norm() < 1e-14);
        assert_eq!(v.mul(v.inverse().unwrap()), Phasor::ONE);
    }

    #[test]
    fn exact_and_float_agree() {
        for bits in 0..5 {
            for k in 0..(1i64 << bits) {
                let e: Cyclo<16> = Cyclo::root_of_unity(k, bits).unwrap();
                let f = C64::root_of_unity(k, bits).unwrap();
                assert!((e.to_complex() - f).norm() < 1e-14, "k={k} bits={bits}");
            }
        }
        assert!(Cyclo::<4>::root_of_unity(1, 3).is_none());
        assert!(Cyclo::<4>::sqrt2_pow(1).is_none());
        for k in -5..6 {
            let e: Cyclo<8> = Cyclo::sqrt2_pow(k).unwrap();
            let f = C64::sqrt2_pow(k).unwrap();
            assert!((e.to_complex() - f).norm() < 1e-14);
        }
        let z: Cyclo<8> = Cyclo::zeta(3);
        let up: Cyclo<16> = Cyclo::from_cyclo(&z).unwrap();
        assert_eq!(up, Cyclo::<16>::zeta(6));
        let s = C64::sum(
            &(0..1000)
                .map(|i| C64::new(0.1 * i as f64, 0.0))
                .collect::<Vec<_>>(),
        );
        assert!((s.re - 0.1 * 999.0 * 1000.0 / 2.0).abs() < 1e-9);
    }
}
