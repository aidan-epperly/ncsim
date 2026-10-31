//! ncalg polynomials as Pauli sums, and states as moment functionals.
//!
//! The Pauli algebra of [`ncalg::relations::pauli`] has generators named
//! `X0, Y0, Z0, X1, …`; a word is the operator product of its letters, left to
//! right, and a polynomial is a Hamiltonian or an observable.  This module
//! turns such polynomials into [`PauliSum`]s ([`PauliSum::from_polynomial`],
//! [`PauliSum::parse`]), Pauli sums back into polynomials in normal words
//! ([`PauliSum::to_polynomial`]), and evaluates the moment functional
//! `L(w) = ⟨ψ|w|ψ⟩` of a state on the words of an NPA relaxation
//! ([`moments`], [`expectation_polynomial`]).  Majorana polynomials of
//! [`ncalg::relations::majorana`] (generators `g0, g1, …`) go through the
//! Jordan–Wigner encoding ([`majorana_pauli`], [`majorana_word`],
//! [`PauliSum::from_majorana_polynomial`], [`majorana_moments`]); any other
//! encoding of generators as Paulis is one closure away
//! ([`word_pauli`], [`PauliSum::from_polynomial_with`], [`moments_with`]).
//!
//! # Coefficients
//!
//! [`IntoAmplitude`] converts ncalg coefficients into an [`Amplitude`] type,
//! [`FromAmplitude`] converts back.  Both return `None` when the value is not
//! representable.  The rules: exact values (integers, rationals, Gaussian
//! rationals, cyclotomic numbers) convert exactly wherever the target contains
//! them — rationals into every `Cyclo<N>`, `a + bi` when `4 | N`, `Cyclo<M>`
//! into `Cyclo<N>` when `M | N` or the value happens to be a Gaussian
//! rational — and approximately into `Complex<f64>`; floating values convert
//! only into floating targets (never into `Cyclo<N>` or a rational type:
//! a float is not an exact value).
//!
//! Coherence of the impls: [`IntoAmplitude<S>`] is implemented once per
//! *source* type, generically in the target `S: Amplitude`, because all the
//! target-specific knowledge already sits in [`Amplitude::from_cyclo`] and
//! [`Amplitude::from_c64`] (`ℚ = ℚ(ζ_1)` and `ℚ(i) = ℚ(ζ_4)` embed into every
//! amplitude type through `from_cyclo`).  No impl is generic in the source,
//! so the impls for `f64`, `BigRational`, …, `Cyclo<M>` never overlap.
//! [`FromAmplitude<S>`] is implemented per (target, source) pair, since
//! [`Amplitude`] has no exact decomposition method; the two sources
//! `Complex<f64>` and `Cyclo<N>` are distinct type constructors, so these do
//! not overlap either.

use crate::error::{Result, SimError};
use crate::pauli::{Pauli, PauliKind, PauliSum};
use crate::scalar::{Amplitude, C64};
use crate::state::QuantumState;
use ncalg::cyclo::Cyclo;
use ncalg::{
    BigInt, BigRational, Coeff, Complex, ComplexCoeff, FreeAlgebra, HybridRational, Polynomial, Var, Word,
};
use num_traits::ToPrimitive;

// ---------------------------------------------------------------------------
// coefficient conversion
// ---------------------------------------------------------------------------

/// Conversion of an ncalg coefficient into an amplitude type (`None` when
/// the value is not representable; see the [module docs](self)).
#[allow(clippy::wrong_self_convention)] // the value is borrowed, as the Coeff methods do
pub trait IntoAmplitude<S: Amplitude> {
    /// The value as an `S`.
    fn into_amplitude(&self) -> Option<S>;
}

/// Conversion of an amplitude into an ncalg coefficient type (`None` when
/// the value is not representable; see the [module docs](self)).
pub trait FromAmplitude<S: Amplitude>: Sized {
    /// The value of `s` as `Self`.
    fn from_amplitude(s: &S) -> Option<Self>;
}

/// The one conversion path of exact values: into a floating type by the
/// correctly rounded `to_f64` of each part; into an exact type as an element
/// of `ℚ(ζ_1) = ℚ` or `ℚ(ζ_4) = ℚ(i)`, then [`Amplitude::from_cyclo`].
fn gaussian_into<S: Amplitude>(re: &BigRational, im: &BigRational) -> Option<S> {
    if !S::EXACT {
        let f = |q: &BigRational| q.to_f64().unwrap_or(f64::NAN);
        return float_into(C64::new(f(re), f(im)));
    }
    if Coeff::is_zero(im) {
        S::from_cyclo(&Cyclo::<1>::from_rational(re.clone()))
    } else {
        let z = Cyclo::<4>::from_rational(re.clone())
            .add_ref(&Cyclo::<4>::imaginary_unit().mul_ref(&Cyclo::<4>::from_rational(im.clone())));
        S::from_cyclo(&z)
    }
}

/// A finite floating value through [`Amplitude::from_c64`].
fn float_into<S: Amplitude>(z: C64) -> Option<S> {
    if z.re.is_finite() && z.im.is_finite() {
        S::from_c64(z)
    } else {
        None
    }
}

/// `(Re c, Im c)` when `c` is a Gaussian rational `a + bi`.
fn gaussian_parts<const N: u32>(c: &Cyclo<N>) -> Option<(BigRational, BigRational)> {
    let re = c.real_part();
    let d = c.sub_ref(&c.conj()); // 2i · Im c
    let im = if Coeff::is_zero(&d) {
        Cyclo::<N>::zero()
    } else {
        if N % 4 != 0 {
            return None;
        }
        // Im c = d / (2i) = −i d / 2
        let half = Cyclo::<N>::from_rational(BigRational::new(BigInt::from(1), BigInt::from(2)));
        d.mul_ref(&Cyclo::<N>::imaginary_unit().negated()).mul_ref(&half)
    };
    (re.is_rational() && im.is_rational()).then(|| (re.coeffs()[0].clone(), im.coeffs()[0].clone()))
}

impl<S: Amplitude> IntoAmplitude<S> for i64 {
    fn into_amplitude(&self) -> Option<S> {
        gaussian_into(&BigRational::from_integer(BigInt::from(*self)), &Coeff::zero())
    }
}
impl<S: Amplitude> IntoAmplitude<S> for BigInt {
    fn into_amplitude(&self) -> Option<S> {
        gaussian_into(&BigRational::from_integer(self.clone()), &Coeff::zero())
    }
}
impl<S: Amplitude> IntoAmplitude<S> for BigRational {
    fn into_amplitude(&self) -> Option<S> {
        gaussian_into(self, &Coeff::zero())
    }
}
impl<S: Amplitude> IntoAmplitude<S> for HybridRational {
    fn into_amplitude(&self) -> Option<S> {
        gaussian_into(&self.to_big(), &Coeff::zero())
    }
}
impl<S: Amplitude> IntoAmplitude<S> for Complex<BigRational> {
    fn into_amplitude(&self) -> Option<S> {
        gaussian_into(&self.re, &self.im)
    }
}
impl<S: Amplitude> IntoAmplitude<S> for Complex<HybridRational> {
    fn into_amplitude(&self) -> Option<S> {
        gaussian_into(&self.re.to_big(), &self.im.to_big())
    }
}
impl<S: Amplitude> IntoAmplitude<S> for f64 {
    fn into_amplitude(&self) -> Option<S> {
        float_into(C64::new(*self, 0.0))
    }
}
impl<S: Amplitude> IntoAmplitude<S> for Complex<f64> {
    fn into_amplitude(&self) -> Option<S> {
        float_into(*self)
    }
}
impl<const M: u32, S: Amplitude> IntoAmplitude<S> for Cyclo<M> {
    fn into_amplitude(&self) -> Option<S> {
        // a Gaussian rational goes the rational way (exact into every target
        // with `i`, correctly rounded into floats); anything else embeds when
        // `M | N` (or evaluates numerically for a floating target)
        if let Some((re, im)) = gaussian_parts(self) {
            return gaussian_into(&re, &im);
        }
        S::from_cyclo(self)
    }
}

impl FromAmplitude<C64> for f64 {
    fn from_amplitude(s: &C64) -> Option<Self> {
        (s.im == 0.0).then_some(s.re)
    }
}
impl FromAmplitude<C64> for Complex<f64> {
    fn from_amplitude(s: &C64) -> Option<Self> {
        Some(*s)
    }
}
impl FromAmplitude<C64> for BigRational {
    fn from_amplitude(_: &C64) -> Option<Self> {
        None
    }
}
impl FromAmplitude<C64> for HybridRational {
    fn from_amplitude(_: &C64) -> Option<Self> {
        None
    }
}
impl FromAmplitude<C64> for Complex<BigRational> {
    fn from_amplitude(_: &C64) -> Option<Self> {
        None
    }
}
impl FromAmplitude<C64> for Complex<HybridRational> {
    fn from_amplitude(_: &C64) -> Option<Self> {
        None
    }
}
impl<const M: u32> FromAmplitude<C64> for Cyclo<M> {
    fn from_amplitude(_: &C64) -> Option<Self> {
        None
    }
}

impl<const N: u32> FromAmplitude<Cyclo<N>> for f64 {
    fn from_amplitude(s: &Cyclo<N>) -> Option<Self> {
        s.is_real().then(|| s.to_complex().re)
    }
}
impl<const N: u32> FromAmplitude<Cyclo<N>> for Complex<f64> {
    fn from_amplitude(s: &Cyclo<N>) -> Option<Self> {
        Some(s.to_complex())
    }
}
impl<const N: u32> FromAmplitude<Cyclo<N>> for BigRational {
    fn from_amplitude(s: &Cyclo<N>) -> Option<Self> {
        s.is_rational().then(|| s.coeffs()[0].clone())
    }
}
impl<const N: u32> FromAmplitude<Cyclo<N>> for HybridRational {
    fn from_amplitude(s: &Cyclo<N>) -> Option<Self> {
        BigRational::from_amplitude(s).map(HybridRational::from_big)
    }
}
impl<const N: u32> FromAmplitude<Cyclo<N>> for Complex<BigRational> {
    fn from_amplitude(s: &Cyclo<N>) -> Option<Self> {
        gaussian_parts(s).map(|(re, im)| Complex::new(re, im))
    }
}
impl<const N: u32> FromAmplitude<Cyclo<N>> for Complex<HybridRational> {
    fn from_amplitude(s: &Cyclo<N>) -> Option<Self> {
        gaussian_parts(s)
            .map(|(re, im)| Complex::new(HybridRational::from_big(re), HybridRational::from_big(im)))
    }
}
impl<const M: u32, const N: u32> FromAmplitude<Cyclo<N>> for Cyclo<M> {
    fn from_amplitude(s: &Cyclo<N>) -> Option<Self> {
        s.into_amplitude()
    }
}

// ---------------------------------------------------------------------------
// words as Pauli operators
// ---------------------------------------------------------------------------

/// The operator of a word: the product of the images of its letters, left
/// to right, where `generator_map(v)` is the Pauli (on `n` qubits) of
/// generator `v` of `alg`, or `None` if that generator has no such image
/// (an error).  Underneath [`pauli_word`] and [`majorana_word`].
pub fn word_pauli(
    alg: &FreeAlgebra,
    w: &[Var],
    n: usize,
    generator_map: &dyn Fn(usize) -> Option<Pauli>,
) -> Result<Pauli> {
    let mut p = Pauli::identity(n);
    for &v in w {
        let v = v as usize;
        if v >= alg.num_vars() {
            return Err(SimError::Invalid(format!(
                "generator index {v} out of range for {alg}"
            )));
        }
        let g = generator_map(v).ok_or_else(|| {
            SimError::Invalid(format!(
                "generator {} of {alg} has no image on {n} qubits",
                alg.name(v as Var)
            ))
        })?;
        if g.num_qubits() != n {
            return Err(SimError::Invalid(format!(
                "image of generator {} acts on {} qubits, not {n}",
                alg.name(v as Var),
                g.num_qubits()
            )));
        }
        p.mul_assign(&g);
    }
    Ok(p)
}

/// The Pauli of a generator named `X3`, `Y0`, `Z12`, … (the naming of
/// [`ncalg::relations::pauli`]) on `n` qubits.
fn pauli_generator(name: &str, n: usize) -> Option<Pauli> {
    let mut chars = name.chars();
    let kind = match chars.next()? {
        'X' => PauliKind::X,
        'Y' => PauliKind::Y,
        'Z' => PauliKind::Z,
        _ => return None,
    };
    let q: usize = chars.as_str().parse().ok()?;
    (q < n).then(|| Pauli::single(n, q, kind))
}

/// The Pauli of a Majorana generator named `g5` (the naming of
/// [`ncalg::relations::majorana`]) on `n` qubits, if `5 < 2n`.
fn majorana_generator(name: &str, n: usize) -> Option<Pauli> {
    let k: usize = name.strip_prefix('g')?.parse().ok()?;
    (k < 2 * n).then(|| majorana_pauli(k, n))
}

/// The operator of a word of the Pauli algebra [`ncalg::relations::pauli`]
/// (generators `X0, Y0, Z0, X1, …`, any order of the letters) on `n` qubits:
/// for example `X0*X0*Z1` is `Z_1` and `X0*Z0` is `−i Y_0`.
pub fn pauli_word(alg: &FreeAlgebra, w: &[Var], n: usize) -> Result<Pauli> {
    word_pauli(alg, w, n, &|v| pauli_generator(alg.name(v as Var), n))
}

/// The Jordan–Wigner image of the Majorana operator `γ_k` on `n` qubits
/// (`k < 2n`): `γ_{2j} = Z_0 ⋯ Z_{j−1} X_j`, `γ_{2j+1} = Z_0 ⋯ Z_{j−1} Y_j`,
/// Hermitian with `γ_k γ_l + γ_l γ_k = 2 δ_kl`.
pub fn majorana_pauli(k: usize, n: usize) -> Pauli {
    assert!(k < 2 * n, "Majorana index {k} needs more than {n} qubits");
    let j = k / 2;
    let kind = if k % 2 == 0 { PauliKind::X } else { PauliKind::Y };
    let mut p = Pauli::single(n, j, kind);
    for q in 0..j {
        p.set_kind(q, PauliKind::Z);
    }
    p
}

/// The operator of a word of the Majorana algebra
/// [`ncalg::relations::majorana`] (generators `g0, g1, …`, `γ_k = gk`) on
/// `n` qubits through Jordan–Wigner ([`majorana_pauli`]).
pub fn majorana_word(alg: &FreeAlgebra, w: &[Var], n: usize) -> Result<Pauli> {
    word_pauli(alg, w, n, &|v| majorana_generator(alg.name(v as Var), n))
}

// ---------------------------------------------------------------------------
// polynomials as Pauli sums
// ---------------------------------------------------------------------------

impl<S: Amplitude> PauliSum<S> {
    /// The operator of a polynomial of the Pauli algebra
    /// [`ncalg::relations::pauli`] on `n` qubits (words need not be normal:
    /// every letter product is multiplied out, see [`pauli_word`]).  Fails
    /// on a generator that is not a Pauli on `n` qubits or a coefficient
    /// `S` cannot hold.
    pub fn from_polynomial<C: Coeff + IntoAmplitude<S>>(
        alg: &FreeAlgebra,
        p: &Polynomial<C>,
        n: usize,
    ) -> Result<Self> {
        Self::from_polynomial_with(alg, p, n, &|v| pauli_generator(alg.name(v as Var), n))
    }

    /// The operator of a polynomial of the Majorana algebra
    /// [`ncalg::relations::majorana`] on `n` qubits through Jordan–Wigner
    /// ([`majorana_pauli`]).
    pub fn from_majorana_polynomial<C: Coeff + IntoAmplitude<S>>(
        alg: &FreeAlgebra,
        p: &Polynomial<C>,
        n: usize,
    ) -> Result<Self> {
        Self::from_polynomial_with(alg, p, n, &|v| majorana_generator(alg.name(v as Var), n))
    }

    /// The operator of a polynomial under an arbitrary encoding of the
    /// generators as Paulis on `n` qubits (see [`word_pauli`]); the
    /// implementation under [`from_polynomial`](Self::from_polynomial) and
    /// [`from_majorana_polynomial`](Self::from_majorana_polynomial).
    pub fn from_polynomial_with<C: Coeff + IntoAmplitude<S>>(
        alg: &FreeAlgebra,
        p: &Polynomial<C>,
        n: usize,
        generator_map: &dyn Fn(usize) -> Option<Pauli>,
    ) -> Result<Self> {
        if !alg.same_as(p.algebra()) {
            return Err(SimError::Invalid(format!(
                "polynomial of {} given with the algebra {alg}",
                p.algebra()
            )));
        }
        let mut h = PauliSum::zero(n);
        for (w, c) in p.iter() {
            let c: S = c
                .into_amplitude()
                .ok_or_else(|| SimError::NotRepresentable(format!("coefficient {c}")))?;
            h.add_term(&word_pauli(alg, w, n, generator_map)?, &c);
        }
        Ok(h)
    }

    /// The polynomial of the Pauli algebra [`ncalg::relations::pauli`]
    /// (`alg` must have the generators `X0, Y0, Z0, X1, …`) in normal words:
    /// each term `c · ⊗_q σ_q` becomes `c` times the product of the letters
    /// in site order.  Fails when a coefficient is not representable in `C`
    /// (see [`FromAmplitude`]).
    pub fn to_polynomial<C: Coeff + FromAmplitude<S>>(&self, alg: &FreeAlgebra) -> Result<Polynomial<C>> {
        let mut terms = Vec::with_capacity(self.len());
        for (p, _) in self.terms() {
            // the Hermitian letter operator and its coefficient (`coefficient`
            // accounts for the phase i^{#Y} between X^x Z^z and ⊗σ)
            let herm = Pauli::from_kinds(&p.kinds());
            let c = self.coefficient(&herm);
            let c = C::from_amplitude(&c)
                .ok_or_else(|| SimError::NotRepresentable(format!("coefficient {c}")))?;
            let mut w = Vec::with_capacity(herm.weight());
            for q in herm.support() {
                let name = format!("{}{q}", herm.kind(q).symbol());
                let v = alg
                    .var_index(&name)
                    .ok_or_else(|| SimError::Invalid(format!("{alg} has no generator {name}")))?;
                w.push(v);
            }
            terms.push((Word::from_vec(w), c));
        }
        Ok(Polynomial::from_terms(alg.clone(), terms))
    }

    /// Parse a polynomial over the Pauli algebra on `n` qubits with ncalg's
    /// parser — `X0*X1 + 1/2*Y0*Y1 - i*Z0` — over `Complex<HybridRational>`
    /// coefficients, and convert.  Only rational coefficients and `i` can be
    /// written this way (a decimal such as `0.25` is read as the rational
    /// `1/4`); irrational values must be built in code.
    pub fn parse(n: usize, text: &str) -> Result<Self> {
        let (alg, _) = ncalg::relations::pauli::<Complex<HybridRational>>(n);
        let p: Polynomial<Complex<HybridRational>> =
            alg.parse(text).map_err(|e| SimError::Parse(e.to_string()))?;
        Self::from_polynomial(&alg, &p, n)
    }
}

// ---------------------------------------------------------------------------
// moments
// ---------------------------------------------------------------------------

/// The moment functional of a state on words of the Pauli algebra:
/// `L(w) = ⟨ψ|w|ψ⟩` for every `w` of `words` (see [`pauli_word`]).
pub fn moments<S: Amplitude, St: QuantumState<S>>(
    state: &St,
    alg: &FreeAlgebra,
    words: &[Word],
) -> Result<Vec<S>> {
    let n = state.num_qubits();
    moments_with(state, alg, words, &|v| pauli_generator(alg.name(v as Var), n))
}

/// The moment functional of a state on words of the Majorana algebra
/// through Jordan–Wigner (see [`majorana_word`]).
pub fn majorana_moments<S: Amplitude, St: QuantumState<S>>(
    state: &St,
    alg: &FreeAlgebra,
    words: &[Word],
) -> Result<Vec<S>> {
    let n = state.num_qubits();
    moments_with(state, alg, words, &|v| majorana_generator(alg.name(v as Var), n))
}

/// The moment functional under an arbitrary encoding of the generators as
/// Paulis (see [`word_pauli`]); the implementation under [`moments`] and
/// [`majorana_moments`].
pub fn moments_with<S: Amplitude, St: QuantumState<S>>(
    state: &St,
    alg: &FreeAlgebra,
    words: &[Word],
    generator_map: &dyn Fn(usize) -> Option<Pauli>,
) -> Result<Vec<S>> {
    use rayon::prelude::*;
    let n = state.num_qubits();
    // the words as Paulis (cheap, sequential: the map need not be Sync), then
    // the expectation values in parallel
    let paulis: Vec<Pauli> = words
        .iter()
        .map(|w| word_pauli(alg, w, n, generator_map))
        .collect::<Result<_>>()?;
    paulis.par_iter().map(|p| state.expectation(p)).collect()
}

/// `⟨ψ|p|ψ⟩` for a polynomial of the Pauli algebra
/// ([`PauliSum::from_polynomial`] then [`QuantumState::expectation_sum`]).
pub fn expectation_polynomial<S, St, C>(state: &St, alg: &FreeAlgebra, p: &Polynomial<C>) -> Result<S>
where
    S: Amplitude,
    St: QuantumState<S>,
    C: Coeff + IntoAmplitude<S>,
{
    let h = PauliSum::<S>::from_polynomial(alg, p, state.num_qubits())?;
    state.expectation_sum(&h)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ncalg::relations;

    fn q(a: i64, b: i64) -> BigRational {
        BigRational::new(BigInt::from(a), BigInt::from(b))
    }

    #[test]
    fn conversions_into_amplitudes() {
        // exact rationals into every cyclotomic field, approximately into floats
        let third = q(1, 3);
        let c: Cyclo<4> = third.into_amplitude().unwrap();
        assert_eq!(c, Cyclo::<4>::from_rational(q(1, 3)));
        let c6: Cyclo<6> = third.into_amplitude().unwrap();
        assert!(c6.is_rational());
        let f: C64 = third.into_amplitude().unwrap();
        assert!((f.re - 1.0 / 3.0).abs() < 1e-15 && f.im == 0.0);
        let h: Cyclo<8> = HybridRational::new(-5, 2).into_amplitude().unwrap();
        assert_eq!(h, Cyclo::<8>::from_rational(q(-5, 2)));
        assert_eq!(7i64.into_amplitude(), Some(Cyclo::<4>::from_i64(7)));
        // Gaussian rationals need i
        let z = Complex::new(q(1, 2), q(-3, 1));
        let c: Cyclo<4> = z.into_amplitude().unwrap();
        assert_eq!(
            c,
            Cyclo::<4>::from_rational(q(1, 2))
                .sub_ref(&Cyclo::<4>::imaginary_unit().mul_ref(&Cyclo::from_i64(3)))
        );
        assert!(<Complex<BigRational> as IntoAmplitude<Cyclo<6>>>::into_amplitude(&z).is_none());
        let zh = Complex::new(HybridRational::new(1, 2), HybridRational::new(-3, 1));
        assert_eq!(zh.into_amplitude(), Some(c.clone()));
        let f: C64 = zh.into_amplitude().unwrap();
        assert_eq!(f, C64::new(0.5, -3.0));
        // floats never into exact types
        assert!(<f64 as IntoAmplitude<Cyclo<4>>>::into_amplitude(&0.5).is_none());
        assert!(<C64 as IntoAmplitude<Cyclo<8>>>::into_amplitude(&C64::new(1.0, 0.0)).is_none());
        assert_eq!(0.5f64.into_amplitude(), Some(C64::new(0.5, 0.0)));
        assert!(<f64 as IntoAmplitude<C64>>::into_amplitude(&f64::NAN).is_none());
        // cyclotomic: embedding when M | N, descent of Gaussian rationals otherwise
        let r2 = Cyclo::<8>::sqrt2();
        let up: Cyclo<16> = r2.into_amplitude().unwrap();
        assert_eq!(up.mul_ref(&up), Cyclo::<16>::from_i64(2));
        assert!(<Cyclo<8> as IntoAmplitude<Cyclo<4>>>::into_amplitude(&r2).is_none());
        let i8 = Cyclo::<8>::imaginary_unit();
        let down: Cyclo<4> = i8.into_amplitude().unwrap();
        assert_eq!(down, Cyclo::<4>::imaginary_unit());
        let f: C64 = r2.into_amplitude().unwrap();
        assert!((f.re - std::f64::consts::SQRT_2).abs() < 1e-15);
    }

    #[test]
    fn conversions_from_amplitudes() {
        let r2 = Cyclo::<8>::sqrt2();
        let c = Cyclo::<8>::from_rational(q(2, 5))
            .add_ref(&Cyclo::<8>::imaginary_unit().mul_ref(&Cyclo::from_i64(-1)));
        assert_eq!(BigRational::from_amplitude(&r2), None);
        // √2 is evaluated numerically (ζ_8 + ζ_8^{-1}): a rounding error of an ulp
        assert!((f64::from_amplitude(&r2).unwrap() - std::f64::consts::SQRT_2).abs() < 1e-15);
        assert_eq!(f64::from_amplitude(&c), None);
        assert_eq!(
            Complex::<BigRational>::from_amplitude(&c),
            Some(Complex::new(q(2, 5), q(-1, 1)))
        );
        assert_eq!(
            Complex::<HybridRational>::from_amplitude(&c),
            Some(Complex::new(
                HybridRational::new(2, 5),
                HybridRational::new(-1, 1)
            ))
        );
        assert_eq!(Complex::<BigRational>::from_amplitude(&r2), None);
        assert_eq!(
            HybridRational::from_amplitude(&Cyclo::<8>::from_i64(4)),
            Some(HybridRational::new(4, 1))
        );
        let c4: Cyclo<4> = Cyclo::from_amplitude(&c).unwrap();
        assert_eq!(c4.to_complex(), c.to_complex());
        assert_eq!(<Cyclo<4> as FromAmplitude<Cyclo<8>>>::from_amplitude(&r2), None);
        let f = C64::new(0.25, -1.0);
        assert_eq!(C64::from_amplitude(&f), Some(f));
        assert_eq!(Complex::<f64>::from_amplitude(&c), Some(c.to_complex()));
        assert_eq!(f64::from_amplitude(&f), None);
        assert_eq!(f64::from_amplitude(&C64::new(0.25, 0.0)), Some(0.25));
        assert_eq!(BigRational::from_amplitude(&f), None);
        assert_eq!(Complex::<HybridRational>::from_amplitude(&f), None);
        assert_eq!(<Cyclo<4> as FromAmplitude<C64>>::from_amplitude(&f), None);
    }

    #[test]
    fn words_become_paulis() {
        let (alg, _) = relations::pauli::<Complex<HybridRational>>(3);
        let w = |s: &str| -> Word {
            let p: Polynomial<Complex<HybridRational>> = alg.parse(s).unwrap();
            p.leading_word().unwrap().clone()
        };
        assert_eq!(
            pauli_word(&alg, &w("X0*X0*Z1"), 3).unwrap(),
            Pauli::from_string(3, "Z1").unwrap()
        );
        assert_eq!(
            pauli_word(&alg, &w("X0*Z0"), 3).unwrap(),
            Pauli::from_string(3, "-i*Y0").unwrap()
        );
        assert_eq!(
            pauli_word(&alg, &w("Z2*Y1*X0"), 3).unwrap(),
            Pauli::from_string(3, "X0*Y1*Z2").unwrap()
        );
        assert_eq!(pauli_word(&alg, &Word::empty(), 3).unwrap(), Pauli::identity(3));
        // too few qubits for the generator
        assert!(matches!(pauli_word(&alg, &w("Z2"), 2), Err(SimError::Invalid(_))));
        // a non-Pauli algebra
        let other = FreeAlgebra::new(["a", "b"]);
        assert!(pauli_word(&other, &Word::letter(0), 1).is_err());
        assert!(pauli_word(&other, &Word::letter(5), 1).is_err());
    }

    #[test]
    fn majorana_jordan_wigner() {
        assert_eq!(majorana_pauli(0, 3), Pauli::from_string(3, "X0").unwrap());
        assert_eq!(majorana_pauli(1, 3), Pauli::from_string(3, "Y0").unwrap());
        assert_eq!(majorana_pauli(4, 3), Pauli::from_string(3, "Z0*Z1*X2").unwrap());
        assert_eq!(majorana_pauli(5, 3), Pauli::from_string(3, "Z0*Z1*Y2").unwrap());
        for k in 0..6 {
            assert!(majorana_pauli(k, 3).is_hermitian());
        }
        let (alg, _) = relations::majorana::<Complex<HybridRational>>(6);
        let w = Word::from_vec(vec![1, 4]);
        assert_eq!(
            majorana_word(&alg, &w, 3).unwrap(),
            majorana_pauli(1, 3).mul(&majorana_pauli(4, 3))
        );
        assert!(majorana_word(&alg, &Word::letter(5), 2).is_err());
    }

    #[test]
    fn parse_and_print_round_trip() {
        let h: PauliSum<Cyclo<4>> = PauliSum::parse(2, "X0*X1 + 1/2*Y0*Y1 - i*Z0 + 3").unwrap();
        assert_eq!(h.len(), 4);
        assert!(!h.is_hermitian());
        assert_eq!(
            h.coefficient(&Pauli::from_string(2, "Z0").unwrap()),
            Cyclo::<4>::imaginary_unit().negated()
        );
        let (alg, _) = relations::pauli::<Complex<HybridRational>>(2);
        let p: Polynomial<Complex<HybridRational>> = h.to_polynomial(&alg).unwrap();
        assert_eq!(p, alg.parse("X0*X1 + 1/2*Y0*Y1 - i*Z0 + 3").unwrap());
        // a non-normal word is multiplied out before printing: X0*Z0 = -i Y0
        let h2: PauliSum<Cyclo<4>> = PauliSum::parse(1, "X0*Z0").unwrap();
        let (alg1, _) = relations::pauli::<Complex<HybridRational>>(1);
        let p2: Polynomial<Complex<HybridRational>> = h2.to_polynomial(&alg1).unwrap();
        assert_eq!(p2, alg1.parse("-i*Y0").unwrap());
        // floats parse as rationals; irrational input does not parse
        let hf: PauliSum<C64> = PauliSum::parse(1, "0.25*X0").unwrap();
        assert_eq!(hf.coefficient(&Pauli::x(1, 0)), C64::new(0.25, 0.0));
        assert!(matches!(
            PauliSum::<C64>::parse(1, "X0 +"),
            Err(SimError::Parse(_))
        ));
        assert!(matches!(PauliSum::<C64>::parse(1, "X3"), Err(SimError::Parse(_))));
        // a float sum does not print into an exact polynomial
        let pf: Result<Polynomial<Complex<HybridRational>>> = hf.to_polynomial(&alg1);
        assert!(matches!(pf, Err(SimError::NotRepresentable(_))));
        let pf: Polynomial<Complex<f64>> = hf.to_polynomial(&alg1).unwrap();
        assert_eq!(pf.to_string(), "0.25*X0");
    }
}
