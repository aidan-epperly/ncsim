//! Linear combinations of Pauli operators: observables and Hamiltonians.

use super::Pauli;
use crate::bits::{self, Bits};
use crate::scalar::Amplitude;

use std::collections::BTreeMap;
use std::fmt;

/// `Σ_k c_k P_k` with the `P_k = X^x Z^z` phase-free and distinct (a phase
/// `i^k` of an inserted operator is folded into its coefficient), stored
/// sorted by `(x, z)` so that equality is structural.
#[derive(Clone, PartialEq)]
pub struct PauliSum<S: Amplitude> {
    n: usize,
    terms: BTreeMap<(Bits, Bits), S>,
}

impl<S: Amplitude> PauliSum<S> {
    /// The zero operator on `n` qubits.
    pub fn zero(n: usize) -> Self {
        PauliSum {
            n,
            terms: BTreeMap::new(),
        }
    }
    /// `c · 1`.
    pub fn scalar(n: usize, c: S) -> Self {
        Self::term(&Pauli::identity(n), c)
    }
    /// A single Pauli (with its phase).
    pub fn from_pauli(p: &Pauli) -> Self {
        Self::term(p, S::one())
    }
    /// `c · p`.
    pub fn term(p: &Pauli, c: S) -> Self {
        let mut s = Self::zero(p.num_qubits());
        s.add_term(p, &c);
        s
    }
    /// Number of qubits.
    pub fn num_qubits(&self) -> usize {
        self.n
    }
    /// Number of terms.
    pub fn len(&self) -> usize {
        self.terms.len()
    }
    /// No terms?
    pub fn is_empty(&self) -> bool {
        self.terms.is_empty()
    }
    /// Add `c · p` (the phase of `p` goes into the coefficient).
    pub fn add_term(&mut self, p: &Pauli, c: &S) {
        assert_eq!(p.num_qubits(), self.n, "term on a different register");
        let c = c.mul_ref(&S::i_pow(p.phase() as i64));
        if c.is_zero() {
            return;
        }
        let key = (p.x_bits().to_vec(), p.z_bits().to_vec());
        match self.terms.get_mut(&key) {
            Some(v) => {
                let abs = v.magnitude() + c.magnitude();
                v.add_assign_ref(&c);
                if S::canceled(v, abs) {
                    self.terms.remove(&key);
                }
            }
            None => {
                self.terms.insert(key, c);
            }
        }
    }
    /// The terms as `(phase-free Pauli, coefficient)`.
    pub fn terms(&self) -> impl Iterator<Item = (Pauli, &S)> + '_ {
        self.terms
            .iter()
            .map(move |((x, z), c)| (Pauli::new(self.n, x.clone(), z.clone(), 0), c))
    }
    /// The coefficient of `p` (so that `coefficient(p) · p` is the term;
    /// `p`'s phase is accounted for).
    pub fn coefficient(&self, p: &Pauli) -> S {
        let key = (p.x_bits().to_vec(), p.z_bits().to_vec());
        match self.terms.get(&key) {
            Some(c) => c.mul_ref(&S::i_pow(-(p.phase() as i64))),
            None => S::zero(),
        }
    }
    /// `self += other`.
    pub fn add_assign(&mut self, other: &Self) {
        assert_eq!(self.n, other.n);
        for (p, c) in other.terms() {
            self.add_term(&p, c);
        }
    }
    /// `self += c · other`.
    pub fn add_scaled(&mut self, other: &Self, c: &S) {
        assert_eq!(self.n, other.n);
        for (p, d) in other.terms() {
            self.add_term(&p, &d.mul_ref(c));
        }
    }
    /// `self + other`.
    pub fn add(&self, other: &Self) -> Self {
        let mut s = self.clone();
        s.add_assign(other);
        s
    }
    /// `self − other`.
    pub fn sub(&self, other: &Self) -> Self {
        let mut s = self.clone();
        s.add_scaled(other, &S::one().negated());
        s
    }
    /// `c · self`.
    pub fn scaled(&self, c: &S) -> Self {
        let mut s = Self::zero(self.n);
        s.add_scaled(self, c);
        s
    }
    /// The operator product `self · other`.
    pub fn mul(&self, other: &Self) -> Self {
        assert_eq!(self.n, other.n);
        let mut s = Self::zero(self.n);
        for (p, c) in self.terms() {
            for (q, d) in other.terms() {
                s.add_term(&p.mul(&q), &c.mul_ref(d));
            }
        }
        s
    }
    /// `[self, other]`.
    pub fn commutator(&self, other: &Self) -> Self {
        self.mul(other).sub(&other.mul(self))
    }
    /// The adjoint `Σ conj(c_k) P_k†`.
    pub fn adjoint(&self) -> Self {
        let mut s = Self::zero(self.n);
        for (p, c) in self.terms() {
            s.add_term(&p.adjoint(), &c.conj());
        }
        s
    }
    /// Hermitian?  (`conj(c) (−1)^{x·z} = c` for every term, up to the
    /// rounding of floating coefficients.)
    pub fn is_hermitian(&self) -> bool {
        self.terms.iter().all(|((x, z), c)| {
            let mut want = c.conj();
            if bits::dot(x, z) {
                want = want.negated();
            }
            let d = c.sub_ref(&want);
            S::canceled(&d, 2.0 * c.magnitude())
        })
    }
    /// Apply a map to every coefficient (dropping zeros).
    pub fn map_coeffs<T: Amplitude>(&self, f: impl Fn(&S) -> T) -> PauliSum<T> {
        let mut s = PauliSum::zero(self.n);
        for (p, c) in self.terms() {
            s.add_term(&p, &f(c));
        }
        s
    }
    /// The dense matrix (row-major, little-endian), for checks on few qubits.
    pub fn matrix(&self) -> Vec<S> {
        let dim = 1usize << self.n;
        let mut m = vec![S::zero(); dim * dim];
        for (p, c) in self.terms() {
            let pm: Vec<S> = p.matrix();
            for (a, b) in m.iter_mut().zip(pm) {
                if !b.is_zero() {
                    a.add_assign_ref(&b.mul_ref(c));
                }
            }
        }
        m
    }
    /// Largest `|c_k|` (0 if empty).
    pub fn max_coefficient(&self) -> f64 {
        self.terms.values().map(|c| c.magnitude()).fold(0.0, f64::max)
    }
    /// Is every term Hermitian as an operator and diagonal (`Z`-type)?
    pub fn is_diagonal(&self) -> bool {
        self.terms.keys().all(|(x, _)| bits::is_zero(x))
    }
}

impl<S: Amplitude> fmt::Display for PauliSum<S> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.terms.is_empty() {
            return write!(f, "0");
        }
        let mut first = true;
        for (p, c) in self.terms() {
            if !first {
                write!(f, " + ")?;
            }
            first = false;
            // letter form: X^x Z^z = i^{-k} ⊗σ with k the number of Y letters
            let k = (bits::and_count(p.x_bits(), p.z_bits()) % 4) as u8;
            let (p, c) = (p.with_phase_added(k), c.mul_ref(&S::i_pow(-(k as i64))));
            if c.is_one() && !p.is_identity() {
                write!(f, "{}", p)?;
            } else if c.needs_parens_for_display() {
                write!(f, "(")?;
                c.fmt_display(f)?;
                write!(f, ")")?;
                if !p.is_identity() {
                    write!(f, "*{}", p)?;
                }
            } else {
                c.fmt_display(f)?;
                if !p.is_identity() {
                    write!(f, "*{}", p)?;
                }
            }
        }
        Ok(())
    }
}

impl<S: Amplitude> fmt::Debug for PauliSum<S> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "PauliSum({} qubits: {})", self.n, self)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::scalar::C64;
    use ncalg::cyclo::Cyclo;
    use ncalg::{Coeff, ComplexCoeff};

    #[test]
    fn sums_and_products() {
        let x = Pauli::parse("X0").unwrap();
        let z = Pauli::parse("Z0").unwrap();
        let hx: PauliSum<Cyclo<4>> = PauliSum::from_pauli(&x);
        let hz: PauliSum<Cyclo<4>> = PauliSum::from_pauli(&z);
        let h = hx.add(&hz);
        assert_eq!(h.len(), 2);
        assert!(h.is_hermitian());
        let h2 = h.mul(&h);
        // (X + Z)^2 = 2 + XZ + ZX = 2
        assert_eq!(h2, PauliSum::scalar(1, Cyclo::from_i64(2)));
        let c = hx.commutator(&hz);
        // [X, Z] = XZ − ZX = −2iY
        let y = Pauli::parse("Y0").unwrap();
        assert_eq!(
            c,
            PauliSum::term(
                &y,
                Cyclo::<4>::imaginary_unit()
                    .negated()
                    .mul_ref(&Cyclo::from_i64(2))
            )
        );
        assert!(!c.is_hermitian());
        assert!(c.adjoint().sub(&c.scaled(&Cyclo::from_i64(-1))).is_empty());
        let f: PauliSum<C64> = h.map_coeffs(|c| c.to_complex());
        assert_eq!(f.to_string(), "Z0 + X0");
        let fc: PauliSum<C64> = c.map_coeffs(|c| c.to_complex());
        assert_eq!(fc.to_string(), "-2i*Y0");
    }
}
