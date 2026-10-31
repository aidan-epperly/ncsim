//! Sums of stabilizer states: stabilizer-rank simulation of non-Clifford
//! circuits.
//!
//! `|ψ⟩ = Σ_j |φ_j⟩` with every term an exact stabilizer vector with its
//! coefficient, of a family `T: `[`StabilizerTerm`]: a Clifford-class
//! [`PathSum`] in normal form (the default), or a [`ChTerm`](super::ChTerm),
//! a CH-form with a coefficient, whose Clifford gates cost `O(n/64)` (`S`,
//! `CZ`, `CX`) and `O(n²/64)` (`H`) at any size; prefer it on large
//! registers.  Clifford gates act term by term (in parallel).  A non-Clifford diagonal
//! gate is a product of factors `I + (ζ^c − 1) Π_i P_{λ_i}` over its
//! non-Clifford terms `c Π_i [λ_i·x]`, where `P_λ = (1 − Z^λ)/2` projects onto
//! `[λ·x] = 1`; each factor at most doubles the number of terms (`T`, `CS`,
//! `CCZ`, `Phase(k, r)`: exactly two).  Non-diagonal non-Clifford gates use
//! their decomposition (`CCX = H·CCZ·H`).
//!
//! Observables are sums over pairs of terms of exact stabilizer inner
//! products (closed-form Gauss sums), `O(χ²)` for `χ` terms, summed pairwise
//! for stability; each term is prepared once (for CH-form terms: its affine
//! form, with Paulis applied to it directly).  Terms with the same canonical affine form are merged by
//! [`StabilizerSum::merge_duplicates`].
//!
//! The rank grows as `2^t` with the number `t` of non-Clifford factors.  For
//! `T` gates, [`StabilizerSum::run_with_t_gadgets`] injects magic states
//! instead: ancillas in `|T⟩^{⊗t}` as `2^{⌈t/2⌉}` terms (pairs
//! `½(|00⟩ + i|11⟩) + (ζ_8/2)(|01⟩ + |10⟩)`), `CX` and a postselected `Z`
//! measurement per gate, so observables cost `2^t` pairs instead of `4^t`.
//! The better decompositions (`|T⟩^{⊗6}` of rank 7, Bravyi–Gosset
//! sparsification) are a possible extension on the same representation.

use super::pathsum::PathSum;
use super::stabilizer::{AffineForm, Stabilizer};
use super::QuantumState;
use crate::bits::{self, Bits};
use crate::circuit::{Circuit, Op};
use crate::error::{check_qubits, unsupported, Result, SimError};
use crate::gate::{Gate, MonomialGate};
use crate::pauli::Pauli;
use crate::phasepoly::PhasePoly;
use crate::scalar::{Amplitude, C64};
use rayon::prelude::*;
use std::collections::HashMap;
use std::fmt;

const BACKEND: &str = "StabilizerSum";

/// A stabilizer vector with an exact coefficient: the terms of a
/// [`StabilizerSum`].  Implemented by Clifford-class [`PathSum`]s (the
/// default) and by [`ChTerm`](super::ChTerm) (a CH-form with a coefficient: `O(n/64)` per
/// C-type gate and `O(n²/64)` per `H` at any size).
pub trait StabilizerTerm<S: Amplitude>: QuantumState<S> {
    /// `|x⟩`.
    fn basis_term(n: usize, x: &[u64]) -> Self;
    /// `c ·` the vector of `af` (`2^{-k/2} i^{Q(u)}` on `shift ⊕ Bu`).
    fn from_canonical(af: &AffineForm, c: S) -> Result<Self>;
    /// `(af, c)` with this term `= c ·` the vector of `af`; proportional terms
    /// give the same `af` (the key of [`StabilizerSum::merge_duplicates`]).
    fn canonical(&self) -> Result<(AffineForm, S)>;
    /// `⟨self|P|other⟩`.
    fn matrix_element(&self, p: &Pauli, other: &Self) -> Result<S>;
    /// What observables over pairs of terms reuse (computed once per term).
    type Prepared: Send + Sync;
    /// Prepare this term for [`StabilizerTerm::prepared_element`].
    fn prepare(&self) -> Result<Self::Prepared>;
    /// `⟨a|P|b⟩` (`P = None`: `⟨a|b⟩`) from prepared terms.
    fn prepared_element(a: &Self::Prepared, p: Option<&Pauli>, b: &Self::Prepared) -> Result<S>;
    /// Is the term a stabilizer vector (for path sums: no non-Clifford part)?
    fn is_stabilizer_term(&self) -> bool;
    /// Is the term zero?
    fn is_zero_term(&self) -> bool;
}

impl<S: Amplitude> StabilizerTerm<S> for PathSum<S> {
    fn basis_term(n: usize, x: &[u64]) -> Self {
        PathSum::basis_state(n, x)
    }
    fn from_canonical(af: &AffineForm, c: S) -> Result<Self> {
        let mut t = PathSum::from_affine_form(af)?;
        t.scale(&c)?;
        Ok(t)
    }
    fn canonical(&self) -> Result<(AffineForm, S)> {
        let af = self.to_affine_form()?;
        let base: PathSum<S> = PathSum::from_affine_form(&af)?;
        Ok((af, self.scalar().div_ref(base.scalar())))
    }
    fn matrix_element(&self, p: &Pauli, other: &Self) -> Result<S> {
        PathSum::matrix_element(self, p, other)
    }
    type Prepared = PathSum<S>;
    fn prepare(&self) -> Result<PathSum<S>> {
        Ok(self.clone())
    }
    fn prepared_element(a: &PathSum<S>, p: Option<&Pauli>, b: &PathSum<S>) -> Result<S> {
        match p {
            Some(p) => a.matrix_element(p, b),
            None => a.inner(b),
        }
    }
    fn is_stabilizer_term(&self) -> bool {
        self.is_clifford()
    }
    fn is_zero_term(&self) -> bool {
        self.is_zero()
    }
}

/// A linear combination of stabilizer states (see the [module documentation](self)),
/// with terms of family `T` (Clifford-class path sums by default;
/// [`ChTerm`](super::ChTerm) for large registers).
#[derive(Clone)]
pub struct StabilizerSum<S: Amplitude, T: StabilizerTerm<S> = PathSum<S>> {
    n: usize,
    terms: Vec<T>,
    _s: std::marker::PhantomData<S>,
}

impl<S: Amplitude, T: StabilizerTerm<S>> StabilizerSum<S, T> {
    fn with_terms(n: usize, terms: Vec<T>) -> Self {
        StabilizerSum {
            n,
            terms,
            _s: std::marker::PhantomData,
        }
    }
    /// `|0…0⟩`.
    pub fn zero_state(n: usize) -> Self {
        Self::with_terms(n, vec![T::basis_term(n, &bits::zero(n))])
    }
    /// `|x⟩`.
    pub fn basis_state(n: usize, x: &[u64]) -> Self {
        Self::with_terms(n, vec![T::basis_term(n, x)])
    }
    /// `|+⟩^{⊗n}`.
    pub fn plus_state(n: usize) -> Result<Self> {
        let af = AffineForm::from_basis(
            n,
            bits::zero(n),
            (0..n).map(|q| bits::unit(n, q)).collect(),
            crate::phasepoly::QuadForm::new(n),
        )?;
        Ok(Self::with_terms(n, vec![T::from_canonical(&af, S::one())?]))
    }
    /// A single stabilizer state (its canonical representative).
    pub fn from_stabilizer(s: &Stabilizer) -> Result<Self> {
        Ok(Self::with_terms(
            s.num_qubits(),
            vec![T::from_canonical(&s.affine_form(), S::one())?],
        ))
    }
    /// From stabilizer terms.
    pub fn from_terms(n: usize, terms: Vec<T>) -> Result<Self> {
        for t in &terms {
            if t.num_qubits() != n {
                return Err(SimError::Invalid("term on the wrong register".into()));
            }
            if !t.is_stabilizer_term() {
                return Err(SimError::Invalid("term is not a stabilizer state".into()));
            }
        }
        let mut s = Self::with_terms(n, terms);
        s.prune();
        Ok(s)
    }
    /// Number of qubits.
    pub fn num_qubits(&self) -> usize {
        self.n
    }
    /// Number of terms (the current stabilizer rank bound).
    pub fn num_terms(&self) -> usize {
        self.terms.len()
    }
    /// The terms.
    pub fn terms(&self) -> &[T] {
        &self.terms
    }
    /// Add a term.
    pub fn push(&mut self, term: T) -> Result<()> {
        if term.num_qubits() != self.n || !term.is_stabilizer_term() {
            return Err(SimError::Invalid(
                "push: not a stabilizer state on this register".into(),
            ));
        }
        if !term.is_zero_term() {
            self.terms.push(term);
        }
        Ok(())
    }
    /// Drop vanished terms.
    fn prune(&mut self) {
        self.terms.retain(|t| !t.is_zero_term());
    }
    /// `|T⟩^{⊗t}` (`|T⟩ = (|0⟩ + ζ_8|1⟩)/√2`) as a sum of `2^{⌈t/2⌉}` stabilizer
    /// terms: each pair is `½(|00⟩ + i|11⟩) + (ζ_8/2)(|01⟩ + |10⟩)`, a single
    /// one `(|0⟩ + ζ_8|1⟩)/√2`.
    pub fn magic_t(t: usize) -> Result<Self> {
        use crate::phasepoly::QuadForm;
        let half = S::sqrt2_pow(-1).ok_or_else(|| SimError::NotRepresentable("1/√2".into()))?;
        let w = S::root_of_unity(1, 3).ok_or_else(|| SimError::NotRepresentable("ζ_8".into()))?;
        let pair = || -> Result<Self> {
            let mut q = QuadForm::new(1);
            q.add_linear(0, 1);
            let even = AffineForm::from_basis(2, bits::zero(2), vec![bits::from_u64(3, 2)], q)?;
            let odd = AffineForm::from_basis(
                2,
                bits::from_u64(1, 2),
                vec![bits::from_u64(3, 2)],
                QuadForm::new(1),
            )?;
            // (|00⟩ + i|11⟩)/√2 · 1/√2  +  (|01⟩ + |10⟩)/√2 · ζ_8/√2
            Ok(Self::with_terms(
                2,
                vec![
                    T::from_canonical(&even, half.clone())?,
                    T::from_canonical(&odd, w.mul_ref(&half))?,
                ],
            ))
        };
        let single = || -> Result<Self> {
            let zero = AffineForm::from_basis(1, bits::zero(1), Vec::new(), QuadForm::new(0))?;
            let one = AffineForm::from_basis(1, bits::from_u64(1, 1), Vec::new(), QuadForm::new(0))?;
            Ok(Self::with_terms(
                1,
                vec![
                    T::from_canonical(&zero, half.clone())?,
                    T::from_canonical(&one, w.mul_ref(&half))?,
                ],
            ))
        };
        let mut parts = Vec::new();
        for _ in 0..t / 2 {
            parts.push(pair()?);
        }
        if t % 2 == 1 {
            parts.push(single()?);
        }
        if parts.is_empty() {
            return Err(SimError::Invalid("magic_t of no qubits".into()));
        }
        QuantumState::tensor_all(&parts)
    }

    /// Run a circuit whose `T` and `T†` gates are replaced by magic-state
    /// gadgets: `T` on `q` is `CX(q → a)` with an ancilla `a` in `|T⟩`, then
    /// the projection of `Z_a` onto `+1` (and a factor `√2`), so the result is
    /// exactly `C|0…0⟩ ⊗ |0…0⟩_ancillas`; `T† = S† T`.  The ancillas start in
    /// [`StabilizerSum::magic_t`], `2^{⌈t/2⌉}` terms instead of the `2^t` of
    /// projector factors.  Returns the state on `n + t` qubits (ancillas after
    /// the register) and `t`.  Other non-Clifford gates are applied as usual.
    pub fn run_with_t_gadgets(circuit: &Circuit, rng: &mut dyn rand::RngCore) -> Result<(Self, usize)> {
        let n = circuit.num_qubits();
        let t = circuit
            .ops()
            .iter()
            .filter(|op| {
                matches!(
                    op,
                    Op::Gate {
                        gate: Gate::T | Gate::Tdg,
                        ..
                    }
                )
            })
            .count();
        let mut state = if t == 0 {
            Self::zero_state(n)
        } else {
            QuantumState::tensor(&Self::zero_state(n), &Self::magic_t(t)?)?
        };
        let total = n + t;
        let r2 = S::sqrt2_pow(1).ok_or_else(|| SimError::NotRepresentable("√2".into()))?;
        let mut next = n;
        for op in circuit.ops() {
            match op {
                Op::Gate {
                    gate: g @ (Gate::T | Gate::Tdg),
                    qubits,
                } => {
                    let (q, a) = (qubits[0], next);
                    next += 1;
                    state.apply_gate(&Gate::Cx, &[q, a])?;
                    state.project(&Pauli::z(total, a), false)?;
                    state.scale(&r2)?;
                    if matches!(g, Gate::Tdg) {
                        state.apply_gate(&Gate::Sdg, &[q])?;
                    }
                }
                Op::Gate { gate, qubits } => state.apply_gate(gate, qubits)?,
                Op::Measure { pauli } => {
                    let p = if t == 0 {
                        pauli.clone()
                    } else {
                        pauli.tensor(&Pauli::identity(t))
                    };
                    state.measure(&p, rng)?;
                }
                // resets and noise name register qubits, valid on the larger register
                other => {
                    state.apply_op(other, rng)?;
                }
            }
        }
        Ok((state, t))
    }

    /// Estimate `⟨ψ|ψ⟩` from `samples` uniformly random stabilizer states `φ`
    /// (Bravyi–Gosset): stabilizer states form a 3-design, so
    /// `2^n E|⟨φ|ψ⟩|² = ⟨ψ|ψ⟩` and the relative standard deviation is at most
    /// `√(2/samples)`.  Cost `O(samples · χ)` affine-form inner products,
    /// instead of the `χ²` of the exact [`QuantumState::norm_sq`]: worth it
    /// when `χ` is much larger than `samples`.
    pub fn estimate_norm_sq(&self, samples: usize, rng: &mut dyn rand::RngCore) -> Result<f64> {
        let phis = random_affine_forms(self.n, samples, rng);
        let (est, _) = self.estimate_with(&phis, None)?;
        Ok(est)
    }

    /// Estimate `⟨ψ|P|ψ⟩ / ⟨ψ|ψ⟩` for a Hermitian Pauli: the norms of
    /// `(1 ± P)ψ / 2` estimated with the same random stabilizer states.
    pub fn estimate_expectation(
        &self,
        p: &Pauli,
        samples: usize,
        rng: &mut dyn rand::RngCore,
    ) -> Result<f64> {
        if !p.is_hermitian() {
            return Err(SimError::Invalid(format!(
                "estimate_expectation of non-Hermitian {p}"
            )));
        }
        let phis = random_affine_forms(self.n, samples, rng);
        let (plus, minus) = self.estimate_with(&phis, Some(p))?;
        let total = plus + minus;
        if total <= 0.0 {
            return Err(SimError::ZeroState("estimate_expectation".into()));
        }
        Ok((plus - minus) / total)
    }

    /// `2^n · mean |⟨φ|χ⟩|²` over the samples, for `χ = ψ` (`p = None`; the
    /// second value is 0) or for `χ = (1 ± P)ψ/2` (both signs).
    fn estimate_with(&self, phis: &[AffineForm], p: Option<&Pauli>) -> Result<(f64, f64)> {
        // each term as c · (affine form), and P · term for the ± branches
        let prepared: Vec<(AffineForm, C64, Option<(AffineForm, C64)>)> = self
            .terms
            .par_iter()
            .map(|t| {
                let (af, c) = t.canonical()?;
                let c = c.to_c64();
                let pt = p.map(|p| {
                    let (w, paf) = af.apply_pauli(p);
                    (paf, c * w.to_c64())
                });
                Ok((af, c, pt))
            })
            .collect::<Result<_>>()?;
        let scale = 2f64.powi(self.n as i32) / phis.len() as f64;
        let per_sample: Vec<(f64, f64)> = phis
            .par_iter()
            .map(|phi| {
                let mut a = C64::new(0.0, 0.0);
                let mut b = C64::new(0.0, 0.0);
                for (af, c, pt) in &prepared {
                    let o = c * phi.inner(af).to_c64();
                    match pt {
                        None => a += o,
                        Some((paf, pc)) => {
                            let po = pc * phi.inner(paf).to_c64();
                            a += (o + po) * 0.5;
                            b += (o - po) * 0.5;
                        }
                    }
                }
                (a.norm_sqr(), b.norm_sqr())
            })
            .collect();
        let (sa, sb) = per_sample
            .iter()
            .fold((0.0, 0.0), |acc, x| (acc.0 + x.0, acc.1 + x.1));
        Ok((sa * scale, sb * scale))
    }

    /// Merge terms with the same canonical affine form (equal states up to
    /// their scalar): exact, `O(χ)` hashing.
    pub fn merge_duplicates(&mut self) -> Result<()> {
        let keyed: Result<Vec<(AffineForm, S)>> = self.terms.par_iter().map(|t| t.canonical()).collect();
        let mut index: HashMap<AffineForm, usize> = HashMap::new();
        let mut merged: Vec<(AffineForm, S)> = Vec::with_capacity(self.terms.len());
        for (af, c) in keyed? {
            match index.get(&af) {
                Some(&i) => merged[i].1 = merged[i].1.add_ref(&c),
                None => {
                    index.insert(af.clone(), merged.len());
                    merged.push((af, c));
                }
            }
        }
        let terms: Result<Vec<T>> = merged
            .par_iter()
            .map(|(af, c)| T::from_canonical(af, c.clone()))
            .collect();
        self.terms = terms?;
        self.prune();
        Ok(())
    }
    /// Apply a Clifford gate to every term.
    pub fn apply_clifford(&mut self, gate: &Gate, qubits: &[usize]) -> Result<()> {
        self.terms
            .par_iter_mut()
            .try_for_each(|t| t.apply_gate(gate, qubits))
    }
    /// Multiply by `I + c · Π_i P_{λ_i}` with `P_λ` the projector onto
    /// `[λ·x] = 1` (`λ` over the gate's qubits): the terms are kept and the
    /// projected, scaled copies appended.
    fn apply_projector_factor(&mut self, parities: &[Bits], qubits: &[usize], c: &S) -> Result<()> {
        let n = self.n;
        let zs: Vec<Pauli> = parities
            .iter()
            .map(|lam| {
                let mut z = bits::zero(n);
                for j in bits::ones(lam) {
                    if j < qubits.len() {
                        bits::set(&mut z, qubits[j], true);
                    }
                }
                Pauli::z_string(n, &z)
            })
            .collect();
        let extra: Result<Vec<T>> = self
            .terms
            .par_iter()
            .map(|t| {
                let mut u = t.clone();
                for z in &zs {
                    u.project(z, true)?;
                    if u.is_zero_term() {
                        break;
                    }
                }
                if !u.is_zero_term() {
                    u.scale(c)?;
                }
                Ok(u)
            })
            .collect();
        for u in extra? {
            if !u.is_zero_term() {
                self.terms.push(u);
            }
        }
        Ok(())
    }
    /// Apply a monomial gate: its non-Clifford diagonal terms as projector
    /// factors, then the Clifford remainder term by term.
    pub fn apply_monomial(&mut self, g: &MonomialGate, qubits: &[usize]) -> Result<()> {
        let k = qubits.len();
        if k != g.num_qubits() {
            return Err(SimError::Arity {
                gate: "monomial".into(),
                expected: k,
                got: qubits.len(),
            });
        }
        let mut phase = g.phase().clone();
        if phase.bits() < 2 {
            phase.raise_bits(2);
        }
        let (q, rest) = phase.split_clifford();
        let bits_r = rest.bits();
        for (key, c) in rest.terms() {
            let z = S::root_of_unity(c as i64, bits_r)
                .ok_or_else(|| SimError::NotRepresentable(format!("ζ_{}^{c}", 1u64 << bits_r)))?;
            let factor = z.sub_ref(&S::one());
            self.apply_projector_factor(key, qubits, &factor)?;
        }
        let cliff_phase = q.to_phase_poly(2);
        let cliff = MonomialGate::new(g.rows().to_vec(), g.shift().to_vec(), cliff_phase, g.scalar())?;
        let const_phase = rest.constant();
        let gate = Gate::Monomial(cliff);
        self.terms
            .par_iter_mut()
            .try_for_each(|t| t.apply_gate(&gate, qubits))?;
        if const_phase != 0 {
            let z = S::root_of_unity(const_phase as i64, bits_r)
                .ok_or_else(|| SimError::NotRepresentable(format!("ζ_{}^{const_phase}", 1u64 << bits_r)))?;
            self.scale(&z)?;
        }
        Ok(())
    }
    /// `Σ_{j,k} ⟨φ_j| P |φ_k⟩` (identity for `None`), using Hermitian symmetry.
    fn pair_sum(&self, p: Option<&Pauli>) -> Result<S> {
        let prepared: Result<Vec<T::Prepared>> = self.terms.par_iter().map(|t| t.prepare()).collect();
        let prepared = prepared?;
        let per_j: Result<Vec<S>> = (0..prepared.len())
            .into_par_iter()
            .map(|j| {
                let mut acc = Vec::with_capacity(prepared.len() - j);
                acc.push(T::prepared_element(&prepared[j], p, &prepared[j])?);
                for k in j + 1..prepared.len() {
                    // ⟨φ_j|P|φ_k⟩ + ⟨φ_k|P|φ_j⟩ = z + conj(z) for Hermitian P
                    let v = T::prepared_element(&prepared[j], p, &prepared[k])?;
                    acc.push(v.add_ref(&v.conj()));
                }
                Ok(S::sum(&acc))
            })
            .collect();
        Ok(S::sum(&per_j?))
    }
}

impl<S: Amplitude, T: StabilizerTerm<S>> QuantumState<S> for StabilizerSum<S, T> {
    const BACKEND: &'static str = BACKEND;

    fn num_qubits(&self) -> usize {
        self.n
    }

    fn apply_gate(&mut self, gate: &Gate, qubits: &[usize]) -> Result<()> {
        check_qubits(qubits, self.n)?;
        if qubits.len() != gate.arity() {
            return Err(SimError::Arity {
                gate: gate.name(),
                expected: gate.arity(),
                got: qubits.len(),
            });
        }
        if gate.is_clifford() {
            return self.apply_clifford(gate, qubits);
        }
        if let Some(m) = gate.monomial() {
            return self.apply_monomial(&m, qubits);
        }
        if let Some(c) = gate.decomposition() {
            for op in c.ops() {
                if let Op::Gate { gate: g, qubits: qs } = op {
                    let mapped: Vec<usize> = qs.iter().map(|&q| qubits[q]).collect();
                    self.apply_gate(g, &mapped)?;
                }
            }
            return Ok(());
        }
        Err(unsupported(BACKEND, format!("gate {gate}")))
    }

    fn norm_sq(&self) -> Result<S> {
        self.pair_sum(None)
    }

    fn amplitude(&self, x: &[u64]) -> Result<S> {
        let amps: Result<Vec<S>> = self.terms.par_iter().map(|t| t.amplitude(x)).collect();
        Ok(S::sum(&amps?))
    }

    fn expectation(&self, p: &Pauli) -> Result<S> {
        if p.num_qubits() != self.n {
            return Err(SimError::Invalid(
                "expectation: operator on the wrong register".into(),
            ));
        }
        if !p.is_hermitian() {
            return Err(SimError::Invalid(format!(
                "expectation of non-Hermitian {p}: use a PauliSum"
            )));
        }
        self.pair_sum(Some(p))
    }

    fn inner(&self, other: &Self) -> Result<S> {
        if other.n != self.n {
            return Err(SimError::Invalid("inner: register sizes differ".into()));
        }
        let per_j: Result<Vec<S>> = self
            .terms
            .par_iter()
            .map(|a| {
                let v: Result<Vec<S>> = other.terms.iter().map(|b| a.inner(b)).collect();
                Ok(S::sum(&v?))
            })
            .collect();
        Ok(S::sum(&per_j?))
    }

    fn project(&mut self, p: &Pauli, outcome: bool) -> Result<S> {
        self.terms
            .par_iter_mut()
            .try_for_each(|t| t.project(p, outcome).map(|_| ()))?;
        self.prune();
        self.norm_sq()
    }

    fn scale(&mut self, c: &S) -> Result<()> {
        for t in self.terms.iter_mut() {
            t.scale(c)?;
        }
        Ok(())
    }

    fn tensor(&self, other: &Self) -> Result<Self> {
        let terms: Result<Vec<T>> = self
            .terms
            .par_iter()
            .flat_map_iter(|a| other.terms.iter().map(move |b| a.tensor(b)))
            .collect();
        Ok(Self::with_terms(self.n + other.n, terms?))
    }
}

impl<S: Amplitude, T: StabilizerTerm<S>> fmt::Debug for StabilizerSum<S, T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "StabilizerSum({} qubits, {} terms)", self.n, self.terms.len())
    }
}

/// `samples` uniformly random stabilizer states on `n` qubits, as affine forms.
fn random_affine_forms(n: usize, samples: usize, rng: &mut dyn rand::RngCore) -> Vec<AffineForm> {
    use rand::SeedableRng;
    let seeds: Vec<u64> = (0..samples).map(|_| rng.next_u64()).collect();
    seeds
        .par_iter()
        .map(|&s| {
            let mut r = rand::rngs::StdRng::seed_from_u64(s);
            crate::clifford::random_stabilizer_state(n, &mut r).affine_form()
        })
        .collect()
}

/// A phase polynomial's non-Clifford terms as projector factors, for callers
/// that build their own gates: `(parities, ζ^c − 1)` per term.
pub fn projector_factors<S: Amplitude>(phase: &PhasePoly) -> Result<Vec<(Vec<Bits>, S)>> {
    let mut p = phase.clone();
    if p.bits() < 2 {
        p.raise_bits(2);
    }
    let (_, rest) = p.split_clifford();
    rest.terms()
        .map(|(k, c)| {
            let z = S::root_of_unity(c as i64, rest.bits())
                .ok_or_else(|| SimError::NotRepresentable(format!("ζ_{}^{c}", rest.modulus())))?;
            Ok((k.clone(), z.sub_ref(&S::one())))
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::circuit::Circuit;
    use crate::scalar::C64;
    use crate::state::StateVector;
    use crate::Cyclo;
    use rand::rngs::StdRng;
    use rand::{Rng, SeedableRng};

    fn close(a: &[C64], b: &[C64]) -> bool {
        a.len() == b.len() && a.iter().zip(b).all(|(x, y)| (x - y).norm() < 1e-9)
    }

    fn random_circuit(n: usize, gates: usize, rng: &mut StdRng) -> Circuit {
        let mut c = Circuit::new(n);
        let one = [
            Gate::H,
            Gate::S,
            Gate::X,
            Gate::Y,
            Gate::SqrtX,
            Gate::T,
            Gate::Tdg,
            Gate::Phase { k: 1, bits: 4 },
        ];
        let two = [Gate::Cx, Gate::Cz, Gate::Swap, Gate::Cs];
        for _ in 0..gates {
            let r = rng.gen::<f64>();
            if r < 0.55 || n < 2 {
                c.gate(one[rng.gen_range(0..one.len())].clone(), &[rng.gen_range(0..n)]);
            } else if r < 0.92 || n < 3 {
                let a = rng.gen_range(0..n);
                let mut b = rng.gen_range(0..n - 1);
                if b >= a {
                    b += 1;
                }
                c.gate(two[rng.gen_range(0..two.len())].clone(), &[a, b]);
            } else {
                let mut qs: Vec<usize> = (0..n).collect();
                for i in 0..3 {
                    let j = rng.gen_range(i..n);
                    qs.swap(i, j);
                }
                c.gate(if rng.gen::<bool>() { Gate::Ccz } else { Gate::Ccx }, &qs[..3]);
            }
        }
        c
    }

    #[test]
    fn t_gates_double_the_rank_and_match_the_vector() {
        let mut c = Circuit::new(3);
        c.h(0)
            .h(1)
            .h(2)
            .t(0)
            .t(1)
            .gate(Gate::Ccz, &[0, 1, 2])
            .h(0)
            .cx(0, 1)
            .t(2);
        let mut s: StabilizerSum<Cyclo<8>> = StabilizerSum::zero_state(3);
        let mut v: StateVector<Cyclo<8>> = StateVector::zero_state(3);
        let mut rng = StdRng::seed_from_u64(1);
        s.run(&c, &mut rng).unwrap();
        v.run(&c, &mut rng).unwrap();
        assert_eq!(s.num_terms(), 16);
        assert_eq!(s.to_vector().unwrap(), v.to_vector().unwrap());
        assert_eq!(s.norm_sq().unwrap(), v.norm_sq().unwrap());
        let p = Pauli::from_string(3, "X0*Z2").unwrap();
        assert_eq!(s.expectation(&p).unwrap(), v.expectation(&p).unwrap());
        s.merge_duplicates().unwrap();
        assert!(s.num_terms() <= 16);
        assert_eq!(s.to_vector().unwrap(), v.to_vector().unwrap());
    }

    #[test]
    fn ch_form_terms_match_the_state_vector_exactly() {
        use crate::state::ChTerm;
        let mut rng = StdRng::seed_from_u64(19);
        for trial in 0..12 {
            let n = 2 + trial % 4;
            let c = random_circuit(n, 16, &mut rng);
            let mut s: StabilizerSum<Cyclo<16>, ChTerm<Cyclo<16>>> = StabilizerSum::zero_state(n);
            let mut p: StabilizerSum<Cyclo<16>> = StabilizerSum::zero_state(n);
            let mut v: StateVector<Cyclo<16>> = StateVector::zero_state(n);
            s.run(&c, &mut rng).unwrap();
            p.run(&c, &mut rng).unwrap();
            v.run(&c, &mut rng).unwrap();
            assert_eq!(
                s.to_vector().unwrap(),
                v.to_vector().unwrap(),
                "trial {trial}: {c}"
            );
            assert_eq!(s.num_terms(), p.num_terms());
            let z = Pauli::from_string(n, "Z0*X1").unwrap();
            assert_eq!(s.expectation(&z).unwrap(), v.expectation(&z).unwrap());
            assert_eq!(s.norm_sq().unwrap(), v.norm_sq().unwrap());
            let mut s2 = s.clone();
            let mut v2 = v.clone();
            assert_eq!(
                s2.project(&Pauli::z(n, 0), true).unwrap(),
                v2.project(&Pauli::z(n, 0), true).unwrap()
            );
            assert_eq!(s.inner(&s2).unwrap(), v.inner(&v2).unwrap());
            s.merge_duplicates().unwrap();
            assert_eq!(
                s.to_vector().unwrap(),
                v.to_vector().unwrap(),
                "trial {trial}: merged"
            );
        }
    }

    #[test]
    fn magic_state_gadgets_match_the_state_vector_with_half_the_terms() {
        use ncalg::Coeff;
        let mut rng = StdRng::seed_from_u64(29);
        for trial in 0..12 {
            let n = 2 + trial % 4;
            let mut c = Circuit::new(n);
            let mut t = 0;
            for _ in 0..30 {
                let q = rng.gen_range(0..n);
                match rng.gen_range(0..6) {
                    0 => c.h(q),
                    1 => c.s(q),
                    2 if t < 9 => {
                        t += 1;
                        c.t(q)
                    }
                    3 if t < 9 => {
                        t += 1;
                        c.gate(Gate::Tdg, &[q])
                    }
                    _ if n > 1 => c.cx(q, (q + 1) % n),
                    _ => c.h(q),
                };
            }
            let mut v: StateVector<Cyclo<8>> = StateVector::zero_state(n);
            v.run(&c, &mut rng).unwrap();
            let (s, tt): (StabilizerSum<Cyclo<8>>, usize) =
                StabilizerSum::run_with_t_gadgets(&c, &mut rng).unwrap();
            assert_eq!(tt, t);
            let full = s.to_vector().unwrap();
            // ψ ⊗ |0⟩_ancillas: the first 2^n amplitudes, zero elsewhere
            let vv = v.to_vector().unwrap();
            assert_eq!(&full[..1 << n], &vv[..], "trial {trial}: {c}");
            assert!(full[1 << n..].iter().all(|a| a.is_zero()));
            // the initial sum had 2^⌈t/2⌉ terms; projections never add terms
            assert!(
                s.num_terms() <= 1 << t.div_ceil(2),
                "trial {trial}: {} terms",
                s.num_terms()
            );
            // the same with CH-form terms
            let (h, _): (StabilizerSum<Cyclo<8>, crate::state::ChTerm<Cyclo<8>>>, usize) =
                StabilizerSum::run_with_t_gadgets(&c, &mut rng).unwrap();
            assert_eq!(
                &h.to_vector().unwrap()[..1 << n],
                &vv[..],
                "trial {trial}: CH-form terms"
            );
        }
        let m: StabilizerSum<Cyclo<8>> = StabilizerSum::magic_t(5).unwrap();
        assert_eq!(m.num_terms(), 8);
        assert!(m.norm_sq().unwrap().is_one());
    }

    #[test]
    fn bravyi_gosset_estimates() {
        let mut rng = StdRng::seed_from_u64(31);
        let c = random_circuit(6, 30, &mut rng);
        let mut s: StabilizerSum<C64> = StabilizerSum::zero_state(6);
        s.run(&c, &mut rng).unwrap();
        let exact = s.norm_sq().unwrap().re;
        let est = s.estimate_norm_sq(4000, &mut rng).unwrap();
        assert!((est / exact - 1.0).abs() < 0.1, "{est} vs {exact}");
        for text in ["Z0", "X1*Z2", "Y3"] {
            let p = Pauli::from_string(6, text).unwrap();
            let e = (s.expectation(&p).unwrap() / s.norm_sq().unwrap()).re;
            let f = s.estimate_expectation(&p, 4000, &mut rng).unwrap();
            assert!((e - f).abs() < 0.1, "<{text}>: {f} vs {e}");
        }
    }

    #[test]
    fn random_circuits_match_the_state_vector() {
        let mut rng = StdRng::seed_from_u64(9);
        for trial in 0..10 {
            let n = 2 + trial % 4;
            let c = random_circuit(n, 14, &mut rng);
            let mut s: StabilizerSum<C64> = StabilizerSum::zero_state(n);
            let mut v: StateVector<C64> = StateVector::zero_state(n);
            s.run(&c, &mut rng).unwrap();
            v.run(&c, &mut rng).unwrap();
            assert!(
                close(&s.to_vector().unwrap(), &v.to_vector().unwrap()),
                "trial {trial}: {c}"
            );
            let q = rng.gen_range(0..n);
            let z = Pauli::z(n, q);
            let a = s.expectation(&z).unwrap();
            let b = v.expectation(&z).unwrap();
            assert!((a - b).norm() < 1e-9);
            let mut s2 = s.clone();
            let mut v2 = v.clone();
            let wa = s2.project(&z, false).unwrap();
            let wb = v2.project(&z, false).unwrap();
            assert!((wa - wb).norm() < 1e-9);
            assert!(close(&s2.to_vector().unwrap(), &v2.to_vector().unwrap()));
            let ip = s.inner(&s2).unwrap();
            let ipv = v.inner(&v2).unwrap();
            assert!((ip - ipv).norm() < 1e-9);
        }
    }
}
