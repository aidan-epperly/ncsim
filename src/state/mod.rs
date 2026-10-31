//! States and the operations every family supports.
//!
//! [`QuantumState`] is implemented by each family (dense [`StateVector`],
//! stabilizer tableaux, path sums, stabilizer-rank sums, matrix product
//! states); a backend declines what it cannot do with
//! [`SimError::Unsupported`].  Circuits run on any of them through
//! [`QuantumState::run`].
//!
//! States are not kept normalized: projections leave `Π ψ`, and
//! [`QuantumState::norm_sq`] gives `⟨ψ|ψ⟩`.  Measurement outcomes are `bool`s
//! with `true` for the eigenvalue `−1` (the bit `1` of a `Z` measurement).

pub mod chform;
pub mod mps;
pub mod pathsum;
pub mod product;
pub mod sparse;
pub mod stabilizer;
pub mod stabsum;
pub mod vector;

pub use chform::{ChForm, ChTerm};
pub use mps::{Mps, MpsOptions};
pub use pathsum::PathSum;
pub use product::measure_commuting_on_product;
pub use sparse::SparseVector;
pub use stabilizer::{AffineForm, Stabilizer};
pub use stabsum::{projector_factors, StabilizerSum, StabilizerTerm};
pub use vector::StateVector;

use crate::bits::{self, Bits};
use crate::circuit::{Circuit, Op};
use crate::error::{Result, SimError};
use crate::gate::Gate;
use crate::pauli::{Pauli, PauliSum};
use crate::scalar::Amplitude;
use rand::{Rng, RngCore};
use std::fmt;

/// A (possibly unnormalized) pure state of `n` qubits with amplitudes in `S`.
pub trait QuantumState<S: Amplitude>: Clone + fmt::Debug + Send + Sync {
    /// The backend's name, for error messages.
    const BACKEND: &'static str;

    /// Number of qubits.
    fn num_qubits(&self) -> usize;

    /// Apply a gate on `qubits` (controls first).
    fn apply_gate(&mut self, gate: &Gate, qubits: &[usize]) -> Result<()>;

    /// `⟨ψ|ψ⟩`.
    fn norm_sq(&self) -> Result<S>;

    /// `⟨x|ψ⟩` for the computational basis state with bits `x` (bit `q` = qubit `q`).
    fn amplitude(&self, x: &[u64]) -> Result<S>;

    /// `⟨ψ|P|ψ⟩`.
    fn expectation(&self, p: &Pauli) -> Result<S>;

    /// `⟨ψ|φ⟩` for `φ = other`.
    fn inner(&self, other: &Self) -> Result<S>;

    /// `ψ ← (1 + (−1)^outcome P)/2 · ψ` for a Hermitian Pauli `P`; returns the
    /// weight `⟨ψ|Π|ψ⟩` of the branch (zero means the state vanished, and the
    /// backend may return [`SimError::ZeroState`] instead).
    fn project(&mut self, p: &Pauli, outcome: bool) -> Result<S>;

    /// `ψ ← c ψ`.
    fn scale(&mut self, c: &S) -> Result<()>;

    /// `ψ ⊗ φ` on `n_ψ + n_φ` qubits for `φ = other`: `ψ` on qubits
    /// `0..n_ψ`, `φ` on the qubits after them.  Backends without a tensor
    /// product decline with [`SimError::Unsupported`].
    fn tensor(&self, other: &Self) -> Result<Self> {
        let _ = other;
        Err(crate::error::unsupported(Self::BACKEND, "tensor product"))
    }

    /// `ψ_0 ⊗ ψ_1 ⊗ ⋯` (`ψ_0` on the lowest qubits), by a balanced tree of
    /// [`QuantumState::tensor`] products.
    fn tensor_all(states: &[Self]) -> Result<Self> {
        match states.len() {
            0 => Err(SimError::Invalid("tensor_all of no states".into())),
            1 => Ok(states[0].clone()),
            k => {
                let (a, b) = states.split_at(k / 2);
                Self::tensor_all(a)?.tensor(&Self::tensor_all(b)?)
            }
        }
    }

    /// `Σ_k c_k ⟨ψ|P_k|ψ⟩`, the terms in parallel and summed pairwise.
    fn expectation_sum(&self, h: &PauliSum<S>) -> Result<S> {
        use rayon::prelude::*;
        let pairs: Vec<(Pauli, &S)> = h.terms().collect();
        let terms: Result<Vec<S>> = pairs
            .par_iter()
            .map(|(p, c)| Ok(c.mul_ref(&self.expectation(p)?)))
            .collect();
        Ok(S::sum(&terms?))
    }

    /// `|⟨x|ψ⟩|² / ⟨ψ|ψ⟩`.
    fn probability(&self, x: &[u64]) -> Result<f64> {
        let a = self.amplitude(x)?.abs_sq_f64();
        let n = self.norm_sq()?.to_c64().re;
        Ok(a / n)
    }

    /// Measure the Hermitian Pauli `p`: draw the outcome with the Born
    /// probabilities and project.  Outcomes within `1e-12` of certain are
    /// treated as certain.
    fn measure(&mut self, p: &Pauli, rng: &mut dyn RngCore) -> Result<bool> {
        if !p.is_hermitian() {
            return Err(SimError::Invalid(format!(
                "measured operator {p} is not Hermitian"
            )));
        }
        let e = self.expectation(p)?.to_c64().re;
        let nn = self.norm_sq()?.to_c64().re;
        let p_plus = ((1.0 + e / nn) / 2.0).clamp(0.0, 1.0);
        let outcome = if p_plus > 1.0 - 1e-12 {
            false
        } else if p_plus < 1e-12 {
            true
        } else {
            rng.gen::<f64>() >= p_plus
        };
        self.project(p, outcome)?;
        Ok(outcome)
    }

    /// Apply one circuit operation; measurements return their outcome.
    fn apply_op(&mut self, op: &Op, rng: &mut dyn RngCore) -> Result<Option<bool>> {
        match op {
            Op::Gate { gate, qubits } => {
                self.apply_gate(gate, qubits)?;
                Ok(None)
            }
            Op::Measure { pauli } => Ok(Some(self.measure(pauli, rng)?)),
            Op::Reset { qubit } => {
                let z = Pauli::z(self.num_qubits(), *qubit);
                if self.measure(&z, rng)? {
                    self.apply_gate(&Gate::X, &[*qubit])?;
                }
                Ok(None)
            }
            Op::Noise { channel, qubits } => {
                // one trajectory: a sampled Pauli error
                if let Some((x, z)) = channel.sample(rng) {
                    let n = self.num_qubits();
                    let mut p = Pauli::identity(n);
                    for (j, &q) in qubits.iter().enumerate() {
                        p.set_raw_bits(q, (x >> j) & 1 == 1, (z >> j) & 1 == 1);
                    }
                    let k = qubits.len();
                    let local = p.restrict(qubits);
                    debug_assert_eq!(local.num_qubits(), k);
                    self.apply_gate(&Gate::Pauli(local), qubits)?;
                }
                Ok(None)
            }
            Op::Tick | Op::Detector { .. } | Op::Observable { .. } => Ok(None),
        }
    }

    /// Run a circuit, returning the measurement record.
    fn run(&mut self, circuit: &Circuit, rng: &mut dyn RngCore) -> Result<Vec<bool>> {
        if circuit.num_qubits() != self.num_qubits() {
            return Err(SimError::Invalid(format!(
                "circuit on {} qubits, state on {}",
                circuit.num_qubits(),
                self.num_qubits()
            )));
        }
        let mut record = Vec::with_capacity(circuit.num_measurements());
        for op in circuit.ops() {
            if let Some(b) = self.apply_op(op, rng)? {
                record.push(b);
            }
        }
        Ok(record)
    }

    /// `shots` samples of the computational basis measurement of all qubits
    /// (default: sequential single-qubit measurements on a copy per shot, the
    /// shots in parallel, each with its own generator seeded from `rng`, so
    /// the result depends only on `rng`).
    fn sample(&self, rng: &mut dyn RngCore, shots: usize) -> Result<Vec<Bits>> {
        use rand::SeedableRng;
        use rayon::prelude::*;
        let n = self.num_qubits();
        let seeds: Vec<u64> = (0..shots).map(|_| rng.next_u64()).collect();
        seeds
            .par_iter()
            .map(|&seed| {
                let mut r = rand::rngs::StdRng::seed_from_u64(seed);
                let mut s = self.clone();
                let mut x = bits::zero(n);
                for q in 0..n {
                    if s.measure(&Pauli::z(n, q), &mut r)? {
                        bits::toggle(&mut x, q);
                    }
                }
                Ok(x)
            })
            .collect()
    }

    /// All `2^n` amplitudes, little-endian (for checks on few qubits).
    fn to_vector(&self) -> Result<Vec<S>> {
        let n = self.num_qubits();
        assert!(n <= 26, "dense vector of more than 26 qubits");
        (0..(1u64 << n))
            .map(|x| self.amplitude(&bits::from_u64(x, n)))
            .collect()
    }

    /// Scale to `⟨ψ|ψ⟩ = 1`.  Exact types need the norm to be a power of `√2`.
    fn normalize(&mut self) -> Result<()> {
        let nn = self.norm_sq()?.to_c64().re;
        if nn <= 0.0 {
            return Err(SimError::ZeroState("normalize".into()));
        }
        let l = -nn.log2();
        let c = if (l - l.round()).abs() < 1e-9 {
            S::sqrt2_pow(l.round() as i64)
        } else {
            S::from_f64(1.0 / nn.sqrt())
        }
        .ok_or_else(|| SimError::NotRepresentable(format!("1/√{nn}")))?;
        self.scale(&c)
    }
}
