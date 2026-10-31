//! # ncsim — quantum circuit simulation on ncalg
//!
//! A library (no binaries) for simulating quantum circuits exactly or
//! numerically, built on the algebra in [`ncalg`]: GF(2) linear algebra
//! (`ncalg::ff`), exact exponential sums of phase polynomials
//! (`ncalg::cyclo::expsum`), cyclotomic scalars (`ncalg::cyclo::Cyclo`) and the
//! Clifford-hierarchy groups of `ncalg::pcgroup`.
//!
//! The crate forbids `unsafe` code.
//!
//! ## Layers
//!
//! * [`scalar`] — the [`Amplitude`] trait: the scalars amplitudes are computed
//!   in (`Complex<f64>`, or exact `Cyclo<N>`), and [`Phasor`], the exact
//!   scalars `√2^p ζ_{2^b}^k` that stabilizer computations produce.
//! * [`pauli`] — bit-packed Pauli operators, packed tables of them and
//!   Pauli sums (Hamiltonians).
//! * [`clifford`] — Clifford unitaries as tableaux of Pauli images.
//! * [`phasepoly`] — phase polynomials `𝔽₂ⁿ → ℤ/2^r` in parity-product form
//!   and ℤ₄ quadratic forms, the diagonal part of every exact state family.
//! * [`gate`], [`circuit`] — gates as data with *views* (`tableau`,
//!   `monomial`, `matrix`), circuits, a Stim-like text format.
//! * [`state`] — the [`QuantumState`] trait and its families: dense state
//!   and sparse vectors, stabilizer tableaux, phase-tracking stabilizer
//!   states (CH-form), path sums, stabilizer-rank sums, matrix product states.
//! * [`frame`] — Pauli-frame sampling of noisy Clifford circuits, 64 shots
//!   per word.
//! * [`bridge`] — ncalg polynomials over the Pauli algebra as Pauli sums,
//!   moment vectors of states for NC-SOS relaxations.
//!
//! Every state family implements the same trait and declines what it cannot
//! do with [`SimError::Unsupported`], so a circuit runs on any backend that
//! supports its gates, and new families or gate kinds plug in without touching
//! the others.

#![forbid(unsafe_code)]
#![warn(missing_docs)]
#![warn(missing_debug_implementations)]
#![allow(clippy::needless_range_loop, clippy::type_complexity)]

pub mod bits;
pub mod bridge;
pub mod circuit;
pub mod clifford;
pub mod error;
pub mod frame;
pub mod gate;
pub mod pauli;
pub mod phasepoly;
pub mod scalar;
pub mod state;

pub use circuit::{Circuit, Noise, Op};
pub use clifford::CliffordTableau;
pub use error::{Result, SimError};
pub use frame::FrameSampler;
pub use gate::{CustomGate, DenseGate, Gate, MonomialGate};
pub use pauli::{Pauli, PauliKind, PauliSum, PauliTable};
pub use phasepoly::{PhasePoly, QuadForm};
pub use scalar::{Amplitude, Phasor, C64};
pub use state::{
    AffineForm, ChForm, ChTerm, Mps, MpsOptions, PathSum, QuantumState, SparseVector, Stabilizer,
    StabilizerSum, StabilizerTerm, StateVector,
};

// re-exported numeric types
pub use ncalg::cyclo::Cyclo;
pub use ncalg::Complex;
