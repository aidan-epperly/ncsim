//! Errors of the simulation API.

use std::fmt;

/// What can go wrong when building or simulating a circuit.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum SimError {
    /// The backend has no implementation of the requested operation
    /// (for example a non-Clifford gate on a stabilizer tableau).
    Unsupported {
        /// The backend that declined.
        backend: &'static str,
        /// What was asked.
        what: String,
    },
    /// A qubit index outside the register.
    QubitOutOfRange {
        /// The index.
        qubit: usize,
        /// The register size.
        num_qubits: usize,
    },
    /// A gate applied to the wrong number of qubits.
    Arity {
        /// The gate.
        gate: String,
        /// Qubits the gate acts on.
        expected: usize,
        /// Qubits given.
        got: usize,
    },
    /// The same qubit listed twice in a gate's targets.
    DuplicateQubit(usize),
    /// A value the scalar type cannot hold exactly (for example `e^{iθ}` in a cyclotomic field).
    NotRepresentable(String),
    /// The state vanished (projection onto an outcome of probability zero).
    ZeroState(String),
    /// Invalid text input.
    Parse(String),
    /// Inconsistent data (non-commuting stabilizers, a singular affine map, …).
    Invalid(String),
}

impl fmt::Display for SimError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            SimError::Unsupported { backend, what } => write!(f, "{backend}: unsupported: {what}"),
            SimError::QubitOutOfRange { qubit, num_qubits } => {
                write!(f, "qubit {qubit} out of range for {num_qubits} qubits")
            }
            SimError::Arity { gate, expected, got } => {
                write!(f, "gate {gate} acts on {expected} qubits, {got} given")
            }
            SimError::DuplicateQubit(q) => write!(f, "qubit {q} listed twice"),
            SimError::NotRepresentable(s) => write!(f, "not representable in the scalar type: {s}"),
            SimError::ZeroState(s) => write!(f, "state vanished: {s}"),
            SimError::Parse(s) => write!(f, "parse error: {s}"),
            SimError::Invalid(s) => write!(f, "invalid: {s}"),
        }
    }
}

impl std::error::Error for SimError {}

/// `Result` with [`SimError`].
pub type Result<T> = std::result::Result<T, SimError>;

/// Check a qubit list against a register size: in range, no repeats.
pub(crate) fn check_qubits(qubits: &[usize], n: usize) -> Result<()> {
    for (i, &q) in qubits.iter().enumerate() {
        if q >= n {
            return Err(SimError::QubitOutOfRange {
                qubit: q,
                num_qubits: n,
            });
        }
        if qubits[..i].contains(&q) {
            return Err(SimError::DuplicateQubit(q));
        }
    }
    Ok(())
}

/// Shorthand for [`SimError::Unsupported`].
pub(crate) fn unsupported(backend: &'static str, what: impl Into<String>) -> SimError {
    SimError::Unsupported {
        backend,
        what: what.into(),
    }
}
