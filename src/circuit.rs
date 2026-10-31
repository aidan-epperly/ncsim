//! Circuits: sequences of gates, measurements and resets on a register, with
//! a Stim-like text format.
//!
//! ```text
//! # comments start with '#'
//! H 0 1            # H on qubit 0, then on qubit 1
//! CX 0 1 2 3       # CX(0,1), CX(2,3): targets are grouped by the gate's arity
//! T 1
//! PHASE(3,4) 2     # diag(1, ζ_16^3)
//! CCZ 0 1 2
//! RZ(0.5) 0        # angle in radians
//! M 0 1            # Z measurements (MX, MY: X and Y)
//! MPP X0*Z1 Y2*Y3  # Pauli product measurements
//! R 0              # reset to |0⟩
//! X_ERROR(0.01) 0 1          # Pauli noise: X_ERROR, Y_ERROR, Z_ERROR,
//! DEPOLARIZE1(0.001) 2       # DEPOLARIZE1(p), DEPOLARIZE2(p) (pairs),
//! PAULI_CHANNEL_1(0.1,0,0.2) 3   # PAULI_CHANNEL_1(px, py, pz)
//! DETECTOR rec[-1] rec[-3]   # parity of earlier measurement results
//! OBSERVABLE_INCLUDE(0) rec[-2]
//! TICK
//! ```
//!
//! Noise acts on every backend as a sampled trajectory (a random Pauli applied
//! with the channel's probabilities); [`FrameSampler`](crate::frame::FrameSampler)
//! samples noisy Clifford circuits in bulk with Pauli frames.
//!
//! The register size is the largest index plus one, or `QUBITS n` given first.

use crate::clifford::CliffordTableau;
use crate::error::{check_qubits, Result, SimError};
use crate::gate::Gate;
use crate::pauli::{Pauli, PauliKind};
use crate::scalar::Amplitude;
use std::fmt;

/// One operation of a circuit.
#[derive(Clone, Debug, PartialEq)]
pub enum Op {
    /// A gate on the listed qubits (controls first).
    Gate {
        /// The gate.
        gate: Gate,
        /// Its qubits.
        qubits: Vec<usize>,
    },
    /// A projective measurement of a Hermitian Pauli on the whole register.
    /// The recorded outcome `b` means eigenvalue `(−1)^b`.
    Measure {
        /// The observable.
        pauli: Pauli,
    },
    /// Reset a qubit to `|0⟩` (measure `Z`, flip if `1`).
    Reset {
        /// The qubit.
        qubit: usize,
    },
    /// A Pauli noise channel on each listed qubit (pairs for two-qubit channels).
    Noise {
        /// The channel.
        channel: Noise,
        /// Its qubits (`channel.arity()` per application).
        qubits: Vec<usize>,
    },
    /// A detector: the parity of earlier measurement results, `rec[-k]` for
    /// each `k` in `lookback` (`1` = the latest).  No effect on the state.
    Detector {
        /// The look-backs (each at least 1).
        lookback: Vec<usize>,
    },
    /// Include earlier measurement results in logical observable `index`.
    Observable {
        /// The observable.
        index: usize,
        /// The look-backs (each at least 1).
        lookback: Vec<usize>,
    },
    /// A layer marker; no effect.
    Tick,
}

/// A Pauli noise channel: with the given probabilities a Pauli error happens.
#[derive(Clone, Debug, PartialEq)]
pub enum Noise {
    /// `X` with probability `p`.
    XError(f64),
    /// `Y` with probability `p`.
    YError(f64),
    /// `Z` with probability `p`.
    ZError(f64),
    /// `X`, `Y`, `Z` with probabilities `px`, `py`, `pz`.
    PauliChannel1(f64, f64, f64),
    /// Each of `X`, `Y`, `Z` with probability `p/3`.
    Depolarize1(f64),
    /// Each of the 15 non-identity two-qubit Paulis with probability `p/15`.
    Depolarize2(f64),
}

impl Noise {
    /// Qubits per application.
    pub fn arity(&self) -> usize {
        match self {
            Noise::Depolarize2(_) => 2,
            _ => 1,
        }
    }
    /// The text-format name.
    pub fn name(&self) -> String {
        match self {
            Noise::XError(p) => format!("X_ERROR({p})"),
            Noise::YError(p) => format!("Y_ERROR({p})"),
            Noise::ZError(p) => format!("Z_ERROR({p})"),
            Noise::PauliChannel1(a, b, c) => format!("PAULI_CHANNEL_1({a},{b},{c})"),
            Noise::Depolarize1(p) => format!("DEPOLARIZE1({p})"),
            Noise::Depolarize2(p) => format!("DEPOLARIZE2({p})"),
        }
    }
    /// `(probability, x bits, z bits)` of each error (local bit `j` = qubit `j`
    /// of one application).
    pub fn errors(&self) -> Vec<(f64, u8, u8)> {
        match *self {
            Noise::XError(p) => vec![(p, 1, 0)],
            Noise::YError(p) => vec![(p, 1, 1)],
            Noise::ZError(p) => vec![(p, 0, 1)],
            Noise::PauliChannel1(a, b, c) => vec![(a, 1, 0), (b, 1, 1), (c, 0, 1)],
            Noise::Depolarize1(p) => vec![(p / 3.0, 1, 0), (p / 3.0, 1, 1), (p / 3.0, 0, 1)],
            Noise::Depolarize2(p) => (1..16u8).map(|k| (p / 15.0, k & 3, k >> 2)).collect(),
        }
    }
    /// Is it a probability distribution (each probability and the total in `[0, 1]`)?
    pub fn is_valid(&self) -> bool {
        let e = self.errors();
        e.iter().all(|(p, _, _)| (0.0..=1.0).contains(p))
            && e.iter().map(|(p, _, _)| p).sum::<f64>() <= 1.0 + 1e-12
    }
    /// Sample one error: `None` (no error) or the local `(x bits, z bits)`.
    pub fn sample(&self, rng: &mut dyn rand::RngCore) -> Option<(u8, u8)> {
        use rand::Rng;
        let mut u = rng.gen::<f64>();
        for (p, x, z) in self.errors() {
            if u < p {
                return Some((x, z));
            }
            u -= p;
        }
        None
    }
}

/// A circuit on `n` qubits.
#[derive(Clone, Debug, PartialEq)]
pub struct Circuit {
    n: usize,
    ops: Vec<Op>,
}

impl Circuit {
    /// The empty circuit on `n` qubits.
    pub fn new(n: usize) -> Self {
        Circuit { n, ops: Vec::new() }
    }
    /// Number of qubits.
    pub fn num_qubits(&self) -> usize {
        self.n
    }
    /// The operations in order.
    pub fn ops(&self) -> &[Op] {
        &self.ops
    }
    /// Number of operations.
    pub fn len(&self) -> usize {
        self.ops.len()
    }
    /// No operations?
    pub fn is_empty(&self) -> bool {
        self.ops.is_empty()
    }
    /// Append an operation (checked).
    pub fn push(&mut self, op: Op) -> Result<()> {
        match &op {
            Op::Gate { gate, qubits } => {
                check_qubits(qubits, self.n)?;
                if qubits.len() != gate.arity() {
                    return Err(SimError::Arity {
                        gate: gate.name(),
                        expected: gate.arity(),
                        got: qubits.len(),
                    });
                }
            }
            Op::Measure { pauli } => {
                if pauli.num_qubits() != self.n {
                    return Err(SimError::Invalid("measurement on the wrong register".into()));
                }
                if !pauli.is_hermitian() {
                    return Err(SimError::Invalid(format!(
                        "measured operator {pauli} is not Hermitian"
                    )));
                }
            }
            Op::Reset { qubit } => check_qubits(&[*qubit], self.n)?,
            Op::Noise { channel, qubits } => {
                check_qubits(qubits, self.n)?;
                if qubits.len() != channel.arity() {
                    return Err(SimError::Arity {
                        gate: channel.name(),
                        expected: channel.arity(),
                        got: qubits.len(),
                    });
                }
                if !channel.is_valid() {
                    return Err(SimError::Invalid(format!(
                        "{} is not a probability distribution",
                        channel.name()
                    )));
                }
            }
            Op::Detector { lookback } | Op::Observable { lookback, .. } => {
                let m = self.num_measurements();
                if let Some(&k) = lookback.iter().find(|&&k| k == 0 || k > m) {
                    return Err(SimError::Invalid(format!(
                        "rec[-{k}] with {m} measurements so far"
                    )));
                }
            }
            Op::Tick => {}
        }
        self.ops.push(op);
        Ok(())
    }
    /// Append a gate (checked).
    pub fn push_gate(&mut self, gate: Gate, qubits: &[usize]) -> Result<&mut Self> {
        self.push(Op::Gate {
            gate,
            qubits: qubits.to_vec(),
        })?;
        Ok(self)
    }
    /// Append a gate; panics on an invalid qubit list (for builders in code).
    pub fn gate(&mut self, gate: Gate, qubits: &[usize]) -> &mut Self {
        self.push_gate(gate, qubits).expect("invalid gate application")
    }
    /// `H q`.
    pub fn h(&mut self, q: usize) -> &mut Self {
        self.gate(Gate::H, &[q])
    }
    /// `S q`.
    pub fn s(&mut self, q: usize) -> &mut Self {
        self.gate(Gate::S, &[q])
    }
    /// `T q`.
    pub fn t(&mut self, q: usize) -> &mut Self {
        self.gate(Gate::T, &[q])
    }
    /// `X q`.
    pub fn x(&mut self, q: usize) -> &mut Self {
        self.gate(Gate::X, &[q])
    }
    /// `Z q`.
    pub fn z(&mut self, q: usize) -> &mut Self {
        self.gate(Gate::Z, &[q])
    }
    /// `CX c t`.
    pub fn cx(&mut self, c: usize, t: usize) -> &mut Self {
        self.gate(Gate::Cx, &[c, t])
    }
    /// `CZ a b`.
    pub fn cz(&mut self, a: usize, b: usize) -> &mut Self {
        self.gate(Gate::Cz, &[a, b])
    }
    /// Measure a Pauli (checked).
    pub fn measure(&mut self, pauli: Pauli) -> Result<&mut Self> {
        self.push(Op::Measure { pauli })?;
        Ok(self)
    }
    /// Measure `Z_q`.
    pub fn measure_z(&mut self, q: usize) -> &mut Self {
        self.push(Op::Measure {
            pauli: Pauli::z(self.n, q),
        })
        .expect("qubit in range");
        self
    }
    /// Reset `q`.
    pub fn reset(&mut self, q: usize) -> &mut Self {
        self.push(Op::Reset { qubit: q }).expect("qubit in range");
        self
    }
    /// A noise channel on `qubits` (checked).
    pub fn noise(&mut self, channel: Noise, qubits: &[usize]) -> Result<&mut Self> {
        self.push(Op::Noise {
            channel,
            qubits: qubits.to_vec(),
        })?;
        Ok(self)
    }
    /// The detectors as absolute measurement indices, in order.
    pub fn detectors(&self) -> Vec<Vec<usize>> {
        let mut m = 0usize;
        let mut out = Vec::new();
        for op in &self.ops {
            match op {
                Op::Measure { .. } => m += 1,
                Op::Detector { lookback } => out.push(lookback.iter().map(|k| m - k).collect()),
                _ => {}
            }
        }
        out
    }
    /// The logical observables as absolute measurement indices (index `i`
    /// collects every `OBSERVABLE_INCLUDE(i)`).
    pub fn observables(&self) -> Vec<Vec<usize>> {
        let mut m = 0usize;
        let mut out: Vec<Vec<usize>> = Vec::new();
        for op in &self.ops {
            match op {
                Op::Measure { .. } => m += 1,
                Op::Observable { index, lookback } => {
                    if out.len() <= *index {
                        out.resize(index + 1, Vec::new());
                    }
                    out[*index].extend(lookback.iter().map(|k| m - k));
                }
                _ => {}
            }
        }
        out
    }
    /// The circuit without its noise channels.
    pub fn without_noise(&self) -> Circuit {
        Circuit {
            n: self.n,
            ops: self
                .ops
                .iter()
                .filter(|op| !matches!(op, Op::Noise { .. }))
                .cloned()
                .collect(),
        }
    }
    /// Append another circuit on the same register.
    pub fn append(&mut self, other: &Circuit) -> Result<()> {
        if other.n != self.n {
            return Err(SimError::Invalid("append: register sizes differ".into()));
        }
        for op in &other.ops {
            self.push(op.clone())?;
        }
        Ok(())
    }
    /// Append `other` (on `other.num_qubits()` qubits) acting on `qubits` of this register.
    pub fn append_on(&mut self, other: &Circuit, qubits: &[usize]) -> Result<()> {
        if qubits.len() != other.n {
            return Err(SimError::Invalid("append_on: qubit count".into()));
        }
        check_qubits(qubits, self.n)?;
        for op in &other.ops {
            let mapped = match op {
                Op::Gate { gate, qubits: qs } => Op::Gate {
                    gate: gate.clone(),
                    qubits: qs.iter().map(|&q| qubits[q]).collect(),
                },
                Op::Measure { pauli } => Op::Measure {
                    pauli: pauli.embed(self.n, qubits),
                },
                Op::Reset { qubit } => Op::Reset {
                    qubit: qubits[*qubit],
                },
                Op::Noise { channel, qubits: qs } => Op::Noise {
                    channel: channel.clone(),
                    qubits: qs.iter().map(|&q| qubits[q]).collect(),
                },
                other => other.clone(),
            };
            self.push(mapped)?;
        }
        Ok(())
    }
    /// Number of measurements.
    pub fn num_measurements(&self) -> usize {
        self.ops
            .iter()
            .filter(|op| matches!(op, Op::Measure { .. }))
            .count()
    }
    /// Number of gates.
    pub fn num_gates(&self) -> usize {
        self.ops.iter().filter(|op| matches!(op, Op::Gate { .. })).count()
    }
    /// Number of non-Clifford gates.
    pub fn non_clifford_count(&self) -> usize {
        self.ops
            .iter()
            .filter(|op| matches!(op, Op::Gate { gate, .. } if !gate.is_clifford()))
            .count()
    }
    /// Only Clifford gates (and no measurements or resets)?
    pub fn is_clifford_unitary(&self) -> bool {
        self.ops.iter().all(|op| match op {
            Op::Gate { gate, .. } => gate.is_clifford(),
            Op::Tick | Op::Detector { .. } | Op::Observable { .. } => true,
            _ => false,
        })
    }
    /// The adjoint circuit (gates reversed and daggered); fails on measurements and resets.
    pub fn dagger(&self) -> Result<Circuit> {
        let mut c = Circuit::new(self.n);
        for op in self.ops.iter().rev() {
            match op {
                Op::Gate { gate, qubits } => c.push_gate(gate.dagger()?, qubits).map(|_| ())?,
                Op::Tick => c.push(Op::Tick)?,
                _ => return Err(SimError::Invalid("dagger of a circuit with measurements".into())),
            }
        }
        Ok(c)
    }
    /// The Clifford tableau of a Clifford unitary circuit.
    pub fn tableau(&self) -> Result<CliffordTableau> {
        let mut t = CliffordTableau::identity(self.n);
        for op in &self.ops {
            match op {
                Op::Gate { gate, qubits } => t.apply_named_gate(gate, qubits)?,
                Op::Tick => {}
                _ => {
                    return Err(SimError::Unsupported {
                        backend: "tableau",
                        what: "measurements".into(),
                    })
                }
            }
        }
        Ok(t)
    }
    /// The dense unitary of a unitary circuit (row-major, little-endian), for
    /// checks on few qubits.
    pub fn unitary<S: Amplitude>(&self) -> Result<Vec<S>> {
        let n = self.n;
        assert!(n <= 12, "dense unitary of more than 12 qubits");
        let dim = 1usize << n;
        let mut u = vec![S::zero(); dim * dim];
        for i in 0..dim {
            u[i * dim + i] = S::one();
        }
        for op in &self.ops {
            match op {
                Op::Gate { gate, qubits } => {
                    let m = gate.matrix::<S>().ok_or_else(|| {
                        SimError::NotRepresentable(format!("matrix of {gate} in the scalar type"))
                    })?;
                    u = apply_matrix_left(&m, qubits, &u, n);
                }
                Op::Tick => {}
                _ => return Err(SimError::Invalid("unitary of a circuit with measurements".into())),
            }
        }
        Ok(u)
    }

    /// Parse the text format.
    pub fn parse(text: &str) -> Result<Circuit> {
        let err = |line: usize, m: String| SimError::Parse(format!("line {}: {m}", line + 1));
        // first pass: register size
        let mut lines: Vec<(usize, Vec<String>)> = Vec::new();
        let mut declared: Option<usize> = None;
        for (ln, raw) in text.lines().enumerate() {
            let line = raw.split('#').next().unwrap().trim();
            if line.is_empty() {
                continue;
            }
            let toks: Vec<String> = line.split_whitespace().map(|s| s.to_string()).collect();
            if toks[0].eq_ignore_ascii_case("QUBITS") {
                let n: usize = toks
                    .get(1)
                    .and_then(|s| s.parse().ok())
                    .ok_or_else(|| err(ln, "QUBITS n".into()))?;
                declared = Some(n);
                continue;
            }
            lines.push((ln, toks));
        }
        let mut max_q = 0usize;
        let mut parsed: Vec<(usize, ParsedLine)> = Vec::new();
        for (ln, toks) in lines {
            let p = parse_line(&toks).map_err(|m| err(ln, m))?;
            match &p {
                ParsedLine::Gate(_, qs) => {
                    for &q in qs {
                        max_q = max_q.max(q + 1);
                    }
                }
                ParsedLine::MeasureBasis(_, qs) | ParsedLine::Reset(qs) | ParsedLine::Noise(_, qs) => {
                    for &q in qs {
                        max_q = max_q.max(q + 1);
                    }
                }
                ParsedLine::MeasurePauli(ps) => {
                    for p in ps {
                        max_q = max_q.max(p.num_qubits());
                    }
                }
                ParsedLine::Tick | ParsedLine::Detector(_) | ParsedLine::Observable(..) => {}
            }
            parsed.push((ln, p));
        }
        let n = declared.unwrap_or(max_q);
        if n < max_q {
            return Err(SimError::Parse(format!(
                "QUBITS {n} but qubit {} is used",
                max_q - 1
            )));
        }
        let mut c = Circuit::new(n);
        for (ln, p) in parsed {
            match p {
                ParsedLine::Gate(g, qs) => {
                    let k = g.arity();
                    if k == 0 || qs.len() % k != 0 {
                        return Err(err(ln, format!("{} takes groups of {k} qubits", g.name())));
                    }
                    for chunk in qs.chunks(k) {
                        c.push_gate(g.clone(), chunk)
                            .map_err(|e| err(ln, e.to_string()))?;
                    }
                }
                ParsedLine::MeasureBasis(kind, qs) => {
                    for q in qs {
                        c.push(Op::Measure {
                            pauli: Pauli::single(n, q, kind),
                        })
                        .map_err(|e| err(ln, e.to_string()))?;
                    }
                }
                ParsedLine::MeasurePauli(ps) => {
                    for p in ps {
                        let p = Pauli::from_string(n, &p.to_sparse_string())
                            .map_err(|e| err(ln, e.to_string()))?;
                        c.push(Op::Measure { pauli: p })
                            .map_err(|e| err(ln, e.to_string()))?;
                    }
                }
                ParsedLine::Reset(qs) => {
                    for q in qs {
                        c.push(Op::Reset { qubit: q })
                            .map_err(|e| err(ln, e.to_string()))?;
                    }
                }
                ParsedLine::Noise(ch, qs) => {
                    let k = ch.arity();
                    if qs.len() % k != 0 {
                        return Err(err(ln, format!("{} takes groups of {k} qubits", ch.name())));
                    }
                    for chunk in qs.chunks(k) {
                        c.noise(ch.clone(), chunk).map_err(|e| err(ln, e.to_string()))?;
                    }
                }
                ParsedLine::Detector(lookback) => c
                    .push(Op::Detector { lookback })
                    .map_err(|e| err(ln, e.to_string()))?,
                ParsedLine::Observable(index, lookback) => c
                    .push(Op::Observable { index, lookback })
                    .map_err(|e| err(ln, e.to_string()))?,
                ParsedLine::Tick => c.push(Op::Tick)?,
            }
        }
        Ok(c)
    }
}

enum ParsedLine {
    Gate(Gate, Vec<usize>),
    MeasureBasis(PauliKind, Vec<usize>),
    MeasurePauli(Vec<Pauli>),
    Reset(Vec<usize>),
    Noise(Noise, Vec<usize>),
    Detector(Vec<usize>),
    Observable(usize, Vec<usize>),
    Tick,
}

fn parse_args(s: &str) -> std::result::Result<(String, Vec<String>), String> {
    match s.find('(') {
        None => Ok((s.to_uppercase(), Vec::new())),
        Some(i) => {
            let name = s[..i].to_uppercase();
            let rest = &s[i + 1..];
            let close = rest.rfind(')').ok_or_else(|| format!("missing ')' in {s}"))?;
            let args = rest[..close].split(',').map(|a| a.trim().to_string()).collect();
            Ok((name, args))
        }
    }
}

fn parse_line(toks: &[String]) -> std::result::Result<ParsedLine, String> {
    let (name, args) = parse_args(&toks[0])?;
    let qubits = || -> std::result::Result<Vec<usize>, String> {
        toks[1..]
            .iter()
            .map(|t| t.parse::<usize>().map_err(|_| format!("bad qubit '{t}'")))
            .collect()
    };
    let int_arg = |i: usize| -> std::result::Result<i64, String> {
        args.get(i)
            .and_then(|a| a.parse().ok())
            .ok_or_else(|| format!("{name} needs integer argument {}", i + 1))
    };
    let float_arg = |i: usize| -> std::result::Result<f64, String> {
        args.get(i)
            .and_then(|a| a.parse().ok())
            .ok_or_else(|| format!("{name} needs a numeric argument"))
    };
    let g = match name.as_str() {
        "I" => Gate::I,
        "X" => Gate::X,
        "Y" => Gate::Y,
        "Z" => Gate::Z,
        "H" => Gate::H,
        "S" => Gate::S,
        "S_DAG" => Gate::Sdg,
        "SQRT_X" => Gate::SqrtX,
        "SQRT_X_DAG" => Gate::SqrtXdg,
        "SQRT_Y" => Gate::SqrtY,
        "SQRT_Y_DAG" => Gate::SqrtYdg,
        "T" => Gate::T,
        "T_DAG" => Gate::Tdg,
        "CX" | "CNOT" => Gate::Cx,
        "CY" => Gate::Cy,
        "CZ" => Gate::Cz,
        "SWAP" => Gate::Swap,
        "CS" => Gate::Cs,
        "CS_DAG" => Gate::Csdg,
        "CCZ" => Gate::Ccz,
        "CCX" | "TOFFOLI" => Gate::Ccx,
        "PHASE" => Gate::Phase {
            k: int_arg(0)?,
            bits: int_arg(1)? as u32,
        },
        "CPHASE" => Gate::CPhase {
            k: int_arg(0)?,
            bits: int_arg(1)? as u32,
        },
        "RX" => Gate::Rx(float_arg(0)?),
        "RY" => Gate::Ry(float_arg(0)?),
        "RZ" => Gate::Rz(float_arg(0)?),
        "M" | "MZ" => return Ok(ParsedLine::MeasureBasis(PauliKind::Z, qubits()?)),
        "MX" => return Ok(ParsedLine::MeasureBasis(PauliKind::X, qubits()?)),
        "MY" => return Ok(ParsedLine::MeasureBasis(PauliKind::Y, qubits()?)),
        "MPP" => {
            let ps: std::result::Result<Vec<Pauli>, String> = toks[1..]
                .iter()
                .map(|t| Pauli::parse(t).map_err(|e| e.to_string()))
                .collect();
            return Ok(ParsedLine::MeasurePauli(ps?));
        }
        "R" => return Ok(ParsedLine::Reset(qubits()?)),
        "DETECTOR" | "OBSERVABLE_INCLUDE" => {
            // rec[-k] targets; DETECTOR's coordinate arguments are ignored
            let recs: std::result::Result<Vec<usize>, String> = toks[1..]
                .iter()
                .map(|t| {
                    t.strip_prefix("rec[-")
                        .and_then(|r| r.strip_suffix(']'))
                        .and_then(|k| k.parse().ok())
                        .ok_or_else(|| format!("bad target '{t}' (expected rec[-k])"))
                })
                .collect();
            return Ok(if name == "DETECTOR" {
                ParsedLine::Detector(recs?)
            } else {
                ParsedLine::Observable(int_arg(0)? as usize, recs?)
            });
        }
        "X_ERROR" => return Ok(ParsedLine::Noise(Noise::XError(float_arg(0)?), qubits()?)),
        "Y_ERROR" => return Ok(ParsedLine::Noise(Noise::YError(float_arg(0)?), qubits()?)),
        "Z_ERROR" => return Ok(ParsedLine::Noise(Noise::ZError(float_arg(0)?), qubits()?)),
        "DEPOLARIZE1" => return Ok(ParsedLine::Noise(Noise::Depolarize1(float_arg(0)?), qubits()?)),
        "DEPOLARIZE2" => return Ok(ParsedLine::Noise(Noise::Depolarize2(float_arg(0)?), qubits()?)),
        "PAULI_CHANNEL_1" => {
            let f = |i: usize| -> std::result::Result<f64, String> {
                args.get(i)
                    .and_then(|a| a.parse().ok())
                    .ok_or_else(|| "PAULI_CHANNEL_1 needs three probabilities".to_string())
            };
            return Ok(ParsedLine::Noise(
                Noise::PauliChannel1(f(0)?, f(1)?, f(2)?),
                qubits()?,
            ));
        }
        "TICK" => return Ok(ParsedLine::Tick),
        other => return Err(format!("unknown instruction '{other}'")),
    };
    Ok(ParsedLine::Gate(g, qubits()?))
}

impl fmt::Display for Circuit {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "QUBITS {}", self.n)?;
        for op in &self.ops {
            match op {
                Op::Gate { gate, qubits } => {
                    let qs: Vec<String> = qubits.iter().map(|q| q.to_string()).collect();
                    writeln!(f, "{} {}", gate.name(), qs.join(" "))?;
                }
                Op::Measure { pauli } => {
                    if pauli.weight() == 1 && pauli.display_phase() == 0 {
                        let q = pauli.support()[0];
                        writeln!(f, "M{} {}", pauli.kind(q).symbol(), q)?;
                    } else {
                        writeln!(f, "MPP {}", pauli.to_sparse_string().replace(' ', ""))?;
                    }
                }
                Op::Reset { qubit } => writeln!(f, "R {qubit}")?,
                Op::Noise { channel, qubits } => {
                    let qs: Vec<String> = qubits.iter().map(|q| q.to_string()).collect();
                    writeln!(f, "{} {}", channel.name(), qs.join(" "))?;
                }
                Op::Detector { lookback } => {
                    let r: Vec<String> = lookback.iter().map(|k| format!("rec[-{k}]")).collect();
                    writeln!(f, "DETECTOR {}", r.join(" "))?;
                }
                Op::Observable { index, lookback } => {
                    let r: Vec<String> = lookback.iter().map(|k| format!("rec[-{k}]")).collect();
                    writeln!(f, "OBSERVABLE_INCLUDE({index}) {}", r.join(" "))?;
                }
                Op::Tick => writeln!(f, "TICK")?,
            }
        }
        Ok(())
    }
}

/// `(G on qubits) · U` for a dense `2^n × 2^n` matrix `U`: multiply each
/// column by the gate.
fn apply_matrix_left<S: Amplitude>(g: &[S], qubits: &[usize], u: &[S], n: usize) -> Vec<S> {
    let dim = 1usize << n;
    let k = qubits.len();
    let gd = 1usize << k;
    let mut out = vec![S::zero(); dim * dim];
    for col in 0..dim {
        // gather the column into the local structure
        for base in 0..dim {
            if qubits.iter().any(|&q| (base >> q) & 1 == 1) {
                continue;
            }
            let idx = |local: usize| -> usize {
                let mut i = base;
                for (j, &q) in qubits.iter().enumerate() {
                    if (local >> j) & 1 == 1 {
                        i |= 1 << q;
                    }
                }
                i
            };
            for a in 0..gd {
                let mut acc = S::zero();
                for b in 0..gd {
                    let x = &u[idx(b) * dim + col];
                    if !x.is_zero() {
                        acc.add_assign_ref(&g[a * gd + b].mul_ref(x));
                    }
                }
                out[idx(a) * dim + col] = acc;
            }
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::scalar::C64;

    #[test]
    fn parse_and_print_round_trip() {
        let text = "H 0\nCX 0 1 1 2\nT 2\nPHASE(3,4) 1\nM 0\nMX 1\nMPP X0*Z2\nR 2\nTICK\n";
        let c = Circuit::parse(text).unwrap();
        assert_eq!(c.num_qubits(), 3);
        assert_eq!(c.num_gates(), 5);
        assert_eq!(c.num_measurements(), 3);
        let printed = c.to_string();
        let c2 = Circuit::parse(&printed).unwrap();
        assert_eq!(c, c2);
        assert!(Circuit::parse("CX 0").is_err());
        assert!(Circuit::parse("QUBITS 1\nH 3").is_err());
        assert!(Circuit::parse("FOO 1").is_err());
    }

    #[test]
    fn tableau_and_unitary_of_bell_circuit() {
        let mut c = Circuit::new(2);
        c.h(0).cx(0, 1);
        let t = c.tableau().unwrap();
        assert_eq!(t.conjugate(&Pauli::z(2, 0)), Pauli::parse("X0*X1").unwrap());
        let u: Vec<C64> = c.unitary().unwrap();
        // first column: (|00> + |11>)/√2
        let r = std::f64::consts::FRAC_1_SQRT_2;
        assert!((u[0] - C64::new(r, 0.0)).norm() < 1e-12);
        assert!((u[3 * 4] - C64::new(r, 0.0)).norm() < 1e-12);
        assert!(u[4].norm() < 1e-12 && u[8].norm() < 1e-12);
        let d = c.dagger().unwrap();
        assert_eq!(d.ops().len(), 2);
        assert!(c.is_clifford_unitary());
        c.t(1);
        assert!(!c.is_clifford_unitary());
        assert_eq!(c.non_clifford_count(), 1);
    }
}
