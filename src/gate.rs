//! Gates as data.  A [`Gate`] is a value; backends ask it for the *view* they
//! understand — a Clifford tableau ([`Gate::tableau`]), an affine permutation
//! with a diagonal phase ([`Gate::monomial`]), a dense matrix
//! ([`Gate::matrix`]) or a decomposition into simpler gates
//! ([`Gate::decomposition`]) — and decline with
//! [`SimError::Unsupported`](crate::SimError) if none applies.
//! New gate kinds implement [`CustomGate`] and plug in the same way.

use crate::bits::{self, Bits};
use crate::circuit::Circuit;
use crate::clifford::CliffordTableau;
use crate::error::{Result, SimError};
use crate::pauli::Pauli;
use crate::phasepoly::PhasePoly;
use crate::scalar::{Amplitude, Phasor, C64};
use ncalg::ff::BitMat;
use std::fmt;
use std::sync::Arc;

// ---------------------------------------------------------------------------
// monomial gates
// ---------------------------------------------------------------------------

/// `U|x⟩ = scalar · ζ_{2^r}^{f(x)} |M x ⊕ shift⟩` on `k` qubits: an invertible
/// affine map on the computational basis times a diagonal phase.  Covers `X`,
/// `CX`, `SWAP`, every diagonal gate (`S`, `T`, `CZ`, `CCZ`, …), Paulis, and
/// the monomial representations of ncalg's Clifford-hierarchy groups.  Closed
/// under composition and inverse.
#[derive(Clone, PartialEq, Eq)]
pub struct MonomialGate {
    k: usize,
    /// Row `i`: the input bits that feed output bit `i`.
    rows: Vec<Bits>,
    shift: Bits,
    phase: PhasePoly,
    scalar: Phasor,
}

impl MonomialGate {
    /// The identity on `k` qubits.
    pub fn identity(k: usize) -> Self {
        MonomialGate {
            k,
            rows: (0..k).map(|i| bits::unit(k, i)).collect(),
            shift: bits::zero(k),
            phase: PhasePoly::new(k, 1),
            scalar: Phasor::ONE,
        }
    }
    /// From the parts; checks that the matrix is invertible.
    pub fn new(rows: Vec<Bits>, shift: Bits, phase: PhasePoly, scalar: Phasor) -> Result<Self> {
        let k = rows.len();
        if phase.num_vars() != k {
            return Err(SimError::Invalid(
                "monomial gate: phase polynomial on the wrong number of qubits".into(),
            ));
        }
        let m = bits::to_bitmat(&rows, k);
        if m.rank() != k {
            return Err(SimError::Invalid("monomial gate: singular affine map".into()));
        }
        let mut g = MonomialGate {
            k,
            rows,
            shift,
            phase,
            scalar,
        };
        for r in g.rows.iter_mut() {
            bits::ensure_len(r, k);
            bits::mask_to(r, k);
        }
        bits::ensure_len(&mut g.shift, k);
        bits::mask_to(&mut g.shift, k);
        Ok(g)
    }
    /// The diagonal gate `ζ^{f(x)}`.
    pub fn diagonal(phase: PhasePoly) -> Self {
        let k = phase.num_vars();
        let mut g = Self::identity(k);
        g.phase = phase;
        g
    }
    /// `X^a`.
    pub fn x_string(k: usize, a: &[u64]) -> Self {
        let mut g = Self::identity(k);
        g.shift = a.to_vec();
        bits::ensure_len(&mut g.shift, k);
        bits::mask_to(&mut g.shift, k);
        g
    }
    /// A qubit permutation: output `i` is input `perm[i]`.
    pub fn permutation(perm: &[usize]) -> Result<Self> {
        let k = perm.len();
        let rows: Vec<Bits> = perm.iter().map(|&j| bits::unit(k, j)).collect();
        Self::new(rows, bits::zero(k), PhasePoly::new(k, 1), Phasor::ONE)
    }
    /// A Pauli operator `i^phase X^a Z^b` as a monomial gate.
    pub fn pauli(p: &Pauli) -> Self {
        let k = p.num_qubits();
        let mut phase = PhasePoly::new(k, 1);
        phase.add_parity_term(1, &[p.z_bits().to_vec()]);
        let mut g = Self::identity(k);
        g.shift = p.x_bits().to_vec();
        g.phase = phase;
        g.scalar = Phasor::i_pow(p.phase() as i64);
        g
    }
    /// Number of qubits.
    pub fn num_qubits(&self) -> usize {
        self.k
    }
    /// The rows of the linear part (`output_i = shift_i ⊕ rows_i · x`).
    pub fn rows(&self) -> &[Bits] {
        &self.rows
    }
    /// The shift.
    pub fn shift(&self) -> &[u64] {
        &self.shift
    }
    /// The phase polynomial.
    pub fn phase(&self) -> &PhasePoly {
        &self.phase
    }
    /// The scalar.
    pub fn scalar(&self) -> Phasor {
        self.scalar
    }
    /// Is the affine part the identity?
    pub fn is_diagonal(&self) -> bool {
        bits::is_zero(&self.shift)
            && self
                .rows
                .iter()
                .enumerate()
                .all(|(i, r)| *r == bits::unit(self.k, i))
    }
    /// Is the phase polynomial Clifford (level ≤ 2)?  (The affine part always is.)
    pub fn is_clifford(&self) -> bool {
        self.phase.is_clifford()
    }
    /// Column `j` of the linear part: the outputs fed by input `j`.
    fn column(&self, j: usize) -> Bits {
        let mut c = bits::zero(self.k);
        for (i, r) in self.rows.iter().enumerate() {
            if bits::get(r, j) {
                bits::toggle(&mut c, i);
            }
        }
        c
    }
    /// The columns of the linear part.
    pub fn columns(&self) -> Vec<Bits> {
        (0..self.k).map(|j| self.column(j)).collect()
    }
    /// `U|x⟩ = s |y⟩`: `(s, y)` for basis bits `x`.
    pub fn apply_to_basis(&self, x: &[u64]) -> (Phasor, Bits) {
        let f = self.phase.eval(x);
        let s = self
            .scalar
            .mul(Phasor::root_of_unity(f as i64, self.phase.bits()));
        let mut y = self.shift.clone();
        for (i, r) in self.rows.iter().enumerate() {
            if bits::dot(r, x) {
                bits::toggle(&mut y, i);
            }
        }
        (s, y)
    }
    /// `self ∘ other` (apply `other` first).
    pub fn compose(&self, other: &MonomialGate) -> MonomialGate {
        assert_eq!(self.k, other.k);
        let k = self.k;
        // rows: row_i(self·other) = Σ_j self_ij row_j(other)
        let rows: Vec<Bits> = self
            .rows
            .iter()
            .map(|r| {
                let mut acc = bits::zero(k);
                for j in bits::ones(r) {
                    bits::xor_into(&mut acc, &other.rows[j]);
                }
                acc
            })
            .collect();
        let mut shift = self.shift.clone();
        for (i, r) in self.rows.iter().enumerate() {
            if bits::dot(r, &other.shift) {
                bits::toggle(&mut shift, i);
            }
        }
        let mut phase = other.phase.clone();
        phase.add_assign(&self.phase.substitute(&other.shift, &other.columns(), k));
        MonomialGate {
            k,
            rows,
            shift,
            phase,
            scalar: self.scalar.mul(other.scalar),
        }
    }
    /// The inverse `U⁻¹ = U†`.
    pub fn inverse(&self) -> MonomialGate {
        let k = self.k;
        let rows = bits::inverse(&self.rows, k).expect("monomial gate: invertible by construction");
        let inv_gate = MonomialGate {
            k,
            rows,
            shift: bits::zero(k),
            phase: PhasePoly::new(k, 1),
            scalar: Phasor::ONE,
        };
        // M⁻¹(y ⊕ b): shift M⁻¹ b
        let mut shift = bits::zero(k);
        for (i, r) in inv_gate.rows.iter().enumerate() {
            if bits::dot(r, &self.shift) {
                bits::toggle(&mut shift, i);
            }
        }
        let mut phase = self.phase.substitute(&shift, &inv_gate.columns(), k);
        phase.scale(-1);
        MonomialGate {
            k,
            rows: inv_gate.rows,
            shift,
            phase,
            scalar: self.scalar.inverse().unwrap(),
        }
    }
    /// The dense matrix (row-major, little-endian).
    pub fn matrix<S: Amplitude>(&self) -> Option<Vec<S>> {
        let dim = 1usize << self.k;
        let mut m = vec![S::zero(); dim * dim];
        for x in 0..dim {
            let (s, y) = self.apply_to_basis(&bits::from_u64(x as u64, self.k));
            let row = bits::to_u64(&y, self.k) as usize;
            m[row * dim + x] = S::from_phasor(s)?;
        }
        Some(m)
    }
    /// The Clifford tableau of a Clifford monomial gate: each `U G U†` is read
    /// off the images of basis states, with `O(k)` work per generator
    /// (no dense matrices).
    pub fn to_tableau(&self) -> Result<CliffordTableau> {
        if !self.is_clifford() {
            return Err(SimError::Invalid(
                "monomial gate: phase polynomial is not Clifford".into(),
            ));
        }
        let k = self.k;
        let inv = self.inverse();
        let image = |g: &Pauli| -> Result<Pauli> {
            // U G U† |x⟩ for x = 0 and x = e_j
            let conj = |x: &Bits| -> (Phasor, Bits) {
                let (s1, y) = inv.apply_to_basis(x);
                let (ph, z) = g.apply_to_basis(&y);
                let (s2, out) = self.apply_to_basis(&z);
                (s1.mul(Phasor::i_pow(ph as i64)).mul(s2), out)
            };
            let (s0, a) = conj(&bits::zero(k));
            let (ph0, b0) = s0.argument();
            if s0.abs_sq_log2() != Some(0) || b0 > 2 {
                return Err(SimError::Invalid(
                    "monomial gate: conjugation leaves the Pauli group".into(),
                ));
            }
            let phase0 = if b0 == 0 { 0 } else { (ph0 << (2 - b0)) as u8 };
            let mut zb = bits::zero(k);
            for j in 0..k {
                let (sj, out) = conj(&bits::unit(k, j));
                let mut want = a.clone();
                bits::toggle(&mut want, j);
                let ratio = sj.mul(s0.inverse().unwrap());
                if out != want || ratio.abs_sq_log2() != Some(0) || ratio.argument().1 > 1 {
                    return Err(SimError::Invalid(
                        "monomial gate: conjugation leaves the Pauli group".into(),
                    ));
                }
                if ratio.argument().1 == 1 {
                    bits::toggle(&mut zb, j);
                }
            }
            Ok(Pauli::new(k, a, zb, phase0))
        };
        let mut xs = Vec::with_capacity(k);
        let mut zs = Vec::with_capacity(k);
        for q in 0..k {
            xs.push(image(&Pauli::x(k, q))?);
        }
        for q in 0..k {
            zs.push(image(&Pauli::z(k, q))?);
        }
        CliffordTableau::from_images(xs, zs)
    }
}

impl fmt::Debug for MonomialGate {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let rows: Vec<String> = self.rows.iter().map(|r| bits::to_string(r, self.k)).collect();
        write!(
            f,
            "MonomialGate(k={}, rows=[{}], shift={}, phase={}, scalar={})",
            self.k,
            rows.join(","),
            bits::to_string(&self.shift, self.k),
            self.phase,
            self.scalar
        )
    }
}

// ---------------------------------------------------------------------------
// gates
// ---------------------------------------------------------------------------

/// A dense gate matrix (row-major, little-endian) on `k` qubits.
#[derive(Clone, Debug, PartialEq)]
pub struct DenseGate {
    /// Number of qubits.
    pub k: usize,
    /// `2^k × 2^k` entries, row-major.
    pub matrix: Vec<C64>,
}

/// A user-defined gate: implement the views a backend may ask for.
pub trait CustomGate: Send + Sync + fmt::Debug {
    /// A name for display.
    fn name(&self) -> String;
    /// Number of qubits.
    fn arity(&self) -> usize;
    /// The Clifford tableau, if the gate is Clifford.
    fn tableau(&self) -> Option<CliffordTableau> {
        None
    }
    /// The monomial form, if the gate is a monomial.
    fn monomial(&self) -> Option<MonomialGate> {
        None
    }
    /// The dense matrix.
    fn matrix(&self) -> Option<DenseGate> {
        None
    }
    /// A decomposition into other gates (on qubits `0..arity`).
    fn decomposition(&self) -> Option<Circuit> {
        None
    }
    /// The adjoint gate.
    fn dagger(&self) -> Option<Gate> {
        None
    }
}

/// A gate.  Named gates follow the usual conventions (controls first;
/// `SqrtX = H S H`, `SqrtY = S SqrtX S†`; `T = diag(1, ζ_8)`).
#[derive(Clone)]
pub enum Gate {
    /// Identity.
    I,
    /// Pauli `X`.
    X,
    /// Pauli `Y`.
    Y,
    /// Pauli `Z`.
    Z,
    /// Hadamard.
    H,
    /// `diag(1, i)`.
    S,
    /// `diag(1, −i)`.
    Sdg,
    /// `√X = H S H`.
    SqrtX,
    /// `√X†`.
    SqrtXdg,
    /// `√Y = S √X S†`.
    SqrtY,
    /// `√Y†`.
    SqrtYdg,
    /// `diag(1, ζ_8)`.
    T,
    /// `diag(1, ζ_8^{-1})`.
    Tdg,
    /// Controlled-`X` (control, target).
    Cx,
    /// Controlled-`Y`.
    Cy,
    /// Controlled-`Z`.
    Cz,
    /// Swap.
    Swap,
    /// Controlled-`S`.
    Cs,
    /// Controlled-`S†`.
    Csdg,
    /// Doubly controlled `Z`.
    Ccz,
    /// Toffoli (controls, target).
    Ccx,
    /// `diag(1, ζ_{2^bits}^k)`.
    Phase {
        /// Numerator `k`.
        k: i64,
        /// Phase bits `r`.
        bits: u32,
    },
    /// Controlled `diag(1, ζ_{2^bits}^k)` (phase on `|11⟩`).
    CPhase {
        /// Numerator `k`.
        k: i64,
        /// Phase bits `r`.
        bits: u32,
    },
    /// The diagonal gate `ζ^{f(x)}` on `f.num_vars()` qubits.
    Diagonal(PhasePoly),
    /// A monomial gate.
    Monomial(MonomialGate),
    /// A Pauli operator (with its phase).
    Pauli(Pauli),
    /// A Clifford unitary given by its tableau (up to global phase).
    Clifford(CliffordTableau),
    /// `exp(−iθX/2)`.
    Rx(f64),
    /// `exp(−iθY/2)`.
    Ry(f64),
    /// `exp(−iθZ/2)`.
    Rz(f64),
    /// A dense unitary (no exactness).
    Unitary(DenseGate),
    /// A user-defined gate.
    Custom(Arc<dyn CustomGate>),
}

fn tableau_of(xs: &[&str], zs: &[&str]) -> CliffordTableau {
    CliffordTableau::from_image_strings(xs, zs).expect("named gate tableau")
}

fn mono_phase(k: usize, bits: u32, terms: &[(i64, &[usize])]) -> MonomialGate {
    let mut p = PhasePoly::new(k, bits);
    for (c, vars) in terms {
        p.add_monomial(*c, vars);
    }
    MonomialGate::diagonal(p)
}

impl Gate {
    /// Number of qubits.
    pub fn arity(&self) -> usize {
        use Gate::*;
        match self {
            I
            | X
            | Y
            | Z
            | H
            | S
            | Sdg
            | SqrtX
            | SqrtXdg
            | SqrtY
            | SqrtYdg
            | T
            | Tdg
            | Phase { .. }
            | Rx(_)
            | Ry(_)
            | Rz(_) => 1,
            Cx | Cy | Cz | Swap | Cs | Csdg | CPhase { .. } => 2,
            Ccz | Ccx => 3,
            Diagonal(p) => p.num_vars(),
            Monomial(m) => m.num_qubits(),
            Pauli(p) => p.num_qubits(),
            Clifford(t) => t.num_qubits(),
            Unitary(d) => d.k,
            Custom(c) => c.arity(),
        }
    }

    /// The name used in circuit text (`H`, `S_DAG`, `PHASE(3,4)`, …).
    pub fn name(&self) -> String {
        use Gate::*;
        match self {
            I => "I".into(),
            X => "X".into(),
            Y => "Y".into(),
            Z => "Z".into(),
            H => "H".into(),
            S => "S".into(),
            Sdg => "S_DAG".into(),
            SqrtX => "SQRT_X".into(),
            SqrtXdg => "SQRT_X_DAG".into(),
            SqrtY => "SQRT_Y".into(),
            SqrtYdg => "SQRT_Y_DAG".into(),
            T => "T".into(),
            Tdg => "T_DAG".into(),
            Cx => "CX".into(),
            Cy => "CY".into(),
            Cz => "CZ".into(),
            Swap => "SWAP".into(),
            Cs => "CS".into(),
            Csdg => "CS_DAG".into(),
            Ccz => "CCZ".into(),
            Ccx => "CCX".into(),
            Phase { k, bits } => format!("PHASE({k},{bits})"),
            CPhase { k, bits } => format!("CPHASE({k},{bits})"),
            Diagonal(p) => format!("DIAG[{p}]"),
            Monomial(m) => format!("MONOMIAL[{m:?}]"),
            Pauli(p) => format!("PAULI[{p}]"),
            Clifford(t) => format!("CLIFFORD[{} qubits]", t.num_qubits()),
            Rx(t) => format!("RX({t})"),
            Ry(t) => format!("RY({t})"),
            Rz(t) => format!("RZ({t})"),
            Unitary(d) => format!("UNITARY[{} qubits]", d.k),
            Custom(c) => c.name(),
        }
    }

    /// The gate as an affine permutation times a diagonal phase, if it is one
    /// (everything but `H`, `√X`, `√Y`, `CCX`, rotations and dense gates).
    pub fn monomial(&self) -> Option<MonomialGate> {
        use Gate::*;
        Some(match self {
            I => MonomialGate::identity(1),
            X => MonomialGate::x_string(1, &[1]),
            Y => MonomialGate::pauli(&crate::pauli::Pauli::y(1, 0)),
            Z => mono_phase(1, 1, &[(1, &[0])]),
            S => mono_phase(1, 2, &[(1, &[0])]),
            Sdg => mono_phase(1, 2, &[(3, &[0])]),
            T => mono_phase(1, 3, &[(1, &[0])]),
            Tdg => mono_phase(1, 3, &[(7, &[0])]),
            Phase { k, bits } => mono_phase(1, *bits, &[(*k, &[0])]),
            CPhase { k, bits } => mono_phase(2, *bits, &[(*k, &[0, 1])]),
            Cz => mono_phase(2, 1, &[(1, &[0, 1])]),
            Cs => mono_phase(2, 2, &[(1, &[0, 1])]),
            Csdg => mono_phase(2, 2, &[(3, &[0, 1])]),
            Ccz => mono_phase(3, 1, &[(1, &[0, 1, 2])]),
            Cx => MonomialGate::new(
                vec![bits::unit(2, 0), bits::from_indices(2, &[0, 1])],
                bits::zero(2),
                PhasePoly::new(2, 1),
                Phasor::ONE,
            )
            .unwrap(),
            Cy => {
                // control 0, target 1: Y on the target when the control is set
                let mut p = PhasePoly::new(2, 2);
                p.add_monomial(1, &[0]);
                p.add_monomial(2, &[0, 1]);
                MonomialGate::new(
                    vec![bits::unit(2, 0), bits::from_indices(2, &[0, 1])],
                    bits::zero(2),
                    p,
                    Phasor::ONE,
                )
                .unwrap()
            }
            Swap => MonomialGate::permutation(&[1, 0]).unwrap(),
            Diagonal(p) => MonomialGate::diagonal(p.clone()),
            Monomial(m) => m.clone(),
            Pauli(p) => MonomialGate::pauli(p),
            Custom(c) => return c.monomial(),
            H | SqrtX | SqrtXdg | SqrtY | SqrtYdg | Ccx | Clifford(_) | Rx(_) | Ry(_) | Rz(_)
            | Unitary(_) => return None,
        })
    }

    /// The Clifford tableau (up to global phase), if the gate is Clifford.
    /// Named gates' tableaux are built once per process.
    pub fn tableau(&self) -> Option<CliffordTableau> {
        use std::sync::OnceLock;
        type T = OnceLock<CliffordTableau>;
        macro_rules! cached {
            ($($v:ident),*) => {
                match self {
                    $( Gate::$v => {
                        #[allow(non_upper_case_globals)]
                        static $v: T = OnceLock::new();
                        return Some($v.get_or_init(|| self.build_tableau().expect("named Clifford")).clone());
                    } )*
                    _ => {}
                }
            };
        }
        cached!(I, X, Y, Z, H, S, Sdg, SqrtX, SqrtXdg, SqrtY, SqrtYdg, Cx, Cy, Cz, Swap);
        self.build_tableau()
    }

    fn build_tableau(&self) -> Option<CliffordTableau> {
        use Gate::*;
        Some(match self {
            I => tableau_of(&["X0"], &["Z0"]),
            X => tableau_of(&["X0"], &["-Z0"]),
            Y => tableau_of(&["-X0"], &["-Z0"]),
            Z => tableau_of(&["-X0"], &["Z0"]),
            H => tableau_of(&["Z0"], &["X0"]),
            S => tableau_of(&["Y0"], &["Z0"]),
            Sdg => tableau_of(&["-Y0"], &["Z0"]),
            SqrtX => tableau_of(&["X0"], &["-Y0"]),
            SqrtXdg => tableau_of(&["X0"], &["Y0"]),
            SqrtY => tableau_of(&["-Z0"], &["X0"]),
            SqrtYdg => tableau_of(&["Z0"], &["-X0"]),
            Cx => tableau_of(&["X0*X1", "X1"], &["Z0", "Z0*Z1"]),
            Cy => tableau_of(&["X0*Y1", "Z0*X1"], &["Z0", "Z0*Z1"]),
            Cz => tableau_of(&["X0*Z1", "Z0*X1"], &["Z0", "Z1"]),
            Swap => tableau_of(&["X1", "X0"], &["Z1", "Z0"]),
            Clifford(t) => t.clone(),
            Pauli(p) => {
                let n = p.num_qubits();
                let sign = |g: &crate::pauli::Pauli| if g.commutes(p) { g.clone() } else { g.negated() };
                let xs: Vec<crate::pauli::Pauli> =
                    (0..n).map(|q| sign(&crate::pauli::Pauli::x(n, q))).collect();
                let zs: Vec<crate::pauli::Pauli> =
                    (0..n).map(|q| sign(&crate::pauli::Pauli::z(n, q))).collect();
                CliffordTableau::from_images(xs, zs).unwrap()
            }
            Custom(c) => return c.tableau(),
            T
            | Tdg
            | Cs
            | Csdg
            | Ccz
            | Ccx
            | Rx(_)
            | Ry(_)
            | Rz(_)
            | Unitary(_)
            | Phase { .. }
            | CPhase { .. }
            | Diagonal(_)
            | Monomial(_) => {
                let m = self.monomial()?;
                return m.to_tableau().ok();
            }
        })
    }

    /// The local lookup table of a named Clifford gate on at most 3 qubits,
    /// computed once per process ([`CliffordTableau::local_lut`]); `None` for
    /// other gates (which build theirs from [`Gate::tableau`]).
    pub fn cached_lut(&self) -> Option<&'static [(u8, u8, u8)]> {
        use std::sync::OnceLock;
        type L = OnceLock<Vec<(u8, u8, u8)>>;
        macro_rules! cache {
            ($($v:ident),*) => {{
                $( static $v: L = OnceLock::new(); )*
                match self {
                    $( Gate::$v => Some($v.get_or_init(|| self.build_tableau().expect("Clifford").local_lut()).as_slice()), )*
                    _ => None,
                }
            }};
        }
        #[allow(non_upper_case_globals)]
        {
            cache!(I, X, Y, Z, H, S, Sdg, SqrtX, SqrtXdg, SqrtY, SqrtYdg, Cx, Cy, Cz, Swap)
        }
    }

    /// Is the gate Clifford?
    pub fn is_clifford(&self) -> bool {
        use Gate::*;
        match self {
            I | X | Y | Z | H | S | Sdg | SqrtX | SqrtXdg | SqrtY | SqrtYdg | Cx | Cy | Cz | Swap
            | Clifford(_) | Pauli(_) => true,
            T | Tdg | Cs | Csdg | Ccz | Ccx => false,
            _ => self.tableau().is_some(),
        }
    }

    /// The dense matrix (row-major, little-endian) in `S`, when every entry
    /// is representable (`H` needs `√2`, rotations need floats).
    pub fn matrix<S: Amplitude>(&self) -> Option<Vec<S>> {
        let half = S::one().add_ref(&S::one()).inverse();
        let i = S::imaginary_unit();
        let one = S::one();
        match self {
            Gate::H => {
                let r = S::sqrt2_pow(-1)?;
                Some(vec![r.clone(), r.clone(), r.clone(), r.negated()])
            }
            Gate::SqrtX => {
                let a = one.add_ref(&i).mul_ref(&half);
                let b = one.sub_ref(&i).mul_ref(&half);
                Some(vec![a.clone(), b.clone(), b, a])
            }
            Gate::SqrtXdg => {
                let a = one.sub_ref(&i).mul_ref(&half);
                let b = one.add_ref(&i).mul_ref(&half);
                Some(vec![a.clone(), b.clone(), b, a])
            }
            Gate::SqrtY => {
                let a = one.add_ref(&i).mul_ref(&half);
                Some(vec![a.clone(), a.negated(), a.clone(), a])
            }
            Gate::SqrtYdg => {
                let a = one.sub_ref(&i).mul_ref(&half);
                Some(vec![a.clone(), a.clone(), a.negated(), a])
            }
            Gate::Ccx => {
                let dim = 8;
                let mut m = vec![S::zero(); dim * dim];
                for x in 0..dim {
                    let y = if x & 3 == 3 { x ^ 4 } else { x };
                    m[y * dim + x] = S::one();
                }
                Some(m)
            }
            Gate::Rx(t) | Gate::Ry(t) | Gate::Rz(t) => {
                let (c, s) = ((t / 2.0).cos(), (t / 2.0).sin());
                let z = C64::new(0.0, 0.0);
                let m = match self {
                    Gate::Rx(_) => [
                        C64::new(c, 0.0),
                        C64::new(0.0, -s),
                        C64::new(0.0, -s),
                        C64::new(c, 0.0),
                    ],
                    Gate::Ry(_) => [
                        C64::new(c, 0.0),
                        C64::new(-s, 0.0),
                        C64::new(s, 0.0),
                        C64::new(c, 0.0),
                    ],
                    _ => [
                        C64::from_polar(1.0, -t / 2.0),
                        z,
                        z,
                        C64::from_polar(1.0, t / 2.0),
                    ],
                };
                m.iter().map(|&e| S::from_c64(e)).collect()
            }
            Gate::Unitary(d) => d.matrix.iter().map(|&e| S::from_c64(e)).collect(),
            Gate::Clifford(t) => t.unitary().ok(),
            Gate::Custom(c) => {
                if let Some(m) = c.monomial() {
                    return m.matrix();
                }
                if let Some(d) = c.matrix() {
                    return d.matrix.iter().map(|&e| S::from_c64(e)).collect();
                }
                if let Some(t) = c.tableau() {
                    return t.unitary().ok();
                }
                c.decomposition().and_then(|circ| circ.unitary().ok())
            }
            _ => self.monomial()?.matrix(),
        }
    }

    /// A decomposition into `H`, Paulis, diagonal and monomial gates on
    /// qubits `0..arity` (for backends that handle those natively), if one is
    /// known.
    pub fn decomposition(&self) -> Option<Circuit> {
        use Gate::*;
        let seq = |gates: &[(Gate, &[usize])]| {
            let mut c = Circuit::new(self.arity());
            for (g, qs) in gates {
                c.push_gate(g.clone(), qs).unwrap();
            }
            c
        };
        match self {
            SqrtX => Some(seq(&[(H, &[0]), (S, &[0]), (H, &[0])])),
            SqrtXdg => Some(seq(&[(H, &[0]), (Sdg, &[0]), (H, &[0])])),
            SqrtY => Some(seq(&[(Sdg, &[0]), (H, &[0]), (S, &[0]), (H, &[0]), (S, &[0])])),
            SqrtYdg => Some(seq(&[(Sdg, &[0]), (H, &[0]), (Sdg, &[0]), (H, &[0]), (S, &[0])])),
            Ccx => Some(seq(&[(H, &[2]), (Ccz, &[0, 1, 2]), (H, &[2])])),
            Clifford(t) => Some(t.to_circuit()),
            Custom(c) => c.decomposition(),
            _ => None,
        }
    }

    /// The adjoint gate.
    pub fn dagger(&self) -> Result<Gate> {
        use Gate::*;
        Ok(match self {
            I | X | Y | Z | H | Cx | Cy | Cz | Swap | Ccz | Ccx => self.clone(),
            S => Sdg,
            Sdg => S,
            SqrtX => SqrtXdg,
            SqrtXdg => SqrtX,
            SqrtY => SqrtYdg,
            SqrtYdg => SqrtY,
            T => Tdg,
            Tdg => T,
            Cs => Csdg,
            Csdg => Cs,
            Phase { k, bits } => Phase { k: -k, bits: *bits },
            CPhase { k, bits } => CPhase { k: -k, bits: *bits },
            Diagonal(p) => {
                let mut q = p.clone();
                q.scale(-1);
                Diagonal(q)
            }
            Monomial(m) => Monomial(m.inverse()),
            Pauli(p) => Pauli(p.adjoint()),
            Clifford(t) => Clifford(t.inverse()),
            Rx(t) => Rx(-t),
            Ry(t) => Ry(-t),
            Rz(t) => Rz(-t),
            Unitary(d) => {
                let dim = 1 << d.k;
                let m = (0..dim * dim)
                    .map(|ij| d.matrix[(ij % dim) * dim + ij / dim].conj())
                    .collect();
                Unitary(DenseGate { k: d.k, matrix: m })
            }
            Custom(c) => c.dagger().ok_or_else(|| SimError::Unsupported {
                backend: "gate",
                what: format!("adjoint of custom gate {}", c.name()),
            })?,
        })
    }

    /// Is the gate diagonal in the computational basis?
    pub fn is_diagonal(&self) -> bool {
        match self {
            Gate::Rz(_) => true,
            g => g.monomial().map(|m| m.is_diagonal()).unwrap_or(false),
        }
    }
}

impl fmt::Debug for Gate {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.name())
    }
}

impl fmt::Display for Gate {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.name())
    }
}

impl PartialEq for Gate {
    fn eq(&self, other: &Gate) -> bool {
        use Gate::*;
        match (self, other) {
            (Phase { k: a, bits: b }, Phase { k: c, bits: d }) => a == c && b == d,
            (CPhase { k: a, bits: b }, CPhase { k: c, bits: d }) => a == c && b == d,
            (Diagonal(a), Diagonal(b)) => a == b,
            (Monomial(a), Monomial(b)) => a == b,
            (Pauli(a), Pauli(b)) => a == b,
            (Clifford(a), Clifford(b)) => a == b,
            (Rx(a), Rx(b)) | (Ry(a), Ry(b)) | (Rz(a), Rz(b)) => a == b,
            (Unitary(a), Unitary(b)) => a == b,
            (Custom(a), Custom(b)) => Arc::ptr_eq(a, b),
            (a, b) => {
                std::mem::discriminant(a) == std::mem::discriminant(b) && a.arity() == b.arity() && {
                    // named gates: same variant
                    !matches!(
                        a,
                        Phase { .. }
                            | CPhase { .. }
                            | Diagonal(_)
                            | Monomial(_)
                            | Pauli(_)
                            | Clifford(_)
                            | Rx(_)
                            | Ry(_)
                            | Rz(_)
                            | Unitary(_)
                            | Custom(_)
                    )
                }
            }
        }
    }
}

/// The `k × k` identity as a [`BitMat`] (helper for callers building monomials).
pub fn identity_bitmat(k: usize) -> BitMat {
    BitMat::identity(k)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn close(a: &[C64], b: &[C64]) -> bool {
        a.len() == b.len() && a.iter().zip(b).all(|(x, y)| (x - y).norm() < 1e-12)
    }

    fn matmul(a: &[C64], b: &[C64], dim: usize) -> Vec<C64> {
        let mut c = vec![C64::new(0.0, 0.0); dim * dim];
        for i in 0..dim {
            for j in 0..dim {
                for k in 0..dim {
                    c[i * dim + j] += a[i * dim + k] * b[k * dim + j];
                }
            }
        }
        c
    }

    fn dagger(a: &[C64], dim: usize) -> Vec<C64> {
        (0..dim * dim)
            .map(|ij| a[(ij % dim) * dim + ij / dim].conj())
            .collect()
    }

    #[test]
    fn views_agree_with_matrices() {
        use Gate::*;
        let gates = [
            I,
            X,
            Y,
            Z,
            H,
            S,
            Sdg,
            SqrtX,
            SqrtXdg,
            SqrtY,
            SqrtYdg,
            T,
            Tdg,
            Cx,
            Cy,
            Cz,
            Swap,
            Cs,
            Csdg,
            Ccz,
            Ccx,
            Phase { k: 3, bits: 4 },
            CPhase { k: 1, bits: 3 },
        ];
        for g in &gates {
            let k = g.arity();
            let dim = 1 << k;
            let m: Vec<C64> = g.matrix().unwrap();
            // unitary
            let id = matmul(&m, &dagger(&m, dim), dim);
            for i in 0..dim {
                for j in 0..dim {
                    let want = if i == j { 1.0 } else { 0.0 };
                    assert!(
                        (id[i * dim + j] - C64::new(want, 0.0)).norm() < 1e-12,
                        "{g} not unitary"
                    );
                }
            }
            // dagger
            let md: Vec<C64> = g.dagger().unwrap().matrix().unwrap();
            assert!(close(&md, &dagger(&m, dim)), "{g} dagger");
            // tableau conjugation agrees with the matrix
            if let Some(t) = g.tableau() {
                for q in 0..k {
                    for p in [crate::pauli::Pauli::x(k, q), crate::pauli::Pauli::z(k, q)] {
                        let pm: Vec<C64> = p.matrix();
                        let lhs = matmul(&matmul(&m, &pm, dim), &dagger(&m, dim), dim);
                        let rhs: Vec<C64> = t.conjugate(&p).matrix();
                        assert!(close(&lhs, &rhs), "{g}: tableau image of {p}");
                    }
                }
            } else {
                assert!(
                    matches!(g, T | Tdg | Cs | Csdg | Ccz | Ccx | Phase { .. } | CPhase { .. }),
                    "{g} should be Clifford"
                );
            }
            // decomposition
            if let Some(c) = g.decomposition() {
                let cm: Vec<C64> = c.unitary().unwrap();
                assert!(close(&cm, &m), "{g} decomposition");
            }
            // monomial inverse and composition
            if let Some(mo) = g.monomial() {
                let inv = mo.inverse();
                assert_eq!(inv.compose(&mo), MonomialGate::identity(k), "{g} inverse");
                let m2: Vec<C64> = mo.compose(&mo).matrix().unwrap();
                assert!(close(&m2, &matmul(&m, &m, dim)), "{g} composition");
            }
        }
        // exact matrices
        let t: Vec<Cyclo<8>> = Gate::T.matrix().unwrap();
        assert_eq!(t[3], Cyclo::<8>::zeta(1));
        assert!(Gate::H.matrix::<Cyclo<4>>().is_none());
        assert!(Gate::H.matrix::<Cyclo<8>>().is_some());
        assert!(Gate::Rz(0.3).matrix::<Cyclo<8>>().is_none());
    }

    use crate::Cyclo;
}
