//! Pauli operators on `n` qubits, bit-packed: [`Pauli`] (one operator),
//! [`PauliTable`] (rows of them, the storage of tableaux) and [`PauliSum`]
//! (linear combinations: observables, Hamiltonians).
//!
//! Convention (see [`ops`]): `P = i^phase · ⊗_q X^{x_q} Z^{z_q}`, phase mod 4,
//! `Y = i X Z`.  Qubit `q` is bit `q` of the packed `x` and `z` vectors.  The
//! stored `phase` is relative to the product `X^x Z^z`; the letter form
//! `i^k · ⊗_q σ_q` with `σ_q ∈ {I, X, Y, Z}` (as printed, parsed and read by
//! [`Pauli::kind`]) has `k = phase − #{q : σ_q = Y}` ([`Pauli::display_phase`]),
//! so a Hermitian operator has `k ∈ {0, 2}`.

mod columns;
pub mod ops;
mod sum;
mod table;

pub use columns::ColumnTable;
pub use sum::PauliSum;
pub use table::PauliTable;

use crate::bits::{self, Bits};
use crate::error::{Result, SimError};
use crate::scalar::Amplitude;
use std::fmt;

/// A single-qubit Pauli.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum PauliKind {
    /// Identity.
    I,
    /// `X`.
    X,
    /// `Y = i X Z`.
    Y,
    /// `Z`.
    Z,
}

impl PauliKind {
    /// From the `(x, z)` bits of `X^x Z^z` (with `Y` for `(1, 1)`).
    pub fn from_bits(x: bool, z: bool) -> Self {
        match (x, z) {
            (false, false) => PauliKind::I,
            (true, false) => PauliKind::X,
            (true, true) => PauliKind::Y,
            (false, true) => PauliKind::Z,
        }
    }
    /// The `(x, z)` bits.
    pub fn bits(self) -> (bool, bool) {
        match self {
            PauliKind::I => (false, false),
            PauliKind::X => (true, false),
            PauliKind::Y => (true, true),
            PauliKind::Z => (false, true),
        }
    }
    /// `I`, `X`, `Y` or `Z`.
    pub fn symbol(self) -> char {
        match self {
            PauliKind::I => 'I',
            PauliKind::X => 'X',
            PauliKind::Y => 'Y',
            PauliKind::Z => 'Z',
        }
    }
    /// From `I`, `X`, `Y`, `Z` or `_`.
    pub fn from_char(c: char) -> Option<Self> {
        match c {
            'I' | '_' => Some(PauliKind::I),
            'X' => Some(PauliKind::X),
            'Y' => Some(PauliKind::Y),
            'Z' => Some(PauliKind::Z),
            _ => None,
        }
    }
}

/// `i^phase · ⊗_q X^{x_q} Z^{z_q}` on `n` qubits.
#[derive(Clone, PartialEq, Eq, Hash)]
pub struct Pauli {
    n: usize,
    x: Bits,
    z: Bits,
    phase: u8,
}

impl Pauli {
    /// The identity on `n` qubits.
    pub fn identity(n: usize) -> Self {
        Pauli {
            n,
            x: bits::zero(n),
            z: bits::zero(n),
            phase: 0,
        }
    }

    /// From packed `x`, `z` bits and a phase (`i^phase`).  Bits beyond `n` are cleared.
    pub fn new(n: usize, mut x: Bits, mut z: Bits, phase: u8) -> Self {
        bits::ensure_len(&mut x, n);
        bits::ensure_len(&mut z, n);
        x.truncate(bits::nwords(n));
        z.truncate(bits::nwords(n));
        bits::mask_to(&mut x, n);
        bits::mask_to(&mut z, n);
        Pauli {
            n,
            x,
            z,
            phase: phase & 3,
        }
    }

    /// The Hermitian operator `⊗_q σ_q` (letter phase `0`).
    pub fn from_kinds(kinds: &[PauliKind]) -> Self {
        let mut p = Pauli::identity(kinds.len());
        for (q, &k) in kinds.iter().enumerate() {
            p.set_kind(q, k);
        }
        p
    }

    /// `i^k · ⊗_q σ_q`.
    pub fn from_kinds_with_phase(kinds: &[PauliKind], k: u8) -> Self {
        let mut p = Self::from_kinds(kinds);
        p.phase = (p.phase + k) & 3;
        p
    }

    /// The letter phase `k` with `P = i^k ⊗_q σ_q`: the stored phase minus
    /// the number of `Y` letters.
    pub fn display_phase(&self) -> u8 {
        let ys = (bits::and_count(&self.x, &self.z) % 4) as u8;
        (self.phase + 4 - ys) & 3
    }

    /// A single-site Pauli.
    pub fn single(n: usize, q: usize, kind: PauliKind) -> Self {
        let mut p = Pauli::identity(n);
        p.set_kind(q, kind);
        p
    }
    /// `X_q`.
    pub fn x(n: usize, q: usize) -> Self {
        Self::single(n, q, PauliKind::X)
    }
    /// `Y_q`.
    pub fn y(n: usize, q: usize) -> Self {
        Self::single(n, q, PauliKind::Y)
    }
    /// `Z_q`.
    pub fn z(n: usize, q: usize) -> Self {
        Self::single(n, q, PauliKind::Z)
    }
    /// `X^a` for a bit string `a` on `n` qubits.
    pub fn x_string(n: usize, a: &[u64]) -> Self {
        Pauli::new(n, a.to_vec(), bits::zero(n), 0)
    }
    /// `Z^b` for a bit string `b` on `n` qubits.
    pub fn z_string(n: usize, b: &[u64]) -> Self {
        Pauli::new(n, bits::zero(n), b.to_vec(), 0)
    }

    /// Number of qubits.
    pub fn num_qubits(&self) -> usize {
        self.n
    }
    /// The phase exponent (`i^phase`).
    pub fn phase(&self) -> u8 {
        self.phase
    }
    /// Set the phase exponent.
    pub fn set_phase(&mut self, phase: u8) {
        self.phase = phase & 3;
    }
    /// Packed `x` bits.
    pub fn x_bits(&self) -> &[u64] {
        &self.x
    }
    /// Packed `z` bits.
    pub fn z_bits(&self) -> &[u64] {
        &self.z
    }
    /// Destructure into `(n, x, z, phase)`.
    pub fn into_parts(self) -> (usize, Bits, Bits, u8) {
        (self.n, self.x, self.z, self.phase)
    }
    /// The Pauli on qubit `q`.
    pub fn kind(&self, q: usize) -> PauliKind {
        PauliKind::from_bits(bits::get(&self.x, q), bits::get(&self.z, q))
    }
    /// Set the letter on qubit `q`, keeping the letter phase (a `Y` put in or
    /// taken out adjusts the stored phase by the `i` in `Y = i X Z`).
    pub fn set_kind(&mut self, q: usize, kind: PauliKind) {
        assert!(q < self.n, "qubit {q} out of range");
        if self.kind(q) == PauliKind::Y {
            self.phase = (self.phase + 3) & 3;
        }
        let (x, z) = kind.bits();
        bits::set(&mut self.x, q, x);
        bits::set(&mut self.z, q, z);
        if kind == PauliKind::Y {
            self.phase = (self.phase + 1) & 3;
        }
    }
    /// The kinds on all qubits.
    pub fn kinds(&self) -> Vec<PauliKind> {
        (0..self.n).map(|q| self.kind(q)).collect()
    }
    /// Number of non-identity sites.
    pub fn weight(&self) -> usize {
        ops::weight(&self.x, &self.z)
    }
    /// The non-identity sites, increasing.
    pub fn support(&self) -> Vec<usize> {
        let mut v: Vec<u64> = self.x.iter().zip(&self.z).map(|(a, b)| a | b).collect();
        bits::mask_to(&mut v, self.n);
        bits::ones(&v)
    }
    /// Is this a scalar `i^phase · 1`?
    pub fn is_identity(&self) -> bool {
        bits::is_zero(&self.x) && bits::is_zero(&self.z)
    }
    /// Hermitian?
    pub fn is_hermitian(&self) -> bool {
        ops::is_hermitian(&self.x, &self.z, self.phase)
    }
    /// Do the two operators commute?
    pub fn commutes(&self, other: &Pauli) -> bool {
        ops::commute(&self.x, &self.z, &other.x, &other.z)
    }
    /// The product `self · other`.
    pub fn mul(&self, other: &Pauli) -> Pauli {
        let mut p = self.clone();
        p.mul_assign(other);
        p
    }
    /// `self ← self · other`.
    pub fn mul_assign(&mut self, other: &Pauli) {
        assert_eq!(self.n, other.n, "Pauli operators on different registers");
        self.phase = ops::product_phase(self.phase, &self.z, other.phase, &other.x);
        bits::xor_into(&mut self.x, &other.x);
        bits::xor_into(&mut self.z, &other.z);
    }
    /// `self ← other · self`.
    pub fn premul_assign(&mut self, other: &Pauli) {
        assert_eq!(self.n, other.n, "Pauli operators on different registers");
        self.phase = ops::product_phase(other.phase, &other.z, self.phase, &self.x);
        bits::xor_into(&mut self.x, &other.x);
        bits::xor_into(&mut self.z, &other.z);
    }
    /// `x ^= dx`, `z ^= dz` (phase untouched).
    pub fn xor_bits(&mut self, dx: &[u64], dz: &[u64]) {
        bits::xor_into(&mut self.x, dx);
        bits::xor_into(&mut self.z, dz);
    }
    /// The adjoint.
    pub fn adjoint(&self) -> Pauli {
        let mut p = self.clone();
        p.phase = ops::adjoint_phase(&self.x, &self.z, self.phase);
        p
    }
    /// `−self`.
    pub fn negated(&self) -> Pauli {
        self.with_phase_added(2)
    }
    /// `i · self`.
    pub fn times_i(&self) -> Pauli {
        self.with_phase_added(1)
    }
    /// `i^k · self`.
    pub fn with_phase_added(&self, k: u8) -> Pauli {
        let mut p = self.clone();
        p.phase = (p.phase + k) & 3;
        p
    }
    /// The operator restricted to `qubits` (in that order), phase kept.  The
    /// operator must be the identity elsewhere for this to be a restriction;
    /// no check is made.
    pub fn restrict(&self, qubits: &[usize]) -> Pauli {
        let mut p = Pauli::identity(qubits.len());
        p.phase = self.phase;
        for (i, &q) in qubits.iter().enumerate() {
            bits::set(&mut p.x, i, bits::get(&self.x, q));
            bits::set(&mut p.z, i, bits::get(&self.z, q));
        }
        p
    }
    /// Place an operator on `qubits.len()` qubits onto `qubits` of an
    /// `n`-qubit register.
    pub fn embed(&self, n: usize, qubits: &[usize]) -> Pauli {
        assert_eq!(self.n, qubits.len(), "embed: qubit count");
        let mut p = Pauli::identity(n);
        p.phase = self.phase;
        for (i, &q) in qubits.iter().enumerate() {
            bits::set(&mut p.x, q, bits::get(&self.x, i));
            bits::set(&mut p.z, q, bits::get(&self.z, i));
        }
        p
    }
    /// Set the raw bits `x_q`, `z_q` without touching the phase exponent
    /// (unlike [`Pauli::set_kind`], which keeps the letter phase).
    pub fn set_raw_bits(&mut self, q: usize, x: bool, z: bool) {
        bits::set(&mut self.x, q, x);
        bits::set(&mut self.z, q, z);
    }
    /// `self ⊗ other` on `n_self + n_other` qubits (`self` on the low
    /// qubits); phases add.
    pub fn tensor(&self, other: &Pauli) -> Pauli {
        let (na, nb) = (self.n, other.n);
        Pauli::new(
            na + nb,
            bits::concat(&self.x, na, &other.x, nb),
            bits::concat(&self.z, na, &other.z, nb),
            (self.phase + other.phase) & 3,
        )
    }
    /// `P |v⟩ = i^k |v ⊕ x⟩`: returns `(k, v ⊕ x)`.
    pub fn apply_to_basis(&self, v: &[u64]) -> (u8, Bits) {
        let k = ops::basis_image_phase(&self.z, self.phase, v);
        let mut w = v.to_vec();
        bits::xor_into(&mut w, &self.x);
        (k, w)
    }
    /// The dense `2^n × 2^n` matrix, row-major, little-endian basis (qubit `q`
    /// is bit `q` of the index).  For checks on few qubits.
    pub fn matrix<S: Amplitude>(&self) -> Vec<S> {
        assert!(self.n <= 20, "dense matrix of more than 20 qubits");
        let dim = 1usize << self.n;
        let mut m = vec![S::zero(); dim * dim];
        for col in 0..dim {
            let v = bits::from_u64(col as u64, self.n);
            let (k, w) = self.apply_to_basis(&v);
            let row = bits::to_u64(&w, self.n) as usize;
            m[row * dim + col] = S::i_pow(k as i64);
        }
        m
    }

    /// Parse either form, inferring the register size: dense `+XZ_Y` (one
    /// letter per qubit, `_` or `I` the identity) or sparse `-i*X0*Z1`
    /// (`n` = largest index + 1).
    pub fn parse(s: &str) -> Result<Self> {
        Self::parse_with(s, None)
    }

    /// Parse either form on a register of `n` qubits.
    pub fn from_string(n: usize, s: &str) -> Result<Self> {
        Self::parse_with(s, Some(n))
    }

    fn parse_with(s: &str, n: Option<usize>) -> Result<Self> {
        let err = |m: String| SimError::Parse(format!("Pauli '{s}': {m}"));
        let t = s.trim();
        // sign / phase prefix
        let (mut phase, mut rest) = (0u8, t);
        for (pre, ph) in [("-i", 3u8), ("+i", 1), ("-", 2), ("+", 0), ("i", 1)] {
            if let Some(r) = rest.strip_prefix(pre) {
                let r = r.trim_start();
                let r = r.strip_prefix('*').unwrap_or(r).trim_start();
                // "i" alone could be followed by nothing (identity) or by a Pauli
                phase = ph;
                rest = r;
                break;
            }
        }
        let dense = !rest.is_empty() && rest.chars().all(|c| "IXYZ_".contains(c));
        if dense {
            let kinds: Vec<PauliKind> = rest.chars().map(|c| PauliKind::from_char(c).unwrap()).collect();
            let p = Pauli::from_kinds_with_phase(&kinds, phase);
            if let Some(n) = n {
                if n != kinds.len() {
                    return Err(err(format!("{} letters for {} qubits", kinds.len(), n)));
                }
            }
            return Ok(p);
        }
        let mut sites: Vec<(usize, PauliKind)> = Vec::new();
        for tok in rest
            .split(|c: char| c == '*' || c.is_whitespace())
            .filter(|t| !t.is_empty())
        {
            if tok == "1" {
                continue;
            }
            let mut chars = tok.chars();
            let k = chars
                .next()
                .and_then(PauliKind::from_char)
                .ok_or_else(|| err(format!("token '{tok}'")))?;
            let idx: usize = chars
                .as_str()
                .parse()
                .map_err(|_| err(format!("token '{tok}'")))?;
            if sites.iter().any(|(q, _)| *q == idx) {
                return Err(err(format!("qubit {idx} repeated")));
            }
            sites.push((idx, k));
        }
        let n = match n {
            Some(n) => n,
            None => sites.iter().map(|(q, _)| q + 1).max().unwrap_or(0),
        };
        let mut p = Pauli::identity(n);
        for (q, k) in sites {
            if q >= n {
                return Err(err(format!("qubit {q} out of range for {n} qubits")));
            }
            p.set_kind(q, k);
        }
        p.phase = (p.phase + phase) & 3;
        Ok(p)
    }

    /// Dense form `+XZ_Y` with a prefix among `+`, `-`, `+i`, `-i` (the letter phase).
    pub fn to_dense_string(&self) -> String {
        let pre = ["+", "+i", "-", "-i"][self.display_phase() as usize];
        let body: String = (0..self.n)
            .map(|q| match self.kind(q) {
                PauliKind::I => '_',
                k => k.symbol(),
            })
            .collect();
        format!("{pre}{body}")
    }

    /// Sparse form `-i*X0*Z1` (`1`, `-1`, `i`, `-i` for scalars), the style of
    /// ncalg polynomials over the Pauli algebra.
    pub fn to_sparse_string(&self) -> String {
        let k = self.display_phase() as usize;
        let pre = ["", "i*", "-", "-i*"][k];
        let sites: Vec<String> = self
            .support()
            .iter()
            .map(|&q| format!("{}{q}", self.kind(q).symbol()))
            .collect();
        if sites.is_empty() {
            return ["1", "i", "-1", "-i"][k].to_string();
        }
        format!("{pre}{}", sites.join("*"))
    }
}

impl fmt::Display for Pauli {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.to_sparse_string())
    }
}

impl fmt::Debug for Pauli {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Pauli({}; {} qubits)", self.to_sparse_string(), self.n)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::scalar::C64;

    #[test]
    fn products_and_phases() {
        let x = Pauli::x(1, 0);
        let y = Pauli::y(1, 0);
        let z = Pauli::z(1, 0);
        assert_eq!(x.mul(&y), z.times_i());
        assert_eq!(y.mul(&x), z.times_i().negated());
        assert_eq!(y.mul(&z), x.times_i());
        assert_eq!(z.mul(&x), y.times_i());
        assert_eq!(y.mul(&y), Pauli::identity(1));
        assert!(y.is_hermitian() && x.is_hermitian());
        assert!(!x.mul(&z).is_hermitian());
        assert_eq!(x.mul(&z).adjoint(), z.mul(&x));
        assert!(!x.commutes(&z));
        let p = Pauli::parse("-i*X0*Z2").unwrap();
        assert_eq!(p.num_qubits(), 3);
        assert_eq!(p.to_dense_string(), "-iX_Z");
        assert_eq!(Pauli::parse("+XZ_Y").unwrap().to_sparse_string(), "X0*Z1*Y3");
        assert_eq!(Pauli::parse("i").unwrap(), Pauli::identity(0).times_i());
        assert_eq!(Pauli::from_string(4, "Y1").unwrap().restrict(&[1]), y);
        assert_eq!(y.embed(4, &[1]), Pauli::from_string(4, "Y1").unwrap());
    }

    #[test]
    fn matrices_multiply() {
        let a = Pauli::parse("X0*Y1").unwrap();
        let b = Pauli::parse("Z0*Z1").unwrap();
        let ma: Vec<C64> = a.matrix();
        let mb: Vec<C64> = b.matrix();
        let mab: Vec<C64> = a.mul(&b).matrix();
        let dim = 4;
        for i in 0..dim {
            for j in 0..dim {
                let mut s = C64::new(0.0, 0.0);
                for k in 0..dim {
                    s += ma[i * dim + k] * mb[k * dim + j];
                }
                assert!((s - mab[i * dim + j]).norm() < 1e-12);
            }
        }
    }
}
