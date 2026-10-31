//! Clifford unitaries as tableaux: `U` is stored, up to a global phase, by the
//! Pauli images `U X_q U†` and `U Z_q U†`.
//!
//! Conjugation of any Pauli, composition, inversion and local application of
//! gates are symplectic linear algebra over `𝔽₂` with phase bookkeeping; the
//! same tableau type describes a gate (on its own qubits) and a whole circuit.

pub mod decompose;
pub mod monomial;
pub mod random;

pub use random::{random_clifford, random_pauli, random_stabilizer_state};

use crate::bits;
use crate::error::{check_qubits, Result, SimError};
use crate::pauli::{Pauli, PauliTable};
use crate::scalar::Amplitude;
use ncalg::ff::BitMat;
use std::fmt;

/// A Clifford unitary on `n` qubits, up to a global phase: rows `0..n` of
/// `images` are `U X_q U†`, rows `n..2n` are `U Z_q U†` (Hermitian Paulis
/// satisfying the Pauli relations).
#[derive(Clone, PartialEq, Eq)]
pub struct CliffordTableau {
    n: usize,
    images: PauliTable,
}

impl CliffordTableau {
    /// The identity.
    pub fn identity(n: usize) -> Self {
        let mut images = PauliTable::new(n);
        for q in 0..n {
            images.push(&Pauli::x(n, q));
        }
        for q in 0..n {
            images.push(&Pauli::z(n, q));
        }
        CliffordTableau { n, images }
    }

    /// From the images of `X_q` and of `Z_q`; checks the Pauli relations.
    pub fn from_images(x_images: Vec<Pauli>, z_images: Vec<Pauli>) -> Result<Self> {
        let n = x_images.len();
        if z_images.len() != n {
            return Err(SimError::Invalid("tableau: X and Z image counts differ".into()));
        }
        let mut images = PauliTable::new(n);
        for p in x_images.iter().chain(z_images.iter()) {
            if p.num_qubits() != n {
                return Err(SimError::Invalid("tableau: image on the wrong register".into()));
            }
            images.push(p);
        }
        let t = CliffordTableau { n, images };
        if !t.is_valid() {
            return Err(SimError::Invalid(
                "tableau: images do not satisfy the Pauli relations".into(),
            ));
        }
        Ok(t)
    }

    /// From the images without checking the Pauli relations (for tables
    /// known to be valid, such as a stabilizer tableau).
    pub(crate) fn from_images_unchecked(images: PauliTable) -> Self {
        CliffordTableau {
            n: images.num_qubits(),
            images,
        }
    }

    /// From image strings (`Pauli::from_string` syntax on `n = xs.len()` qubits).
    pub fn from_image_strings(xs: &[&str], zs: &[&str]) -> Result<Self> {
        let n = xs.len();
        let parse = |s: &&str| Pauli::from_string(n, s);
        let x: Result<Vec<Pauli>> = xs.iter().map(parse).collect();
        let z: Result<Vec<Pauli>> = zs.iter().map(parse).collect();
        Self::from_images(x?, z?)
    }

    /// Are the images Hermitian and do they satisfy the Pauli relations
    /// (`X_q`, `Z_q` anticommute, everything else commutes)?
    pub fn is_valid(&self) -> bool {
        let n = self.n;
        let t = &self.images;
        if t.rows() != 2 * n {
            return false;
        }
        for i in 0..2 * n {
            if !t.is_hermitian(i) {
                return false;
            }
            for j in i + 1..2 * n {
                let want_anti = j == i + n;
                if t.rows_commute(i, j) == want_anti {
                    return false;
                }
            }
        }
        true
    }

    /// Number of qubits.
    pub fn num_qubits(&self) -> usize {
        self.n
    }
    /// `U X_q U†`.
    pub fn x_image(&self, q: usize) -> Pauli {
        self.images.row(q)
    }
    /// `U Z_q U†`.
    pub fn z_image(&self, q: usize) -> Pauli {
        self.images.row(self.n + q)
    }
    /// All images as a table (`X` images first).
    pub fn images(&self) -> &PauliTable {
        &self.images
    }
    /// The `2n × 2n` symplectic matrix (row `i` = `(x | z)` of image `i`).
    pub fn symplectic(&self) -> BitMat {
        self.images.symplectic()
    }
    /// The phase exponents of the images.
    pub fn phases(&self) -> Vec<u8> {
        (0..2 * self.n).map(|i| self.images.phase(i)).collect()
    }
    /// From a symplectic matrix and phases (checked).
    pub fn from_symplectic(m: &BitMat, phases: &[u8]) -> Result<Self> {
        let t = PauliTable::from_symplectic(m, phases);
        let n = t.num_qubits();
        let c = CliffordTableau { n, images: t };
        if !c.is_valid() {
            return Err(SimError::Invalid(
                "tableau: not symplectic or not Hermitian".into(),
            ));
        }
        Ok(c)
    }

    /// `U P U†`.
    pub fn conjugate(&self, p: &Pauli) -> Pauli {
        assert_eq!(p.num_qubits(), self.n, "conjugate: register size");
        let mut out = Pauli::identity(self.n);
        out.set_phase(p.phase());
        for q in bits::ones(p.x_bits()) {
            if q < self.n {
                self.images.mul_pauli_by_row(&mut out, q);
            }
        }
        for q in bits::ones(p.z_bits()) {
            if q < self.n {
                self.images.mul_pauli_by_row(&mut out, self.n + q);
            }
        }
        out
    }

    /// `U† P U`.
    pub fn conjugate_inverse(&self, p: &Pauli) -> Pauli {
        self.inverse().conjugate(p)
    }

    /// Conjugate `p` (on a larger register) by this tableau acting on `qubits`.
    pub fn conjugate_local(&self, p: &mut Pauli, qubits: &[usize]) {
        debug_assert_eq!(qubits.len(), self.n);
        let img = self.conjugate(&p.restrict(qubits));
        // raw bits, not letters: the phase is carried by `img.phase()`
        for (i, &q) in qubits.iter().enumerate() {
            p.set_raw_bits(q, bits::get(img.x_bits(), i), bits::get(img.z_bits(), i));
        }
        p.set_phase(img.phase());
    }

    /// The local lookup table of a tableau on at most 3 qubits: entry
    /// `x | z << k` is `(image x bits, image z bits, phase)` of
    /// `U X^x Z^z U†` (used by [`PauliTable::conjugate_rows_lut`]).
    pub fn local_lut(&self) -> Vec<(u8, u8, u8)> {
        let k = self.n;
        assert!(k <= 3, "local_lut: more than 3 qubits");
        (0..(1usize << (2 * k)))
            .map(|idx| {
                let x = bits::from_u64((idx & ((1 << k) - 1)) as u64, k);
                let z = bits::from_u64((idx >> k) as u64, k);
                let img = self.conjugate(&Pauli::new(k, x, z, 0));
                (
                    bits::to_u64(img.x_bits(), k) as u8,
                    bits::to_u64(img.z_bits(), k) as u8,
                    img.phase(),
                )
            })
            .collect()
    }

    /// Conjugate every row of a table by this tableau acting on `qubits`:
    /// the update of a stabilizer or Clifford tableau by a gate.  For gates
    /// on at most three qubits through the `4^k` local lookup table.
    pub fn conjugate_table(&self, t: &mut PauliTable, qubits: &[usize]) {
        assert_eq!(qubits.len(), self.n, "conjugate_table: gate arity");
        let k = self.n;
        if k == 0 {
            return;
        }
        if k <= 3 {
            t.conjugate_rows_lut(&self.local_lut(), qubits);
        } else {
            for i in 0..t.rows() {
                let mut p = t.row(i);
                self.conjugate_local(&mut p, qubits);
                t.set_row(i, &p);
            }
        }
    }

    /// `self ∘ other` (apply `other` first, then `self`).
    pub fn compose(&self, other: &Self) -> Self {
        assert_eq!(self.n, other.n, "compose: register size");
        let mut images = PauliTable::new(self.n);
        for i in 0..2 * self.n {
            images.push(&self.conjugate(&other.images.row(i)));
        }
        CliffordTableau { n: self.n, images }
    }

    /// `other ∘ self` (apply `self` first).
    pub fn then(&self, other: &Self) -> Self {
        other.compose(self)
    }

    /// `U ← G ∘ U` for a gate `G` on `qubits`.
    pub fn apply_gate(&mut self, gate: &CliffordTableau, qubits: &[usize]) -> Result<()> {
        check_qubits(qubits, self.n)?;
        if qubits.len() != gate.n {
            return Err(SimError::Arity {
                gate: "Clifford".into(),
                expected: gate.n,
                got: qubits.len(),
            });
        }
        gate.conjugate_table(&mut self.images, qubits);
        Ok(())
    }

    /// `U ← G ∘ U` for a gate given as a [`Gate`](crate::gate::Gate) (named gates through their
    /// cached lookup tables); an error for non-Clifford gates.
    pub fn apply_named_gate(&mut self, gate: &crate::gate::Gate, qubits: &[usize]) -> Result<()> {
        check_qubits(qubits, self.n)?;
        if qubits.len() != gate.arity() {
            return Err(SimError::Arity {
                gate: gate.name(),
                expected: gate.arity(),
                got: qubits.len(),
            });
        }
        if let Some(lut) = gate.cached_lut() {
            self.images.conjugate_rows_lut(lut, qubits);
            return Ok(());
        }
        let g = gate.tableau().ok_or_else(|| SimError::Unsupported {
            backend: "tableau",
            what: format!("non-Clifford gate {gate}"),
        })?;
        self.apply_gate(&g, qubits)
    }

    /// `U ← U ∘ G` for a gate `G` on `qubits`.
    pub fn prepend_gate(&mut self, gate: &CliffordTableau, qubits: &[usize]) -> Result<()> {
        check_qubits(qubits, self.n)?;
        if qubits.len() != gate.n {
            return Err(SimError::Arity {
                gate: "Clifford".into(),
                expected: gate.n,
                got: qubits.len(),
            });
        }
        let k = gate.n;
        let mut new_rows = Vec::with_capacity(2 * k);
        for i in 0..2 * k {
            let g = gate.images.row(i).embed(self.n, qubits);
            new_rows.push(self.conjugate(&g));
        }
        for (i, &q) in qubits.iter().enumerate() {
            self.images.set_row(q, &new_rows[i]);
            self.images.set_row(self.n + q, &new_rows[k + i]);
        }
        Ok(())
    }

    /// The inverse `U†`: the symplectic inverse `Λ Mᵀ Λ`, phases fixed by
    /// conjugating each preimage.
    pub fn inverse(&self) -> Self {
        let n = self.n;
        let m = self.symplectic();
        let mut inv = BitMat::zeros(2 * n, 2 * n);
        // (Λ Mᵀ Λ)[i][j] = Mᵀ[σ(i)][σ(j)] = M[σ(j)][σ(i)], σ swapping the halves
        let swap = |i: usize| if i < n { i + n } else { i - n };
        for i in 0..2 * n {
            for j in 0..2 * n {
                if m.get(swap(j), swap(i)) {
                    inv.set(i, j, true);
                }
            }
        }
        let mut t = PauliTable::from_symplectic(&inv, &vec![0u8; 2 * n]);
        for i in 0..2 * n {
            let img = self.conjugate(&t.row(i));
            debug_assert!(img.weight() == 1, "inverse: preimage does not map to a generator");
            t.set_phase(i, (4 - img.phase()) & 3);
        }
        CliffordTableau { n, images: t }
    }

    /// The dense unitary (row-major, little-endian) with the global phase
    /// fixed so that the first nonzero entry of the first column is positive:
    /// for checks on few qubits.
    pub fn unitary<S: Amplitude>(&self) -> Result<Vec<S>> {
        assert!(self.n <= 12, "dense unitary of more than 12 qubits");
        let n = self.n;
        let dim = 1usize << n;
        let half = S::one().add_ref(&S::one()).inverse();
        for x0 in 0..dim {
            let mut v = vec![S::zero(); dim];
            v[x0] = S::one();
            for q in 0..n {
                let pv = apply_pauli_dense(&self.z_image(q), &v);
                for (a, b) in v.iter_mut().zip(pv) {
                    *a = a.add_ref(&b).mul_ref(&half);
                }
            }
            if v.iter().all(|c| c.is_zero()) {
                continue;
            }
            // the entries are i^k 2^{-j/2}: normalize and rotate the first one to be positive
            let norm2: f64 = v.iter().map(|c| c.abs_sq_f64()).sum();
            let j = (-norm2.log2()).round() as i64;
            let scale = S::sqrt2_pow(j)
                .ok_or_else(|| SimError::NotRepresentable(format!("normalization 2^({j}/2)")))?;
            let first = v.iter().find(|c| !c.is_zero()).unwrap().to_c64();
            let k = (first.arg() / std::f64::consts::FRAC_PI_2).round() as i64;
            let rot = scale.mul_ref(&S::i_pow(-k));
            for c in v.iter_mut() {
                *c = c.mul_ref(&rot);
            }
            let mut u = vec![S::zero(); dim * dim];
            for x in 0..dim {
                let mut w = v.clone();
                for q in 0..n {
                    if (x >> q) & 1 == 1 {
                        w = apply_pauli_dense(&self.x_image(q), &w);
                    }
                }
                for row in 0..dim {
                    u[row * dim + x] = w[row].clone();
                }
            }
            return Ok(u);
        }
        Err(SimError::Invalid("tableau: no stabilized vector found".into()))
    }
}

/// `P v` for a dense vector.
pub(crate) fn apply_pauli_dense<S: Amplitude>(p: &Pauli, v: &[S]) -> Vec<S> {
    let n = p.num_qubits();
    assert!(n <= 30, "dense Pauli action on more than 30 qubits");
    let dim = 1usize << n;
    let (xm, zm) = (p.x_bits()[0] as usize, p.z_bits()[0] as usize);
    let ipow: Vec<S> = (0..4).map(|k| S::i_pow(k)).collect();
    let mut out = vec![S::zero(); dim];
    for (col, a) in v.iter().enumerate() {
        if a.is_exactly_zero() {
            continue;
        }
        let k = (p.phase() as usize + 2 * (zm & col).count_ones() as usize) & 3;
        out[col ^ xm] = a.mul_ref(&ipow[k]);
    }
    out
}

impl fmt::Debug for CliffordTableau {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "CliffordTableau({} qubits)", self.n)?;
        for q in 0..self.n {
            writeln!(f, "  X{q} -> {}", self.x_image(q))?;
        }
        for q in 0..self.n {
            writeln!(f, "  Z{q} -> {}", self.z_image(q))?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::scalar::C64;

    fn h() -> CliffordTableau {
        CliffordTableau::from_image_strings(&["Z0"], &["X0"]).unwrap()
    }
    fn s() -> CliffordTableau {
        CliffordTableau::from_image_strings(&["Y0"], &["Z0"]).unwrap()
    }
    fn cx() -> CliffordTableau {
        CliffordTableau::from_image_strings(&["X0*X1", "X1"], &["Z0", "Z0*Z1"]).unwrap()
    }

    #[test]
    fn relations_and_inverse() {
        let hs = h().compose(&s()); // H S
        assert!(hs.is_valid());
        let inv = hs.inverse();
        assert_eq!(hs.compose(&inv), CliffordTableau::identity(1));
        assert_eq!(inv.compose(&hs), CliffordTableau::identity(1));
        // S X S† = Y, S Y S† = −X
        assert_eq!(s().conjugate(&Pauli::x(1, 0)), Pauli::y(1, 0));
        assert_eq!(s().conjugate(&Pauli::y(1, 0)), Pauli::x(1, 0).negated());
        // (H S)^3 ∝ identity as a Clifford: tableau is identity up to phase
        let hs3 = hs.compose(&hs).compose(&hs);
        assert_eq!(hs3, CliffordTableau::identity(1));
        assert!(CliffordTableau::from_image_strings(&["X0"], &["X0"]).is_err());
    }

    #[test]
    fn local_application_matches_composition() {
        let mut t = CliffordTableau::identity(3);
        t.apply_gate(&h(), &[1]).unwrap();
        t.apply_gate(&cx(), &[1, 2]).unwrap();
        t.apply_gate(&s(), &[0]).unwrap();
        // the same by embedding and composing
        let big = |g: &CliffordTableau, qs: &[usize]| {
            let xs: Vec<Pauli> = (0..3)
                .map(|q| match qs.iter().position(|&x| x == q) {
                    Some(i) => g.x_image(i).embed(3, qs),
                    None => Pauli::x(3, q),
                })
                .collect();
            let zs: Vec<Pauli> = (0..3)
                .map(|q| match qs.iter().position(|&x| x == q) {
                    Some(i) => g.z_image(i).embed(3, qs),
                    None => Pauli::z(3, q),
                })
                .collect();
            CliffordTableau::from_images(xs, zs).unwrap()
        };
        let want = big(&s(), &[0])
            .compose(&big(&cx(), &[1, 2]))
            .compose(&big(&h(), &[1]));
        assert_eq!(t, want);
        let mut u = CliffordTableau::identity(3);
        u.prepend_gate(&s(), &[0]).unwrap();
        u.prepend_gate(&cx(), &[1, 2]).unwrap();
        u.prepend_gate(&h(), &[1]).unwrap();
        assert_eq!(u, want);
        // conjugation of a 3-qubit Pauli by the composite equals conjugating step by step
        let p = Pauli::parse("Y0*X1*Z2").unwrap();
        let mut q = p.clone();
        h().conjugate_local(&mut q, &[1]);
        cx().conjugate_local(&mut q, &[1, 2]);
        s().conjugate_local(&mut q, &[0]);
        assert_eq!(q, t.conjugate(&p));
        assert_eq!(t.conjugate_inverse(&q), p);
    }

    #[test]
    fn dense_unitary_conjugates_correctly() {
        let t = h().compose(&s());
        let u: Vec<C64> = t.unitary().unwrap();
        let dim = 2;
        let mm = |a: &[C64], b: &[C64]| {
            let mut c = vec![C64::new(0.0, 0.0); dim * dim];
            for i in 0..dim {
                for j in 0..dim {
                    for k in 0..dim {
                        c[i * dim + j] += a[i * dim + k] * b[k * dim + j];
                    }
                }
            }
            c
        };
        let udag: Vec<C64> = (0..dim * dim)
            .map(|ij| u[(ij % dim) * dim + ij / dim].conj())
            .collect();
        for p in [Pauli::x(1, 0), Pauli::z(1, 0)] {
            let pm: Vec<C64> = p.matrix();
            let lhs = mm(&mm(&u, &pm), &udag);
            let rhs: Vec<C64> = t.conjugate(&p).matrix();
            for (a, b) in lhs.iter().zip(&rhs) {
                assert!((a - b).norm() < 1e-12);
            }
        }
    }
}
