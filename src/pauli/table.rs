//! Rows of Pauli operators in one packed allocation: the storage of
//! stabilizer and Clifford tableaux.

use super::{ops, Pauli, PauliKind};
use crate::bits::{self, Bits};
use ncalg::ff::BitMat;
use std::fmt;

/// Tables with at least this many rows run row loops in parallel.
pub(crate) const PAR_ROWS: usize = 4096;
/// Rows per parallel task.
pub(crate) const PAR_CHUNK: usize = 1024;

/// `rows` Pauli operators on `n` qubits, each `i^phase X^x Z^z`, with the
/// `x` and `z` bits of row `i` stored contiguously.
#[derive(Clone, PartialEq, Eq)]
pub struct PauliTable {
    n: usize,
    words: usize,
    x: Vec<u64>,
    z: Vec<u64>,
    phase: Vec<u8>,
}

impl PauliTable {
    /// An empty table on `n` qubits.
    pub fn new(n: usize) -> Self {
        PauliTable {
            n,
            words: bits::nwords(n),
            x: Vec::new(),
            z: Vec::new(),
            phase: Vec::new(),
        }
    }
    /// A table from rows.
    pub fn from_rows(n: usize, rows: &[Pauli]) -> Self {
        let mut t = Self::new(n);
        for r in rows {
            t.push(r);
        }
        t
    }
    /// Number of qubits.
    pub fn num_qubits(&self) -> usize {
        self.n
    }
    /// Words per row.
    pub fn words(&self) -> usize {
        self.words
    }
    /// Number of rows.
    pub fn rows(&self) -> usize {
        self.phase.len()
    }
    /// Append a row.
    pub fn push(&mut self, p: &Pauli) {
        assert_eq!(p.num_qubits(), self.n, "row on a different register");
        self.x.extend_from_slice(p.x_bits());
        self.z.extend_from_slice(p.z_bits());
        self.phase.push(p.phase());
    }
    /// Row `i` as a [`Pauli`].
    pub fn row(&self, i: usize) -> Pauli {
        Pauli::new(
            self.n,
            self.x_row(i).to_vec(),
            self.z_row(i).to_vec(),
            self.phase[i],
        )
    }
    /// Overwrite row `i`.
    pub fn set_row(&mut self, i: usize, p: &Pauli) {
        assert_eq!(p.num_qubits(), self.n);
        let w = self.words;
        self.x[i * w..(i + 1) * w].copy_from_slice(p.x_bits());
        self.z[i * w..(i + 1) * w].copy_from_slice(p.z_bits());
        self.phase[i] = p.phase();
    }
    /// The `x` and `z` bits of row `i`.
    #[inline]
    pub fn row_bits(&self, i: usize) -> (&[u64], &[u64]) {
        (self.x_row(i), self.z_row(i))
    }
    /// The `x` bits of row `i`.
    #[inline]
    pub fn x_row(&self, i: usize) -> &[u64] {
        &self.x[i * self.words..(i + 1) * self.words]
    }
    /// The `z` bits of row `i`.
    #[inline]
    pub fn z_row(&self, i: usize) -> &[u64] {
        &self.z[i * self.words..(i + 1) * self.words]
    }
    /// Mutable `x` bits of row `i`.
    #[inline]
    pub fn x_row_mut(&mut self, i: usize) -> &mut [u64] {
        let w = self.words;
        &mut self.x[i * w..(i + 1) * w]
    }
    /// Mutable `z` bits of row `i`.
    #[inline]
    pub fn z_row_mut(&mut self, i: usize) -> &mut [u64] {
        let w = self.words;
        &mut self.z[i * w..(i + 1) * w]
    }
    /// The phase exponent of row `i`.
    #[inline]
    pub fn phase(&self, i: usize) -> u8 {
        self.phase[i]
    }
    /// Set the phase exponent of row `i`.
    #[inline]
    pub fn set_phase(&mut self, i: usize, phase: u8) {
        self.phase[i] = phase & 3;
    }
    /// Add `k` to the phase exponent of row `i`.
    #[inline]
    pub fn add_phase(&mut self, i: usize, k: u8) {
        self.phase[i] = (self.phase[i] + k) & 3;
    }
    /// The Pauli of row `i` on qubit `q`.
    pub fn get(&self, i: usize, q: usize) -> PauliKind {
        PauliKind::from_bits(bits::get(self.x_row(i), q), bits::get(self.z_row(i), q))
    }
    /// Set the letter of row `i` on qubit `q`, keeping the letter phase
    /// (as [`Pauli::set_kind`]).
    pub fn set(&mut self, i: usize, q: usize, kind: PauliKind) {
        if self.get(i, q) == PauliKind::Y {
            self.add_phase(i, 3);
        }
        let (x, z) = kind.bits();
        bits::set(self.x_row_mut(i), q, x);
        bits::set(self.z_row_mut(i), q, z);
        if kind == PauliKind::Y {
            self.add_phase(i, 1);
        }
    }
    /// `row[dst] ← row[dst] · row[src]`.
    pub fn mul_rows(&mut self, dst: usize, src: usize) {
        if dst == src {
            // P · P = i^{2 phase} (−1)^{x·z}: a scalar
            let ph = (2 * self.phase[dst] + 2 * bits::dot(self.x_row(dst), self.z_row(dst)) as u8) & 3;
            self.x_row_mut(dst).fill(0);
            self.z_row_mut(dst).fill(0);
            self.phase[dst] = ph;
            return;
        }
        let ph = ops::product_phase(self.phase[dst], self.z_row(dst), self.phase[src], self.x_row(src));
        let w = self.words;
        let (d, s) = (dst * w, src * w);
        for k in 0..w {
            let v = self.x[s + k];
            self.x[d + k] ^= v;
            let v = self.z[s + k];
            self.z[d + k] ^= v;
        }
        self.phase[dst] = ph;
    }
    /// `row[i] ← row[i] · p`.
    pub fn mul_row_by(&mut self, i: usize, p: &Pauli) {
        let ph = ops::product_phase(self.phase[i], self.z_row(i), p.phase(), p.x_bits());
        bits::xor_into(self.x_row_mut(i), p.x_bits());
        bits::xor_into(self.z_row_mut(i), p.z_bits());
        self.phase[i] = ph;
    }
    /// `row[i] ← p · row[i]`.
    pub fn premul_row_by(&mut self, i: usize, p: &Pauli) {
        let ph = ops::product_phase(p.phase(), p.z_bits(), self.phase[i], self.x_row(i));
        bits::xor_into(self.x_row_mut(i), p.x_bits());
        bits::xor_into(self.z_row_mut(i), p.z_bits());
        self.phase[i] = ph;
    }
    /// `p ← p · row[i]` (no allocation beyond `p`).
    pub fn mul_pauli_by_row(&self, p: &mut Pauli, i: usize) {
        let ph = ops::product_phase(p.phase(), p.z_bits(), self.phase[i], self.x_row(i));
        p.xor_bits(self.x_row(i), self.z_row(i));
        p.set_phase(ph);
    }
    /// Does row `i` commute with `p`?
    pub fn commutes_with(&self, i: usize, p: &Pauli) -> bool {
        ops::commute(self.x_row(i), self.z_row(i), p.x_bits(), p.z_bits())
    }
    /// Do rows `i` and `j` commute?
    pub fn rows_commute(&self, i: usize, j: usize) -> bool {
        ops::commute(self.x_row(i), self.z_row(i), self.x_row(j), self.z_row(j))
    }
    /// Is row `i` Hermitian?
    pub fn is_hermitian(&self, i: usize) -> bool {
        ops::is_hermitian(self.x_row(i), self.z_row(i), self.phase[i])
    }
    /// Conjugate every row by a gate on `qubits` (at most 3) given by its
    /// local lookup table: entry `x | z << k` of the row's letters on the gate
    /// qubits is `(image x bits, image z bits, phase added)`
    /// ([`CliffordTableau::local_lut`](crate::clifford::CliffordTableau::local_lut)).
    /// Rows without support on the gate qubits are skipped; large tables are
    /// processed in parallel.
    pub fn conjugate_rows_lut(&mut self, lut: &[(u8, u8, u8)], qubits: &[usize]) {
        let k = qubits.len();
        assert!(
            k <= 3 && lut.len() == 1 << (2 * k),
            "conjugate_rows_lut: table size"
        );
        let mut pos = [(0usize, 0u64); 3];
        for (j, &q) in qubits.iter().enumerate() {
            pos[j] = (q / 64, 1u64 << (q % 64));
        }
        let pos = &pos[..k];
        let w = self.words;
        let body = |xs: &mut [u64], zs: &mut [u64], ph: &mut [u8]| {
            for r in 0..ph.len() {
                let (xr, zr) = (&mut xs[r * w..(r + 1) * w], &mut zs[r * w..(r + 1) * w]);
                let mut idx = 0usize;
                for (j, &(wd, m)) in pos.iter().enumerate() {
                    if xr[wd] & m != 0 {
                        idx |= 1 << j;
                    }
                    if zr[wd] & m != 0 {
                        idx |= 1 << (k + j);
                    }
                }
                if idx == 0 {
                    continue;
                }
                let (ix, iz, add) = lut[idx];
                for (j, &(wd, m)) in pos.iter().enumerate() {
                    if (ix >> j) & 1 == 1 {
                        xr[wd] |= m;
                    } else {
                        xr[wd] &= !m;
                    }
                    if (iz >> j) & 1 == 1 {
                        zr[wd] |= m;
                    } else {
                        zr[wd] &= !m;
                    }
                }
                ph[r] = (ph[r] + add) & 3;
            }
        };
        let rows = self.rows();
        if rows >= PAR_ROWS {
            use rayon::prelude::*;
            let ch = PAR_CHUNK;
            self.x
                .par_chunks_mut(ch * w)
                .zip(self.z.par_chunks_mut(ch * w))
                .zip(self.phase.par_chunks_mut(ch))
                .for_each(|((xs, zs), ph)| body(xs, zs, ph));
        } else {
            body(&mut self.x, &mut self.z, &mut self.phase);
        }
    }
    /// Run `f(i, x_i, z_i, phase_i)` on every row, in parallel for tables of
    /// at least `PAR_ROWS` rows (the rows are independent).
    pub fn for_each_row_mut<F>(&mut self, f: F)
    where
        F: Fn(usize, &mut [u64], &mut [u64], &mut u8) + Sync,
    {
        let w = self.words;
        let body = |base: usize, xs: &mut [u64], zs: &mut [u64], ph: &mut [u8]| {
            for (r, ((xr, zr), p)) in xs
                .chunks_mut(w)
                .zip(zs.chunks_mut(w))
                .zip(ph.iter_mut())
                .enumerate()
            {
                f(base + r, xr, zr, p);
            }
        };
        if self.rows() >= PAR_ROWS {
            use rayon::prelude::*;
            let ch = PAR_CHUNK;
            self.x
                .par_chunks_mut(ch * w)
                .zip(self.z.par_chunks_mut(ch * w))
                .zip(self.phase.par_chunks_mut(ch))
                .enumerate()
                .for_each(|(c, ((xs, zs), ph))| body(c * ch, xs, zs, ph));
        } else {
            body(0, &mut self.x, &mut self.z, &mut self.phase);
        }
    }
    /// Swap two rows.
    pub fn swap_rows(&mut self, i: usize, j: usize) {
        if i == j {
            return;
        }
        let w = self.words;
        for k in 0..w {
            self.x.swap(i * w + k, j * w + k);
            self.z.swap(i * w + k, j * w + k);
        }
        self.phase.swap(i, j);
    }
    /// Remove row `i`, shifting later rows up.
    pub fn remove_row(&mut self, i: usize) {
        let w = self.words;
        self.x.drain(i * w..(i + 1) * w);
        self.z.drain(i * w..(i + 1) * w);
        self.phase.remove(i);
    }
    /// Keep only the first `rows` rows.
    pub fn truncate(&mut self, rows: usize) {
        self.x.truncate(rows * self.words);
        self.z.truncate(rows * self.words);
        self.phase.truncate(rows);
    }
    /// The rows as Paulis.
    pub fn iter(&self) -> impl Iterator<Item = Pauli> + '_ {
        (0..self.rows()).map(move |i| self.row(i))
    }
    /// The `x` parts as a `rows × n` GF(2) matrix.
    pub fn x_part(&self) -> BitMat {
        let rows: Vec<Bits> = (0..self.rows()).map(|i| self.x_row(i).to_vec()).collect();
        bits::to_bitmat(&rows, self.n)
    }
    /// The `z` parts as a `rows × n` GF(2) matrix.
    pub fn z_part(&self) -> BitMat {
        let rows: Vec<Bits> = (0..self.rows()).map(|i| self.z_row(i).to_vec()).collect();
        bits::to_bitmat(&rows, self.n)
    }
    /// The symplectic representation: row `i` is `(x_i | z_i)`, a `rows × 2n` matrix.
    pub fn symplectic(&self) -> BitMat {
        let mut m = BitMat::zeros(self.rows(), 2 * self.n);
        for i in 0..self.rows() {
            for j in bits::ones(self.x_row(i)) {
                m.set(i, j, true);
            }
            for j in bits::ones(self.z_row(i)) {
                m.set(i, self.n + j, true);
            }
        }
        m
    }
    /// A table from a symplectic matrix (`rows × 2n`) and phases.
    pub fn from_symplectic(m: &BitMat, phases: &[u8]) -> Self {
        assert_eq!(m.cols() % 2, 0);
        let n = m.cols() / 2;
        assert_eq!(m.rows(), phases.len());
        let mut t = Self::new(n);
        for i in 0..m.rows() {
            let mut x = bits::zero(n);
            let mut z = bits::zero(n);
            for j in 0..n {
                if m.get(i, j) {
                    bits::toggle(&mut x, j);
                }
                if m.get(i, n + j) {
                    bits::toggle(&mut z, j);
                }
            }
            t.push(&Pauli::new(n, x, z, phases[i]));
        }
        t
    }
}

impl fmt::Debug for PauliTable {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "PauliTable({} qubits, {} rows)", self.n, self.rows())?;
        for i in 0..self.rows() {
            writeln!(f, "  {}", self.row(i).to_dense_string())?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn row_products_match_paulis() {
        let a = Pauli::parse("-i*X0*Y1*Z2").unwrap();
        let b = Pauli::parse("Y0*Y1*Y2").unwrap();
        let mut t = PauliTable::from_rows(3, &[a.clone(), b.clone()]);
        t.mul_rows(0, 1);
        assert_eq!(t.row(0), a.mul(&b));
        assert_eq!(t.row(1), b);
        t.mul_row_by(1, &a);
        assert_eq!(t.row(1), b.mul(&a));
        t.premul_row_by(1, &a);
        assert_eq!(t.row(1), a.mul(&b).mul(&a));
        t.mul_rows(0, 0);
        assert!(t.row(0).is_identity());
        assert_eq!(t.row(0), a.mul(&b).mul(&a.mul(&b)));
        let s = t.symplectic();
        assert_eq!(PauliTable::from_symplectic(&s, &[t.phase(0), t.phase(1)]), t);
    }
}
