//! Rows of Pauli operators stored by *columns*: for each qubit `q` the bits
//! `x_i[q]` (and `z_i[q]`) of every row `i` are one packed vector, and the
//! phases are two bit planes over the rows.  The layout of Stim's tableaux:
//! a gate on `k` qubits touches `2k` columns, `O(rows/64)` words, instead of
//! one word in every row; the rows anticommuting with a Pauli `P` are found in
//! `O(|P| · rows/64)`; multiplying a row into a masked set of rows, and the
//! ordered product of a masked set, are bit-sliced (`O(n · rows/64)`).
//!
//! Same convention as [`PauliTable`]: row `i` is `i^{phase_i} X^{x_i} Z^{z_i}`.

use super::{Pauli, PauliTable};
use crate::bits::{self, Bits};
use rayon::prelude::*;
use std::fmt;

/// Column loops touching at least this many words run in parallel.
const PAR_WORDS: usize = 1 << 15;

/// Pauli rows stored by columns: for each qubit the bits of every row are
/// one packed vector, the phases two bit planes over the rows (the layout of
/// Stim's tableaux; costs in the source module documentation).
#[derive(Clone, PartialEq, Eq)]
pub struct ColumnTable {
    n: usize,
    rows: usize,
    rw: usize,
    x: Vec<u64>,
    z: Vec<u64>,
    p0: Vec<u64>,
    p1: Vec<u64>,
}

/// `p += c` (mod 4) on the rows of mask `m`, for one word of the phase planes.
#[inline]
fn add_phase_word(p0: &mut u64, p1: &mut u64, m: u64, c: u8) {
    if c & 2 != 0 {
        *p1 ^= m;
    }
    if c & 1 != 0 {
        let carry = *p0 & m;
        *p0 ^= m;
        *p1 ^= carry;
    }
}

/// Bit `b` of the result: parity of bits `0..b` of `v` (exclusive prefix XOR).
#[inline]
fn prefix_parity_exclusive(v: u64) -> u64 {
    let mut t = v;
    t ^= t << 1;
    t ^= t << 2;
    t ^= t << 4;
    t ^= t << 8;
    t ^= t << 16;
    t ^= t << 32;
    t ^ v
}

impl ColumnTable {
    /// `rows` identity rows on `n` qubits.
    pub fn new(n: usize, rows: usize) -> Self {
        let rw = bits::nwords(rows);
        ColumnTable {
            n,
            rows,
            rw,
            x: vec![0; n * rw],
            z: vec![0; n * rw],
            p0: vec![0; rw],
            p1: vec![0; rw],
        }
    }
    /// The `2n` rows `X_0, …, X_{n−1}, Z_0, …, Z_{n−1}`: the tableau of `|0…0⟩`
    /// (destabilizers, then stabilizers) and of the identity Clifford.
    pub fn destabilizer_basis(n: usize) -> Self {
        let mut c = Self::new(n, 2 * n);
        let rw = c.rw;
        for q in 0..n {
            c.x[q * rw + q / 64] |= 1u64 << (q % 64);
            let r = n + q;
            c.z[q * rw + r / 64] |= 1u64 << (r % 64);
        }
        c
    }
    /// From a row-major table.
    pub fn from_table(t: &PauliTable) -> Self {
        let mut c = Self::new(t.num_qubits(), t.rows());
        let rw = c.rw;
        for i in 0..t.rows() {
            let (w, m) = (i / 64, 1u64 << (i % 64));
            for q in bits::ones(t.x_row(i)) {
                c.x[q * rw + w] |= m;
            }
            for q in bits::ones(t.z_row(i)) {
                c.z[q * rw + w] |= m;
            }
            let ph = t.phase(i);
            if ph & 1 != 0 {
                c.p0[w] |= m;
            }
            if ph & 2 != 0 {
                c.p1[w] |= m;
            }
        }
        c
    }
    /// The row-major table.
    pub fn to_table(&self) -> PauliTable {
        let mut t = PauliTable::new(self.n);
        for i in 0..self.rows {
            t.push(&self.row(i));
        }
        t
    }
    /// Number of qubits.
    pub fn num_qubits(&self) -> usize {
        self.n
    }
    /// Number of rows.
    pub fn rows(&self) -> usize {
        self.rows
    }
    /// Words per column.
    pub fn column_words(&self) -> usize {
        self.rw
    }
    /// The `x` bits of column `q` (bit `i` = row `i`).
    #[inline]
    pub fn x_col(&self, q: usize) -> &[u64] {
        &self.x[q * self.rw..(q + 1) * self.rw]
    }
    /// The `z` bits of column `q`.
    #[inline]
    pub fn z_col(&self, q: usize) -> &[u64] {
        &self.z[q * self.rw..(q + 1) * self.rw]
    }
    /// Mutable `x` and `z` bits of column `q` (bit `i` = row `i`).
    #[inline]
    pub fn cols_mut(&mut self, q: usize) -> (&mut [u64], &mut [u64]) {
        let rw = self.rw;
        (
            &mut self.x[q * rw..(q + 1) * rw],
            &mut self.z[q * rw..(q + 1) * rw],
        )
    }
    /// `row_i ← row_i · p` (letters only, phases untouched) for the rows in `mask`.
    pub fn xor_pauli_masked(&mut self, p: &Pauli, mask: &[u64]) {
        let rw = self.rw;
        for q in bits::ones(p.x_bits()) {
            for (a, b) in self.x[q * rw..(q + 1) * rw].iter_mut().zip(mask) {
                *a ^= b;
            }
        }
        for q in bits::ones(p.z_bits()) {
            for (a, b) in self.z[q * rw..(q + 1) * rw].iter_mut().zip(mask) {
                *a ^= b;
            }
        }
    }
    /// The phase exponent of row `i`.
    #[inline]
    pub fn phase(&self, i: usize) -> u8 {
        let (w, b) = (i / 64, i % 64);
        (((self.p0[w] >> b) & 1) | (((self.p1[w] >> b) & 1) << 1)) as u8
    }
    /// Set the phase exponent of row `i`.
    pub fn set_phase(&mut self, i: usize, ph: u8) {
        let (w, m) = (i / 64, 1u64 << (i % 64));
        self.p0[w] = if ph & 1 != 0 {
            self.p0[w] | m
        } else {
            self.p0[w] & !m
        };
        self.p1[w] = if ph & 2 != 0 {
            self.p1[w] | m
        } else {
            self.p1[w] & !m
        };
    }
    /// Add `k` to the phase exponent of every row in `mask`.
    pub fn add_phase_masked(&mut self, mask: &[u64], k: u8) {
        for w in 0..self.rw {
            add_phase_word(&mut self.p0[w], &mut self.p1[w], mask[w], k & 3);
        }
    }
    /// Row `i` as a [`Pauli`] (`O(n)` bit gathers).
    pub fn row(&self, i: usize) -> Pauli {
        let (w, b) = (i / 64, i % 64);
        let mut x = bits::zero(self.n);
        let mut z = bits::zero(self.n);
        for q in 0..self.n {
            if (self.x[q * self.rw + w] >> b) & 1 == 1 {
                bits::toggle(&mut x, q);
            }
            if (self.z[q * self.rw + w] >> b) & 1 == 1 {
                bits::toggle(&mut z, q);
            }
        }
        Pauli::new(self.n, x, z, self.phase(i))
    }
    /// Overwrite row `i`.
    pub fn set_row(&mut self, i: usize, p: &Pauli) {
        assert_eq!(p.num_qubits(), self.n, "set_row: register size");
        let (w, m) = (i / 64, 1u64 << (i % 64));
        for q in 0..self.n {
            let k = q * self.rw + w;
            self.x[k] = if bits::get(p.x_bits(), q) {
                self.x[k] | m
            } else {
                self.x[k] & !m
            };
            self.z[k] = if bits::get(p.z_bits(), q) {
                self.z[k] | m
            } else {
                self.z[k] & !m
            };
        }
        self.set_phase(i, p.phase());
    }
    /// The rows anticommuting with `p`, as a mask over the rows:
    /// `x_i·z_p + z_i·x_p` summed column by column over the support of `p`.
    pub fn anticommute_mask(&self, p: &Pauli) -> Bits {
        assert_eq!(p.num_qubits(), self.n, "anticommute_mask: register size");
        let mut m = vec![0u64; self.rw];
        for q in bits::ones(p.z_bits()) {
            bits::xor_into(&mut m, self.x_col(q));
        }
        for q in bits::ones(p.x_bits()) {
            bits::xor_into(&mut m, self.z_col(q));
        }
        m
    }
    /// `row_i ← row_i · row_src` for every row `i` in `mask` (`src ∉ mask`).
    pub fn rowsum_masked(&mut self, mask: &[u64], src: usize) {
        debug_assert!(!bits::get(mask, src), "rowsum_masked: the source is in the mask");
        let (sw, sb) = (src / 64, src % 64);
        let rw = self.rw;
        let xs: Vec<usize> = (0..self.n)
            .filter(|&q| (self.x[q * rw + sw] >> sb) & 1 == 1)
            .collect();
        let zs: Vec<usize> = (0..self.n)
            .filter(|&q| (self.z[q * rw + sw] >> sb) & 1 == 1)
            .collect();
        // (i^a X^x Z^z)(i^b X^x' Z^z') = i^{a + b + 2 z·x'}: d_i = z_i · x_src
        let mut d = vec![0u64; rw];
        for &q in &xs {
            bits::xor_into(&mut d, &self.z[q * rw..(q + 1) * rw]);
        }
        let ph = self.phase(src);
        for w in 0..rw {
            let m = mask[w];
            if m == 0 {
                continue;
            }
            add_phase_word(&mut self.p0[w], &mut self.p1[w], m & d[w], 2);
            add_phase_word(&mut self.p0[w], &mut self.p1[w], m, ph);
        }
        let xor_cols = |data: &mut [u64], cols: &[usize]| {
            let mut sel = vec![false; self.n];
            for &q in cols {
                sel[q] = true;
            }
            let body = |(q, col): (usize, &mut [u64])| {
                if sel[q] {
                    for (a, b) in col.iter_mut().zip(mask) {
                        *a ^= b;
                    }
                }
            };
            // memory-bound: a sequential sweep beats splitting across threads
            data.chunks_mut(rw).enumerate().for_each(body);
        };
        let (mut x, mut z) = (std::mem::take(&mut self.x), std::mem::take(&mut self.z));
        xor_cols(&mut x, &xs);
        xor_cols(&mut z, &zs);
        self.x = x;
        self.z = z;
    }
    /// The ordered product `Π_{i ∈ mask, increasing} row_i` (identity for an
    /// empty mask): bits by column parities, the phase by bit-sliced prefix
    /// parities, `O(n · rows/64)`.
    pub fn product_masked(&self, mask: &[u64]) -> Pauli {
        let rw = self.rw;
        let mut x = bits::zero(self.n);
        let mut z = bits::zero(self.n);
        let mut ph = 0usize;
        for w in 0..rw {
            ph += (self.p0[w] & mask[w]).count_ones() as usize
                + 2 * (self.p1[w] & mask[w]).count_ones() as usize;
        }
        // per column: (x parity, z parity, Σ_{j<k} z_j x_k)
        let column = |q: usize| -> (bool, bool, usize) {
            let (xc, zc) = (&self.x[q * rw..(q + 1) * rw], &self.z[q * rw..(q + 1) * rw]);
            let (mut px, mut pz) = (0u32, 0u32);
            let mut carry = 0u64;
            let mut pairs = 0usize;
            for w in 0..rw {
                let (xm, zm) = (xc[w] & mask[w], zc[w] & mask[w]);
                if xm == 0 && zm == 0 {
                    continue;
                }
                // z parity strictly before each selected x
                let pre = prefix_parity_exclusive(zm) ^ carry;
                pairs += (xm & pre).count_ones() as usize;
                if zm.count_ones() & 1 == 1 {
                    carry = !carry;
                }
                px ^= xm.count_ones() & 1;
                pz ^= zm.count_ones() & 1;
            }
            (px == 1, pz == 1, pairs)
        };
        let cols: Vec<(bool, bool, usize)> = if self.n * rw >= PAR_WORDS {
            (0..self.n)
                .into_par_iter()
                .with_min_len((PAR_WORDS / 4 / rw).max(1))
                .map(column)
                .collect()
        } else {
            (0..self.n).map(column).collect()
        };
        let mut pairs = 0usize;
        for (q, &(bx, bz, c)) in cols.iter().enumerate() {
            if bx {
                bits::toggle(&mut x, q);
            }
            if bz {
                bits::toggle(&mut z, q);
            }
            pairs += c;
        }
        Pauli::new(self.n, x, z, ((ph + 2 * pairs) & 3) as u8)
    }

    // ---- gates: O(rows/64) words per column touched ----

    /// `H` on qubit `q` (`X ↔ Z`, `Y → −Y`).
    pub fn apply_h(&mut self, q: usize) {
        let rw = self.rw;
        for w in 0..rw {
            let (a, b) = (self.x[q * rw + w], self.z[q * rw + w]);
            self.p1[w] ^= a & b;
            self.x[q * rw + w] = b;
            self.z[q * rw + w] = a;
        }
    }
    /// `S^e` on qubit `q` (`e` mod 4): `X → i^e X Z^e`.
    pub fn apply_s_pow(&mut self, q: usize, e: u8) {
        let e = e & 3;
        let rw = self.rw;
        for w in 0..rw {
            let a = self.x[q * rw + w];
            match e {
                0 => {}
                2 => self.p1[w] ^= a,
                _ => {
                    add_phase_word(&mut self.p0[w], &mut self.p1[w], a, e);
                    self.z[q * rw + w] ^= a;
                }
            }
        }
    }
    /// The Pauli `X^a Z^b` on qubit `q` (conjugation flips the signs of the
    /// letters it anticommutes with).
    pub fn apply_pauli_1q(&mut self, q: usize, a: bool, b: bool) {
        let rw = self.rw;
        for w in 0..rw {
            let mut m = 0;
            if a {
                m ^= self.z[q * rw + w];
            }
            if b {
                m ^= self.x[q * rw + w];
            }
            self.p1[w] ^= m;
        }
    }
    /// `CX(c → t)`: `X_c → X_c X_t`, `Z_t → Z_c Z_t`.
    pub fn apply_cx(&mut self, c: usize, t: usize) {
        let rw = self.rw;
        for w in 0..rw {
            let v = self.x[c * rw + w];
            self.x[t * rw + w] ^= v;
            let u = self.z[t * rw + w];
            self.z[c * rw + w] ^= u;
        }
    }
    /// `CZ(a, b)`: `X_a → X_a Z_b`, `X_b → Z_a X_b`.
    pub fn apply_cz(&mut self, a: usize, b: usize) {
        let rw = self.rw;
        for w in 0..rw {
            let (xa, xb) = (self.x[a * rw + w], self.x[b * rw + w]);
            self.p1[w] ^= xa & xb;
            self.z[a * rw + w] ^= xb;
            self.z[b * rw + w] ^= xa;
        }
    }
    /// `SWAP(a, b)`.
    pub fn apply_swap(&mut self, a: usize, b: usize) {
        let rw = self.rw;
        for w in 0..rw {
            self.x.swap(a * rw + w, b * rw + w);
            self.z.swap(a * rw + w, b * rw + w);
        }
    }
    /// A gate on at most 3 qubits by its local lookup table
    /// ([`CliffordTableau::local_lut`](crate::clifford::CliffordTableau::local_lut)):
    /// one mask per local letter pattern.
    pub fn apply_lut(&mut self, lut: &[(u8, u8, u8)], qubits: &[usize]) {
        let k = qubits.len();
        assert!(k <= 3 && lut.len() == 1 << (2 * k), "apply_lut: table size");
        let rw = self.rw;
        let mut xs = [0u64; 3];
        let mut zs = [0u64; 3];
        for w in 0..rw {
            let mut any = 0u64;
            for (j, &q) in qubits.iter().enumerate() {
                xs[j] = self.x[q * rw + w];
                zs[j] = self.z[q * rw + w];
                any |= xs[j] | zs[j];
            }
            if any == 0 {
                continue;
            }
            let mut nx = [0u64; 3];
            let mut nz = [0u64; 3];
            for (idx, &(ix, iz, add)) in lut.iter().enumerate().skip(1) {
                let mut m = any;
                for j in 0..k {
                    m &= if (idx >> j) & 1 == 1 { xs[j] } else { !xs[j] };
                    m &= if (idx >> (k + j)) & 1 == 1 { zs[j] } else { !zs[j] };
                }
                if m == 0 {
                    continue;
                }
                for j in 0..k {
                    if (ix >> j) & 1 == 1 {
                        nx[j] |= m;
                    }
                    if (iz >> j) & 1 == 1 {
                        nz[j] |= m;
                    }
                }
                add_phase_word(&mut self.p0[w], &mut self.p1[w], m, add);
            }
            for (j, &q) in qubits.iter().enumerate() {
                self.x[q * rw + w] = nx[j];
                self.z[q * rw + w] = nz[j];
            }
        }
    }
}

impl fmt::Debug for ColumnTable {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "ColumnTable({} qubits, {} rows)", self.n, self.rows)?;
        for i in 0..self.rows {
            writeln!(f, "  {}", self.row(i).to_dense_string())?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gate::Gate;
    use crate::pauli::PauliKind;
    use rand::rngs::StdRng;
    use rand::{Rng, SeedableRng};

    fn random_table(n: usize, rows: usize, rng: &mut StdRng) -> PauliTable {
        let mut t = PauliTable::new(n);
        for _ in 0..rows {
            let kinds: Vec<PauliKind> = (0..n)
                .map(|_| [PauliKind::I, PauliKind::X, PauliKind::Y, PauliKind::Z][rng.gen_range(0..4)])
                .collect();
            t.push(&Pauli::from_kinds_with_phase(&kinds, rng.gen_range(0..4)));
        }
        t
    }

    #[test]
    fn gates_match_the_row_major_lookup_tables() {
        let mut rng = StdRng::seed_from_u64(4);
        let gates = [
            Gate::I,
            Gate::X,
            Gate::Y,
            Gate::Z,
            Gate::H,
            Gate::S,
            Gate::Sdg,
            Gate::SqrtX,
            Gate::SqrtXdg,
            Gate::SqrtY,
            Gate::SqrtYdg,
            Gate::Cx,
            Gate::Cy,
            Gate::Cz,
            Gate::Swap,
        ];
        for trial in 0..60 {
            let (n, rows) = (3 + trial % 70, 1 + (trial * 7) % 140);
            let mut rt = random_table(n, rows, &mut rng);
            let mut ct = ColumnTable::from_table(&rt);
            assert_eq!(ct.to_table(), rt);
            for _ in 0..40 {
                let g = &gates[rng.gen_range(0..gates.len())];
                let a = rng.gen_range(0..n);
                let qs = if g.arity() == 1 {
                    vec![a]
                } else {
                    vec![a, (a + 1 + rng.gen_range(0..n - 1)) % n]
                };
                rt.conjugate_rows_lut(g.cached_lut().unwrap(), &qs);
                match g {
                    Gate::H => ct.apply_h(qs[0]),
                    Gate::S => ct.apply_s_pow(qs[0], 1),
                    Gate::Sdg => ct.apply_s_pow(qs[0], 3),
                    Gate::Z => ct.apply_s_pow(qs[0], 2),
                    Gate::X => ct.apply_pauli_1q(qs[0], true, false),
                    Gate::Y => ct.apply_pauli_1q(qs[0], true, true),
                    Gate::Cx => ct.apply_cx(qs[0], qs[1]),
                    Gate::Cz => ct.apply_cz(qs[0], qs[1]),
                    Gate::Swap => ct.apply_swap(qs[0], qs[1]),
                    _ => ct.apply_lut(g.cached_lut().unwrap(), &qs),
                }
                assert_eq!(ct.to_table(), rt, "trial {trial}: {g} on {qs:?}");
            }
            // rowsum and ordered products against Pauli multiplication
            let rows = rt.rows();
            if rows >= 2 {
                let src = rng.gen_range(0..rows);
                let mut mask = bits::zero(rows);
                for i in 0..rows {
                    if i != src && rng.gen::<bool>() {
                        bits::set(&mut mask, i, true);
                    }
                }
                let mut want = rt.clone();
                for i in bits::ones(&mask) {
                    want.mul_rows(i, src);
                }
                let prod_want = bits::ones(&mask)
                    .iter()
                    .fold(Pauli::identity(n), |acc, &i| acc.mul(&rt.row(i)));
                assert_eq!(ct.product_masked(&mask), prod_want, "trial {trial}: product");
                ct.rowsum_masked(&mask, src);
                assert_eq!(ct.to_table(), want, "trial {trial}: rowsum");
            }
            let kinds: Vec<PauliKind> = (0..n)
                .map(|_| [PauliKind::I, PauliKind::X, PauliKind::Y, PauliKind::Z][rng.gen_range(0..4)])
                .collect();
            let p = Pauli::from_kinds(&kinds);
            let m = ct.anticommute_mask(&p);
            let rt = ct.to_table();
            for i in 0..rt.rows() {
                assert_eq!(bits::get(&m, i), !rt.commutes_with(i, &p));
            }
        }
    }
}
