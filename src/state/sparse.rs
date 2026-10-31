//! Sparse state vectors: the nonzero amplitudes only, as `(basis state,
//! amplitude)` pairs sorted by basis state, on any number of qubits.
//!
//! For circuits whose states keep few nonzero amplitudes on many qubits:
//! reversible arithmetic on superposed inputs, permutation-heavy circuits,
//! low-rank superpositions.  A monomial gate (every permutation and diagonal
//! gate, Paulis, `CX`, `CCX`, `CCZ`, `T`, …) maps each entry to one entry in
//! place, `O(N)` for `N` nonzero amplitudes; a dense `k`-qubit gate maps it to
//! at most `2^k` entries, merged by a parallel sort, `O(N log N)`.  The order
//! is restored lazily: only merges and lookups sort.  Exact zeros are dropped; with float
//! amplitudes an optional cutoff drops entries of smaller modulus.

use super::QuantumState;
use crate::bits::{self, Bits};
use crate::circuit::Op;
use crate::error::{check_qubits, Result, SimError};
use crate::gate::{Gate, MonomialGate};
use crate::pauli::Pauli;
use crate::scalar::Amplitude;
use rand::{Rng, RngCore};
use rayon::prelude::*;
use std::fmt;

const BACKEND: &str = "SparseVector";

/// The nonzero amplitudes of a state (see the [module documentation](self)).
#[derive(Clone)]
pub struct SparseVector<S: Amplitude> {
    n: usize,
    /// Unique basis states, no exact zeros; sorted when `sorted`.
    entries: Vec<(Bits, S)>,
    sorted: bool,
    cutoff: f64,
}

impl<S: Amplitude> SparseVector<S> {
    /// `|x⟩`.
    pub fn basis_state(n: usize, x: &[u64]) -> Self {
        let mut b = bits::zero(n);
        bits::xor_into(&mut b, x);
        bits::mask_to(&mut b, n);
        SparseVector {
            n,
            entries: vec![(b, S::one())],
            sorted: true,
            cutoff: 0.0,
        }
    }
    /// `|0…0⟩`.
    pub fn zero_state(n: usize) -> Self {
        Self::basis_state(n, &bits::zero(n))
    }
    /// From `(basis state, amplitude)` pairs (duplicates are added).
    pub fn from_entries(n: usize, entries: Vec<(Bits, S)>) -> Self {
        let mut s = SparseVector {
            n,
            entries: entries
                .into_iter()
                .map(|(mut b, a)| {
                    bits::ensure_len(&mut b, n);
                    b.truncate(bits::nwords(n));
                    bits::mask_to(&mut b, n);
                    (b, a)
                })
                .collect(),
            sorted: false,
            cutoff: 0.0,
        };
        s.normalize_entries();
        s
    }
    /// Drop amplitudes of modulus below `cutoff` after each operation (floats;
    /// `0` keeps every nonzero amplitude, the default).
    pub fn with_cutoff(mut self, cutoff: f64) -> Self {
        self.cutoff = cutoff;
        self.normalize_entries();
        self
    }
    /// Number of qubits.
    pub fn num_qubits(&self) -> usize {
        self.n
    }
    /// Number of nonzero amplitudes.
    pub fn num_entries(&self) -> usize {
        self.entries.len()
    }
    /// The nonzero amplitudes (sorted by basis state after [`SparseVector::sort`]).
    pub fn entries(&self) -> &[(Bits, S)] {
        &self.entries
    }
    /// Sort the entries by basis state (done lazily by the operations that need it).
    pub fn sort(&mut self) {
        if !self.sorted {
            self.entries.par_sort_unstable_by(|a, b| a.0.cmp(&b.0));
            self.sorted = true;
        }
    }
    /// The entries in sorted order: borrowed when sorted, else a sorted copy.
    fn sorted_view(&self) -> std::borrow::Cow<'_, [(Bits, S)]> {
        if self.sorted {
            std::borrow::Cow::Borrowed(&self.entries)
        } else {
            let mut e = self.entries.clone();
            e.par_sort_unstable_by(|a, b| a.0.cmp(&b.0));
            std::borrow::Cow::Owned(e)
        }
    }
    fn find_in<'a>(view: &'a [(Bits, S)], x: &[u64]) -> Option<&'a S> {
        view.binary_search_by(|(b, _)| b.as_slice().cmp(x))
            .ok()
            .map(|i| &view[i].1)
    }
    /// Sort, add duplicates, drop zeros (and entries below the cutoff).
    fn normalize_entries(&mut self) {
        let mut e = std::mem::take(&mut self.entries);
        e.par_sort_unstable_by(|a, b| a.0.cmp(&b.0));
        let mut out: Vec<(Bits, S)> = Vec::with_capacity(e.len());
        for (b, a) in e {
            match out.last_mut() {
                Some((lb, la)) if *lb == b => la.add_assign_ref(&a),
                _ => out.push((b, a)),
            }
        }
        let cut = self.cutoff * self.cutoff;
        out.retain(|(_, a)| !a.is_exactly_zero() && (cut == 0.0 || a.abs_sq_f64() >= cut));
        self.entries = out;
        self.sorted = true;
    }
    /// Apply a monomial gate: every entry moves to one entry.
    pub fn apply_monomial(&mut self, g: &MonomialGate, qubits: &[usize]) -> Result<()> {
        check_qubits(qubits, self.n)?;
        let k = qubits.len();
        if k != g.num_qubits() {
            return Err(SimError::Arity {
                gate: "monomial".into(),
                expected: g.num_qubits(),
                got: k,
            });
        }
        let images: Vec<(S, usize)> = (0..1usize << k)
            .map(|x| {
                let (s, y) = g.apply_to_basis(&bits::from_u64(x as u64, k));
                let c = S::from_phasor(s).ok_or_else(|| SimError::NotRepresentable(format!("phase {s}")))?;
                Ok((c, bits::to_u64(&y, k) as usize))
            })
            .collect::<Result<_>>()?;
        let permutes = images.iter().enumerate().any(|(x, (_, y))| x != *y);
        for_each_entry(&mut self.entries, |(b, a)| {
            let x = local(b, qubits);
            let (c, y) = &images[x];
            *a = a.mul_ref(c);
            if *y != x {
                set_local(b, qubits, *y);
            }
        });
        if permutes {
            self.sorted = false;
        }
        Ok(())
    }
    /// Apply a dense `2^k × 2^k` matrix on `qubits`.
    pub fn apply_matrix(&mut self, m: &[S], qubits: &[usize]) -> Result<()> {
        check_qubits(qubits, self.n)?;
        let k = qubits.len();
        let gd = 1usize << k;
        assert_eq!(m.len(), gd * gd, "matrix size");
        // column b of the matrix: its nonzero entries (a, m_ab)
        let cols: Vec<Vec<(usize, S)>> = (0..gd)
            .map(|b| {
                (0..gd)
                    .filter(|&a| !m[a * gd + b].is_exactly_zero())
                    .map(|a| (a, m[a * gd + b].clone()))
                    .collect()
            })
            .collect();
        // a permutation with phases (one nonzero per column, distinct rows, as
        // CCX): every entry moves to one entry, in place and without a merge
        let mut targets: Vec<usize> = cols.iter().filter(|c| c.len() == 1).map(|c| c[0].0).collect();
        targets.sort_unstable();
        targets.dedup();
        if targets.len() == gd && cols.iter().all(|c| c.len() == 1) {
            let mut moved = false;
            for (b, c) in cols.iter().enumerate() {
                moved |= c[0].0 != b;
            }
            for_each_entry(&mut self.entries, |(key, v)| {
                let b = local(key, qubits);
                let (a, c) = &cols[b][0];
                *v = c.mul_ref(v);
                if *a != b {
                    set_local(key, qubits, *a);
                }
            });
            if moved {
                self.sorted = false;
            }
            return Ok(());
        }
        let out: Vec<(Bits, S)> = self
            .entries
            .par_iter()
            .flat_map_iter(|(key, v)| {
                let b = local(key, qubits);
                cols[b].iter().map(move |(a, c)| {
                    let mut k2 = key.clone();
                    set_local(&mut k2, qubits, *a);
                    (k2, c.mul_ref(v))
                })
            })
            .collect();
        self.entries = out;
        self.normalize_entries();
        Ok(())
    }
    /// `ψ ← P ψ` for a Pauli `P`.
    pub fn apply_pauli(&mut self, p: &Pauli) {
        let ipow: Vec<S> = (0..4).map(|k| S::i_pow(k)).collect();
        for_each_entry(&mut self.entries, |(b, a)| {
            let k = crate::pauli::ops::basis_image_phase(p.z_bits(), p.phase(), b);
            *a = a.mul_ref(&ipow[k as usize]);
            bits::xor_into(b, p.x_bits());
        });
        if !bits::is_zero(p.x_bits()) {
            self.sorted = false;
        }
    }
}

/// Entries per parallel task (smaller sets run sequentially).
const PAR_ENTRIES: usize = 2048;

/// `f` on every entry, in parallel for large states.
fn for_each_entry<S: Amplitude, F: Fn(&mut (Bits, S)) + Sync + Send>(entries: &mut [(Bits, S)], f: F) {
    if entries.len() >= 2 * PAR_ENTRIES {
        entries.par_iter_mut().with_min_len(PAR_ENTRIES).for_each(f);
    } else {
        entries.iter_mut().for_each(f);
    }
}

/// The bits of `b` on `qubits`, as a local index.
#[inline]
fn local(b: &[u64], qubits: &[usize]) -> usize {
    qubits
        .iter()
        .enumerate()
        .map(|(j, &q)| (bits::get(b, q) as usize) << j)
        .sum()
}

/// Write the local index `y` into the bits of `b` on `qubits`.
#[inline]
fn set_local(b: &mut [u64], qubits: &[usize], y: usize) {
    for (j, &q) in qubits.iter().enumerate() {
        bits::set(b, q, (y >> j) & 1 == 1);
    }
}

impl<S: Amplitude> QuantumState<S> for SparseVector<S> {
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
        if let Gate::Pauli(p) = gate {
            self.apply_pauli(&p.embed(self.n, qubits));
            return Ok(());
        }
        if let Some(m) = gate.monomial() {
            return self.apply_monomial(&m, qubits);
        }
        if let Some(m) = gate.matrix::<S>() {
            return self.apply_matrix(&m, qubits);
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
        Err(SimError::NotRepresentable(format!(
            "gate {gate} in the scalar type"
        )))
    }

    fn norm_sq(&self) -> Result<S> {
        let e = &self.entries;
        Ok(S::sum(&e.par_iter().map(|(_, a)| a.abs_sq()).collect::<Vec<S>>()))
    }

    fn amplitude(&self, x: &[u64]) -> Result<S> {
        let mut b = bits::zero(self.n);
        bits::xor_into(&mut b, x);
        bits::mask_to(&mut b, self.n);
        let found = if self.sorted {
            Self::find_in(&self.entries, &b)
        } else {
            self.entries.iter().find(|(k, _)| *k == b).map(|(_, a)| a)
        };
        Ok(found.cloned().unwrap_or_else(S::zero))
    }

    fn expectation(&self, p: &Pauli) -> Result<S> {
        if p.num_qubits() != self.n {
            return Err(SimError::Invalid(
                "expectation: operator on the wrong register".into(),
            ));
        }
        let ipow: Vec<S> = (0..4).map(|k| S::i_pow(k)).collect();
        let view = self.sorted_view();
        // ⟨ψ|P|ψ⟩ = Σ_c conj ψ(c ⊕ x) i^{k(c)} ψ(c)
        let terms: Vec<S> = view
            .par_iter()
            .filter_map(|(c, a)| {
                let mut d = c.clone();
                bits::xor_into(&mut d, p.x_bits());
                let b = Self::find_in(&view, &d)?;
                let k = crate::pauli::ops::basis_image_phase(p.z_bits(), p.phase(), c);
                Some(b.conj().mul_ref(&ipow[k as usize].mul_ref(a)))
            })
            .collect();
        Ok(S::sum(&terms))
    }

    fn inner(&self, other: &Self) -> Result<S> {
        if other.n != self.n {
            return Err(SimError::Invalid("inner: register sizes differ".into()));
        }
        let view = other.sorted_view();
        let terms: Vec<S> = self
            .entries
            .par_iter()
            .filter_map(|(c, a)| Self::find_in(&view, c).map(|b| a.conj().mul_ref(b)))
            .collect();
        Ok(S::sum(&terms))
    }

    fn project(&mut self, p: &Pauli, outcome: bool) -> Result<S> {
        if !p.is_hermitian() {
            return Err(SimError::Invalid(format!("projector of non-Hermitian {p}")));
        }
        let half = S::one().add_ref(&S::one()).inverse();
        let ipow: Vec<S> = (0..4)
            .map(|k| {
                let c = S::i_pow(k).mul_ref(&half);
                if outcome {
                    c.negated()
                } else {
                    c
                }
            })
            .collect();
        // (1 ± P)/2: entry c contributes ψ(c)/2 at c and ± i^{k(c)} ψ(c)/2 at c ⊕ x
        let out: Vec<(Bits, S)> = self
            .entries
            .par_iter()
            .flat_map_iter(|(c, a)| {
                let mut d = c.clone();
                bits::xor_into(&mut d, p.x_bits());
                let k = crate::pauli::ops::basis_image_phase(p.z_bits(), p.phase(), c);
                [(c.clone(), a.mul_ref(&half)), (d, ipow[k as usize].mul_ref(a))].into_iter()
            })
            .collect();
        self.entries = out;
        self.normalize_entries();
        self.norm_sq()
    }

    fn scale(&mut self, c: &S) -> Result<()> {
        for_each_entry(&mut self.entries, |(_, a)| *a = a.mul_ref(c));
        self.normalize_entries();
        Ok(())
    }

    fn tensor(&self, other: &Self) -> Result<Self> {
        let (na, nb) = (self.n, other.n);
        let entries: Vec<(Bits, S)> = other
            .entries
            .par_iter()
            .flat_map_iter(|(kb, b)| {
                self.entries
                    .iter()
                    .map(move |(ka, a)| (bits::concat(ka, na, kb, nb), a.mul_ref(b)))
            })
            .collect();
        Ok(SparseVector {
            n: na + nb,
            entries,
            sorted: false,
            cutoff: self.cutoff,
        })
    }

    fn sample(&self, rng: &mut dyn RngCore, shots: usize) -> Result<Vec<Bits>> {
        let mut cum = Vec::with_capacity(self.entries.len());
        let mut acc = 0.0;
        for (_, a) in &self.entries {
            acc += a.abs_sq_f64();
            cum.push(acc);
        }
        if acc <= 0.0 {
            return Err(SimError::ZeroState("sample".into()));
        }
        Ok((0..shots)
            .map(|_| {
                let u = rng.gen::<f64>() * acc;
                let i = cum.partition_point(|&c| c <= u).min(cum.len() - 1);
                self.entries[i].0.clone()
            })
            .collect())
    }

    fn to_vector(&self) -> Result<Vec<S>> {
        assert!(self.n <= 26, "dense vector of more than 26 qubits");
        let mut v = vec![S::zero(); 1usize << self.n];
        for (b, a) in &self.entries {
            v[bits::to_u64(b, self.n) as usize] = a.clone();
        }
        Ok(v)
    }
}

impl<S: Amplitude> fmt::Debug for SparseVector<S> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "SparseVector({} qubits, {} entries)",
            self.n,
            self.entries.len()
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::circuit::Circuit;
    use crate::scalar::C64;
    use crate::state::StateVector;
    use crate::Cyclo;
    use rand::rngs::StdRng;
    use rand::SeedableRng;

    fn random_circuit(n: usize, gates: usize, rng: &mut StdRng) -> Circuit {
        let mut c = Circuit::new(n);
        let one = [
            Gate::H,
            Gate::S,
            Gate::T,
            Gate::X,
            Gate::Y,
            Gate::SqrtX,
            Gate::Rz(0.4),
        ];
        let two = [Gate::Cx, Gate::Cz, Gate::Swap, Gate::Cs];
        for _ in 0..gates {
            let r = rng.gen::<f64>();
            if r < 0.5 {
                c.gate(one[rng.gen_range(0..one.len())].clone(), &[rng.gen_range(0..n)]);
            } else if r < 0.9 {
                let a = rng.gen_range(0..n);
                let b = (a + 1 + rng.gen_range(0..n - 1)) % n;
                c.gate(two[rng.gen_range(0..two.len())].clone(), &[a, b]);
            } else {
                let mut qs: Vec<usize> = (0..n).collect();
                for i in 0..3 {
                    let j = rng.gen_range(i..n);
                    qs.swap(i, j);
                }
                c.gate(Gate::Ccx, &qs[..3]);
            }
        }
        c
    }

    #[test]
    fn matches_the_dense_vector() {
        let mut rng = StdRng::seed_from_u64(13);
        for trial in 0..30 {
            let n = 3 + trial % 6;
            let c = random_circuit(n, 30, &mut rng);
            let mut s: SparseVector<C64> = SparseVector::zero_state(n);
            let mut v: StateVector<C64> = StateVector::zero_state(n);
            for op in c.ops() {
                if let Op::Gate { gate, qubits } = op {
                    s.apply_gate(gate, qubits).unwrap();
                    v.apply_gate(gate, qubits).unwrap();
                }
            }
            let sv = s.to_vector().unwrap();
            let vv = v.to_vector().unwrap();
            assert!(
                sv.iter().zip(&vv).all(|(a, b)| (a - b).norm() < 1e-9),
                "trial {trial}"
            );
            let kinds: Vec<crate::pauli::PauliKind> = (0..n)
                .map(|_| {
                    [
                        crate::pauli::PauliKind::I,
                        crate::pauli::PauliKind::X,
                        crate::pauli::PauliKind::Y,
                        crate::pauli::PauliKind::Z,
                    ][rng.gen_range(0..4)]
                })
                .collect();
            let p = Pauli::from_kinds(&kinds);
            assert!((s.expectation(&p).unwrap() - v.expectation(&p).unwrap()).norm() < 1e-9);
            let mut s2 = s.clone();
            let mut v2 = v.clone();
            let b = rng.gen::<bool>();
            let ws = s2.project(&p, b).unwrap();
            let wv = v2.project(&p, b).unwrap();
            assert!((ws - wv).norm() < 1e-9);
            assert!((s.inner(&s2).unwrap() - v.inner(&v2).unwrap()).norm() < 1e-9);
            let mut s3 = s.clone();
            s3.apply_pauli(&p);
            let mut v3 = v.clone();
            v3.apply_pauli(&p);
            assert!(s3
                .to_vector()
                .unwrap()
                .iter()
                .zip(v3.amplitudes())
                .all(|(a, b)| (a - b).norm() < 1e-9));
        }
    }

    #[test]
    fn reversible_arithmetic_on_many_qubits_stays_sparse_and_exact() {
        // a ripple of Toffolis and CNOTs on 200 qubits with 4 superposed inputs
        let n = 200;
        let mut s: SparseVector<Cyclo<8>> = SparseVector::zero_state(n);
        for q in [0, 1] {
            s.apply_gate(&Gate::H, &[q]).unwrap();
        }
        for q in 0..n - 2 {
            s.apply_gate(&Gate::Ccx, &[q, q + 1, q + 2]).unwrap();
            s.apply_gate(&Gate::Cx, &[q, q + 2]).unwrap();
        }
        assert_eq!(s.num_entries(), 4);
        assert!(ncalg::Coeff::is_one(&s.norm_sq().unwrap()));
        let t: SparseVector<Cyclo<8>> = s
            .tensor(&SparseVector::basis_state(3, &bits::from_u64(5, 3)))
            .unwrap();
        assert_eq!(t.num_qubits(), 203);
        assert_eq!(t.num_entries(), 4);
    }
}
