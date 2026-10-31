//! Stabilizer states as tableaux, stored by the **inverse** Clifford (the
//! layout of Stim's tableau simulator): `ψ = U|0…0⟩` is kept as the rows
//! `V X_q V†`, `V Z_q V†` of `V = U⁻¹`, with signs.
//!
//! * A Clifford gate on `k` qubits recombines `2k` rows: `O(k n/64)`.
//! * Measuring a Pauli `P` pulls it back, `V P V†`, in `O(|P| n/64)`. When the
//!   pull-back has no `X` part the outcome is deterministic and is its sign,
//!   so deterministic measurements need no product of stabilizer rows.
//! * A random outcome folds the pull-back to `±X_a` by a fan-out of frame
//!   `CX`/`CZ` gates and one `S`, then applies `H_a` in the frame. This is one
//!   pass over the rows that reads only the nonzero words of the fold: `O(n²/64)` at worst.
//! * The stabilizer generators and destabilizers (the forward images) are
//!   recovered on demand in `O(n²/64)` each, as a column of the symplectic
//!   inverse with its sign fixed by a pull-back.
//!
//! A tableau is a *ray*: it has no global phase and no norm.  Amplitudes and
//! inner products are those of a canonical representative, the
//! [`AffineForm`] `ψ(c ⊕ Bu) = 2^{-k/2} i^{Q(u)}` with `ψ(c) > 0`, so they are
//! meaningful up to a global phase (moduli are exact).  Projections keep the
//! state normalized and return the branch probability.  For phase-exact
//! vectors use [`ChForm`](super::ChForm) (or the path-sum families); quantities
//! that do not depend on the phase of the ray, such as `⟨ψ|U|ψ⟩` for a
//! Clifford circuit `U`, are exact here ([`Stabilizer::circuit_expectation`]).

use super::QuantumState;
use crate::bits::{self, Bits};
use crate::clifford::CliffordTableau;
use crate::error::{check_qubits, unsupported, Result, SimError};
use crate::gate::Gate;
use crate::pauli::{Pauli, PauliTable};
use crate::phasepoly::QuadForm;
use crate::scalar::{Amplitude, Phasor};
use rand::{Rng, RngCore};
use std::fmt;

const BACKEND: &str = "Stabilizer";

/// A stabilizer state in affine form: `ψ(shift ⊕ Σ_i u_i basis_i) = 2^{-k/2} i^{form(u)}`
/// and zero off that affine space.  The basis rows are in reduced echelon
/// form with pivot columns `pivots`, so coordinates are read off directly.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct AffineForm {
    /// Number of qubits.
    pub n: usize,
    /// A point of the support.
    pub shift: Bits,
    /// Independent directions of the support (`k` of them).
    pub basis: Vec<Bits>,
    /// Pivot column of each basis vector.
    pub pivots: Vec<usize>,
    /// The ℤ₄ quadratic form on the `k` coordinates.
    pub form: QuadForm,
}

impl AffineForm {
    /// From any independent basis of the support and the form in its
    /// coordinates: the basis is brought to reduced echelon form and the form
    /// transformed accordingly.
    pub fn from_basis(n: usize, shift: Bits, basis: Vec<Bits>, form: QuadForm) -> Result<AffineForm> {
        let k = basis.len();
        assert_eq!(
            form.num_vars(),
            k,
            "from_basis: form on the wrong number of coordinates"
        );
        let (red, pivots, t) = bits::echelon_with_transform(&basis, n)
            .ok_or_else(|| SimError::Invalid("affine form: dependent basis".into()))?;
        // reduced_j = Σ_i T_ji basis_i, so old coordinates u = Tᵀ u′: column j of Tᵀ is row j of T
        let form = form.substitute(&bits::zero(k), &t, k);
        let mut shift = shift;
        bits::ensure_len(&mut shift, n);
        bits::mask_to(&mut shift, n);
        Ok(AffineForm {
            n,
            shift,
            basis: red,
            pivots,
            form,
        })
    }
    /// `k = dim` of the support.
    pub fn dim(&self) -> usize {
        self.basis.len()
    }
    /// The coordinates `u` with `x = shift ⊕ Σ u_i basis_i`, if `x` is in the support.
    pub fn coordinates(&self, x: &[u64]) -> Option<Bits> {
        let k = self.dim();
        let mut d = bits::zero(self.n);
        bits::xor_into(&mut d, x);
        bits::xor_into(&mut d, &self.shift);
        bits::mask_to(&mut d, self.n);
        let mut u = bits::zero(k);
        let mut acc = bits::zero(self.n);
        for (i, &p) in self.pivots.iter().enumerate() {
            if bits::get(&d, p) {
                bits::set(&mut u, i, true);
                bits::xor_into(&mut acc, &self.basis[i]);
            }
        }
        bits::mask_to(&mut acc, self.n);
        (acc == d).then_some(u)
    }
    /// `ψ(x)`.
    pub fn amplitude(&self, x: &[u64]) -> Phasor {
        match self.coordinates(x) {
            None => Phasor::Zero,
            Some(u) => Phasor::new(-(self.dim() as i32), self.form.eval(&u) as i64, 2),
        }
    }
    /// `⟨self|other⟩`: the Gauss sum of `Q_other(u) − Q_self(v)` over the
    /// solutions of `B_other u ⊕ B_self v = shift_other ⊕ shift_self`.
    pub fn inner(&self, other: &AffineForm) -> Phasor {
        assert_eq!(self.n, other.n, "inner: register size");
        let n = self.n;
        let (ks, ko) = (self.dim(), other.dim());
        let m = ks + ko;
        // rows of the system by qubit: bit j = other.basis[j][q], bit ko + j = self.basis[j][q]
        let mut sys: Vec<Bits> = vec![bits::zero(m); n];
        for (j, b) in other.basis.iter().enumerate() {
            for i in bits::ones(b) {
                if i < n {
                    bits::set(&mut sys[i], j, true);
                }
            }
        }
        for (j, b) in self.basis.iter().enumerate() {
            for i in bits::ones(b) {
                if i < n {
                    bits::set(&mut sys[i], ko + j, true);
                }
            }
        }
        let mut rhs = other.shift.clone();
        bits::xor_into(&mut rhs, &self.shift);
        bits::mask_to(&mut rhs, n);
        let Some((mut w0, cols)) = bits::solve_affine(&sys, m, &rhs) else {
            return Phasor::Zero;
        };
        bits::ensure_len(&mut w0, m);
        let q = other.form.direct_sum(&self.form.negated());
        let r = q.substitute(&w0, &cols, cols.len());
        r.gauss_sum().mul(Phasor::sqrt2_pow(-(m as i32)))
    }
    /// `P · ψ = c · ψ'` for a Pauli `P = i^k X^x Z^z`: the same basis, the
    /// shift moved by `x`, `2 (z·b_j)` added to the linear coefficients and
    /// `c = i^{k + 2 z·shift}`; `O(k n/64)`.
    pub fn apply_pauli(&self, p: &Pauli) -> (Phasor, AffineForm) {
        assert_eq!(p.num_qubits(), self.n, "apply_pauli: register size");
        let mut out = self.clone();
        bits::xor_into(&mut out.shift, p.x_bits());
        bits::mask_to(&mut out.shift, self.n);
        for (j, b) in self.basis.iter().enumerate() {
            if bits::dot(p.z_bits(), b) {
                out.form.add_linear(j, 2);
            }
        }
        let c = p.phase() as i64 + 2 * bits::dot(p.z_bits(), &self.shift) as i64;
        (Phasor::i_pow(c), out)
    }
    /// `shots` points of the support, uniformly (the Born distribution of a
    /// stabilizer state: all amplitudes on the support have the same modulus).
    pub fn sample(&self, rng: &mut dyn RngCore, shots: usize) -> Vec<Bits> {
        (0..shots)
            .map(|_| {
                let mut x = self.shift.clone();
                for b in &self.basis {
                    if rng.gen::<bool>() {
                        bits::xor_into(&mut x, b);
                    }
                }
                bits::mask_to(&mut x, self.n);
                x
            })
            .collect()
    }
    /// All `2^n` amplitudes.
    pub fn to_vector<S: Amplitude>(&self) -> Result<Vec<S>> {
        let n = self.n;
        assert!(n <= 26, "dense vector of more than 26 qubits");
        let k = self.dim();
        let mut v = vec![S::zero(); 1usize << n];
        let scale =
            S::sqrt2_pow(-(k as i64)).ok_or_else(|| SimError::NotRepresentable(format!("2^(-{k}/2)")))?;
        for u in 0..(1u64 << k) {
            let ub = bits::from_u64(u, k);
            let mut x = self.shift.clone();
            for i in bits::ones(&ub) {
                bits::xor_into(&mut x, &self.basis[i]);
            }
            bits::mask_to(&mut x, n);
            let idx = bits::to_u64(&x, n) as usize;
            v[idx] = scale.mul_ref(&S::i_pow(self.form.eval(&ub) as i64));
        }
        Ok(v)
    }
}

/// A stabilizer state `ψ = U|0…0⟩`, stored by the inverse `V = U⁻¹` (see the
/// [module documentation](self)): rows `0..n` of the table are `V X_q V†`,
/// rows `n..2n` are `V Z_q V†`, with signs.  The destabilizers and stabilizer
/// generators are the forward images `U X_q U†`, `U Z_q U†`
/// ([`Stabilizer::destabilizer`], [`Stabilizer::stabilizer`]).
#[derive(Clone, PartialEq, Eq)]
pub struct Stabilizer {
    n: usize,
    inv: PauliTable,
}

impl Stabilizer {
    /// `|0…0⟩`.
    pub fn zero_state(n: usize) -> Self {
        Stabilizer {
            n,
            inv: CliffordTableau::identity(n).images().clone(),
        }
    }
    /// `U |0…0⟩` for a Clifford tableau `U` (its inverse is stored, `O(n³/64)`).
    pub fn from_clifford(u: &CliffordTableau) -> Self {
        Stabilizer {
            n: u.num_qubits(),
            inv: u.inverse().images().clone(),
        }
    }
    /// `|x⟩`.
    pub fn basis_state(n: usize, x: &[u64]) -> Self {
        let mut s = Self::zero_state(n);
        for q in bits::ones(x) {
            if q < n {
                // V ← V X_q: the image of Z_q changes sign
                s.inv.add_phase(n + q, 2);
            }
        }
        s
    }
    /// `|+⟩^{⊗n}`.
    pub fn plus_state(n: usize) -> Self {
        let mut s = Self::zero_state(n);
        for q in 0..n {
            s.inv.swap_rows(q, n + q);
        }
        s
    }
    /// The state stabilized by `n` independent, commuting, Hermitian Pauli
    /// generators on `n` qubits; destabilizers are completed by symplectic
    /// Gram–Schmidt.
    pub fn from_stabilizers(gens: &[Pauli]) -> Result<Self> {
        let n = gens.len();
        let invalid = |m: &str| SimError::Invalid(format!("from_stabilizers: {m}"));
        for (i, g) in gens.iter().enumerate() {
            if g.num_qubits() != n {
                return Err(invalid("each generator must act on n qubits, n = their number"));
            }
            if !g.is_hermitian() {
                return Err(invalid(&format!("{g} is not Hermitian")));
            }
            for h in &gens[..i] {
                if !g.commutes(h) {
                    return Err(invalid(&format!("{g} and {h} do not commute")));
                }
            }
        }
        let stab = PauliTable::from_rows(n, gens);
        let sym = stab.symplectic();
        if sym.rank() != n {
            return Err(invalid("generators are dependent"));
        }
        // ω(d, s_j) = x·z_j + z·x_j: the system rows are (z_j | x_j).  One
        // elimination with its transform T (T·rows = reduced, pivots P) gives
        // every d_i with rows_j · d_i = δ_ij: d_i = Σ_k T_ki e_{P_k}.
        let rows: Vec<Bits> = (0..n)
            .map(|j| bits::concat(stab.z_row(j), n, stab.x_row(j), n))
            .collect();
        let (_, pivots, t) =
            bits::echelon_with_transform(&rows, 2 * n).ok_or_else(|| invalid("no destabilizers found"))?;
        let mut destab: Vec<Pauli> = Vec::with_capacity(n);
        for i in 0..n {
            let mut x = bits::zero(n);
            let mut z = bits::zero(n);
            for (k, &pk) in pivots.iter().enumerate() {
                if bits::get(&t[k], i) {
                    if pk < n {
                        bits::toggle(&mut x, pk);
                    } else {
                        bits::toggle(&mut z, pk - n);
                    }
                }
            }
            let mut p = Pauli::new(n, x, z, 0);
            // orthogonalise against the earlier destabilizers: d_i += ω(d_i, d_j) s_j
            for j in 0..i {
                if !p.commutes(&destab[j]) {
                    p.mul_assign(&gens[j]);
                }
            }
            let ph = bits::dot(p.x_bits(), p.z_bits()) as u8;
            p.set_phase(ph);
            destab.push(p);
        }
        let mut table = PauliTable::new(n);
        for d in &destab {
            table.push(d);
        }
        for g in gens {
            table.push(g);
        }
        Ok(Self::from_forward(&table))
    }
    /// From the forward rows (destabilizers, then stabilizers).
    fn from_forward(table: &PauliTable) -> Self {
        let u = CliffordTableau::from_images_unchecked(table.clone());
        Stabilizer {
            n: table.num_qubits(),
            inv: u.inverse().images().clone(),
        }
    }
    /// `self ⊗ other` (`self` on qubits `0..n_self`): block-diagonal, `O(n²/64)`.
    pub fn tensor(&self, other: &Stabilizer) -> Stabilizer {
        Self::tensor_all(&[self.clone(), other.clone()])
    }
    /// `ψ_0 ⊗ ψ_1 ⊗ ⋯` (`ψ_0` on the lowest qubits) in one pass.
    pub fn tensor_all(states: &[Stabilizer]) -> Stabilizer {
        let n: usize = states.iter().map(|s| s.n).sum();
        let mut inv = PauliTable::new(n);
        let offsets: Vec<usize> = states
            .iter()
            .scan(0, |acc, s| {
                let o = *acc;
                *acc += s.n;
                Some(o)
            })
            .collect();
        for half in [0usize, 1] {
            for (s, &o) in states.iter().zip(&offsets) {
                for q in 0..s.n {
                    let r = half * s.n + q;
                    let mut x = bits::zero(n);
                    let mut z = bits::zero(n);
                    bits::or_shifted(&mut x, s.inv.x_row(r), s.n, o);
                    bits::or_shifted(&mut z, s.inv.z_row(r), s.n, o);
                    inv.push(&Pauli::new(n, x, z, s.inv.phase(r)));
                }
            }
        }
        Stabilizer { n, inv }
    }
    /// This state on `qubits` (in that order) of an `n`-qubit register whose
    /// other qubits are `|0⟩`.
    pub fn embed(&self, n: usize, qubits: &[usize]) -> Result<Stabilizer> {
        if qubits.len() != self.n {
            return Err(SimError::Invalid(format!(
                "embed: {} qubits for a state on {}",
                qubits.len(),
                self.n
            )));
        }
        check_qubits(qubits, n)?;
        let mut s = Stabilizer::zero_state(n);
        for (i, &q) in qubits.iter().enumerate() {
            s.inv.set_row(q, &self.inv.row(i).embed(n, qubits));
            s.inv.set_row(n + q, &self.inv.row(self.n + i).embed(n, qubits));
        }
        Ok(s)
    }
    /// Apply a Clifford gate: `U ← G U`, so `V ← V G⁻¹` (rows recombined).
    pub fn apply_gate(&mut self, gate: &Gate, qubits: &[usize]) -> Result<()> {
        check_qubits(qubits, self.n)?;
        if qubits.len() != gate.arity() {
            return Err(SimError::Arity {
                gate: gate.name(),
                expected: gate.arity(),
                got: qubits.len(),
            });
        }
        let n = self.n;
        let t = &mut self.inv;
        match gate {
            Gate::I => {}
            Gate::H => t.swap_rows(qubits[0], n + qubits[0]),
            Gate::S => {
                // S† X S = −Y = i³ X Z
                t.mul_rows(qubits[0], n + qubits[0]);
                t.add_phase(qubits[0], 3);
            }
            Gate::Sdg => {
                t.mul_rows(qubits[0], n + qubits[0]);
                t.add_phase(qubits[0], 1);
            }
            Gate::X => t.add_phase(n + qubits[0], 2),
            Gate::Z => t.add_phase(qubits[0], 2),
            Gate::Y => {
                t.add_phase(qubits[0], 2);
                t.add_phase(n + qubits[0], 2);
            }
            Gate::Cx => {
                let (c, tg) = (qubits[0], qubits[1]);
                t.mul_rows(c, tg);
                t.mul_rows(n + tg, n + c);
            }
            Gate::Cz => {
                let (a, b) = (qubits[0], qubits[1]);
                t.mul_rows(a, n + b);
                t.mul_rows(b, n + a);
            }
            Gate::Swap => {
                let (a, b) = (qubits[0], qubits[1]);
                t.swap_rows(a, b);
                t.swap_rows(n + a, n + b);
            }
            Gate::Pauli(p) => {
                // V ← V P: the images of the letters anticommuting with P change sign
                for (j, &q) in qubits.iter().enumerate() {
                    let (x, z) = p.kind(j).bits();
                    if z {
                        t.add_phase(q, 2);
                    }
                    if x {
                        t.add_phase(n + q, 2);
                    }
                }
            }
            _ => {
                let tab = gate
                    .tableau()
                    .ok_or_else(|| unsupported(BACKEND, format!("non-Clifford gate {gate}")))?;
                return self.apply_tableau(&tab, qubits);
            }
        }
        Ok(())
    }
    /// Run a Clifford circuit with measurements, returning the record.
    pub fn run(&mut self, circuit: &crate::circuit::Circuit, rng: &mut dyn RngCore) -> Result<Vec<bool>> {
        QuantumState::<crate::scalar::C64>::run(self, circuit, rng)
    }
    /// Measure a Hermitian Pauli (same as [`Stabilizer::measure_pauli`]).
    pub fn measure(&mut self, p: &Pauli, rng: &mut dyn RngCore) -> Result<bool> {
        self.measure_pauli(p, rng)
    }
    /// Number of qubits.
    pub fn num_qubits(&self) -> usize {
        self.n
    }
    /// The forward image `U Q U†` of basis row `i` (`i < n`: `X_i`, else
    /// `Z_{i−n}`): its bits are a column of the inverse (`R⁻¹ = Λ Rᵀ Λ`
    /// symplectically), its sign is fixed by pulling it back.  `O(n²/64)`.
    fn forward_row(&self, i: usize) -> Pauli {
        let n = self.n;
        // σ swaps the halves; entry j of the forward row is bit σ(i) of row σ(j)
        let (col, from_z) = if i < n { (i, true) } else { (i - n, false) };
        let mut x = bits::zero(n);
        let mut z = bits::zero(n);
        for j in 0..n {
            let (rz, rx) = (self.inv.row_bits(n + j), self.inv.row_bits(j));
            let pick = |r: (&[u64], &[u64])| {
                if from_z {
                    bits::get(r.1, col)
                } else {
                    bits::get(r.0, col)
                }
            };
            if pick(rz) {
                bits::set(&mut x, j, true);
            }
            if pick(rx) {
                bits::set(&mut z, j, true);
            }
        }
        let ph = bits::dot(&x, &z) as u8;
        let p = Pauli::new(n, x, z, ph);
        // V p V† must be +X_i / +Z_i
        let back = self.pull_back(&p);
        debug_assert!(bits::popcount(back.x_bits()) + bits::popcount(back.z_bits()) == 1);
        if back.phase() == 2 {
            p.negated()
        } else {
            p
        }
    }
    /// Stabilizer generator `i`, `U Z_i U†` (`O(n²/64)`).
    pub fn stabilizer(&self, i: usize) -> Pauli {
        self.forward_row(self.n + i)
    }
    /// Destabilizer `i`, `U X_i U†` (anticommutes with generator `i` only).
    pub fn destabilizer(&self, i: usize) -> Pauli {
        self.forward_row(i)
    }
    /// The stabilizer generators.
    pub fn stabilizers(&self) -> Vec<Pauli> {
        use rayon::prelude::*;
        (0..self.n).into_par_iter().map(|i| self.stabilizer(i)).collect()
    }
    /// The forward table (destabilizers, then stabilizers), `O(n³/64)`.
    pub fn table(&self) -> PauliTable {
        use rayon::prelude::*;
        let rows: Vec<Pauli> = (0..2 * self.n)
            .into_par_iter()
            .map(|i| self.forward_row(i))
            .collect();
        PauliTable::from_rows(self.n, &rows)
    }
    /// The stored inverse table: rows `V X_q V†`, then `V Z_q V†`.
    pub fn inverse_table(&self) -> &PauliTable {
        &self.inv
    }
    /// Apply a Clifford tableau on `qubits` (any size): the rows of the
    /// gate qubits are recombined by the images of its inverse, `O(k² n/64)`.
    pub fn apply_tableau(&mut self, t: &CliffordTableau, qubits: &[usize]) -> Result<()> {
        check_qubits(qubits, self.n)?;
        if qubits.len() != t.num_qubits() {
            return Err(SimError::Arity {
                gate: "Clifford".into(),
                expected: t.num_qubits(),
                got: qubits.len(),
            });
        }
        let n = self.n;
        let k = qubits.len();
        let gi = t.inverse();
        let inv = &self.inv;
        let new: Vec<Pauli> = (0..2 * k)
            .map(|i| {
                let w = if i < k { gi.x_image(i) } else { gi.z_image(i - k) };
                let mut out = Pauli::identity(n);
                out.set_phase(w.phase());
                for j in bits::ones(w.x_bits()) {
                    inv.mul_pauli_by_row(&mut out, qubits[j]);
                }
                for j in bits::ones(w.z_bits()) {
                    inv.mul_pauli_by_row(&mut out, n + qubits[j]);
                }
                out
            })
            .collect();
        for (j, &q) in qubits.iter().enumerate() {
            self.inv.set_row(q, &new[j]);
            self.inv.set_row(n + q, &new[k + j]);
        }
        Ok(())
    }
    /// `V P V†` (`ψ = U|0⟩`, `V = U⁻¹`), `O(|P| n/64)`: deterministic iff it has no `X` part.
    pub fn pull_back(&self, p: &Pauli) -> Pauli {
        let n = self.n;
        let mut out = Pauli::identity(n);
        out.set_phase(p.phase());
        for q in bits::ones(p.x_bits()) {
            self.inv.mul_pauli_by_row(&mut out, q);
        }
        for q in bits::ones(p.z_bits()) {
            self.inv.mul_pauli_by_row(&mut out, n + q);
        }
        out
    }
    /// `⟨ψ|P|ψ⟩`: `0`, or `i^k` with `V P V† = i^k Z^B` (`O(|P| n/64)`).
    pub fn expectation_phasor(&self, p: &Pauli) -> Phasor {
        assert_eq!(p.num_qubits(), self.n, "expectation: register size");
        let q = self.pull_back(p);
        if bits::is_zero(q.x_bits()) {
            Phasor::i_pow(q.phase() as i64)
        } else {
            Phasor::Zero
        }
    }
    /// Is `P ψ = ψ`?
    pub fn is_stabilized_by(&self, p: &Pauli) -> bool {
        self.expectation_phasor(p) == Phasor::ONE
    }
    /// Is the outcome of measuring `p` deterministic?  `O(|p| n/64)`.
    pub fn is_deterministic(&self, p: &Pauli) -> bool {
        bits::is_zero(self.pull_back(p).x_bits())
    }
    /// Project onto the `outcome` eigenspace of `p` (`None`: draw it with
    /// `rng`); returns the outcome and its probability (`1` or `1/2`).
    fn collapse(&mut self, p: &Pauli, forced: Option<bool>, rng: &mut dyn RngCore) -> Result<(bool, Phasor)> {
        if !p.is_hermitian() {
            return Err(SimError::Invalid(format!(
                "measured operator {p} is not Hermitian"
            )));
        }
        assert_eq!(p.num_qubits(), self.n, "measurement: register size");
        let q = self.pull_back(p);
        let Some(a) = bits::lowest_one(q.x_bits()) else {
            // V P V† = ±Z^B and Z^B|0⟩ = |0⟩: deterministic
            let val = q.phase() == 2;
            return match forced {
                Some(o) if o != val => Err(SimError::ZeroState(format!(
                    "outcome {} of {p} has probability 0",
                    o as u8
                ))),
                _ => Ok((val, Phasor::ONE)),
            };
        };
        let outcome = forced.unwrap_or_else(|| rng.gen::<bool>());
        // fold q to ±X_a in the frame: CX(a → A) clears the other X bits
        // (z_a ⊕= z·A), CZ(a, B) the other Z bits, S on a removes Z_a
        let mut set_a = q.x_bits().to_vec();
        bits::set(&mut set_a, a, false);
        let mut zq = q.z_bits().to_vec();
        if bits::dot(&zq, &set_a) {
            bits::toggle(&mut zq, a);
        }
        let za = bits::get(&zq, a);
        let mut set_b = zq;
        bits::set(&mut set_b, a, false);
        let phase = if za { (q.phase() + 1) & 3 } else { q.phase() };
        debug_assert!(phase % 2 == 0, "folded to a non-Hermitian X_a");
        // (1 + (−1)^outcome σ X_a)/2 |0⟩ ∝ |+⟩ or |−⟩ = H_a X_a^m |0⟩
        let minus = (phase == 2) ^ outcome;
        // the nonzero words of the sets: rows are read only there
        let sparse = |v: &[u64]| -> Vec<(usize, u64)> {
            v.iter()
                .enumerate()
                .filter(|(_, &w)| w != 0)
                .map(|(i, &w)| (i, w))
                .collect()
        };
        let (sa, sb) = (sparse(&set_a), sparse(&set_b));
        let (wa, ma) = (a / 64, 1u64 << (a % 64));
        let dot = |row: &[u64], set: &[(usize, u64)]| -> bool {
            set.iter()
                .fold(0u32, |acc, &(i, w)| acc ^ (row[i] & w).count_ones())
                & 1
                == 1
        };
        // one pass over the rows: CX fan-out, CZ fan-out, S, H, X on a, in this order
        self.inv.for_each_row_mut(|_, x, z, ph| {
            if !sa.is_empty() {
                let t = dot(z, &sa);
                if x[wa] & ma != 0 {
                    for &(i, w) in &sa {
                        x[i] ^= w;
                    }
                }
                if t {
                    z[wa] ^= ma;
                }
            }
            if !sb.is_empty() {
                let xb = dot(x, &sb);
                if x[wa] & ma != 0 {
                    for &(i, w) in &sb {
                        z[i] ^= w;
                    }
                    if xb {
                        *ph = (*ph + 2) & 3;
                    }
                }
                if xb {
                    z[wa] ^= ma;
                }
            }
            if za && x[wa] & ma != 0 {
                z[wa] ^= ma;
                *ph = (*ph + 1) & 3;
            }
            let (xa, zz) = (x[wa] & ma != 0, z[wa] & ma != 0);
            if xa && zz {
                *ph = (*ph + 2) & 3;
            }
            if xa != zz {
                x[wa] ^= ma;
                z[wa] ^= ma;
            }
            // X on a flips the sign of rows with z_a, which after H is the old x_a
            if minus && xa {
                *ph = (*ph + 2) & 3;
            }
        });
        Ok((outcome, Phasor::new(-2, 0, 0)))
    }
    /// Measure the Hermitian Pauli `p` (random outcome drawn with `rng` when
    /// the outcome is not determined).
    pub fn measure_pauli(&mut self, p: &Pauli, rng: &mut dyn RngCore) -> Result<bool> {
        Ok(self.collapse(p, None, rng)?.0)
    }
    /// Project onto the `outcome` eigenspace of `p`; returns the probability
    /// (`1/2` or `1`), or [`SimError::ZeroState`] for an impossible outcome.
    pub fn project_pauli(&mut self, p: &Pauli, outcome: bool) -> Result<Phasor> {
        let mut r = rand::rngs::mock::StepRng::new(0, 0);
        Ok(self.collapse(p, Some(outcome), &mut r)?.1)
    }
    /// The affine form (canonical representative with `ψ(shift) > 0`).
    pub fn affine_form(&self) -> AffineForm {
        let n = self.n;
        let g = PauliTable::from_rows(n, &self.stabilizers());
        let mut g = g;
        // reduced echelon form of the X parts, by products of generators
        let mut pivots = Vec::new();
        let mut rank = 0;
        for q in 0..n {
            let Some(r) = (rank..n).find(|&r| bits::get(g.x_row(r), q)) else {
                continue;
            };
            g.swap_rows(rank, r);
            for r2 in 0..n {
                if r2 != rank && bits::get(g.x_row(r2), q) {
                    g.mul_rows(r2, rank);
                }
            }
            pivots.push(q);
            rank += 1;
        }
        let k = rank;
        // Z-type rows: (−1)^s Z^b fixes b·x = s on the support
        let sys: Vec<Bits> = (k..n).map(|r| g.z_row(r).to_vec()).collect();
        let mut rhs = bits::zero(n - k);
        for (idx, r) in (k..n).enumerate() {
            debug_assert!(g.phase(r) % 2 == 0);
            if g.phase(r) == 2 {
                bits::set(&mut rhs, idx, true);
            }
        }
        let (mut shift, _) =
            bits::solve_affine(&sys, n, &rhs).expect("a stabilizer state has a nonempty support");
        bits::ensure_len(&mut shift, n);
        let mut form = QuadForm::new(k);
        for i in 0..k {
            let (a, b, r) = (g.x_row(i), g.z_row(i), g.phase(i) as i64);
            let l = -r + 2 * bits::dot(b, &shift) as i64 + 2 * bits::dot(b, a) as i64;
            form.add_linear(i, l);
            for j in i + 1..k {
                if bits::dot(b, g.x_row(j)) {
                    form.add_quad(i, j);
                }
            }
        }
        let basis = (0..k).map(|i| g.x_row(i).to_vec()).collect();
        AffineForm {
            n,
            shift,
            basis,
            pivots,
            form,
        }
    }
    /// The tableau of an affine form.
    pub fn from_affine_form(af: &AffineForm) -> Result<Self> {
        let (n, k) = (af.n, af.dim());
        let a_rows = bits::to_bitmat(&af.basis, n);
        let mut gens = Vec::with_capacity(n);
        for i in 0..k {
            // b_i with b_i·a_j = J_ij (j ≠ i), b_i·a_i ≡ L_i (mod 2): the basis is in
            // reduced echelon form (a_j has a 1 at pivot P_j and 0 at the other
            // pivots), so b_i = Σ_j rhs_j e_{P_j}
            let mut b = bits::zero(n);
            for j in 0..k {
                let rhs = if j == i {
                    af.form.linear(i) % 2 == 1
                } else {
                    af.form.has_quad(i, j)
                };
                if rhs {
                    bits::set(&mut b, af.pivots[j], true);
                }
            }
            debug_assert!((0..k).all(|j| bits::dot(&b, &af.basis[j])
                == if j == i {
                    af.form.linear(i) % 2 == 1
                } else {
                    af.form.has_quad(i, j)
                }));
            let a = af.basis[i].clone();
            let r = (-(af.form.linear(i) as i64)
                + 2 * bits::dot(&b, &af.shift) as i64
                + 2 * bits::dot(&b, &a) as i64)
                .rem_euclid(4) as u8;
            gens.push(Pauli::new(n, a, b, r));
        }
        let ann = a_rows.nullspace();
        for b in bits::from_bitmat(&ann) {
            let ph = 2 * bits::dot(&b, &af.shift) as u8;
            gens.push(Pauli::new(n, bits::zero(n), b, ph));
        }
        Self::from_stabilizers(&gens)
    }
    /// `⟨ψ|U|ψ⟩` for a unitary Clifford circuit `U` (gates with their usual
    /// matrices; no measurements), exactly.  It does not depend on the global
    /// phase of `ψ`, so it is well defined on a ray; computed on the
    /// phase-tracking [`ChForm`](super::ChForm) in `O(n²/64)` per `H` and
    /// `O(n³/64)` for the final overlap.
    pub fn circuit_expectation(&self, u: &crate::circuit::Circuit) -> Result<Phasor> {
        super::ChForm::from_stabilizer(self).circuit_expectation(u)
    }
    /// The eigenvalue of the unitary Clifford circuit `U` on `ψ` when `ψ` is
    /// an eigenvector (`U ψ = λ ψ`), else `None`.
    pub fn circuit_eigenvalue(&self, u: &crate::circuit::Circuit) -> Result<Option<Phasor>> {
        let e = self.circuit_expectation(u)?;
        Ok((e.abs_sq_log2() == Some(0)).then_some(e))
    }
    /// `ψ(x)` of the canonical representative.
    pub fn amplitude_phasor(&self, x: &[u64]) -> Phasor {
        self.affine_form().amplitude(x)
    }
    /// `⟨self|other⟩` of the canonical representatives.
    pub fn inner_phasor(&self, other: &Stabilizer) -> Phasor {
        self.affine_form().inner(&other.affine_form())
    }
}

impl<S: Amplitude> QuantumState<S> for Stabilizer {
    const BACKEND: &'static str = BACKEND;

    fn num_qubits(&self) -> usize {
        self.n
    }

    fn apply_gate(&mut self, gate: &Gate, qubits: &[usize]) -> Result<()> {
        Stabilizer::apply_gate(self, gate, qubits)
    }

    fn norm_sq(&self) -> Result<S> {
        Ok(S::one())
    }

    fn amplitude(&self, x: &[u64]) -> Result<S> {
        let p = self.amplitude_phasor(x);
        S::from_phasor(p).ok_or_else(|| SimError::NotRepresentable(format!("amplitude {p}")))
    }

    fn expectation(&self, p: &Pauli) -> Result<S> {
        if p.num_qubits() != self.n {
            return Err(SimError::Invalid(
                "expectation: operator on the wrong register".into(),
            ));
        }
        Ok(S::from_phasor(self.expectation_phasor(p)).expect("i^k is representable"))
    }

    fn inner(&self, other: &Self) -> Result<S> {
        if other.n != self.n {
            return Err(SimError::Invalid("inner: register sizes differ".into()));
        }
        let p = self.inner_phasor(other);
        S::from_phasor(p).ok_or_else(|| SimError::NotRepresentable(format!("inner product {p}")))
    }

    fn project(&mut self, p: &Pauli, outcome: bool) -> Result<S> {
        let w = self.project_pauli(p, outcome)?;
        S::from_phasor(w).ok_or_else(|| SimError::NotRepresentable(format!("probability {w}")))
    }

    fn scale(&mut self, _c: &S) -> Result<()> {
        Err(unsupported(BACKEND, "scaling (a tableau holds no norm or phase)"))
    }

    fn measure(&mut self, p: &Pauli, rng: &mut dyn RngCore) -> Result<bool> {
        self.measure_pauli(p, rng)
    }

    fn tensor(&self, other: &Self) -> Result<Self> {
        Ok(Stabilizer::tensor(self, other))
    }

    /// Uniform on the affine support: one affine form, then `O(k n/64)` per shot.
    fn sample(&self, rng: &mut dyn RngCore, shots: usize) -> Result<Vec<Bits>> {
        Ok(self.affine_form().sample(rng, shots))
    }

    fn tensor_all(states: &[Self]) -> Result<Self> {
        if states.is_empty() {
            return Err(SimError::Invalid("tensor_all of no states".into()));
        }
        Ok(Stabilizer::tensor_all(states))
    }

    fn to_vector(&self) -> Result<Vec<S>> {
        self.affine_form().to_vector()
    }
}

impl fmt::Debug for Stabilizer {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Stabilizer({} qubits; stabilizers", self.n)?;
        for i in 0..self.n {
            write!(f, " {}", self.stabilizer(i).to_dense_string())?;
        }
        write!(f, ")")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::circuit::Circuit;
    use crate::scalar::C64;
    use crate::state::StateVector;
    use rand::rngs::StdRng;
    use rand::SeedableRng;

    fn random_clifford_circuit(n: usize, gates: usize, rng: &mut StdRng) -> Circuit {
        let mut c = Circuit::new(n);
        let one = [
            Gate::H,
            Gate::S,
            Gate::Sdg,
            Gate::X,
            Gate::Y,
            Gate::Z,
            Gate::SqrtX,
            Gate::SqrtY,
        ];
        let two = [Gate::Cx, Gate::Cz, Gate::Cy, Gate::Swap];
        for _ in 0..gates {
            if rng.gen::<f64>() < 0.5 || n < 2 {
                let g = one[rng.gen_range(0..one.len())].clone();
                let q = rng.gen_range(0..n);
                c.gate(g, &[q]);
            } else {
                let g = two[rng.gen_range(0..two.len())].clone();
                let a = rng.gen_range(0..n);
                let mut b = rng.gen_range(0..n - 1);
                if b >= a {
                    b += 1;
                }
                c.gate(g, &[a, b]);
            }
        }
        c
    }

    /// `v ≈ e^{iθ} w` for some `θ`.
    fn equal_up_to_phase(v: &[C64], w: &[C64]) -> bool {
        let i = (0..v.len())
            .max_by(|&a, &b| v[a].norm().partial_cmp(&v[b].norm()).unwrap())
            .unwrap();
        if v[i].norm() < 1e-9 {
            return w.iter().all(|c| c.norm() < 1e-9);
        }
        let r = w[i] / v[i];
        (r.norm() - 1.0).abs() < 1e-9 && v.iter().zip(w).all(|(a, b)| (a * r - b).norm() < 1e-9)
    }

    #[test]
    fn ghz_from_gates_and_from_generators() {
        let mut c = Circuit::new(3);
        c.h(0).cx(0, 1).cx(1, 2);
        let mut s = Stabilizer::zero_state(3);
        let mut rng = StdRng::seed_from_u64(7);
        QuantumState::<C64>::run(&mut s, &c, &mut rng).unwrap();
        let gens: Vec<Pauli> = ["X0*X1*X2", "Z0*Z1", "Z1*Z2"]
            .iter()
            .map(|t| Pauli::from_string(3, t).unwrap())
            .collect();
        let g = Stabilizer::from_stabilizers(&gens).unwrap();
        for p in &gens {
            assert!(s.is_stabilized_by(p));
            assert!(g.is_stabilized_by(p));
        }
        assert!(!s.is_stabilized_by(&Pauli::from_string(3, "Z0").unwrap()));
        assert_eq!(
            s.expectation_phasor(&Pauli::from_string(3, "Z0").unwrap()),
            Phasor::Zero
        );
        assert_eq!(
            s.expectation_phasor(&Pauli::from_string(3, "-X0*X1*X2").unwrap()),
            Phasor::sign(1)
        );
        assert_eq!(s.inner_phasor(&g).abs_sq_log2(), Some(0));
        let v: Vec<C64> = QuantumState::<C64>::to_vector(&s).unwrap();
        let r = std::f64::consts::FRAC_1_SQRT_2;
        assert!((v[0] - C64::new(r, 0.0)).norm() < 1e-12 && (v[7] - C64::new(r, 0.0)).norm() < 1e-12);
        assert!(v[1..7].iter().all(|c| c.norm() < 1e-12));
        // round trip through the affine form
        let af = s.affine_form();
        assert_eq!(af.dim(), 1);
        let back = Stabilizer::from_affine_form(&af).unwrap();
        assert_eq!(back.affine_form(), af);
        // measurements: Z0 random then Z1 determined and equal
        let mut ones = 0;
        for _ in 0..100 {
            let mut t = s.clone();
            let a = t
                .measure_pauli(&Pauli::from_string(3, "Z0").unwrap(), &mut rng)
                .unwrap();
            let b = t
                .measure_pauli(&Pauli::from_string(3, "Z1").unwrap(), &mut rng)
                .unwrap();
            assert_eq!(a, b);
            ones += a as usize;
            assert!(t.is_stabilized_by(&Pauli::from_string(3, if a { "-Z0" } else { "Z0" }).unwrap()));
        }
        assert!((25..=75).contains(&ones));
        let mut t = s.clone();
        assert_eq!(
            t.project_pauli(&Pauli::from_string(3, "X0*X1*X2").unwrap(), false)
                .unwrap(),
            Phasor::ONE
        );
        assert!(t
            .project_pauli(&Pauli::from_string(3, "X0*X1*X2").unwrap(), true)
            .is_err());
        assert_eq!(
            t.project_pauli(&Pauli::from_string(3, "X1").unwrap(), true)
                .unwrap(),
            Phasor::new(-2, 0, 0)
        );
    }

    #[test]
    fn forward_rows_from_the_inverse_tableau() {
        let mut rng = StdRng::seed_from_u64(77);
        for trial in 0..30 {
            let n = 1 + trial % 70;
            let c = random_clifford_circuit(n, 6 * n, &mut rng);
            let mut s = Stabilizer::zero_state(n);
            s.run(&c, &mut rng).unwrap();
            // a few random measurements as well
            for _ in 0..3 {
                let q = rng.gen_range(0..n);
                s.measure(&Pauli::x(n, q), &mut rng).unwrap();
            }
            let gens = s.stabilizers();
            for (i, g) in gens.iter().enumerate() {
                assert!(s.is_stabilized_by(g), "trial {trial}: generator {i}");
                let d = s.destabilizer(i);
                for (j, h) in gens.iter().enumerate() {
                    assert_eq!(d.commutes(h), i != j, "trial {trial}: destabilizer {i} vs {j}");
                }
            }
            // rebuilding from the generators gives the same state
            let t = Stabilizer::from_stabilizers(&gens).unwrap();
            for g in &gens {
                assert!(t.is_stabilized_by(g));
            }
            assert_eq!(t.affine_form(), s.affine_form());
        }
    }

    #[test]
    fn random_circuits_match_the_state_vector() {
        let mut rng = StdRng::seed_from_u64(2024);
        for trial in 0..20 {
            let n = 1 + trial % 5;
            let c = random_clifford_circuit(n, 30, &mut rng);
            let mut s = Stabilizer::zero_state(n);
            let mut v: StateVector<C64> = StateVector::zero_state(n);
            QuantumState::<C64>::run(&mut s, &c, &mut rng).unwrap();
            v.run(&c, &mut rng).unwrap();
            let sv: Vec<C64> = QuantumState::<C64>::to_vector(&s).unwrap();
            assert!(
                equal_up_to_phase(&sv, &v.to_vector().unwrap()),
                "trial {trial}: {c}"
            );
            // expectation values of random Paulis
            for _ in 0..10 {
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
                let a: C64 = QuantumState::<C64>::expectation(&s, &p).unwrap();
                let b = v.expectation(&p).unwrap();
                assert!((a - b).norm() < 1e-9, "trial {trial}: <{p}>");
            }
            // inner products between two random stabilizer states, up to phase in modulus
            let c2 = random_clifford_circuit(n, 30, &mut rng);
            let mut s2 = Stabilizer::zero_state(n);
            let mut v2: StateVector<C64> = StateVector::zero_state(n);
            QuantumState::<C64>::run(&mut s2, &c2, &mut rng).unwrap();
            v2.run(&c2, &mut rng).unwrap();
            let ip: C64 = QuantumState::<C64>::inner(&s, &s2).unwrap();
            let ipv = v.inner(&v2).unwrap();
            assert!(
                (ip.norm() - ipv.norm()).abs() < 1e-9,
                "trial {trial}: |<s|s2>| {} vs {}",
                ip.norm(),
                ipv.norm()
            );
            // the canonical representatives' inner product equals the vectors' inner product
            let sv2: Vec<C64> = QuantumState::<C64>::to_vector(&s2).unwrap();
            let direct: C64 = sv.iter().zip(&sv2).map(|(a, b)| a.conj() * b).sum();
            assert!((ip - direct).norm() < 1e-9, "trial {trial}: phase of <s|s2>");
            // measurement probabilities
            let q = rng.gen_range(0..n);
            let z = Pauli::z(n, q);
            let e = v.expectation(&z).unwrap().re;
            let mut t = s.clone();
            match t.project_pauli(&z, true) {
                Ok(w) => assert!((w.to_c64().re - (1.0 - e) / 2.0).abs() < 1e-9),
                Err(_) => assert!((e - 1.0).abs() < 1e-9),
            }
        }
    }
}
