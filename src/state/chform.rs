//! Stabilizer states with their global phase: the CH-form of Bravyi, Browne,
//! Calpin, Campbell, Gosset and Howard (Quantum 3, 181 (2019), §4.1),
//!
//! ```text
//! |ψ⟩ = ω · U_C · U_H · |s⟩,
//! ```
//!
//! with `ω` an exact [`Phasor`], `U_C` a *C-type* Clifford (a product of `S`,
//! `CZ` and `CX`, so `U_C|0⟩ = |0⟩` and no phase is hidden in it), `U_H` a
//! product of Hadamards on the qubits marked by `v`, and `s` a basis state.
//! `U_C` is stored by its inverse action on Paulis, `U_C† X_p U_C` and
//! `U_C† Z_p U_C` (a [`PauliTable`] of `2n` rows with exact phases).
//!
//! Costs: `S`, `CZ`, `CX`, `SWAP` and Paulis `O(n/64)` words (left
//! multiplication of `U_C` updates two rows); `H` and Pauli projections
//! `O(n²/64)` (a superposition of two basis states folded back into the form
//! by one right multiplication of `U_C` with a fan-out of `CX`/`CZ`);
//! amplitudes and expectation values `O(n²/64)`; inner products `O(n³/64)`
//! through the [`AffineForm`].  Unlike the tableau family the state is a
//! vector, not a ray: `⟨ψ|U|ψ⟩`, relative phases of branches and the global
//! phase after a circuit are exact.  States are unnormalized in the sense of
//! [`QuantumState`]: a projection multiplies `ω` by the branch amplitude.

use super::stabilizer::{AffineForm, Stabilizer};
use super::QuantumState;
use crate::bits::{self, Bits};
use crate::circuit::{Circuit, Op};
use crate::clifford::CliffordTableau;
use crate::error::{check_qubits, unsupported, Result, SimError};
use crate::gate::Gate;
use crate::pauli::{Pauli, PauliTable};
use crate::phasepoly::QuadForm;
use crate::scalar::{Amplitude, Phasor};
use rand::{Rng, RngCore};
use std::fmt;

const BACKEND: &str = "ChForm";

/// A stabilizer state with its global phase (see the [module documentation](self)).
#[derive(Clone, PartialEq, Eq)]
pub struct ChForm {
    n: usize,
    /// Rows `p`: `U_C† X_p U_C`; rows `n + p`: `U_C† Z_p U_C` (pure `Z` strings).
    inv: PauliTable,
    v: Bits,
    s: Bits,
    omega: Phasor,
}

/// `i^k X^x Z^z` pushed through `U_H` and applied to `|s⟩`: `(k′, s ⊕ x′)`.
fn through_h_on(p: &Pauli, v: &[u64], s: &[u64]) -> (u8, Bits) {
    let (x, z) = (p.x_bits(), p.z_bits());
    let mut xn = x.to_vec();
    let mut zn = z.to_vec();
    let mut yv = 0usize;
    for w in 0..xn.len() {
        let swap = (x[w] ^ z[w]) & v[w];
        xn[w] ^= swap;
        zn[w] ^= swap;
        yv += (x[w] & z[w] & v[w]).count_ones() as usize;
    }
    // H (X Z) H = Z X = −X Z on each Hadamard qubit carrying both letters
    let k = (p.phase() as usize + 2 * yv + 2 * bits::dot(&zn, s) as usize) & 3;
    let mut t = s.to_vec();
    bits::xor_into(&mut t, &xn);
    (k as u8, t)
}

impl ChForm {
    /// `|x⟩`.
    pub fn basis_state(n: usize, x: &[u64]) -> Self {
        let mut inv = PauliTable::new(n);
        for q in 0..n {
            inv.push(&Pauli::x(n, q));
        }
        for q in 0..n {
            inv.push(&Pauli::z(n, q));
        }
        let mut s = bits::zero(n);
        bits::xor_into(&mut s, x);
        bits::mask_to(&mut s, n);
        ChForm {
            n,
            inv,
            v: bits::zero(n),
            s,
            omega: Phasor::ONE,
        }
    }
    /// `|0…0⟩`.
    pub fn zero_state(n: usize) -> Self {
        Self::basis_state(n, &bits::zero(n))
    }
    /// `|+⟩^{⊗n}`.
    pub fn plus_state(n: usize) -> Self {
        let mut c = Self::zero_state(n);
        for q in 0..n {
            bits::set(&mut c.v, q, true);
        }
        c
    }
    /// The vector `af` (amplitude `2^{-k/2} i^{Q(u)}` at `shift ⊕ Bu`), exactly.
    pub fn from_affine_form(af: &AffineForm) -> Self {
        let n = af.n;
        let mut c = Self::zero_state(n);
        for &p in &af.pivots {
            bits::set(&mut c.v, p, true);
        }
        // Σ_u |Σ u_i e_{p_i}⟩ → Σ_u |Σ u_i b_i⟩: the basis is in reduced echelon
        // form, so every other bit of b_i is a non-pivot qubit still at 0
        for (i, b) in af.basis.iter().enumerate() {
            let p = af.pivots[i];
            for j in bits::ones(b) {
                if j != p && j < n {
                    c.left_cx(p, j);
                }
            }
        }
        // i^{Q(u)} with u_i the bit of pivot p_i
        let k = af.dim();
        for i in 0..k {
            let p = af.pivots[i];
            let l = af.form.linear(i);
            if l != 0 {
                c.left_s(p, l);
            }
            for j in i + 1..k {
                if af.form.has_quad(i, j) {
                    c.left_cz(p, af.pivots[j]);
                }
            }
        }
        c.apply_pauli(&Pauli::x_string(n, &af.shift));
        c.omega = c.omega.mul(Phasor::i_pow(af.form.constant() as i64));
        c
    }
    /// The canonical representative of a tableau state (`ψ(shift) > 0`, as
    /// [`Stabilizer::affine_form`]).
    pub fn from_stabilizer(st: &Stabilizer) -> Self {
        Self::from_affine_form(&st.affine_form())
    }
    /// Number of qubits.
    pub fn num_qubits(&self) -> usize {
        self.n
    }
    /// The global factor `ω`.
    pub fn omega(&self) -> Phasor {
        self.omega
    }
    /// Is the state zero?
    pub fn is_zero(&self) -> bool {
        self.omega.is_zero()
    }
    /// `|ω|²` as a phasor (a power of 2, or zero).
    pub fn norm_sq_phasor(&self) -> Phasor {
        match self.omega.abs_sq_log2() {
            None => Phasor::Zero,
            Some(p) => Phasor::sqrt2_pow(2 * p),
        }
    }

    // ---- left multiplication of U_C by C-type gates: O(n/64) ----

    /// `U_C ← S^e U_C` on qubit `q` (`e` mod 4: `S`, `Z`, `S†`).
    fn left_s(&mut self, q: usize, e: u8) {
        let n = self.n;
        match e & 3 {
            0 => {}
            2 => self.inv.add_phase(q, 2),
            e => {
                // S^{-e} X S^e = i^{−e} X Z  (e = 1: −Y, e = 3: Y)
                self.inv.mul_rows(q, n + q);
                self.inv.add_phase(q, 4 - e);
            }
        }
    }
    /// `U_C ← CX(c → t) U_C`.
    fn left_cx(&mut self, c: usize, t: usize) {
        let n = self.n;
        self.inv.mul_rows(c, t);
        self.inv.mul_rows(n + t, n + c);
    }
    /// `U_C ← CZ(a, b) U_C`.
    fn left_cz(&mut self, a: usize, b: usize) {
        let n = self.n;
        self.inv.mul_rows(a, n + b);
        self.inv.mul_rows(b, n + a);
    }
    /// `U_C ← SWAP(a, b) U_C`.
    fn left_swap(&mut self, a: usize, b: usize) {
        let n = self.n;
        self.inv.swap_rows(a, b);
        self.inv.swap_rows(n + a, n + b);
    }
    /// `U_C ← C U_C` for a C-type gate `C` on `qubits` given by `C† P C`
    /// for its local `X_j` (`xs`) and `Z_j` (`zs`).
    fn left_ctype(&mut self, xs: &[Pauli], zs: &[Pauli], qubits: &[usize]) {
        let n = self.n;
        let rows: Vec<Pauli> = xs
            .iter()
            .chain(zs)
            .map(|w| {
                let mut out = Pauli::identity(n);
                out.set_phase(w.phase());
                for j in bits::ones(w.x_bits()) {
                    self.inv.mul_pauli_by_row(&mut out, qubits[j]);
                }
                for j in bits::ones(w.z_bits()) {
                    self.inv.mul_pauli_by_row(&mut out, n + qubits[j]);
                }
                out
            })
            .collect();
        let k = qubits.len();
        for (j, &q) in qubits.iter().enumerate() {
            self.inv.set_row(q, &rows[j]);
            self.inv.set_row(n + q, &rows[k + j]);
        }
    }
    /// `U_C† P U_C`.
    fn pull_back(&self, p: &Pauli) -> Pauli {
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
    /// `P ψ` for a Pauli `P` (any phase).
    pub fn apply_pauli(&mut self, p: &Pauli) {
        assert_eq!(p.num_qubits(), self.n, "apply_pauli: register size");
        if bits::is_zero(p.x_bits()) && bits::is_zero(p.z_bits()) {
            self.omega = self.omega.mul(Phasor::i_pow(p.phase() as i64));
            return;
        }
        let q = self.pull_back(p);
        let (k, t) = through_h_on(&q, &self.v, &self.s);
        self.s = t;
        self.omega = self.omega.mul(Phasor::i_pow(k as i64));
    }

    // ---- right multiplication of U_C: O(n) rows, word-parallel ----

    /// `U_C ← U_C S_q^e`: every row `R ← S^{-e} R S^e`.
    fn right_s(&mut self, q: usize, e: u8) {
        let e = e & 3;
        if e == 0 {
            return;
        }
        // S^{-e} X S^e = i^{-e} X Z for odd e, −X for e = 2; Z is fixed
        let add = if e == 2 { 2 } else { 4 - e };
        let (wd, m) = (q / 64, 1u64 << (q % 64));
        self.inv.for_each_row_mut(|_, xr, zr, ph| {
            if xr[wd] & m != 0 {
                if e != 2 {
                    zr[wd] ^= m;
                }
                *ph = (*ph + add) & 3;
            }
        });
    }
    /// `U_C ← U_C W` with `W = Π_{j∈a} CX(q → j) Π_{k∈b} CZ(q, k)` (`q ∉ a, b`).
    fn right_fanout_a(&mut self, q: usize, a: &[u64], b: &[u64]) {
        let (wd, m) = (q / 64, 1u64 << (q % 64));
        self.inv.for_each_row_mut(|_, xr, zr, ph| {
            let xq = xr[wd] & m != 0;
            let xb = bits::dot(xr, b);
            let za = bits::dot(zr, a);
            if xq {
                bits::xor_into(xr, a);
                bits::xor_into(zr, b);
                if xb {
                    *ph = (*ph + 2) & 3;
                }
            }
            if xb ^ za {
                zr[wd] ^= m;
            }
        });
    }
    /// `U_C ← U_C W` with `W = Π_{j∈b} CX(j → q)` (`q ∉ b`).
    fn right_fanout_b(&mut self, q: usize, b: &[u64]) {
        let (wd, m) = (q / 64, 1u64 << (q % 64));
        self.inv.for_each_row_mut(|_, xr, zr, _| {
            if bits::dot(xr, b) {
                xr[wd] ^= m;
            }
            if zr[wd] & m != 0 {
                bits::xor_into(zr, b);
            }
        });
    }

    /// `ψ ← ω U_C U_H (|t⟩ + i^δ |u⟩)` (the current `s` is discarded): the
    /// superposition folded back into CH-form.
    fn superpose(&mut self, t: Bits, u: Bits, delta: u8) {
        let n = self.n;
        let delta = delta & 3;
        let mut y = t.clone();
        bits::xor_into(&mut y, &u);
        if bits::is_zero(&y) {
            self.s = t;
            self.omega = self.omega.mul(match delta {
                0 => Phasor::sqrt2_pow(2),
                1 => Phasor::new(1, 1, 3),
                2 => Phasor::Zero,
                _ => Phasor::new(1, -1, 3),
            });
            return;
        }
        let mut a = y.clone();
        let mut b = y.clone();
        for w in 0..y.len() {
            a[w] &= !self.v[w];
            b[w] &= self.v[w];
        }
        let tq = |q: usize| bits::get(&t, q);
        let fold = |q: usize| -> (bool, u8, Bits) {
            // |a⟩ + i^δ|a ⊕ 1⟩ on qubit q = i^{aδ} (|0⟩ + i^{δ'}|1⟩), δ' = (−1)^a δ
            let aq = tq(q);
            let dp = if aq { (4 - delta) & 3 } else { delta };
            let mut s2 = t.clone();
            if aq {
                let mut d = y.clone();
                bits::toggle(&mut d, q);
                bits::xor_into(&mut s2, &d);
            }
            bits::set(&mut s2, q, false);
            (aq, dp, s2)
        };
        if let Some(q) = bits::lowest_one(&a) {
            let (aq, dp, s2) = fold(q);
            let mut a2 = a.clone();
            bits::toggle(&mut a2, q);
            self.right_fanout_a(q, &a2, &b);
            self.right_s(q, dp);
            bits::set(&mut self.v, q, true);
            self.s = s2;
            let c = if aq { delta } else { 0 };
            self.omega = self.omega.mul(Phasor::new(1, c as i64, 2));
        } else {
            let q = bits::lowest_one(&b).expect("y ≠ 0");
            let (aq, dp, mut s2) = fold(q);
            let mut b2 = b.clone();
            bits::toggle(&mut b2, q);
            self.right_fanout_b(q, &b2);
            // V S^{δ'} H_q with H_q in V: H S^{δ'} H
            let mut extra = Phasor::ONE;
            match dp {
                0 => bits::set(&mut self.v, q, false),
                2 => {
                    bits::set(&mut self.v, q, false);
                    bits::set(&mut s2, q, true);
                }
                1 => {
                    // H S H = ζ_8 S† H S†
                    self.right_s(q, 3);
                    extra = Phasor::new(0, 1, 3);
                }
                _ => {
                    // H S† H = ζ_8^{-1} S H S
                    self.right_s(q, 1);
                    extra = Phasor::new(0, -1, 3);
                }
            }
            self.s = s2;
            let c = if aq { delta } else { 0 };
            self.omega = self.omega.mul(Phasor::new(1, c as i64, 2)).mul(extra);
        }
        debug_assert_eq!(self.s.len(), bits::nwords(n));
    }

    /// `H` on qubit `q`.
    pub fn apply_h(&mut self, q: usize) {
        let n = self.n;
        if self.omega.is_zero() {
            return;
        }
        // H_q = (X_q + Z_q)/√2 pulled back through U_C and U_H onto |s⟩
        let (kx, t) = through_h_on(&self.inv.row(q), &self.v, &self.s);
        let (kz, u) = through_h_on(&self.inv.row(n + q), &self.v, &self.s);
        self.omega = self.omega.mul(Phasor::new(-1, kx as i64, 2));
        self.superpose(t, u, (kz + 4 - kx) & 3);
    }

    /// `(1 + (−1)^outcome P)/2 · ψ`: returns the new `⟨ψ|ψ⟩`.
    pub fn project_pauli(&mut self, p: &Pauli, outcome: bool) -> Result<Phasor> {
        if !p.is_hermitian() {
            return Err(SimError::Invalid(format!("projector of non-Hermitian {p}")));
        }
        assert_eq!(p.num_qubits(), self.n, "project: register size");
        let q = self.pull_back(p);
        let (k, t) = through_h_on(&q, &self.v, &self.s);
        let s = self.s.clone();
        self.omega = self.omega.mul(Phasor::sqrt2_pow(-2));
        self.superpose(s, t, k + 2 * outcome as u8);
        Ok(self.norm_sq_phasor())
    }

    /// Measure the Hermitian Pauli `p`: the outcome is drawn with the Born
    /// probabilities and the state becomes `Πψ` (as [`QuantumState::project`];
    /// a random outcome halves `⟨ψ|ψ⟩`).
    pub fn measure_pauli(&mut self, p: &Pauli, rng: &mut dyn RngCore) -> Result<bool> {
        if !p.is_hermitian() {
            return Err(SimError::Invalid(format!(
                "measured operator {p} is not Hermitian"
            )));
        }
        let q = self.pull_back(p);
        let (k, t) = through_h_on(&q, &self.v, &self.s);
        if t == self.s {
            // P ψ = i^k ψ, k ∈ {0, 2}
            return Ok(k == 2);
        }
        let outcome = rng.gen::<bool>();
        let s = self.s.clone();
        self.omega = self.omega.mul(Phasor::sqrt2_pow(-2));
        self.superpose(s, t, k + 2 * outcome as u8);
        Ok(outcome)
    }

    /// `⟨ψ|P|ψ⟩` exactly.
    pub fn expectation_phasor(&self, p: &Pauli) -> Phasor {
        let q = self.pull_back(p);
        let (k, t) = through_h_on(&q, &self.v, &self.s);
        if t != self.s {
            return Phasor::Zero;
        }
        self.norm_sq_phasor().mul(Phasor::i_pow(k as i64))
    }

    /// `⟨x|ψ⟩` exactly.
    pub fn amplitude_phasor(&self, x: &[u64]) -> Phasor {
        if self.omega.is_zero() {
            return Phasor::Zero;
        }
        // U_C†|x⟩ = (Π_p U_C† X_p U_C)^{x_p} |0⟩ = i^φ |w⟩
        let mut prod = Pauli::identity(self.n);
        for p in bits::ones(x) {
            if p < self.n {
                self.inv.mul_pauli_by_row(&mut prod, p);
            }
        }
        let w = prod.x_bits();
        // ⟨w|U_H|s⟩: w = s off V, (−1)^{w·s} 2^{-1/2} per Hadamard qubit
        for i in 0..w.len() {
            if (w[i] ^ self.s[i]) & !self.v[i] != 0 {
                return Phasor::Zero;
            }
        }
        let h = bits::popcount(&self.v) as i32;
        let mut ws = bits::zero(self.n);
        for i in 0..w.len() {
            ws[i] = w[i] & self.s[i] & self.v[i];
        }
        let sign = bits::popcount(&ws) as i64;
        self.omega
            .mul(Phasor::i_pow(-(prod.phase() as i64) + 2 * sign))
            .mul(Phasor::sqrt2_pow(-h))
    }

    /// `ψ = c · af` with `af` the canonical affine form; `c = 0` for the zero state.
    pub fn to_affine_form(&self) -> (Phasor, AffineForm) {
        let n = self.n;
        // U_C|y⟩ = i^{−φ(G y)} |G y⟩ with φ(x) = Σ γ_p x_p + 2 Σ_{p<p'} (M_p·F_p') x_p x_p'
        let mut phi = QuadForm::new(n);
        // J_pp' = M_p · F_p' for p < p', rows in parallel
        let upper: Vec<Bits> = {
            use rayon::prelude::*;
            (0..n)
                .into_par_iter()
                .map(|p| {
                    let mut r = bits::zero(n);
                    for p2 in p + 1..n {
                        if bits::dot(self.inv.z_row(p), self.inv.x_row(p2)) {
                            bits::toggle(&mut r, p2);
                        }
                    }
                    r
                })
                .collect()
        };
        for p in 0..n {
            phi.add_linear(p, self.inv.phase(p) as i64);
            for p2 in bits::ones(&upper[p]) {
                phi.add_quad(p, p2);
            }
        }
        // the support: x = G s0 ⊕ Σ_{j∈V} u_j G e_j, G_p = Z part of row n + p
        let vs = bits::ones(&self.v);
        let mut s0 = self.s.clone();
        for i in 0..s0.len() {
            s0[i] &= !self.v[i];
        }
        let g_mul = |y: &[u64]| -> Bits {
            let mut out = bits::zero(n);
            for p in 0..n {
                if bits::dot(self.inv.z_row(n + p), y) {
                    bits::toggle(&mut out, p);
                }
            }
            out
        };
        let shift = g_mul(&s0);
        let cols: Vec<Bits> = vs.iter().map(|&j| g_mul(&bits::unit(n, j))).collect();
        let h = vs.len();
        let mut form = phi.substitute(&shift, &cols, h).negated();
        for (i, &j) in vs.iter().enumerate() {
            if bits::get(&self.s, j) {
                form.add_linear(i, 2);
            }
        }
        let c0 = form.constant();
        form.add_constant(-(c0 as i64));
        let af = AffineForm::from_basis(n, shift, cols, form).expect("G is invertible");
        // from_basis may move a constant into the form when it changes coordinates
        let c1 = af.form.constant();
        let mut af = af;
        af.form.add_constant(-(c1 as i64));
        let scalar = self.omega.mul(Phasor::i_pow(c0 as i64 + c1 as i64));
        (scalar, af)
    }

    /// The ray (forgetting `ω`).
    pub fn to_stabilizer(&self) -> Result<Stabilizer> {
        Stabilizer::from_affine_form(&self.to_affine_form().1)
    }

    /// `⟨self|other⟩` exactly.
    pub fn inner_phasor(&self, other: &ChForm) -> Phasor {
        assert_eq!(self.n, other.n, "inner: register size");
        if self.omega.is_zero() || other.omega.is_zero() {
            return Phasor::Zero;
        }
        let (ca, a) = self.to_affine_form();
        let (cb, b) = other.to_affine_form();
        ca.conj().mul(cb).mul(a.inner(&b))
    }

    /// `self ⊗ other` (`self` on the low qubits).
    pub fn tensor(&self, other: &ChForm) -> ChForm {
        let (na, nb) = (self.n, other.n);
        let (ia, ib) = (Pauli::identity(na), Pauli::identity(nb));
        let mut inv = PauliTable::new(na + nb);
        for r in 0..na {
            inv.push(&self.inv.row(r).tensor(&ib));
        }
        for r in 0..nb {
            inv.push(&ia.tensor(&other.inv.row(r)));
        }
        for r in 0..na {
            inv.push(&self.inv.row(na + r).tensor(&ib));
        }
        for r in 0..nb {
            inv.push(&ia.tensor(&other.inv.row(nb + r)));
        }
        ChForm {
            n: na + nb,
            inv,
            v: bits::concat(&self.v, na, &other.v, nb),
            s: bits::concat(&self.s, na, &other.s, nb),
            omega: self.omega.mul(other.omega),
        }
    }

    /// `ψ ← c ψ` for an exact phasor.
    pub fn scale_phasor(&mut self, c: Phasor) {
        self.omega = self.omega.mul(c);
    }

    /// `⟨ψ|U|ψ⟩ / ⟨ψ|ψ⟩` for a unitary Clifford circuit `U` (no measurements).
    pub fn circuit_expectation(&self, circuit: &Circuit) -> Result<Phasor> {
        let mut u = self.clone();
        u.apply_unitary_circuit(circuit)?;
        let num = self.inner_phasor(&u);
        let nn = self.norm_sq_phasor();
        if nn.is_zero() {
            return Err(SimError::ZeroState("circuit_expectation".into()));
        }
        Ok(num.mul(nn.inverse().expect("nonzero")))
    }

    /// Apply the gates of a circuit (an error for measurements and resets).
    pub fn apply_unitary_circuit(&mut self, circuit: &Circuit) -> Result<()> {
        if circuit.num_qubits() != self.n {
            return Err(SimError::Invalid(format!(
                "circuit on {} qubits, state on {}",
                circuit.num_qubits(),
                self.n
            )));
        }
        for op in circuit.ops() {
            match op {
                Op::Gate { gate, qubits } => self.apply(gate, qubits)?,
                Op::Tick => {}
                _ => {
                    return Err(SimError::Invalid(
                        "apply_unitary_circuit: the circuit measures".into(),
                    ))
                }
            }
        }
        Ok(())
    }

    /// Apply a Clifford gate (exact phase).  Named gates have their usual
    /// matrices; a monomial Clifford is split as `s · X^b · C` with `C`
    /// C-type; gates given only by a tableau take the phase of
    /// [`CliffordTableau::unitary`], as in the dense backend.
    pub fn apply(&mut self, gate: &Gate, qubits: &[usize]) -> Result<()> {
        check_qubits(qubits, self.n)?;
        if qubits.len() != gate.arity() {
            return Err(SimError::Arity {
                gate: gate.name(),
                expected: gate.arity(),
                got: qubits.len(),
            });
        }
        let n = self.n;
        match gate {
            Gate::I => Ok(()),
            Gate::H => {
                self.apply_h(qubits[0]);
                Ok(())
            }
            Gate::S => {
                self.left_s(qubits[0], 1);
                Ok(())
            }
            Gate::Sdg => {
                self.left_s(qubits[0], 3);
                Ok(())
            }
            Gate::Z => {
                self.left_s(qubits[0], 2);
                Ok(())
            }
            Gate::X => {
                self.apply_pauli(&Pauli::x(n, qubits[0]));
                Ok(())
            }
            Gate::Y => {
                self.apply_pauli(&Pauli::y(n, qubits[0]));
                Ok(())
            }
            Gate::Cx => {
                self.left_cx(qubits[0], qubits[1]);
                Ok(())
            }
            Gate::Cz => {
                self.left_cz(qubits[0], qubits[1]);
                Ok(())
            }
            Gate::Swap => {
                self.left_swap(qubits[0], qubits[1]);
                Ok(())
            }
            Gate::Pauli(p) => {
                self.apply_pauli(&p.embed(n, qubits));
                Ok(())
            }
            Gate::Clifford(t) => self.apply_tableau_gate(t, qubits),
            _ => {
                if let Some(m) = gate.monomial() {
                    if m.to_tableau().is_ok() {
                        return self.apply_monomial_clifford(&m, qubits);
                    }
                    return Err(unsupported(BACKEND, format!("non-Clifford gate {gate}")));
                }
                if let Some(c) = gate.decomposition() {
                    for op in c.ops() {
                        if let Op::Gate { gate: g, qubits: qs } = op {
                            let mapped: Vec<usize> = qs.iter().map(|&q| qubits[q]).collect();
                            self.apply(g, &mapped)?;
                        }
                    }
                    return Ok(());
                }
                if let Some(t) = gate.tableau() {
                    return self.apply_tableau_gate(&t, qubits);
                }
                Err(unsupported(BACKEND, format!("non-Clifford gate {gate}")))
            }
        }
    }

    fn apply_monomial_clifford(&mut self, m: &crate::gate::MonomialGate, qubits: &[usize]) -> Result<()> {
        let k = m.num_qubits();
        let (s0, b) = m.apply_to_basis(&bits::zero(k));
        // C = s0^{-1} X^b U is C-type; C P C† = X^b (U P U†) X^b
        let tu = m.to_tableau()?;
        let xb = Pauli::x_string(k, &b);
        let flip = |p: Pauli| if p.commutes(&xb) { p } else { p.negated() };
        let tc = CliffordTableau::from_images(
            (0..k).map(|j| flip(tu.x_image(j))).collect(),
            (0..k).map(|j| flip(tu.z_image(j))).collect(),
        )?;
        let ci = tc.inverse();
        let xs: Vec<Pauli> = (0..k).map(|j| ci.x_image(j)).collect();
        let zs: Vec<Pauli> = (0..k).map(|j| ci.z_image(j)).collect();
        self.left_ctype(&xs, &zs, qubits);
        if !bits::is_zero(&b) {
            self.apply_pauli(&xb.embed(self.n, qubits));
        }
        self.omega = self.omega.mul(s0);
        Ok(())
    }

    fn apply_tableau_gate(&mut self, t: &CliffordTableau, qubits: &[usize]) -> Result<()> {
        let k = t.num_qubits();
        let circ = t.to_circuit();
        // the phase convention of CliffordTableau::unitary: the first nonzero
        // entry of the first column is positive
        let mut col = ChForm::zero_state(k);
        col.apply_unitary_circuit(&circ)?;
        let first = (0..1u64 << k)
            .map(|x| col.amplitude_phasor(&bits::from_u64(x, k)))
            .find(|a| !a.is_zero())
            .expect("a unitary column is nonzero");
        let (ph, b) = first.argument();
        for op in circ.ops() {
            if let Op::Gate { gate: g, qubits: qs } = op {
                let mapped: Vec<usize> = qs.iter().map(|&q| qubits[q]).collect();
                self.apply(g, &mapped)?;
            }
        }
        self.omega = self.omega.mul(Phasor::root_of_unity(-(ph as i64), b));
        Ok(())
    }
}

/// The phasor nearest to a float scalar, if it is one (`√2^p ζ_8^k` within `1e-12`).
fn phasor_of(c: crate::scalar::C64) -> Option<Phasor> {
    let r = c.norm();
    if r == 0.0 {
        return Some(Phasor::Zero);
    }
    let p = (2.0 * r.log2()).round() as i32;
    let k = (c.arg() / (std::f64::consts::PI / 4.0)).round() as i64;
    let cand = Phasor::new(p, k, 3);
    ((cand.to_c64() - c).norm() <= 1e-12 * r.max(1.0)).then_some(cand)
}

impl<S: Amplitude> QuantumState<S> for ChForm {
    const BACKEND: &'static str = BACKEND;

    fn num_qubits(&self) -> usize {
        self.n
    }

    fn apply_gate(&mut self, gate: &Gate, qubits: &[usize]) -> Result<()> {
        self.apply(gate, qubits)
    }

    fn norm_sq(&self) -> Result<S> {
        let p = self.norm_sq_phasor();
        S::from_phasor(p).ok_or_else(|| SimError::NotRepresentable(format!("norm {p}")))
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
        let e = self.expectation_phasor(p);
        S::from_phasor(e).ok_or_else(|| SimError::NotRepresentable(format!("expectation {e}")))
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
        S::from_phasor(w).ok_or_else(|| SimError::NotRepresentable(format!("weight {w}")))
    }

    fn scale(&mut self, c: &S) -> Result<()> {
        let p = phasor_of(c.to_c64())
            .ok_or_else(|| unsupported(BACKEND, format!("scaling by {c} (not √2^p ζ_8^k)")))?;
        self.scale_phasor(p);
        Ok(())
    }

    fn measure(&mut self, p: &Pauli, rng: &mut dyn RngCore) -> Result<bool> {
        self.measure_pauli(p, rng)
    }

    fn tensor(&self, other: &Self) -> Result<Self> {
        Ok(ChForm::tensor(self, other))
    }

    /// Uniform on the affine support (see [`AffineForm::sample`]).
    fn sample(&self, rng: &mut dyn RngCore, shots: usize) -> Result<Vec<Bits>> {
        if self.omega.is_zero() {
            return Err(SimError::ZeroState("sample".into()));
        }
        Ok(self.to_affine_form().1.sample(rng, shots))
    }
}

/// A CH-form with a coefficient in `S`: `c · ω U_C U_H |s⟩`, the term type of
/// a stabilizer-rank sum on large registers
/// ([`StabilizerSum`](super::StabilizerSum)`<S, ChTerm<S>>`), where
/// coefficients such as `ζ_8 − 1` are not phasors.
#[derive(Clone, Debug, PartialEq)]
pub struct ChTerm<S: Amplitude> {
    coeff: S,
    state: ChForm,
}

impl<S: Amplitude> ChTerm<S> {
    /// `c · ψ`.
    pub fn new(coeff: S, state: ChForm) -> Self {
        ChTerm { coeff, state }
    }
    /// The coefficient `c`.
    pub fn coefficient(&self) -> &S {
        &self.coeff
    }
    /// The CH-form `ψ`.
    pub fn state(&self) -> &ChForm {
        &self.state
    }
    fn phasor(p: Phasor) -> Result<S> {
        S::from_phasor(p).ok_or_else(|| SimError::NotRepresentable(format!("{p}")))
    }
}

impl<S: Amplitude> QuantumState<S> for ChTerm<S> {
    const BACKEND: &'static str = "ChTerm";

    fn num_qubits(&self) -> usize {
        self.state.n
    }
    fn apply_gate(&mut self, gate: &Gate, qubits: &[usize]) -> Result<()> {
        self.state.apply(gate, qubits)
    }
    fn norm_sq(&self) -> Result<S> {
        Ok(self
            .coeff
            .abs_sq()
            .mul_ref(&Self::phasor(self.state.norm_sq_phasor())?))
    }
    fn amplitude(&self, x: &[u64]) -> Result<S> {
        Ok(self.coeff.mul_ref(&Self::phasor(self.state.amplitude_phasor(x))?))
    }
    fn expectation(&self, p: &Pauli) -> Result<S> {
        Ok(self
            .coeff
            .abs_sq()
            .mul_ref(&Self::phasor(self.state.expectation_phasor(p))?))
    }
    fn inner(&self, other: &Self) -> Result<S> {
        let ip = Self::phasor(self.state.inner_phasor(&other.state))?;
        Ok(self.coeff.conj().mul_ref(&other.coeff).mul_ref(&ip))
    }
    fn project(&mut self, p: &Pauli, outcome: bool) -> Result<S> {
        let w = self.state.project_pauli(p, outcome)?;
        Ok(self.coeff.abs_sq().mul_ref(&Self::phasor(w)?))
    }
    fn scale(&mut self, c: &S) -> Result<()> {
        self.coeff = self.coeff.mul_ref(c);
        Ok(())
    }
    fn tensor(&self, other: &Self) -> Result<Self> {
        Ok(ChTerm {
            coeff: self.coeff.mul_ref(&other.coeff),
            state: self.state.tensor(&other.state),
        })
    }
}

impl<S: Amplitude> super::stabsum::StabilizerTerm<S> for ChTerm<S> {
    fn basis_term(n: usize, x: &[u64]) -> Self {
        ChTerm {
            coeff: S::one(),
            state: ChForm::basis_state(n, x),
        }
    }
    fn from_canonical(af: &AffineForm, c: S) -> Result<Self> {
        Ok(ChTerm {
            coeff: c,
            state: ChForm::from_affine_form(af),
        })
    }
    fn canonical(&self) -> Result<(AffineForm, S)> {
        let (w, af) = self.state.to_affine_form();
        Ok((af, self.coeff.mul_ref(&Self::phasor(w)?)))
    }
    fn matrix_element(&self, p: &Pauli, other: &Self) -> Result<S> {
        let mut b = other.state.clone();
        b.apply_pauli(p);
        let ip = Self::phasor(self.state.inner_phasor(&b))?;
        Ok(self.coeff.conj().mul_ref(&other.coeff).mul_ref(&ip))
    }
    /// `(c, af)` with the term `= c ·` the vector of `af`.
    type Prepared = (S, AffineForm);
    fn prepare(&self) -> Result<(S, AffineForm)> {
        let (af, c) = self.canonical()?;
        Ok((c, af))
    }
    fn prepared_element(a: &(S, AffineForm), p: Option<&Pauli>, b: &(S, AffineForm)) -> Result<S> {
        let (w, ip) = match p {
            Some(p) => {
                let (w, pb) = b.1.apply_pauli(p);
                (w, a.1.inner(&pb))
            }
            None => (Phasor::ONE, a.1.inner(&b.1)),
        };
        let v = Self::phasor(w.mul(ip))?;
        Ok(a.0.conj().mul_ref(&b.0).mul_ref(&v))
    }
    fn is_stabilizer_term(&self) -> bool {
        true
    }
    fn is_zero_term(&self) -> bool {
        self.state.is_zero() || self.coeff.is_exactly_zero()
    }
}

impl fmt::Debug for ChForm {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "ChForm({} qubits; ω = {}, v = {}, s = {})",
            self.n,
            self.omega,
            bits::to_string(&self.v, self.n),
            bits::to_string(&self.s, self.n)
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::clifford::{random_clifford, random_stabilizer_state};
    use crate::pauli::PauliKind;
    use crate::scalar::C64;
    use crate::state::StateVector;
    use rand::rngs::StdRng;
    use rand::SeedableRng;

    fn random_pauli_on(n: usize, rng: &mut StdRng) -> Pauli {
        let kinds: Vec<PauliKind> = (0..n)
            .map(|_| [PauliKind::I, PauliKind::X, PauliKind::Y, PauliKind::Z][rng.gen_range(0..4)])
            .collect();
        Pauli::from_kinds(&kinds)
    }

    fn random_gate(n: usize, rng: &mut StdRng) -> (Gate, Vec<usize>) {
        let one = [
            Gate::H,
            Gate::S,
            Gate::Sdg,
            Gate::X,
            Gate::Y,
            Gate::Z,
            Gate::SqrtX,
            Gate::SqrtXdg,
            Gate::SqrtY,
            Gate::SqrtYdg,
            Gate::I,
        ];
        let two = [Gate::Cx, Gate::Cz, Gate::Cy, Gate::Swap];
        let r = rng.gen_range(0..20);
        if r == 0 && n >= 2 {
            let t = random_clifford(2, rng);
            let a = rng.gen_range(0..n);
            let b = (a + 1 + rng.gen_range(0..n - 1)) % n;
            return (Gate::Clifford(t), vec![a, b]);
        }
        if r == 1 {
            let k = 2.min(n);
            let qs: Vec<usize> = (0..k).collect();
            return (Gate::Pauli(random_pauli_on(k, rng)), qs);
        }
        if r < 11 || n < 2 {
            (
                one[rng.gen_range(0..one.len())].clone(),
                vec![rng.gen_range(0..n)],
            )
        } else {
            let a = rng.gen_range(0..n);
            let b = (a + 1 + rng.gen_range(0..n - 1)) % n;
            (two[rng.gen_range(0..two.len())].clone(), vec![a, b])
        }
    }

    fn close(a: &[C64], b: &[C64]) -> bool {
        a.iter().zip(b).all(|(x, y)| (x - y).norm() < 1e-9)
    }

    #[test]
    fn exact_phases_match_the_state_vector() {
        let mut rng = StdRng::seed_from_u64(5);
        for trial in 0..200 {
            let n = 1 + trial % 6;
            let mut c = ChForm::zero_state(n);
            let mut v: StateVector<C64> = StateVector::zero_state(n);
            for step in 0..40 {
                if step % 9 == 8 {
                    let p = random_pauli_on(n, &mut rng);
                    let b = rng.gen::<bool>();
                    let wv = v.project(&p, b).unwrap();
                    let wc: C64 = QuantumState::<C64>::project(&mut c, &p, b).unwrap();
                    assert!((wv - wc).norm() < 1e-9, "trial {trial} step {step}: weight");
                    if wv.norm() < 1e-12 {
                        break;
                    }
                    continue;
                }
                let (g, qs) = random_gate(n, &mut rng);
                v.apply_gate(&g, &qs).unwrap();
                c.apply(&g, &qs).unwrap();
                let cv: Vec<C64> = QuantumState::<C64>::to_vector(&c).unwrap();
                assert!(
                    close(&cv, &v.to_vector().unwrap()),
                    "trial {trial} step {step}: {g} on {qs:?}"
                );
            }
            let vv = v.to_vector().unwrap();
            for _ in 0..5 {
                let p = random_pauli_on(n, &mut rng);
                let a: C64 = QuantumState::<C64>::expectation(&c, &p).unwrap();
                assert!(
                    (a - v.expectation(&p).unwrap()).norm() < 1e-9,
                    "trial {trial}: <{p}>"
                );
            }
            let (sc, af) = c.to_affine_form();
            if !sc.is_zero() {
                let afv: Vec<C64> = af.to_vector().unwrap();
                let scaled: Vec<C64> = afv.iter().map(|a| a * sc.to_c64()).collect();
                assert!(close(&scaled, &vv), "trial {trial}: affine form");
            }
            let mut d = ChForm::zero_state(n);
            let mut w: StateVector<C64> = StateVector::zero_state(n);
            for _ in 0..30 {
                let (g, qs) = random_gate(n, &mut rng);
                w.apply_gate(&g, &qs).unwrap();
                d.apply(&g, &qs).unwrap();
            }
            let ip: C64 = QuantumState::<C64>::inner(&c, &d).unwrap();
            assert!((ip - v.inner(&w).unwrap()).norm() < 1e-9, "trial {trial}: inner");
        }
    }

    #[test]
    fn measurement_statistics_and_post_states() {
        let mut rng = StdRng::seed_from_u64(6);
        for trial in 0..60 {
            let n = 2 + trial % 4;
            let mut c = ChForm::zero_state(n);
            for _ in 0..25 {
                let (g, qs) = random_gate(n, &mut rng);
                c.apply(&g, &qs).unwrap();
            }
            let p = random_pauli_on(n, &mut rng);
            let e = c.expectation_phasor(&p).to_c64().re;
            let mut d = c.clone();
            let b = d.measure_pauli(&p, &mut rng).unwrap();
            if e.abs() > 1.0 - 1e-9 {
                assert_eq!(b, e < 0.0);
                assert_eq!(d, c);
            } else {
                let mut v: StateVector<C64> =
                    StateVector::from_amplitudes(n, QuantumState::<C64>::to_vector(&c).unwrap()).unwrap();
                v.project(&p, b).unwrap();
                let dv: Vec<C64> = QuantumState::<C64>::to_vector(&d).unwrap();
                assert!(close(&dv, &v.to_vector().unwrap()), "trial {trial}");
            }
        }
    }

    #[test]
    fn from_stabilizer_is_the_canonical_representative_and_eigenvalues_are_exact() {
        let mut rng = StdRng::seed_from_u64(9);
        for trial in 0..40 {
            let n = 1 + trial % 7;
            let st = random_stabilizer_state(n, &mut rng);
            let c = ChForm::from_stabilizer(&st);
            let a: Vec<C64> = QuantumState::<C64>::to_vector(&c).unwrap();
            let b: Vec<C64> = QuantumState::<C64>::to_vector(&st).unwrap();
            assert!(close(&a, &b), "trial {trial}");
            assert_eq!(c.to_stabilizer().unwrap().affine_form(), st.affine_form());
            let g = st.stabilizer(rng.gen_range(0..n));
            let mut circ = Circuit::new(n);
            circ.gate(Gate::Pauli(g.clone()), &(0..n).collect::<Vec<_>>());
            assert_eq!(c.circuit_expectation(&circ).unwrap(), Phasor::ONE);
        }
    }

    #[test]
    fn tensor_matches_kron() {
        let mut rng = StdRng::seed_from_u64(3);
        for _ in 0..20 {
            let (na, nb) = (rng.gen_range(1..4), rng.gen_range(1..4));
            let mut a = ChForm::zero_state(na);
            let mut b = ChForm::zero_state(nb);
            for _ in 0..15 {
                let (g, qs) = random_gate(na, &mut rng);
                a.apply(&g, &qs).unwrap();
                let (g, qs) = random_gate(nb, &mut rng);
                b.apply(&g, &qs).unwrap();
            }
            let va: StateVector<C64> =
                StateVector::from_amplitudes(na, QuantumState::<C64>::to_vector(&a).unwrap()).unwrap();
            let vb: StateVector<C64> =
                StateVector::from_amplitudes(nb, QuantumState::<C64>::to_vector(&b).unwrap()).unwrap();
            let t: Vec<C64> = QuantumState::<C64>::to_vector(&a.tensor(&b)).unwrap();
            assert!(close(&t, &va.kron(&vb).unwrap().to_vector().unwrap()));
        }
    }
}

#[cfg(test)]
mod clifford_eigenvalue_tests {
    use super::*;
    use crate::clifford::random_stabilizer_state;
    use rand::rngs::StdRng;
    use rand::SeedableRng;

    /// Eigenvalues of non-Pauli Cliffords and of phased Paulis on stabilizer
    /// states, read exactly from a ray.
    #[test]
    fn eigenvalues_of_non_pauli_cliffords_on_rays() {
        // ⟨ψ|U|ψ⟩ against the dense vectors for U = (X S)^{⊗2} and random Clifford circuits
        use crate::scalar::C64;
        use crate::state::StateVector;
        let mut rng = StdRng::seed_from_u64(1);
        let mut xs2 = Circuit::new(2);
        xs2.s(0).x(0).s(1).x(1);
        let mut eigen = 0;
        for trial in 0..60 {
            let n = if trial < 30 { 2 } else { 1 + trial % 5 };
            let st = random_stabilizer_state(n, &mut rng);
            let u = if trial < 30 {
                xs2.clone()
            } else {
                let mut c = Circuit::new(n);
                for _ in 0..12 {
                    let q = rng.gen_range(0..n);
                    match rng.gen_range(0..4) {
                        0 => c.h(q),
                        1 => c.s(q),
                        2 => c.x(q),
                        _ if n > 1 => c.cx(q, (q + 1) % n),
                        _ => c.s(q),
                    };
                }
                c
            };
            let mut v: StateVector<C64> =
                StateVector::from_amplitudes(n, QuantumState::<C64>::to_vector(&st).unwrap()).unwrap();
            let v0 = v.clone();
            for op in u.ops() {
                if let Op::Gate { gate, qubits } = op {
                    v.apply_gate(gate, qubits).unwrap();
                }
            }
            let want = v0.inner(&v).unwrap();
            let got = st.circuit_expectation(&u).unwrap().to_c64();
            assert!((want - got).norm() < 1e-9, "trial {trial}: {want} vs {got}");
            if let Some(l) = st.circuit_eigenvalue(&u).unwrap() {
                assert!((l.to_c64().norm() - 1.0).abs() < 1e-12);
                eigen += 1;
            }
        }
        // some random stabilizer states are eigenvectors
        assert!(eigen > 0);
        // a phased Pauli string i^c Z^z X^a as a gate: eigenvalue ±i^c on its eigenstates
        for trial in 0..30 {
            let n = 1 + trial % 5;
            let st = random_stabilizer_state(n, &mut rng);
            let g = st.stabilizer(trial % n);
            let mut c = Circuit::new(n);
            c.gate(Gate::Pauli(g.negated()), &(0..n).collect::<Vec<_>>());
            assert_eq!(st.circuit_eigenvalue(&c).unwrap(), Some(Phasor::sign(1)));
            let d = st.destabilizer(trial % n);
            let mut c = Circuit::new(n);
            let d = if d.is_hermitian() { d } else { d.times_i() };
            c.gate(Gate::Pauli(d), &(0..n).collect::<Vec<_>>());
            assert_eq!(st.circuit_eigenvalue(&c).unwrap(), None);
        }
    }
}
