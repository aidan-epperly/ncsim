//! Path sums: exact states of the form
//!
//! ```text
//! |ψ⟩ = scalar · Σ_{y ∈ 𝔽₂^m} i^{Q(y)} ζ^{g(y)} |shift ⊕ A y⟩
//! ```
//!
//! with `Q` a ℤ₄ quadratic form (the Clifford part), `g` a phase polynomial
//! mod `2^r` without Clifford terms (the non-Clifford part) and `A` an
//! `n × m` matrix over `𝔽₂` whose row `q` gives output bit `q` as an affine
//! function of the path variables `y`.
//!
//! * Diagonal and permutation gates (every [`MonomialGate`]: `X`, `CX`,
//!   `SWAP`, `S`, `T`, `CZ`, `CCZ`, `Phase(k, r)`, Paulis, …) act by affine
//!   substitution; `H` adds a path variable; a Pauli projector adds one too.
//! * After every operation [`PathSum::reduce`] sums out each path variable
//!   that is absent from the outputs and from the non-Clifford part (Gauss
//!   sums, [`QuadForm::eliminate`]) and changes variables along kernel
//!   directions of `A`, so a Clifford circuit keeps `m = rank A`: the state
//!   stays in the affine form of a stabilizer state with its global phase
//!   tracked.  Non-Clifford variables remain and are the cost.  A rank cache
//!   (a left inverse and the left kernel of the Clifford columns of `A`)
//!   answers "which kernel direction did this `H` create" without a
//!   nullspace, so Clifford gates cost `O(n²/64)` at worst.
//! * Amplitudes, norms, inner products and Pauli expectation values are
//!   exponential sums of phase polynomials, evaluated exactly by ncalg
//!   (`cyclo::expsum`): closed form for the Clifford class, otherwise
//!   `O(terms · 2^width)` with the width of the non-Clifford core
//!   ([`PathSum::amplitude_plan`]).
//!
//! The scalar type `S` carries the global factor; with `S = Cyclo<8>` every
//! Clifford+T computation is exact, with `S = Complex<f64>` the exact sums are
//! converted at the end.

use super::stabilizer::{AffineForm, Stabilizer};
use super::QuantumState;
use crate::bits::{self, Bits};
use crate::circuit::Op;
use crate::error::{check_qubits, unsupported, Result, SimError};
use crate::gate::{Gate, MonomialGate};
use crate::pauli::Pauli;
use crate::phasepoly::{Elimination, PhasePoly, QuadForm};
use crate::scalar::{Amplitude, Phasor};
use ncalg::cyclo::expsum::ExpSumPlan;
use ncalg::ff::BitMat;
use std::fmt;

const BACKEND: &str = "PathSum";

/// A path-sum state (see the [module documentation](self)).
#[derive(Clone)]
pub struct PathSum<S: Amplitude> {
    n: usize,
    m: usize,
    shift: Bits,
    rows: Vec<Bits>,
    clifford: QuadForm,
    rest: PhasePoly,
    scalar: S,
    /// Left inverse and left kernel of the Clifford columns of `A`, kept
    /// through the gates that update them cheaply (see [`RankCache`]).
    cache: Option<RankCache>,
}

/// For the Clifford-only variables `C` (full column rank of `A_C`, the normal
/// form): `ell[j]` (`j ∈ C`, an `n`-bit vector) with `ell_j · col_i = δ_ji`, and
/// `w`, a basis of the left kernel `{w : wᵀ A_C = 0}`.  An `H` on qubit `q`
/// then needs no nullspace: if some `w` has `w_q = 1` the row of `q` is
/// redundant and no kernel appears, otherwise `v = (ell_j[q])_j` is the one
/// new kernel direction.  Monomial Clifford gates act on both by the inverse
/// of their linear part; operations without an update rule drop the cache and
/// `reduce` rebuilds it by one elimination.
#[derive(Clone, Debug)]
struct RankCache {
    cvars: Bits,
    ell: Vec<Option<Bits>>,
    w: Vec<Bits>,
}

impl<S: Amplitude> PathSum<S> {
    /// `|x⟩`.
    pub fn basis_state(n: usize, x: &[u64]) -> Self {
        let mut shift = bits::zero(n);
        bits::xor_into(&mut shift, x);
        bits::mask_to(&mut shift, n);
        PathSum {
            n,
            m: 0,
            shift,
            rows: vec![bits::zero(0); n],
            clifford: QuadForm::new(0),
            rest: PhasePoly::new(0, 2),
            scalar: S::one(),
            cache: None,
        }
    }
    /// `|0…0⟩`.
    pub fn zero_state(n: usize) -> Self {
        Self::basis_state(n, &bits::zero(n))
    }
    /// `|+⟩^{⊗n}` (needs `2^{-n/2}` in `S`).
    pub fn plus_state(n: usize) -> Result<Self> {
        let scalar =
            S::sqrt2_pow(-(n as i64)).ok_or_else(|| SimError::NotRepresentable(format!("2^(-{n}/2)")))?;
        Ok(PathSum {
            n,
            m: n,
            shift: bits::zero(n),
            rows: (0..n).map(|q| bits::unit(n, q)).collect(),
            clifford: QuadForm::new(n),
            rest: PhasePoly::new(n, 2),
            scalar,
            cache: None,
        })
    }
    /// From a stabilizer state in affine form (`ψ(shift) = 2^{-k/2} > 0`).
    pub fn from_affine_form(af: &AffineForm) -> Result<Self> {
        let (n, k) = (af.n, af.dim());
        let scalar =
            S::sqrt2_pow(-(k as i64)).ok_or_else(|| SimError::NotRepresentable(format!("2^(-{k}/2)")))?;
        let rows: Vec<Bits> = (0..n)
            .map(|q| {
                let mut r = bits::zero(k);
                for (i, b) in af.basis.iter().enumerate() {
                    if bits::get(b, q) {
                        bits::toggle(&mut r, i);
                    }
                }
                r
            })
            .collect();
        Ok(PathSum {
            n,
            m: k,
            shift: af.shift.clone(),
            rows,
            clifford: af.form.clone(),
            rest: PhasePoly::new(k, 2),
            scalar,
            cache: None,
        })
    }
    /// From a stabilizer tableau (its canonical representative).
    pub fn from_stabilizer(s: &Stabilizer) -> Result<Self> {
        Self::from_affine_form(&s.affine_form())
    }
    /// Number of qubits.
    pub fn num_qubits(&self) -> usize {
        self.n
    }
    /// Number of path variables `m`.
    pub fn num_path_vars(&self) -> usize {
        self.m
    }
    /// The global factor.
    pub fn scalar(&self) -> &S {
        &self.scalar
    }
    /// The Clifford part `Q`.
    pub fn clifford_part(&self) -> &QuadForm {
        &self.clifford
    }
    /// The non-Clifford part `g`.
    pub fn nonclifford_part(&self) -> &PhasePoly {
        &self.rest
    }
    /// Is the state in the Clifford class (no non-Clifford terms)?
    pub fn is_clifford(&self) -> bool {
        self.rest.is_zero()
    }
    /// Is the state identically zero?
    pub fn is_zero(&self) -> bool {
        self.scalar.is_zero()
    }
    /// Check the rank cache against its definition (tests).
    #[cfg(test)]
    pub(crate) fn rank_cache_consistent(&self) -> bool {
        let Some(c) = &self.cache else { return true };
        if !self.cache_valid() {
            return true; // stale caches are never used
        }
        let cvars: Vec<usize> = bits::ones(&c.cvars).into_iter().filter(|&j| j < self.m).collect();
        let col = |i: usize| -> Bits {
            let mut b = bits::zero(self.n);
            for q in 0..self.n {
                if bits::get(&self.rows[q], i) {
                    bits::set(&mut b, q, true);
                }
            }
            b
        };
        let cols: Vec<Bits> = cvars.iter().map(|&i| col(i)).collect();
        for &j in &cvars {
            let Some(Some(l)) = c.ell.get(j) else { return false };
            for (ii, &i) in cvars.iter().enumerate() {
                if bits::dot(l, &cols[ii]) != (i == j) {
                    return false;
                }
            }
        }
        for w in &c.w {
            if cols.iter().any(|cl| bits::dot(w, cl)) || bits::is_zero(w) {
                return false;
            }
        }
        c.w.len() + cvars.len() == self.n
    }
    /// The output shift.
    pub fn shift(&self) -> &[u64] {
        &self.shift
    }
    /// Row `q` of `A` (the path variables feeding output bit `q`).
    pub fn row(&self, q: usize) -> &[u64] {
        &self.rows[q]
    }

    /// `self ⊗ other` (`self` on the low qubits): disjoint path variables,
    /// the direct sum of the phase functions, the product of the scalars.
    pub fn tensor(&self, other: &PathSum<S>) -> PathSum<S> {
        let (na, nb) = (self.n, other.n);
        let (ma, mb) = (self.m, other.m);
        let m = ma + mb;
        let mut rows = Vec::with_capacity(na + nb);
        for r in &self.rows {
            let mut r = r.clone();
            bits::ensure_len(&mut r, m);
            rows.push(r);
        }
        for r in &other.rows {
            let mut x = bits::zero(m);
            bits::or_shifted(&mut x, r, mb, ma);
            rows.push(x);
        }
        let bits_r = self.rest.bits().max(other.rest.bits());
        let mut ra = self.rest.clone();
        ra.raise_bits(bits_r);
        let mut rb = other.rest.clone();
        rb.raise_bits(bits_r);
        let mut rest = ra.extend(m);
        rest.add_assign(&rb.embed(ma, m));
        PathSum {
            n: na + nb,
            m,
            shift: bits::concat(&self.shift, na, &other.shift, nb),
            rows,
            clifford: self.clifford.direct_sum(&other.clifford),
            rest,
            scalar: self.scalar.mul_ref(&other.scalar),
            cache: None,
        }
    }

    fn set_zero(&mut self) {
        self.cache = None;
        self.scalar = S::zero();
        self.m = 0;
        self.rows = vec![bits::zero(0); self.n];
        self.clifford = QuadForm::new(0);
        self.rest = PhasePoly::new(0, self.rest.bits());
    }
    fn mul_scalar(&mut self, c: S) {
        self.scalar = self.scalar.mul_ref(&c);
    }
    fn mul_phasor(&mut self, p: Phasor) -> Result<()> {
        let c = S::from_phasor(p).ok_or_else(|| SimError::NotRepresentable(format!("factor {p}")))?;
        self.mul_scalar(c);
        Ok(())
    }
    /// Add a path variable; returns its index.
    fn add_variable(&mut self) -> usize {
        let j = self.m;
        self.m += 1;
        for r in self.rows.iter_mut() {
            bits::ensure_len(r, self.m);
        }
        self.clifford.push_var();
        self.rest = self.rest.extend(self.m);
        if let Some(c) = self.cache.as_mut() {
            bits::ensure_len(&mut c.cvars, self.m);
            c.ell.push(None);
        }
        j
    }
    /// The Clifford-only variables (absent from the non-Clifford part).
    fn clifford_mask(&self) -> Bits {
        let gv = self.rest.used_vars();
        let mut c = bits::zero(self.m);
        for j in 0..self.m {
            if !bits::get(&gv, j) {
                bits::set(&mut c, j, true);
            }
        }
        c
    }
    /// Is the cache valid for the current Clifford-only variables?
    fn cache_valid(&self) -> bool {
        match &self.cache {
            Some(c) => {
                let mut cv = c.cvars.clone();
                cv.resize(bits::nwords(self.m), 0);
                bits::mask_to(&mut cv, self.m);
                cv == self.clifford_mask()
            }
            None => false,
        }
    }
    /// Is the cache valid for the Clifford-only set `mask`?
    fn cache_valid_for(&self, mask: &Bits) -> bool {
        match &self.cache {
            Some(c) => {
                let mut cv = c.cvars.clone();
                cv.resize(bits::nwords(self.m), 0);
                bits::mask_to(&mut cv, self.m);
                cv == *mask
            }
            None => false,
        }
    }
    /// Build the cache by one elimination over the rows restricted to the
    /// Clifford-only variables; `None` (and no cache) if `A_C` has a kernel.
    fn build_rank_cache(&self, cmask: &Bits) -> Option<RankCache> {
        let n = self.n;
        let cvars: Vec<usize> = bits::ones(cmask).into_iter().filter(|&j| j < self.m).collect();
        let c = cvars.len();
        let mut pos = vec![usize::MAX; self.m];
        for (i, &j) in cvars.iter().enumerate() {
            pos[j] = i;
        }
        let to_n = |comb: Vec<u64>| -> Bits {
            let mut b = bits::zero(n);
            for (x, y) in b.iter_mut().zip(&comb) {
                *x = *y;
            }
            bits::mask_to(&mut b, n);
            b
        };
        let mut ech = ncalg::ff::BitEchelon::new(c);
        let mut w = Vec::new();
        for q in 0..n {
            let mut r = bits::zero(c);
            for j in bits::ones(&self.rows[q]) {
                if j < self.m && pos[j] != usize::MAX {
                    bits::set(&mut r, pos[j], true);
                }
            }
            if let Err(comb) = ech.insert(r) {
                w.push(to_n(comb));
            }
        }
        if ech.rank() < c {
            return None;
        }
        let mut ell = vec![None; self.m];
        for (i, &j) in cvars.iter().enumerate() {
            let mut e = bits::unit(c, i);
            let comb = ech.reduce(&mut e);
            debug_assert!(bits::is_zero(&e));
            ell[j] = Some(to_n(comb));
        }
        Some(RankCache {
            cvars: cmask.clone(),
            ell,
            w,
        })
    }
    /// Apply `x ← x · G^{-1}` on the gate qubits to every cached vector
    /// (both `ell` and `w` transform this way under `A ← Ĝ A`).
    fn cache_apply_linear(&mut self, inv: &[Bits], qubits: &[usize]) {
        let Some(c) = self.cache.as_mut() else { return };
        let k = qubits.len();
        let apply = |x: &mut Bits| {
            let mut a = 0usize;
            for (r, &q) in qubits.iter().enumerate() {
                if bits::get(x, q) {
                    a |= 1 << r;
                }
            }
            if a == 0 {
                return;
            }
            let mut out = 0u64;
            for r in 0..k {
                if (a >> r) & 1 == 1 {
                    out ^= inv[r][0];
                }
            }
            for (i, &q) in qubits.iter().enumerate() {
                bits::set(x, q, (out >> i) & 1 == 1);
            }
        };
        for x in c.ell.iter_mut().flatten() {
            apply(x);
        }
        for x in c.w.iter_mut() {
            apply(x);
        }
    }
    /// Move constant terms of `Q` and `g` into the scalar.
    fn absorb_constants(&mut self) -> Result<()> {
        let c = self.clifford.constant();
        if c != 0 {
            self.mul_scalar(S::i_pow(c as i64));
            self.clifford.add_constant(-(c as i64));
        }
        let k = self.rest.constant();
        if k != 0 {
            let z = S::root_of_unity(k as i64, self.rest.bits())
                .ok_or_else(|| SimError::NotRepresentable(format!("ζ_{}^{k}", self.rest.modulus())))?;
            self.mul_scalar(z);
            self.rest.add_constant(-(k as i64));
        }
        Ok(())
    }
    /// Variables appearing in some output row.
    fn row_vars(&self) -> Bits {
        let mut u = bits::zero(self.m);
        for r in &self.rows {
            for (a, b) in u.iter_mut().zip(r) {
                *a |= *b;
            }
        }
        u
    }

    /// Apply a monomial gate on `qubits`.
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
        if self.is_zero() {
            return Ok(());
        }
        let m = self.m;
        // the gate's inputs as affine functions of y
        let mut local_shift = bits::zero(k);
        for (j, &q) in qubits.iter().enumerate() {
            if bits::get(&self.shift, q) {
                bits::toggle(&mut local_shift, j);
            }
        }
        // phase
        if !g.phase().is_zero() {
            let mut ph = g.phase().clone();
            if ph.bits() < 2 {
                ph.raise_bits(2);
            }
            let gate_rows: Vec<Bits> = qubits.iter().map(|&q| self.rows[q].clone()).collect();
            let sub = ph.substitute_rows(&local_shift, &gate_rows, m);
            let rest = sub.split_clifford_into(&mut self.clifford);
            if rest.is_zero() {
                self.rest.add_assign(&rest);
            } else {
                // variables that gain non-Clifford terms leave C: their rows of
                // L are left-kernel vectors of the remaining columns
                let before = self.clifford_mask();
                self.rest.add_assign(&rest);
                let after = self.clifford_mask();
                let joined = (0..m).any(|j| bits::get(&after, j) && !bits::get(&before, j));
                let valid = self.cache_valid_for(&before);
                match self.cache.as_mut() {
                    Some(c) if valid && !joined => {
                        for j in 0..m {
                            if bits::get(&before, j) && !bits::get(&after, j) {
                                bits::set(&mut c.cvars, j, false);
                                if let Some(l) = c.ell[j].take() {
                                    c.w.push(l);
                                }
                            }
                        }
                    }
                    _ => self.cache = None,
                }
            }
        }
        let identity_rows = g.rows().iter().enumerate().all(|(i, r)| bits::ones(r) == vec![i]);
        if self.cache.is_some() && !identity_rows {
            match bits::inverse(g.rows(), k) {
                Some(inv) if k <= 64 => self.cache_apply_linear(&inv, qubits),
                _ => self.cache = None,
            }
        }
        // permutation
        let old_rows: Vec<Bits> = qubits.iter().map(|&q| self.rows[q].clone()).collect();
        let old_shift: Vec<bool> = qubits.iter().map(|&q| bits::get(&self.shift, q)).collect();
        for (i, &q) in qubits.iter().enumerate() {
            let mut r = bits::zero(m);
            let mut s = bits::get(g.shift(), i);
            for j in bits::ones(&g.rows()[i]) {
                bits::xor_into(&mut r, &old_rows[j]);
                s ^= old_shift[j];
            }
            self.rows[q] = r;
            bits::set(&mut self.shift, q, s);
        }
        self.mul_phasor(g.scalar())?;
        self.absorb_constants()
    }

    /// Apply `H` on qubit `q`: a new path variable `t` with the phase
    /// `(−1)^{x_q(y) t}` and the factor `1/√2`, then reduce.
    pub fn apply_h(&mut self, q: usize) -> Result<()> {
        check_qubits(&[q], self.n)?;
        if self.is_zero() {
            return Ok(());
        }
        // with a valid cache: is the row of q redundant (a left-kernel vector
        // through q), or which kernel direction appears (v = L e_q)?
        enum Plan {
            Rebuild,
            Redundant(usize),
            Essential(Bits),
        }
        let plan = if self.cache_valid() {
            let c = self.cache.as_ref().expect("valid cache");
            match c.w.iter().position(|w| bits::get(w, q)) {
                Some(k) => Plan::Redundant(k),
                None => {
                    let mut v = bits::zero(self.m);
                    for j in bits::ones(&c.cvars) {
                        if let Some(Some(l)) = c.ell.get(j) {
                            if bits::get(l, q) {
                                bits::set(&mut v, j, true);
                            }
                        }
                    }
                    Plan::Essential(v)
                }
            }
        } else {
            self.cache = None;
            Plan::Rebuild
        };
        let t = self.add_variable();
        let m = self.m;
        let (s, lam) = (bits::get(&self.shift, q), self.rows[q].clone());
        self.clifford.add_double_product(&lam, &bits::unit(m, t));
        if s {
            self.clifford.add_linear(t, 2);
        }
        self.rows[q] = bits::unit(m, t);
        bits::set(&mut self.shift, q, false);
        let n = self.n;
        match plan {
            Plan::Rebuild => {}
            Plan::Redundant(k) => {
                // L ← L ⊕ (L e_q) k0ᵀ, W ← {w ⊕ w_q k0}, then the new variable reads row q
                let c = self.cache.as_mut().expect("cache");
                let k0 = c.w.swap_remove(k);
                for l in c.ell.iter_mut().flatten() {
                    if bits::get(l, q) {
                        bits::xor_into(l, &k0);
                    }
                }
                for w in c.w.iter_mut() {
                    if bits::get(w, q) {
                        bits::xor_into(w, &k0);
                    }
                }
                c.ell[t] = Some(bits::unit(n, q));
                bits::set(&mut c.cvars, t, true);
            }
            Plan::Essential(v) => match bits::lowest_one(&v) {
                // the variables feeding only row q: change variables along v so
                // that t0 leaves the outputs (reduce sums it out)
                Some(t0) => {
                    self.change_variables(&v, t0);
                    if let Some(c) = self.cache.as_mut() {
                        let l0 = c.ell[t0].take().expect("t0 in C");
                        bits::set(&mut c.cvars, t0, false);
                        for j in bits::ones(&c.cvars) {
                            if let Some(Some(l)) = c.ell.get_mut(j) {
                                if bits::get(l, q) {
                                    bits::xor_into(l, &l0);
                                }
                            }
                        }
                        c.ell[t] = Some(bits::unit(n, q));
                        bits::set(&mut c.cvars, t, true);
                    }
                }
                None => self.cache = None,
            },
        }
        self.mul_phasor(Phasor::sqrt2_pow(-1))?;
        self.reduce()
    }

    /// Substitute `y_k = value ⊕ μ·y` everywhere (`k ∉ μ`).
    fn substitute_var(&mut self, k: usize, value: bool, mu: &[u64]) {
        // columns: col_i ⊕= μ_i col_k, col_k leaves; the rows of L for the
        // other variables still invert, and ell_k ⊕ Σ_{i∈μ∩C} ell_i joins the
        // left kernel (needs k ∈ C)
        // (an algebraic update: valid whenever the cache was, even while an
        // eliminated variable is still waiting for `compact`)
        // (non-Clifford variables in μ change only columns outside A_C)
        match self.cache.as_mut() {
            Some(c) if bits::get(&c.cvars, k) => {
                let mut wnew = c.ell[k].take().expect("k in C");
                for i in bits::ones(mu) {
                    if !bits::get(&c.cvars, i) {
                        continue;
                    }
                    if let Some(Some(l)) = c.ell.get(i) {
                        bits::xor_into(&mut wnew, l);
                    }
                }
                bits::set(&mut c.cvars, k, false);
                c.w.push(wnew);
            }
            _ => self.cache = None,
        }
        self.clifford.substitute_var(k, value, mu);
        self.rest.substitute_var(k, value, mu);
        for q in 0..self.n {
            if bits::get(&self.rows[q], k) {
                bits::set(&mut self.rows[q], k, false);
                bits::xor_into(&mut self.rows[q], mu);
                if value {
                    bits::toggle(&mut self.shift, q);
                }
            }
        }
    }

    /// Sum out a Clifford variable absent from the outputs and from `g`.
    fn eliminate_var(&mut self, j: usize) -> Result<()> {
        if let Some(c) = &self.cache {
            if bits::get(&c.cvars, j) {
                self.cache = None;
            }
        }
        match self.clifford.eliminate(j) {
            Elimination::Factor(ph) => self.mul_phasor(ph),
            Elimination::Constraint { lambda, value } => {
                let k = bits::lowest_one(&lambda).expect("nonzero constraint");
                let mut mu = lambda.clone();
                bits::toggle(&mut mu, k);
                self.substitute_var(k, value, &mu);
                self.mul_scalar(S::from_i64(2));
                Ok(())
            }
            Elimination::Zero => {
                self.set_zero();
                Ok(())
            }
        }
    }

    /// Change variables along a kernel vector `v` of `A` (`A v = 0`, `v_t = 1`,
    /// `v` disjoint from the non-Clifford variables) so that column `t` of
    /// `A` becomes zero: `y_s = z_s ⊕ v_s z_t` for `s ≠ t`.
    fn change_variables(&mut self, v: &Bits, t: usize) {
        let mut set = v.clone();
        bits::set(&mut set, t, false);
        self.clifford.shear(t, &set);
        for r in self.rows.iter_mut() {
            bits::set(r, t, false);
        }
        // L ← M L: ell_s ⊕= ell_t for s in the shear
        if let Some(c) = self.cache.as_mut() {
            let lt = c.ell.get(t).cloned().flatten();
            match lt {
                Some(lt) => {
                    for s in bits::ones(&set) {
                        match c.ell.get_mut(s) {
                            Some(Some(l)) => bits::xor_into(l, &lt),
                            _ => {
                                self.cache = None;
                                return;
                            }
                        }
                    }
                }
                None => self.cache = None,
            }
        }
    }

    /// Drop variables that appear nowhere.
    fn compact(&mut self) {
        let mut used = self.row_vars();
        for (a, b) in used.iter_mut().zip(self.rest.used_vars()) {
            *a |= b;
        }
        for j in 0..self.m {
            if !self.clifford.is_isolated(j) {
                bits::set(&mut used, j, true);
            }
        }
        let keep: Vec<usize> = bits::ones(&used).into_iter().filter(|&j| j < self.m).collect();
        if keep.len() == self.m {
            return;
        }
        let new_m = keep.len();
        let mut keep_mask = bits::zero(self.m);
        for &j in &keep {
            bits::set(&mut keep_mask, j, true);
        }
        let dropped: Vec<usize> = (0..self.m).filter(|&j| !bits::get(&keep_mask, j)).collect();
        if dropped.len() <= 4 {
            // delete the bits, highest first: O(n m/64) per variable
            for &j in dropped.iter().rev() {
                for r in self.rows.iter_mut() {
                    bits::remove_bit(r, j);
                }
                self.clifford.remove_var(j);
                if let Some(c) = self.cache.as_mut() {
                    bits::remove_bit(&mut c.cvars, j);
                    c.ell.remove(j);
                }
            }
            for r in self.rows.iter_mut() {
                r.truncate(bits::nwords(new_m));
            }
        } else {
            for r in self.rows.iter_mut() {
                let mut nr = bits::zero(new_m);
                for (i, &j) in keep.iter().enumerate() {
                    if bits::get(r, j) {
                        bits::toggle(&mut nr, i);
                    }
                }
                *r = nr;
            }
            self.clifford = self.clifford.project(&keep);
            if let Some(c) = self.cache.as_mut() {
                let mut cv = bits::zero(new_m);
                for (i, &j) in keep.iter().enumerate() {
                    if bits::get(&c.cvars, j) {
                        bits::set(&mut cv, i, true);
                    }
                }
                c.cvars = cv;
                c.ell = keep.iter().map(|&j| c.ell[j].clone()).collect();
            }
        }
        if let Some(c) = self.cache.as_mut() {
            c.cvars.truncate(bits::nwords(new_m));
            bits::mask_to(&mut c.cvars, new_m);
        }
        self.rest = self.rest.project(&keep);
        self.m = new_m;
    }

    /// Bring the state to normal form: sum out every path variable that is
    /// absent from the outputs and from the non-Clifford part, including the
    /// ones exposed by changing variables along kernel directions of `A`
    /// among the Clifford-only variables.
    pub fn reduce(&mut self) -> Result<()> {
        loop {
            if self.is_zero() {
                return Ok(());
            }
            let rv = self.row_vars();
            let gv = self.rest.used_vars();
            if let Some(j) = (0..self.m).find(|&j| !bits::get(&rv, j) && !bits::get(&gv, j)) {
                self.eliminate_var(j)?;
                self.compact();
                continue;
            }
            // kernel directions among the Clifford-only variables: none while
            // the cache is valid (full column rank is its invariant)
            if self.cache_valid() {
                break;
            }
            self.cache = None;
            let cmask = self.clifford_mask();
            if let Some(c) = self.build_rank_cache(&cmask) {
                self.cache = Some(c);
                break;
            }
            let cvars: Vec<usize> = (0..self.m).filter(|&j| !bits::get(&gv, j)).collect();
            let mut a = BitMat::zeros(self.n, cvars.len());
            for (c, &j) in cvars.iter().enumerate() {
                for q in 0..self.n {
                    if bits::get(&self.rows[q], j) {
                        a.set(q, c, true);
                    }
                }
            }
            let null = a.nullspace();
            if null.rows() == 0 {
                break;
            }
            let mut v = bits::zero(self.m);
            for c in bits::ones(null.row_words(0)) {
                if c < cvars.len() {
                    bits::set(&mut v, cvars[c], true);
                }
            }
            let t = bits::lowest_one(&v).expect("nonzero kernel vector");
            self.change_variables(&v, t);
        }
        self.absorb_constants()
    }

    /// The solutions of `A y = rhs` as `(particular, kernel basis)`, or `None`.
    fn solve_outputs(&self, rhs: &[bool]) -> Option<(Bits, Vec<Bits>)> {
        let (mut w0, null) = bits::solve_affine(&self.rows, self.m, &bits::from_bools(rhs))?;
        bits::ensure_len(&mut w0, self.m);
        Some((w0, null))
    }

    /// `Σ_{y ∈ w0 + span(cols)} i^{q(y)} ζ^{g(y)}`.
    fn restricted_sum(q: &QuadForm, g: &PhasePoly, w0: &[u64], cols: &[Bits]) -> Result<S> {
        let k = cols.len();
        let rq = q.substitute(w0, cols, k);
        let rg = g.substitute(w0, cols, k);
        if rg.is_zero() {
            let ph = rq.gauss_sum();
            S::from_phasor(ph).ok_or_else(|| SimError::NotRepresentable(format!("Gauss sum {ph}")))
        } else {
            let bits = rg.bits().max(2);
            let mut f = rq.to_phase_poly(bits);
            f.add_assign(&rg);
            f.exp_sum::<S>()
        }
    }

    /// `⟨self| P |other⟩` for `P = i^{ph} X^a Z^b` (`None`: the identity):
    /// one exponential sum over the pairs `(u, v)` with `x_other(u) ⊕ a = x_self(v)`.
    fn bilinear(&self, other: &PathSum<S>, p: Option<&Pauli>) -> Result<S> {
        if self.n != other.n {
            return Err(SimError::Invalid("register sizes differ".into()));
        }
        if self.is_zero() || other.is_zero() {
            return Ok(S::zero());
        }
        let n = self.n;
        let (mo, ms) = (other.m, self.m);
        let tot = mo + ms;
        let sys: Vec<Bits> = (0..n)
            .map(|q| bits::concat(&other.rows[q], mo, &self.rows[q], ms))
            .collect();
        let mut rhs = bits::zero(n);
        bits::xor_into(&mut rhs, &other.shift);
        bits::xor_into(&mut rhs, &self.shift);
        if let Some(p) = p {
            bits::xor_into(&mut rhs, p.x_bits());
        }
        bits::mask_to(&mut rhs, n);
        let Some((mut w0, cols)) = bits::solve_affine(&sys, tot, &rhs) else {
            return Ok(S::zero());
        };
        bits::ensure_len(&mut w0, tot);
        // i^{Q_o(u) − Q_s(v) + ph + 2 [b·x_o(u)]}
        let mut q = other.clifford.direct_sum(&self.clifford.negated());
        if let Some(p) = p {
            q.add_constant(p.phase() as i64);
            let mut mu = bits::zero(tot);
            let mut sb = false;
            for qb in bits::ones(p.z_bits()) {
                if qb < n {
                    bits::xor_into(&mut mu, &other.rows[qb]);
                    sb ^= bits::get(&other.shift, qb);
                }
            }
            if sb {
                q.add_constant(2);
            }
            q.add_parity(2, &mu);
        }
        let bits_r = self.rest.bits().max(other.rest.bits());
        let mut g = other.rest.extend(tot);
        g.raise_bits(bits_r);
        let mut gs = self.rest.embed(mo, tot);
        gs.raise_bits(bits_r);
        gs.scale(-1);
        g.add_assign(&gs);
        let sum = Self::restricted_sum(&q, &g, &w0, &cols)?;
        Ok(self.scalar.conj().mul_ref(&other.scalar).mul_ref(&sum))
    }

    /// `⟨self| P |other⟩` for a Pauli `P`.
    pub fn matrix_element(&self, p: &Pauli, other: &PathSum<S>) -> Result<S> {
        if p.num_qubits() != self.n {
            return Err(SimError::Invalid(
                "matrix_element: operator on the wrong register".into(),
            ));
        }
        self.bilinear(other, Some(p))
    }

    /// The phase polynomial whose exponential sum is `⟨x|ψ⟩`, for the cost
    /// estimate [`amplitude_plan`](Self::amplitude_plan).
    fn amplitude_polynomial(&self, x: &[u64]) -> Option<PhasePoly> {
        let rhs: Vec<bool> = (0..self.n)
            .map(|q| bits::get(x, q) ^ bits::get(&self.shift, q))
            .collect();
        let (w0, cols) = self.solve_outputs(&rhs)?;
        let k = cols.len();
        let bits_r = self.rest.bits().max(2);
        let mut f = self.clifford.substitute(&w0, &cols, k).to_phase_poly(bits_r);
        f.add_assign(&self.rest.substitute(&w0, &cols, k));
        Some(f)
    }

    /// The cost structure of the amplitude `⟨x|ψ⟩` (variables, Clifford
    /// eliminations, non-Clifford core and its width); `None` if `x` is
    /// outside the support.
    pub fn amplitude_plan(&self, x: &[u64]) -> Option<ExpSumPlan> {
        self.amplitude_polynomial(x).map(|f| f.exp_sum_plan())
    }

    /// The affine form of a Clifford-class state in normal form
    /// (`m = rank A`); the scalar is dropped (it is `2^{-m/2}` up to a phase).
    pub fn to_affine_form(&self) -> Result<AffineForm> {
        if !self.is_clifford() {
            return Err(SimError::Invalid("not a Clifford-class state".into()));
        }
        let basis: Vec<Bits> = (0..self.m)
            .map(|j| {
                let mut b = bits::zero(self.n);
                for q in 0..self.n {
                    if bits::get(&self.rows[q], j) {
                        bits::toggle(&mut b, q);
                    }
                }
                b
            })
            .collect();
        AffineForm::from_basis(self.n, self.shift.clone(), basis, self.clifford.clone())
    }

    /// The stabilizer tableau of a Clifford-class state.
    pub fn to_stabilizer(&self) -> Result<Stabilizer> {
        Stabilizer::from_affine_form(&self.to_affine_form()?)
    }
}

impl<S: Amplitude> QuantumState<S> for PathSum<S> {
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
        if let Gate::H = gate {
            return self.apply_h(qubits[0]);
        }
        if let Some(m) = gate.monomial() {
            return self.apply_monomial(&m, qubits);
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
        Err(unsupported(
            BACKEND,
            format!("gate {gate} (no monomial form or decomposition)"),
        ))
    }

    fn norm_sq(&self) -> Result<S> {
        // Clifford class with A of full column rank (the cache's invariant when
        // every variable is Clifford): distinct paths reach distinct basis
        // states, so ⟨ψ|ψ⟩ = |scalar|² 2^m
        if self.rest.is_zero() && self.cache_valid() {
            let p = S::sqrt2_pow(2 * self.m as i64)
                .ok_or_else(|| SimError::NotRepresentable(format!("2^{}", self.m)))?;
            return Ok(self.scalar.abs_sq().mul_ref(&p));
        }
        self.bilinear(self, None)
    }

    fn amplitude(&self, x: &[u64]) -> Result<S> {
        if self.is_zero() {
            return Ok(S::zero());
        }
        let rhs: Vec<bool> = (0..self.n)
            .map(|q| bits::get(x, q) ^ bits::get(&self.shift, q))
            .collect();
        let Some((w0, cols)) = self.solve_outputs(&rhs) else {
            return Ok(S::zero());
        };
        let sum = Self::restricted_sum(&self.clifford, &self.rest, &w0, &cols)?;
        Ok(self.scalar.mul_ref(&sum))
    }

    fn expectation(&self, p: &Pauli) -> Result<S> {
        if p.num_qubits() != self.n {
            return Err(SimError::Invalid(
                "expectation: operator on the wrong register".into(),
            ));
        }
        self.bilinear(self, Some(p))
    }

    fn inner(&self, other: &Self) -> Result<S> {
        self.bilinear(other, None)
    }

    fn project(&mut self, p: &Pauli, outcome: bool) -> Result<S> {
        if !p.is_hermitian() {
            return Err(SimError::Invalid(format!("projector of non-Hermitian {p}")));
        }
        if p.num_qubits() != self.n {
            return Err(SimError::Invalid(
                "project: operator on the wrong register".into(),
            ));
        }
        if self.is_zero() {
            return Ok(S::zero());
        }
        // (1 + (−1)^o P)/2 = ½ Σ_{t} ((−1)^o P)^t with a new path variable t
        // whose column of A is the X support a of P
        let valid = self.cache_valid();
        if !valid {
            self.cache = None;
        }
        let t = self.add_variable();
        let m = self.m;
        let mut mu = bits::zero(m);
        let mut sb = false;
        for q in bits::ones(p.z_bits()) {
            if q < self.n {
                bits::xor_into(&mut mu, &self.rows[q]);
                sb ^= bits::get(&self.shift, q);
            }
        }
        let lin = p.phase() as i64 + 2 * sb as i64 + 2 * outcome as i64;
        self.clifford.add_linear(t, lin);
        self.clifford.add_double_product(&mu, &bits::unit(m, t));
        for q in bits::ones(p.x_bits()) {
            if q < self.n {
                bits::toggle(&mut self.rows[q], t);
            }
        }
        if valid {
            let mut a = bits::zero(self.n);
            bits::xor_into(&mut a, p.x_bits());
            bits::mask_to(&mut a, self.n);
            let c = self.cache.as_mut().expect("valid cache");
            match c.w.iter().position(|w| bits::dot(w, &a)) {
                Some(k) => {
                    // a outside the column space: full rank, ell_t = w0
                    let w0 = c.w.swap_remove(k);
                    for l in c.ell.iter_mut().flatten() {
                        if bits::dot(l, &a) {
                            bits::xor_into(l, &w0);
                        }
                    }
                    for w in c.w.iter_mut() {
                        if bits::dot(w, &a) {
                            bits::xor_into(w, &w0);
                        }
                    }
                    c.ell[t] = Some(w0);
                    bits::set(&mut c.cvars, t, true);
                }
                None => {
                    // a = A (L a): change variables along v = L a ⊕ e_t so that t
                    // leaves the outputs; the old columns, and the cache, are unchanged
                    let mut v = bits::zero(m);
                    for j in bits::ones(&c.cvars) {
                        if let Some(Some(l)) = c.ell.get(j) {
                            if bits::dot(l, &a) {
                                bits::set(&mut v, j, true);
                            }
                        }
                    }
                    bits::set(&mut v, t, true);
                    let kept = self.cache.take();
                    self.change_variables(&v, t);
                    self.cache = kept;
                }
            }
        }
        self.mul_phasor(Phasor::sqrt2_pow(-2))?;
        self.reduce()?;
        self.norm_sq()
    }

    fn scale(&mut self, c: &S) -> Result<()> {
        self.mul_scalar(c.clone());
        Ok(())
    }

    fn tensor(&self, other: &Self) -> Result<Self> {
        Ok(PathSum::tensor(self, other))
    }
}

impl<S: Amplitude> fmt::Debug for PathSum<S> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "PathSum({} qubits, {} path vars, scalar {}, Q = {:?}, g = {}, shift {}",
            self.n,
            self.m,
            self.scalar,
            self.clifford,
            self.rest,
            bits::to_string(&self.shift, self.n)
        )?;
        for q in 0..self.n {
            write!(f, ", x{q}={}", bits::to_string(&self.rows[q], self.m))?;
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
    use crate::Cyclo;
    use ncalg::Coeff;
    use rand::rngs::StdRng;
    use rand::{Rng, SeedableRng};

    fn random_circuit(n: usize, gates: usize, non_clifford: bool, rng: &mut StdRng) -> Circuit {
        let mut c = Circuit::new(n);
        let mut one = vec![
            Gate::H,
            Gate::S,
            Gate::Sdg,
            Gate::X,
            Gate::Y,
            Gate::Z,
            Gate::SqrtX,
            Gate::SqrtY,
        ];
        let mut two = vec![Gate::Cx, Gate::Cz, Gate::Cy, Gate::Swap];
        if non_clifford {
            one.push(Gate::T);
            one.push(Gate::Tdg);
            two.push(Gate::Cs);
        }
        for _ in 0..gates {
            let r = rng.gen::<f64>();
            if r < 0.5 || n < 2 {
                let g = one[rng.gen_range(0..one.len())].clone();
                c.gate(g, &[rng.gen_range(0..n)]);
            } else if r < 0.9 || n < 3 {
                let g = two[rng.gen_range(0..two.len())].clone();
                let a = rng.gen_range(0..n);
                let mut b = rng.gen_range(0..n - 1);
                if b >= a {
                    b += 1;
                }
                c.gate(g, &[a, b]);
            } else if non_clifford {
                let mut qs: Vec<usize> = (0..n).collect();
                for i in 0..3 {
                    let j = rng.gen_range(i..n);
                    qs.swap(i, j);
                }
                c.gate(if rng.gen::<bool>() { Gate::Ccz } else { Gate::Ccx }, &qs[..3]);
            }
        }
        c
    }

    fn close(a: &[C64], b: &[C64]) -> bool {
        a.len() == b.len() && a.iter().zip(b).all(|(x, y)| (x - y).norm() < 1e-9)
    }

    #[test]
    fn clifford_circuits_stay_in_affine_normal_form() {
        let mut rng = StdRng::seed_from_u64(11);
        for trial in 0..20 {
            let n = 1 + trial % 6;
            let c = random_circuit(n, 40, false, &mut rng);
            let mut p: PathSum<C64> = PathSum::zero_state(n);
            let mut v: StateVector<C64> = StateVector::zero_state(n);
            p.run(&c, &mut rng).unwrap();
            v.run(&c, &mut rng).unwrap();
            assert!(p.is_clifford());
            let pv = p.to_vector().unwrap();
            assert!(close(&pv, &v.to_vector().unwrap()), "trial {trial}: {c}");
            // normal form: m = rank of the support
            let af = p.to_affine_form().unwrap();
            assert_eq!(af.dim(), p.num_path_vars());
            let s = p.to_stabilizer().unwrap();
            let sv: Vec<C64> = QuantumState::<C64>::to_vector(&s).unwrap();
            // same ray
            let i = (0..pv.len())
                .max_by(|&a, &b| pv[a].norm().partial_cmp(&pv[b].norm()).unwrap())
                .unwrap();
            let r = pv[i] / sv[i];
            assert!(pv.iter().zip(&sv).all(|(a, b)| (a - b * r).norm() < 1e-9));
        }
    }

    #[test]
    fn non_clifford_circuits_match_the_state_vector() {
        let mut rng = StdRng::seed_from_u64(5);
        for trial in 0..12 {
            let n = 2 + trial % 4;
            let c = random_circuit(n, 25, true, &mut rng);
            let mut p: PathSum<C64> = PathSum::zero_state(n);
            let mut v: StateVector<C64> = StateVector::zero_state(n);
            p.run(&c, &mut rng).unwrap();
            v.run(&c, &mut rng).unwrap();
            assert!(
                close(&p.to_vector().unwrap(), &v.to_vector().unwrap()),
                "trial {trial}: {c}"
            );
            let nn = p.norm_sq().unwrap();
            assert!((nn - C64::new(1.0, 0.0)).norm() < 1e-9);
            for _ in 0..5 {
                let kinds: Vec<crate::pauli::PauliKind> = (0..n)
                    .map(|_| {
                        use crate::pauli::PauliKind::*;
                        [I, X, Y, Z][rng.gen_range(0..4)]
                    })
                    .collect();
                let pauli = Pauli::from_kinds(&kinds);
                let a = p.expectation(&pauli).unwrap();
                let b = v.expectation(&pauli).unwrap();
                assert!((a - b).norm() < 1e-9, "trial {trial}: <{pauli}>");
            }
            // inner product with a second state
            let c2 = random_circuit(n, 25, true, &mut rng);
            let mut p2: PathSum<C64> = PathSum::zero_state(n);
            let mut v2: StateVector<C64> = StateVector::zero_state(n);
            p2.run(&c2, &mut rng).unwrap();
            v2.run(&c2, &mut rng).unwrap();
            let a = p.inner(&p2).unwrap();
            let b = v.inner(&v2).unwrap();
            assert!((a - b).norm() < 1e-9, "trial {trial}: inner");
            // projection
            let q = rng.gen_range(0..n);
            let z = Pauli::z(n, q);
            let mut pp = p.clone();
            let mut vv = v.clone();
            let wa = pp.project(&z, true).unwrap();
            let wb = vv.project(&z, true).unwrap();
            assert!((wa - wb).norm() < 1e-9);
            assert!(close(&pp.to_vector().unwrap(), &vv.to_vector().unwrap()));
        }
    }

    #[test]
    fn exact_t_gates() {
        // T on |+>^3 then CCZ: amplitudes are ζ8 powers / 2^{3/2}
        let mut c = Circuit::new(3);
        c.h(0).h(1).h(2).t(0).t(1).t(2).gate(Gate::Ccz, &[0, 1, 2]);
        let mut p: PathSum<Cyclo<8>> = PathSum::zero_state(3);
        let mut rng = StdRng::seed_from_u64(1);
        p.run(&c, &mut rng).unwrap();
        assert_eq!(p.num_path_vars(), 3);
        assert!(!p.is_clifford());
        let scale = Cyclo::<8>::sqrt2_pow(-3).unwrap();
        for x in 0..8u64 {
            let xb = bits::from_u64(x, 3);
            let k = x.count_ones() as i64 + if x == 7 { 4 } else { 0 };
            let want = scale.mul_ref(&Cyclo::<8>::zeta(k));
            assert_eq!(p.amplitude(&xb).unwrap(), want, "x = {x}");
        }
        let nn = p.norm_sq().unwrap();
        assert!(nn.is_one());
        // then H everywhere: the sum over paths has width 3
        c.h(0).h(1).h(2);
        let mut p: PathSum<Cyclo<8>> = PathSum::zero_state(3);
        p.run(&c, &mut rng).unwrap();
        let plan = p.amplitude_plan(&bits::zero(3)).unwrap();
        assert!(plan.core <= 3 && plan.width <= 3);
        let v: StateVector<C64> = {
            let mut v = StateVector::zero_state(3);
            v.run(&c, &mut rng).unwrap();
            v
        };
        let pv: Vec<C64> = p.to_vector().unwrap().iter().map(|a| a.to_complex()).collect();
        assert!(close(&pv, &v.to_vector().unwrap()));
    }

    #[test]
    fn stabilizer_round_trip_and_unsupported() {
        let mut rng = StdRng::seed_from_u64(3);
        let c = random_circuit(5, 30, false, &mut rng);
        let mut s = Stabilizer::zero_state(5);
        QuantumState::<C64>::run(&mut s, &c, &mut rng).unwrap();
        let p: PathSum<C64> = PathSum::from_stabilizer(&s).unwrap();
        assert_eq!(p.num_path_vars(), s.affine_form().dim());
        let a: Vec<C64> = QuantumState::<C64>::to_vector(&s).unwrap();
        assert!(close(&p.to_vector().unwrap(), &a));
        let mut p: PathSum<Cyclo<4>> = PathSum::zero_state(1);
        assert!(matches!(
            p.apply_gate(&Gate::H, &[0]),
            Err(SimError::NotRepresentable(_))
        ));
        let mut p: PathSum<C64> = PathSum::zero_state(1);
        assert!(matches!(
            p.apply_gate(&Gate::Rz(0.3), &[0]),
            Err(SimError::Unsupported { .. })
        ));
    }

    /// The rank cache stays consistent through Clifford and Clifford+T
    /// circuits with projections, and the states match the dense vector.
    #[test]
    fn rank_cache_is_consistent_and_states_are_unchanged() {
        let mut rng = StdRng::seed_from_u64(77);
        for trial in 0..40 {
            let n = 2 + trial % 9;
            let nc = trial % 2 == 1;
            let c = random_circuit(n, 60, nc, &mut rng);
            let mut p: PathSum<C64> = PathSum::zero_state(n);
            let mut v: StateVector<C64> = StateVector::zero_state(n);
            let (mut valid, mut steps) = (0usize, 0usize);
            for (step, op) in c.ops().iter().enumerate() {
                valid += p.cache_valid() as usize;
                steps += 1;
                if let Op::Gate { gate, qubits } = op {
                    p.apply_gate(gate, qubits).unwrap();
                    v.apply_gate(gate, qubits).unwrap();
                }
                assert!(p.rank_cache_consistent(), "trial {trial} step {step}: {p:?}");
                if step % 17 == 16 {
                    let z = Pauli::z(n, rng.gen_range(0..n));
                    let b = rng.gen::<bool>();
                    let wp = p.project(&z, b).unwrap();
                    let wv = v.project(&z, b).unwrap();
                    assert!((wp - wv).norm() < 1e-9);
                    if wv.norm() < 1e-12 {
                        break;
                    }
                }
            }
            let pv = p.to_vector().unwrap();
            let vv = v.to_vector().unwrap();
            assert!(
                pv.iter().zip(&vv).all(|(a, b)| (a - b).norm() < 1e-9),
                "trial {trial}: {c}"
            );
            // valid except before the first reduce and while a T-type term is fresh
            let need = if nc { steps / 4 } else { steps / 2 };
            assert!(
                valid >= need,
                "trial {trial}: cache valid at {valid} of {steps} steps"
            );
        }
        // the closed-form norm of Clifford-class states equals the Gauss sum
        for trial in 0..20 {
            let n = 3 + trial % 20;
            let c = random_circuit(n, 80, false, &mut rng);
            let mut p: PathSum<C64> = PathSum::zero_state(n);
            for op in c.ops() {
                if let Op::Gate { gate, qubits } = op {
                    p.apply_gate(gate, qubits).unwrap();
                }
            }
            let a = p.norm_sq().unwrap();
            let b = p.bilinear(&p, None).unwrap();
            assert!(
                (a - b).norm() < 1e-9 && (a.re - 1.0).abs() < 1e-9,
                "trial {trial}: {a} vs {b}"
            );
        }
        // larger registers: consistency only, and the cache is in use
        for trial in 0..6 {
            let n = 40 + 10 * trial;
            let c = random_circuit(n, 400, trial % 2 == 1, &mut rng);
            let mut p: PathSum<C64> = PathSum::zero_state(n);
            for op in c.ops() {
                if let Op::Gate { gate, qubits } = op {
                    p.apply_gate(gate, qubits).unwrap();
                }
            }
            assert!(p.rank_cache_consistent(), "large trial {trial}");
            // Clifford circuits never drop it (non-Clifford terms may, until the next reduce)
            if trial % 2 == 0 {
                assert!(p.cache_valid(), "large trial {trial}: the cache was dropped");
            }
        }
    }
}
