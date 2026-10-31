//! Phase polynomials `f: 𝔽₂ⁿ → ℤ/2^r` — the diagonal part of every exact
//! state family — in two forms:
//!
//! * [`PhasePoly`]: parity-product form `c + Σ_T c_T Π_{λ∈T} [λ·x]`, the
//!   representation of `ncalg::cyclo::expsum::PhasePolynomial`, closed under
//!   affine substitution.  ncalg's type exposes no term access, substitution
//!   of a single variable or variable extension, which simulation needs, so
//!   this is the one copy here; [`PhasePoly::to_ncalg`] hands a polynomial to
//!   ncalg's exact exponential sums.  Candidate for ncalg (then: re-exports).
//! * [`QuadForm`]: a ℤ₄ quadratic form `c + Σ a_j x_j + 2 Σ_{j<k} B_jk x_j x_k`,
//!   the Clifford class (`i^{q(x)}`), with single-variable Gauss-sum
//!   elimination ([`QuadForm::eliminate`]): the step that keeps path sums of
//!   Clifford circuits at zero free variables.  ncalg's `expsum` does the same
//!   elimination internally (its private `Reduced`); candidate for ncalg.
//!
//! The level of a term `c Π_{k≤s}[λ_k·x]` is `s + r − v₂(c) − 1`: 1 for `Z`
//! phases, 2 for the Clifford class (`S`, `CZ`), 3 for `T`, `CS`, `CCZ`.

use crate::bits::{self, Bits};
use crate::error::{Result, SimError};
use crate::scalar::{Amplitude, Phasor};
use ncalg::cyclo::expsum::{ExpSumPlan, PhasePolynomial as NcPhasePolynomial};
use ncalg::cyclo::Cyclo;
use std::collections::BTreeMap;
use std::fmt;

// ---------------------------------------------------------------------------
// ℤ₄ quadratic forms
// ---------------------------------------------------------------------------

/// `q(x) = c + Σ_j a_j x_j + 2 Σ_{j<k} B_jk x_j x_k` (mod 4) on `x ∈ 𝔽₂ⁿ`:
/// the phase `i^{q(x)}` of a Clifford-class diagonal.
#[derive(Clone, PartialEq, Eq, Hash)]
pub struct QuadForm {
    n: usize,
    constant: u8,
    linear: Vec<u8>,
    /// `quad[j]` = the neighbors `{k : B_jk = 1}`; symmetric, zero diagonal.
    quad: Vec<Bits>,
}

/// The result of summing a ℤ₄ quadratic form over one variable
/// ([`QuadForm::eliminate`]).
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Elimination {
    /// `Σ_{x_j} i^{q} = factor · i^{q'}` with `q'` the updated form on the
    /// other variables (`a_j` odd: `1 + i^{a_j}(−1)^{L}`; `a_j = 0`, `L ≡ 0`: `2`).
    Factor(Phasor),
    /// `Σ_{x_j} i^{q} = 2 · [L(x) = value] · i^{q'}`: an affine constraint on
    /// the other variables (`lambda` nonzero); the caller substitutes it.
    Constraint {
        /// The linear form `L`.
        lambda: Bits,
        /// The required value of `L(x)`.
        value: bool,
    },
    /// The sum vanishes identically (`a_j = 2`, `L ≡ 0`).
    Zero,
}

impl QuadForm {
    /// The zero form on `n` variables.
    pub fn new(n: usize) -> Self {
        QuadForm {
            n,
            constant: 0,
            linear: vec![0; n],
            quad: vec![bits::zero(n); n],
        }
    }
    /// Number of variables.
    pub fn num_vars(&self) -> usize {
        self.n
    }
    /// The constant `c`.
    pub fn constant(&self) -> u8 {
        self.constant
    }
    /// `a_j`.
    pub fn linear(&self, j: usize) -> u8 {
        self.linear[j]
    }
    /// `B_jk`.
    pub fn has_quad(&self, j: usize, k: usize) -> bool {
        bits::get(&self.quad[j], k)
    }
    /// The neighbors of `j`.
    pub fn neighbors(&self, j: usize) -> &[u64] {
        &self.quad[j]
    }
    /// Is the form zero?
    pub fn is_zero(&self) -> bool {
        self.constant == 0
            && self.linear.iter().all(|&a| a == 0)
            && self.quad.iter().all(|r| bits::is_zero(r))
    }
    /// `+ c`.
    pub fn add_constant(&mut self, c: i64) {
        self.constant = (self.constant as i64 + c).rem_euclid(4) as u8;
    }
    /// `+ c x_j`.
    pub fn add_linear(&mut self, j: usize, c: i64) {
        self.linear[j] = (self.linear[j] as i64 + c).rem_euclid(4) as u8;
    }
    /// `+ 2 x_j x_k` (`+ 2 x_j` when `j = k`).
    pub fn add_quad(&mut self, j: usize, k: usize) {
        if j == k {
            self.add_linear(j, 2);
        } else {
            bits::toggle(&mut self.quad[j], k);
            bits::toggle(&mut self.quad[k], j);
        }
    }
    /// `+ c · [λ·x]`, the integer value of a parity: `c Σ_{i∈λ} x_i − 2c Σ_{i<j∈λ} x_i x_j`.
    pub fn add_parity(&mut self, c: i64, lambda: &[u64]) {
        let c = c.rem_euclid(4);
        if c == 0 {
            return;
        }
        let vars: Vec<usize> = bits::ones(lambda).into_iter().filter(|&i| i < self.n).collect();
        for &i in &vars {
            self.add_linear(i, c);
        }
        if c % 2 == 1 && vars.len() > 1 {
            // every pair inside λ: row i ⊕= λ \ {i}, O(|λ| n/64)
            let mut lam = bits::zero(self.n);
            for &i in &vars {
                bits::set(&mut lam, i, true);
            }
            for &i in &vars {
                bits::xor_into(&mut self.quad[i], &lam);
                bits::toggle(&mut self.quad[i], i);
            }
        }
    }
    /// `+ 2 [λ·x][μ·x] = 2 Σ_{i∈λ, j∈μ} x_i x_j` (mod 4).
    pub fn add_double_product(&mut self, lambda: &[u64], mu: &[u64]) {
        // the pair {a, b} toggles iff [a∈λ][b∈μ] ⊕ [b∈λ][a∈μ]; x_i² = x_i on λ ∩ μ
        let n = self.n;
        let mut l = bits::zero(n);
        let mut m = bits::zero(n);
        for (d, s) in l.iter_mut().zip(lambda) {
            *d = *s;
        }
        for (d, s) in m.iter_mut().zip(mu) {
            *d = *s;
        }
        bits::mask_to(&mut l, n);
        bits::mask_to(&mut m, n);
        let mut both = l.clone();
        for (b, y) in both.iter_mut().zip(&m) {
            *b &= *y;
        }
        let mut either = l.clone();
        for (b, y) in either.iter_mut().zip(&m) {
            *b |= *y;
        }
        for a in bits::ones(&either) {
            let (in_l, in_m) = (bits::get(&l, a), bits::get(&m, a));
            if in_l {
                bits::xor_into(&mut self.quad[a], &m);
            }
            if in_m {
                bits::xor_into(&mut self.quad[a], &l);
            }
            // the diagonal: x_a x_a counted [a∈λ][a∈μ] twice → 2 x_a once
            bits::set(&mut self.quad[a], a, false);
        }
        for i in bits::ones(&both) {
            self.add_linear(i, 2);
        }
    }
    /// Append a variable `x_n` (in place; the quadratic rows grow only when
    /// they cross a word).
    pub fn push_var(&mut self) {
        self.n += 1;
        self.linear.push(0);
        let n = self.n;
        for r in self.quad.iter_mut() {
            bits::ensure_len(r, n);
        }
        self.quad.push(bits::zero(n));
    }
    /// `+ other`.
    pub fn add_assign(&mut self, other: &QuadForm) {
        assert_eq!(self.n, other.n);
        self.add_constant(other.constant as i64);
        for j in 0..self.n {
            self.add_linear(j, other.linear[j] as i64);
            bits::xor_into(&mut self.quad[j], &other.quad[j]);
        }
    }
    /// `−q`.
    pub fn negated(&self) -> QuadForm {
        let mut q = self.clone();
        q.constant = (4 - q.constant) & 3;
        for a in q.linear.iter_mut() {
            *a = (4 - *a) & 3;
        }
        q
    }
    /// `q(x) + other(y)` on the variables `(x, y)` (`self`'s first).
    pub fn direct_sum(&self, other: &QuadForm) -> QuadForm {
        let m = self.n + other.n;
        let mut q = self.extend(m);
        q.add_constant(other.constant as i64);
        for j in 0..other.n {
            q.linear[self.n + j] = other.linear[j];
            for k in bits::ones(&other.quad[j]) {
                bits::set(&mut q.quad[self.n + j], self.n + k, true);
            }
        }
        q
    }
    /// `q(x)` mod 4.
    pub fn eval(&self, x: &[u64]) -> u8 {
        let mut acc = self.constant as usize;
        let mut twice_pairs = 0usize;
        for j in bits::ones(x) {
            if j >= self.n {
                break;
            }
            acc += self.linear[j] as usize;
            twice_pairs += bits::and_count(&self.quad[j], x);
        }
        (acc + twice_pairs) as u8 & 3 // 2 · pairs = twice_pairs
    }
    /// Substitute `x_j → 1 − x_j`.
    pub fn flip(&mut self, j: usize) {
        let a = self.linear[j] as i64;
        self.add_constant(a);
        self.linear[j] = (-a).rem_euclid(4) as u8;
        for k in bits::ones(&self.quad[j]) {
            self.add_linear(k, 2);
        }
    }
    /// Is `x_j` absent from the form?
    pub fn is_isolated(&self, j: usize) -> bool {
        self.linear[j] == 0 && bits::is_zero(&self.quad[j])
    }
    /// Remove `x_j`'s terms, returning `(a_j, neighbors)`.
    fn take_var(&mut self, j: usize) -> (u8, Bits) {
        let a = self.linear[j];
        let nb = std::mem::replace(&mut self.quad[j], bits::zero(self.n));
        self.linear[j] = 0;
        for k in bits::ones(&nb) {
            bits::set(&mut self.quad[k], j, false);
        }
        (a, nb)
    }
    /// Substitute `x_j = value ⊕ [λ·x]` (`j ∉ λ`); afterwards `x_j` is isolated.
    pub fn substitute_var(&mut self, j: usize, value: bool, lambda: &[u64]) {
        debug_assert!(!bits::get(lambda, j), "substitute_var: λ contains the variable");
        let (a, nb) = self.take_var(j);
        if a != 0 {
            if value {
                self.add_constant(a as i64);
                self.add_parity(-(a as i64), lambda);
            } else {
                self.add_parity(a as i64, lambda);
            }
        }
        // Σ_{k∈nb} 2 x_j x_k = 2 [λ·x][nb·x] + (value ? Σ 2 x_k : 0)
        if !bits::is_zero(&nb) {
            self.add_double_product(lambda, &nb);
            if value {
                for k in bits::ones(&nb) {
                    self.add_linear(k, 2);
                }
            }
        }
    }
    /// The form in new variables `t ∈ 𝔽₂^m` after `x = shift ⊕ Σ_t t_i · columns_i`
    /// (each column an `n`-bit vector).
    pub fn substitute(&self, shift: &[u64], columns: &[Bits], m: usize) -> QuadForm {
        assert_eq!(columns.len(), m);
        let n = self.n;
        // x = x0 ⊕ C t with C the n × m matrix of the columns, rows μ_i.
        //   Σ a_i x_i = Σ a_i x0_i + Σ a'_i [μ_i·t],  a'_i = (−1)^{x0_i} a_i,
        //   [μ·t] = Σ_a μ_a t_a − 2 Σ_{a<b} μ_a μ_b t_a t_b             (mod 4);
        //   2 Σ_{i<j} B_ij x_i x_j = 2 (x0ᵀU x0 + (B x0)ᵀ C t + tᵀ Cᵀ U C t)  (U = upper B),
        // so the new quadratic part is the off-diagonal of Cᵀ (B + D) C with
        // D = diag(a_i odd), and diag(Cᵀ U C) joins the linear part.  GF(2)
        // products on packed rows: O((n·deg + n m) · m / 64).
        let x0 = |i: usize| bits::get(shift, i);
        let mu: Vec<Bits> = (0..n)
            .map(|i| {
                let mut r = bits::zero(m);
                for (t, col) in columns.iter().enumerate() {
                    if bits::get(col, i) {
                        bits::toggle(&mut r, t);
                    }
                }
                r
            })
            .collect();
        let mut out = QuadForm::new(m);
        // constant: c + Σ a_i x0_i + 2 x0ᵀ U x0
        let mut c = self.constant as i64;
        let mut pairs = 0usize;
        for i in 0..n {
            if x0(i) {
                c += self.linear[i] as i64;
                pairs += bits::and_count(&self.quad[i], shift);
            }
        }
        c += 2 * ((pairs / 2) % 2) as i64;
        out.add_constant(c);
        // P = (B + D) C, row i = ⊕_{j ∈ B_i} μ_j ⊕ [a_i odd] μ_i
        let p: Vec<Bits> = (0..n)
            .map(|i| {
                let mut r = bits::zero(m);
                for j in bits::ones(&self.quad[i]) {
                    if j < n {
                        bits::xor_into(&mut r, &mu[j]);
                    }
                }
                if self.linear[i] % 2 == 1 {
                    bits::xor_into(&mut r, &mu[i]);
                }
                r
            })
            .collect();
        // B x0
        let bx0: Bits = {
            let mut v = bits::zero(n);
            for i in 0..n {
                if bits::dot(&self.quad[i], shift) {
                    bits::toggle(&mut v, i);
                }
            }
            v
        };
        let column = |col: &Bits| -> (u8, Bits) {
            let ones = bits::ones(col);
            // linear: Σ_i a'_i μ_ia + 2 ((B x0)·col_a + diag(Cᵀ U C)_a)
            let mut l = 0i64;
            let mut inner_pairs = 0usize;
            for &i in &ones {
                if i >= n {
                    continue;
                }
                let ai = self.linear[i] as i64;
                l += if x0(i) { -ai } else { ai };
                inner_pairs += bits::and_count(&self.quad[i], col);
            }
            l += 2 * (bits::dot(&bx0, col) as i64 + ((inner_pairs / 2) % 2) as i64);
            // quadratic: row a of Cᵀ P
            let mut r = bits::zero(m);
            for &i in &ones {
                if i < n {
                    bits::xor_into(&mut r, &p[i]);
                }
            }
            (l.rem_euclid(4) as u8, r)
        };
        let results: Vec<(u8, Bits)> = if n * m >= 1 << 16 {
            use rayon::prelude::*;
            columns.par_iter().map(column).collect()
        } else {
            columns.iter().map(column).collect()
        };
        for (a, (l, mut r)) in results.into_iter().enumerate() {
            bits::set(&mut r, a, false);
            out.linear[a] = l;
            out.quad[a] = r;
        }
        debug_assert!((0..m).all(|a| (0..m).all(|b| out.has_quad(a, b) == out.has_quad(b, a))));
        out
    }
    /// The same form on `m ≥ n` variables.
    pub fn extend(&self, m: usize) -> QuadForm {
        assert!(m >= self.n);
        let mut q = QuadForm::new(m);
        q.constant = self.constant;
        q.linear[..self.n].copy_from_slice(&self.linear);
        for j in 0..self.n {
            let mut r = self.quad[j].clone();
            bits::ensure_len(&mut r, m);
            q.quad[j] = r;
        }
        q
    }
    /// Substitute `x_s → x_s ⊕ x_t` for every `s` in the set `shear`
    /// (`t ∉ shear`): `O(|shear| · n/64)`, the change of variables along a
    /// kernel direction.
    pub fn shear(&mut self, t: usize, shear: &[u64]) {
        debug_assert!(!bits::get(shear, t), "shear: the set contains t");
        let set: Vec<usize> = bits::ones(shear).into_iter().filter(|&s| s < self.n).collect();
        if set.is_empty() {
            return;
        }
        // quadratic part, on the original form: an edge {i, j} gives
        // [j∈S] x_i x_t + [i∈S] x_j x_t + [i, j ∈ S] x_t (all times 2)
        let mut p = bits::zero(self.n);
        let mut inside = 0usize;
        for &s in &set {
            bits::xor_into(&mut p, &self.quad[s]);
            inside += bits::and_count(&self.quad[s], shear);
        }
        let mut lt = 2 * ((inside / 2) % 2) as i64;
        // linear part: a_s [x_s ⊕ x_t] = a_s x_s + a_s x_t − 2 a_s x_s x_t
        let odd: Vec<usize> = set.iter().cloned().filter(|&s| self.linear[s] % 2 == 1).collect();
        for &s in &set {
            lt += self.linear[s] as i64;
        }
        for x in bits::ones(&p) {
            if x >= self.n {
                continue;
            }
            if x == t {
                lt += 2;
            } else {
                self.add_quad(x, t);
            }
        }
        for s in odd {
            self.add_quad(s, t);
        }
        self.add_linear(t, lt);
    }
    /// Remove the isolated variable `x_j` (the later ones are renumbered down).
    pub fn remove_var(&mut self, j: usize) {
        assert!(self.is_isolated(j), "remove_var: variable {j} is used");
        self.linear.remove(j);
        self.quad.remove(j);
        for r in self.quad.iter_mut() {
            bits::remove_bit(r, j);
        }
        self.n -= 1;
    }
    /// Keep the variables `keep` (renumbered in that order); every other
    /// variable must be isolated.
    pub fn project(&self, keep: &[usize]) -> QuadForm {
        let mut pos = vec![usize::MAX; self.n];
        for (i, &k) in keep.iter().enumerate() {
            pos[k] = i;
        }
        let mut q = QuadForm::new(keep.len());
        q.constant = self.constant;
        for j in 0..self.n {
            if pos[j] == usize::MAX {
                assert!(self.is_isolated(j), "project: dropped variable {j} is used");
                continue;
            }
            q.linear[pos[j]] = self.linear[j];
            for k in bits::ones(&self.quad[j]) {
                if pos[k] == usize::MAX {
                    panic!("project: dropped variable {k} is used");
                }
                bits::set(&mut q.quad[pos[j]], pos[k], true);
            }
        }
        q
    }
    /// Sum out `x_j`: `Σ_{x_j ∈ {0,1}} i^{q(x)}`.  Afterwards `x_j` is isolated
    /// and the form is the remaining `q'` (see [`Elimination`]).
    pub fn eliminate(&mut self, j: usize) -> Elimination {
        let (a, nb) = self.take_var(j);
        if a % 2 == 1 {
            // 1 + i^a (−1)^{L} = (1 + i^a) · i^{−a [L]}
            self.add_parity(-(a as i64), &nb);
            let phase = if a == 1 { 1 } else { 7 };
            Elimination::Factor(Phasor::new(1, phase, 3))
        } else {
            let value = a == 2;
            if bits::is_zero(&nb) {
                if value {
                    Elimination::Zero
                } else {
                    Elimination::Factor(Phasor::new(2, 0, 0))
                }
            } else {
                Elimination::Constraint { lambda: nb, value }
            }
        }
    }
    /// `Σ_x i^{q(x)}` in closed form (through ncalg's Clifford Gauss sum).
    pub fn gauss_sum(&self) -> Phasor {
        // the variables one by one with `eliminate` (row-XOR updates, no
        // conversion to a general phase polynomial); a constraint fixes a later
        // variable, which then leaves the sum
        let mut q = self.clone();
        let mut acc = Phasor::ONE;
        let mut fixed = bits::zero(self.n);
        for j in 0..self.n {
            if bits::get(&fixed, j) {
                continue;
            }
            match q.eliminate(j) {
                Elimination::Factor(ph) => acc = acc.mul(ph),
                Elimination::Constraint { lambda, value } => {
                    let k = bits::lowest_one(&lambda).expect("nonzero constraint");
                    let mut mu = lambda.clone();
                    bits::toggle(&mut mu, k);
                    q.substitute_var(k, value, &mu);
                    bits::set(&mut fixed, k, true);
                    acc = acc.mul(Phasor::sqrt2_pow(2));
                }
                Elimination::Zero => return Phasor::Zero,
            }
        }
        acc.mul(Phasor::i_pow(q.constant() as i64))
    }

    /// The form as a phase polynomial mod `2^bits` (`bits ≥ 2`): `2^{bits−2} q`.
    pub fn to_phase_poly(&self, bits: u32) -> PhasePoly {
        assert!(bits >= 2, "a ℤ₄ form needs at least 2 phase bits");
        let unit = 1i64 << (bits - 2);
        let mut p = PhasePoly::new(self.n, bits);
        p.add_constant(unit * self.constant as i64);
        for j in 0..self.n {
            if self.linear[j] != 0 {
                p.add_monomial(unit * self.linear[j] as i64, &[j]);
            }
            for k in bits::ones(&self.quad[j]) {
                if k > j {
                    p.add_monomial(2 * unit, &[j, k]);
                }
            }
        }
        p
    }
}

impl fmt::Debug for QuadForm {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "QuadForm({})", self.to_phase_poly(2))
    }
}

// ---------------------------------------------------------------------------
// parity-product phase polynomials
// ---------------------------------------------------------------------------

/// `f(x) = c + Σ_T c_T Π_{λ∈T} [λ·x]` with values mod `2^bits`, `[λ·x] ∈ {0,1}`
/// the parity of the bits of `x` selected by `λ`.  Terms are keyed by the
/// sorted list of distinct nonzero parities; coefficients are nonzero.
#[derive(Clone, Eq)]
pub struct PhasePoly {
    n: usize,
    bits: u32,
    constant: u64,
    terms: BTreeMap<Vec<Bits>, u64>,
}

impl PhasePoly {
    /// The zero polynomial on `n` variables mod `2^bits` (`1 ≤ bits ≤ 32`).
    pub fn new(n: usize, bits: u32) -> Self {
        assert!((1..=32).contains(&bits), "phase bits must be in 1..=32");
        PhasePoly {
            n,
            bits,
            constant: 0,
            terms: BTreeMap::new(),
        }
    }
    /// Number of variables.
    pub fn num_vars(&self) -> usize {
        self.n
    }
    /// `r`: values are mod `2^r`, `ζ = e^{2πi/2^r}`.
    pub fn bits(&self) -> u32 {
        self.bits
    }
    /// `2^r`.
    pub fn modulus(&self) -> u64 {
        1u64 << self.bits
    }
    /// The constant term.
    pub fn constant(&self) -> u64 {
        self.constant
    }
    /// Number of non-constant terms.
    pub fn num_terms(&self) -> usize {
        self.terms.len()
    }
    /// Zero (no terms, zero constant)?
    pub fn is_zero(&self) -> bool {
        self.constant == 0 && self.terms.is_empty()
    }
    /// The terms `(parities, coefficient)`.
    pub fn terms(&self) -> impl Iterator<Item = (&Vec<Bits>, u64)> + '_ {
        self.terms.iter().map(|(k, &c)| (k, c))
    }
    fn reduce(&self, c: i64) -> u64 {
        c.rem_euclid(self.modulus() as i64) as u64
    }
    /// `+ c`.
    pub fn add_constant(&mut self, c: i64) {
        self.constant = self.reduce(self.constant as i64 + c);
    }
    /// `+ c · Π_i [λ_i·x]`.  Zero parities make the term vanish; repeats collapse.
    pub fn add_parity_term(&mut self, c: i64, forms: &[Bits]) {
        let c = self.reduce(c);
        if c == 0 {
            return;
        }
        let mut key: Vec<Bits> = Vec::with_capacity(forms.len());
        for f in forms {
            let mut g = f.clone();
            bits::ensure_len(&mut g, self.n);
            g.truncate(bits::nwords(self.n));
            bits::mask_to(&mut g, self.n);
            if bits::is_zero(&g) {
                return;
            }
            key.push(g);
        }
        key.sort();
        key.dedup();
        if key.is_empty() {
            self.add_constant(c as i64);
            return;
        }
        let m = self.modulus();
        match self.terms.get_mut(&key) {
            Some(v) => {
                *v = (*v + c) % m;
                if *v == 0 {
                    self.terms.remove(&key);
                }
            }
            None => {
                self.terms.insert(key, c);
            }
        }
    }
    /// `+ c · Π_{i ∈ vars} x_i`.
    pub fn add_monomial(&mut self, c: i64, vars: &[usize]) {
        let forms: Vec<Bits> = vars.iter().map(|&v| bits::unit(self.n, v)).collect();
        self.add_parity_term(c, &forms);
    }
    /// `+ other` (the modulus is raised to the larger of the two).
    pub fn add_assign(&mut self, other: &PhasePoly) {
        assert_eq!(self.n, other.n, "add_assign: variable count");
        if other.bits > self.bits {
            self.raise_bits(other.bits);
        }
        let scale = 1i64 << (self.bits - other.bits);
        self.add_constant(scale * other.constant as i64);
        for (k, c) in other.terms() {
            self.add_parity_term(scale * c as i64, k);
        }
    }
    /// `c · self`.
    pub fn scale(&mut self, c: i64) {
        let m = self.modulus();
        self.constant = self.reduce(self.constant as i64 * c);
        let old = std::mem::take(&mut self.terms);
        for (k, v) in old {
            let w = (v as i64 * c).rem_euclid(m as i64) as u64;
            if w != 0 {
                self.terms.insert(k, w);
            }
        }
    }
    /// Represent the same function mod `2^bits` for a larger `bits`
    /// (coefficients scaled by `2^{bits − r}`).
    pub fn raise_bits(&mut self, bits: u32) {
        assert!(bits >= self.bits && bits <= 32);
        let scale = 1u64 << (bits - self.bits);
        self.constant *= scale;
        for v in self.terms.values_mut() {
            *v *= scale;
        }
        self.bits = bits;
    }
    /// `f(x)` mod `2^r`.
    pub fn eval(&self, x: &[u64]) -> u64 {
        let mut acc = self.constant;
        for (k, c) in &self.terms {
            if k.iter().all(|l| bits::dot(l, x)) {
                acc = (acc + c) % self.modulus();
            }
        }
        acc
    }
    /// The level of a term with `s` parities and coefficient `c`.
    pub fn term_level(c: u64, s: usize, bits: u32) -> u32 {
        debug_assert!(c != 0);
        (s as u32 + bits).saturating_sub(c.trailing_zeros() + 1)
    }
    /// The largest term level (0 without terms).
    pub fn max_level(&self) -> u32 {
        self.terms
            .iter()
            .map(|(k, &c)| Self::term_level(c, k.len(), self.bits))
            .max()
            .unwrap_or(0)
    }
    /// Every term of level at most 2?
    pub fn is_clifford(&self) -> bool {
        self.max_level() <= 2
    }
    /// Does any term involve `x_j`?
    pub fn uses_var(&self, j: usize) -> bool {
        self.terms.keys().any(|k| k.iter().any(|l| bits::get(l, j)))
    }
    /// The variables appearing in some parity.
    pub fn used_vars(&self) -> Bits {
        let mut u = bits::zero(self.n);
        for k in self.terms.keys() {
            for l in k {
                for w in bits::ones(l) {
                    bits::set(&mut u, w, true);
                }
            }
        }
        u
    }
    /// `+ c Π_k (flip_k ? 1 − [μ_k] : [μ_k])`, expanded.
    fn add_flipped_term(&mut self, c: i64, factors: &[(Bits, bool)]) {
        let mut plain = Vec::new();
        let mut flipped = Vec::new();
        for (mu, flip) in factors {
            let mut g = mu.clone();
            bits::ensure_len(&mut g, self.n);
            bits::mask_to(&mut g, self.n);
            if bits::is_zero(&g) {
                if *flip {
                    continue; // factor 1
                }
                return; // factor 0
            }
            if *flip {
                flipped.push(g);
            } else {
                plain.push(g);
            }
        }
        for sub in 0..(1usize << flipped.len()) {
            let mut key = plain.clone();
            let mut sign = 1i64;
            for (i, mu) in flipped.iter().enumerate() {
                if (sub >> i) & 1 == 1 {
                    key.push(mu.clone());
                    sign = -sign;
                }
            }
            self.add_parity_term(sign * c, &key);
        }
    }
    /// The polynomial in new variables `t ∈ 𝔽₂^m` after
    /// `x = shift ⊕ Σ_t t_i · columns_i` (each column an `n`-bit vector).
    pub fn substitute(&self, shift: &[u64], columns: &[Bits], m: usize) -> PhasePoly {
        assert_eq!(columns.len(), m);
        let mut out = PhasePoly::new(m, self.bits);
        out.add_constant(self.constant as i64);
        for (k, c) in &self.terms {
            let factors: Vec<(Bits, bool)> = k
                .iter()
                .map(|l| {
                    let flip = bits::dot(l, shift);
                    let mut mu = bits::zero(m);
                    for (t, col) in columns.iter().enumerate() {
                        if bits::dot(l, col) {
                            bits::toggle(&mut mu, t);
                        }
                    }
                    (mu, flip)
                })
                .collect();
            out.add_flipped_term(*c as i64, &factors);
        }
        out
    }
    /// Substitute `x_k = value ⊕ [λ·x]` (`k ∉ λ`) in place; afterwards `x_k` is unused.
    pub fn substitute_var(&mut self, k: usize, value: bool, lambda: &[u64]) {
        debug_assert!(!bits::get(lambda, k));
        let affected: Vec<Vec<Bits>> = self
            .terms
            .keys()
            .filter(|key| key.iter().any(|l| bits::get(l, k)))
            .cloned()
            .collect();
        for key in affected {
            let c = self.terms.remove(&key).unwrap();
            let factors: Vec<(Bits, bool)> = key
                .iter()
                .map(|l| {
                    if bits::get(l, k) {
                        let mut mu = l.clone();
                        bits::toggle(&mut mu, k);
                        bits::xor_into(&mut mu, lambda);
                        (mu, value)
                    } else {
                        (l.clone(), false)
                    }
                })
                .collect();
            self.add_flipped_term(c as i64, &factors);
        }
    }
    /// `−f`.
    pub fn negated(&self) -> PhasePoly {
        let mut p = self.clone();
        p.scale(-1);
        p
    }
    /// The same polynomial with variable `i` renamed `offset + i`, on `total` variables.
    pub fn embed(&self, offset: usize, total: usize) -> PhasePoly {
        assert!(offset + self.n <= total);
        let mut p = PhasePoly::new(total, self.bits);
        p.constant = self.constant;
        for (k, c) in &self.terms {
            let key: Vec<Bits> = k
                .iter()
                .map(|l| {
                    let mut g = bits::zero(total);
                    for w in bits::ones(l) {
                        bits::set(&mut g, offset + w, true);
                    }
                    g
                })
                .collect();
            p.terms.insert(key, *c);
        }
        p
    }
    /// The same polynomial on `m ≥ n` variables.
    pub fn extend(&self, m: usize) -> PhasePoly {
        assert!(m >= self.n);
        let mut p = PhasePoly::new(m, self.bits);
        p.constant = self.constant;
        for (k, c) in &self.terms {
            let key: Vec<Bits> = k
                .iter()
                .map(|l| {
                    let mut g = l.clone();
                    bits::ensure_len(&mut g, m);
                    g
                })
                .collect();
            p.terms.insert(key, *c);
        }
        p
    }
    /// Keep the variables `keep` (renumbered in that order); the others must be unused.
    pub fn project(&self, keep: &[usize]) -> PhasePoly {
        let mut pos = vec![usize::MAX; self.n];
        for (i, &k) in keep.iter().enumerate() {
            pos[k] = i;
        }
        let mut p = PhasePoly::new(keep.len(), self.bits);
        p.constant = self.constant;
        for (k, c) in &self.terms {
            let key: Vec<Bits> = k
                .iter()
                .map(|l| {
                    let mut g = bits::zero(keep.len());
                    for w in bits::ones(l) {
                        assert!(pos[w] != usize::MAX, "project: dropped variable {w} is used");
                        bits::set(&mut g, pos[w], true);
                    }
                    g
                })
                .collect();
            p.add_parity_term(*c as i64, &key);
        }
        p
    }
    /// Split into the Clifford part, as a ℤ₄ form `q` with `2^{r−2} q`, and
    /// the remaining (level ≥ 3) terms plus the constant.  Needs `r ≥ 2`.
    pub fn split_clifford(&self) -> (QuadForm, PhasePoly) {
        assert!(self.bits >= 2, "split_clifford needs at least 2 phase bits");
        let unit = 1u64 << (self.bits - 2);
        let mut q = QuadForm::new(self.n);
        let mut rest = PhasePoly::new(self.n, self.bits);
        if self.constant % unit == 0 {
            q.add_constant((self.constant / unit) as i64);
        } else {
            rest.add_constant(self.constant as i64);
        }
        for (k, &c) in &self.terms {
            match k.len() {
                1 if c % unit == 0 => q.add_parity((c / unit) as i64, &k[0]),
                2 if c % (2 * unit) == 0 => {
                    if (c / (2 * unit)) % 2 == 1 {
                        q.add_double_product(&k[0], &k[1]);
                    }
                }
                _ => rest.add_parity_term(c as i64, k),
            }
        }
        (q, rest)
    }
    /// Add the Clifford-level terms to `q` in place (as [`PhasePoly::split_clifford`]
    /// would split them, without building a new form) and return the rest.
    pub fn split_clifford_into(&self, q: &mut QuadForm) -> PhasePoly {
        assert!(self.bits >= 2, "split_clifford needs at least 2 phase bits");
        assert_eq!(q.num_vars(), self.n, "split_clifford_into: variable count");
        let unit = 1u64 << (self.bits - 2);
        let mut rest = PhasePoly::new(self.n, self.bits);
        if self.constant % unit == 0 {
            q.add_constant((self.constant / unit) as i64);
        } else {
            rest.add_constant(self.constant as i64);
        }
        for (k, &c) in &self.terms {
            match k.len() {
                1 if c % unit == 0 => q.add_parity((c / unit) as i64, &k[0]),
                2 if c % (2 * unit) == 0 => {
                    if (c / (2 * unit)) % 2 == 1 {
                        q.add_double_product(&k[0], &k[1]);
                    }
                }
                _ => rest.add_parity_term(c as i64, k),
            }
        }
        rest
    }
    /// The polynomial after `x_i = shift_i ⊕ rows_i · y` for `y ∈ 𝔽₂^m`
    /// (`rows` the `n` affine maps as `m`-bit vectors; the transpose of
    /// [`PhasePoly::substitute`]'s columns, cheaper when `n ≪ m`).
    pub fn substitute_rows(&self, shift: &[u64], rows: &[Bits], m: usize) -> PhasePoly {
        assert_eq!(rows.len(), self.n);
        let mut out = PhasePoly::new(m, self.bits);
        out.add_constant(self.constant as i64);
        for (k, c) in &self.terms {
            let factors: Vec<(Bits, bool)> = k
                .iter()
                .map(|l| {
                    let flip = bits::dot(l, shift);
                    let mut mu = bits::zero(m);
                    for i in bits::ones(l) {
                        if i < self.n {
                            bits::xor_into(&mut mu, &rows[i]);
                        }
                    }
                    bits::mask_to(&mut mu, m);
                    (mu, flip)
                })
                .collect();
            out.add_flipped_term(*c as i64, &factors);
        }
        out
    }
    /// `2^{bits−2} q` as a polynomial.
    pub fn from_quadform(q: &QuadForm, bits: u32) -> PhasePoly {
        q.to_phase_poly(bits)
    }
    /// The same polynomial as ncalg's type (for its exponential sums).
    pub fn to_ncalg(&self) -> NcPhasePolynomial {
        let mut p = NcPhasePolynomial::new(self.n, self.bits);
        p.add_constant(self.constant as i64);
        for (k, &c) in &self.terms {
            let forms: Vec<Vec<usize>> = k.iter().map(|l| bits::ones(l)).collect();
            let refs: Vec<&[usize]> = forms.iter().map(|f| f.as_slice()).collect();
            p.add_parity_term(c as i64, &refs);
        }
        p
    }
    /// `Σ_{x ∈ 𝔽₂ⁿ} ζ^{f(x)}` exactly (ncalg), converted to `S`.  Polynomial for
    /// the Clifford class, otherwise exponential in [`ExpSumPlan::width`].
    /// Needs `bits ≤ 7`.
    pub fn exp_sum<S: Amplitude>(&self) -> Result<S> {
        exp_sum_dispatch(&self.to_ncalg(), self.bits)
    }
    /// `Σ_{x ∈ shift + span(basis)} ζ^{f(x)}` (basis vectors independent).
    pub fn exp_sum_over<S: Amplitude>(&self, shift: &[u64], basis: &[Bits]) -> Result<S> {
        self.substitute(shift, basis, basis.len()).exp_sum()
    }
    /// The closed form `0` or `√2^p ζ^k` of a Clifford-class sum; `None` if
    /// some term is not Clifford.
    pub fn exp_sum_phasor(&self) -> Option<Phasor> {
        self.to_ncalg()
            .clifford_sum()
            .map(|cs| Phasor::from_clifford_sum(&cs))
    }
    /// The cost structure of [`exp_sum`](Self::exp_sum).
    pub fn exp_sum_plan(&self) -> ExpSumPlan {
        self.to_ncalg().exp_sum_plan()
    }
}

impl PartialEq for PhasePoly {
    /// Equality as functions `x ↦ ζ^{f(x)}`: the same terms once both are
    /// written mod the larger modulus.
    fn eq(&self, other: &PhasePoly) -> bool {
        if self.n != other.n {
            return false;
        }
        let bits = self.bits.max(other.bits);
        let (a, b) = (1u64 << (bits - self.bits), 1u64 << (bits - other.bits));
        if self.constant * a != other.constant * b || self.terms.len() != other.terms.len() {
            return false;
        }
        self.terms
            .iter()
            .zip(other.terms.iter())
            .all(|((k1, c1), (k2, c2))| k1 == k2 && c1 * a == c2 * b)
    }
}

fn exp_sum_dispatch<S: Amplitude>(p: &NcPhasePolynomial, bits: u32) -> Result<S> {
    fn conv<S: Amplitude, const N: u32>(c: Cyclo<N>) -> Result<S> {
        S::from_cyclo(&c).ok_or_else(|| SimError::NotRepresentable(format!("an element of Q(ζ_{N})")))
    }
    match bits {
        0..=2 => conv(p.exp_sum::<4>()),
        3 => conv(p.exp_sum::<8>()),
        4 => conv(p.exp_sum::<16>()),
        5 => conv(p.exp_sum::<32>()),
        6 => conv(p.exp_sum::<64>()),
        7 => conv(p.exp_sum::<128>()),
        _ => Err(SimError::Unsupported {
            backend: "exp_sum",
            what: format!("exponential sums with {bits} phase bits (at most 7)"),
        }),
    }
}

fn fmt_parity(l: &[u64], n: usize) -> String {
    let vars: Vec<String> = bits::ones(l)
        .into_iter()
        .filter(|&i| i < n)
        .map(|i| format!("x{i}"))
        .collect();
    if vars.len() == 1 {
        vars[0].clone()
    } else {
        format!("[{}]", vars.join("+"))
    }
}

impl fmt::Display for PhasePoly {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let mut parts = Vec::new();
        if self.constant != 0 {
            parts.push(self.constant.to_string());
        }
        for (k, c) in &self.terms {
            let body: Vec<String> = k.iter().map(|l| fmt_parity(l, self.n)).collect();
            if *c == 1 {
                parts.push(body.join("*"));
            } else {
                parts.push(format!("{c}*{}", body.join("*")));
            }
        }
        if parts.is_empty() {
            write!(f, "0 (mod {})", self.modulus())
        } else {
            write!(f, "{} (mod {})", parts.join(" + "), self.modulus())
        }
    }
}

impl fmt::Debug for PhasePoly {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "PhasePoly[{} vars]({})", self.n, self)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::scalar::C64;

    fn all_points(n: usize) -> Vec<Bits> {
        (0..(1u64 << n)).map(|v| bits::from_u64(v, n)).collect()
    }

    #[test]
    fn shear_and_remove_var_match_substitution() {
        use rand::rngs::StdRng;
        use rand::{Rng, SeedableRng};
        let mut rng = StdRng::seed_from_u64(23);
        for trial in 0..300 {
            let n = 2 + trial % 9;
            let mut q = QuadForm::new(n);
            q.add_constant(rng.gen_range(0..4));
            for i in 0..n {
                q.add_linear(i, rng.gen_range(0..4));
                for j in i + 1..n {
                    if rng.gen::<bool>() {
                        q.add_quad(i, j);
                    }
                }
            }
            let t = rng.gen_range(0..n);
            let mut set = bits::zero(n);
            for s in 0..n {
                if s != t && rng.gen::<bool>() {
                    bits::set(&mut set, s, true);
                }
            }
            let mut cols: Vec<Bits> = (0..n).map(|s| bits::unit(n, s)).collect();
            let mut v = set.clone();
            bits::set(&mut v, t, true);
            cols[t] = v;
            let want = q.substitute(&bits::zero(n), &cols, n);
            let mut got = q.clone();
            got.shear(t, &set);
            for x in all_points(n) {
                assert_eq!(got.eval(&x), want.eval(&x), "trial {trial}");
            }
        }
        let mut q = QuadForm::new(4);
        q.add_linear(0, 1);
        q.add_quad(0, 3);
        q.add_linear(2, 3);
        q.remove_var(1);
        assert_eq!(
            (q.num_vars(), q.linear(0), q.linear(1), q.has_quad(0, 2)),
            (3, 1, 3, true)
        );
    }

    #[test]
    fn gauss_sum_by_elimination_matches_ncalg() {
        use rand::rngs::StdRng;
        use rand::{Rng, SeedableRng};
        let mut rng = StdRng::seed_from_u64(41);
        for trial in 0..400 {
            let n = trial % 14;
            let mut q = QuadForm::new(n);
            q.add_constant(rng.gen_range(0..4));
            for i in 0..n {
                q.add_linear(i, rng.gen_range(0..4));
                for j in i + 1..n {
                    if rng.gen::<f64>() < 0.3 {
                        q.add_quad(i, j);
                    }
                }
            }
            // ncalg's Clifford exponential sum as the reference
            let want = q.to_phase_poly(2).exp_sum_phasor().unwrap();
            assert_eq!(q.gauss_sum(), want, "trial {trial}: {q:?}");
        }
    }

    #[test]
    fn substitute_matches_evaluation() {
        use rand::rngs::StdRng;
        use rand::{Rng, SeedableRng};
        let mut rng = StdRng::seed_from_u64(17);
        for trial in 0..300 {
            let n = 1 + trial % 9;
            let m = 1 + (trial / 9) % 7;
            let mut q = QuadForm::new(n);
            q.add_constant(rng.gen_range(0..4));
            for i in 0..n {
                q.add_linear(i, rng.gen_range(0..4));
                for j in i + 1..n {
                    if rng.gen::<bool>() {
                        q.add_quad(i, j);
                    }
                }
            }
            let shift = bits::from_u64(rng.gen_range(0..1u64 << n), n);
            let cols: Vec<Bits> = (0..m)
                .map(|_| bits::from_u64(rng.gen_range(0..1u64 << n), n))
                .collect();
            let r = q.substitute(&shift, &cols, m);
            for t in all_points(m) {
                let mut x = shift.clone();
                for (a, c) in cols.iter().enumerate() {
                    if bits::get(&t, a) {
                        bits::xor_into(&mut x, c);
                    }
                }
                assert_eq!(r.eval(&t), q.eval(&x), "trial {trial}");
            }
        }
    }

    #[test]
    fn quadform_matches_parity_definition() {
        // q = 3[x0+x1+x2] + 2[x0][x1+x2] + 1
        let mut q = QuadForm::new(3);
        q.add_constant(1);
        q.add_parity(3, &bits::from_indices(3, &[0, 1, 2]));
        q.add_double_product(&bits::from_indices(3, &[0]), &bits::from_indices(3, &[1, 2]));
        for x in all_points(3) {
            let p = (bits::popcount(&x) % 2) as i64;
            let a = bits::get(&x, 0) as i64;
            let b = (bits::get(&x, 1) as i64 + bits::get(&x, 2) as i64) % 2;
            let want = (1 + 3 * p + 2 * a * b).rem_euclid(4) as u8;
            assert_eq!(q.eval(&x), want, "x = {:?}", bits::to_u8s(&x, 3));
        }
        // the same through the parity-product form
        let p = q.to_phase_poly(2);
        for x in all_points(3) {
            assert_eq!(p.eval(&x) as u8, q.eval(&x));
        }
        let (q2, rest) = p.split_clifford();
        assert_eq!(q2, q);
        assert!(rest.is_zero());
    }

    #[test]
    fn substitutions_agree_with_evaluation() {
        let mut q = QuadForm::new(4);
        q.add_parity(1, &bits::from_indices(4, &[0, 3]));
        q.add_parity(2, &bits::from_indices(4, &[1]));
        q.add_quad(0, 1);
        q.add_quad(2, 3);
        q.add_constant(3);
        // x = shift ⊕ M t with 2 new variables
        let shift = bits::from_indices(4, &[1, 2]);
        let cols = vec![bits::from_indices(4, &[0, 1]), bits::from_indices(4, &[1, 2, 3])];
        let r = q.substitute(&shift, &cols, 2);
        let p = q.to_phase_poly(3);
        let rp = p.substitute(&shift, &cols, 2);
        for t in all_points(2) {
            let mut x = shift.clone();
            for (i, c) in cols.iter().enumerate() {
                if bits::get(&t, i) {
                    bits::xor_into(&mut x, c);
                }
            }
            assert_eq!(r.eval(&t), q.eval(&x));
            assert_eq!(rp.eval(&t), p.eval(&x));
        }
        // substitute_var: x_3 = 1 ⊕ x0 ⊕ x2
        let mut qs = q.clone();
        let lam = bits::from_indices(4, &[0, 2]);
        qs.substitute_var(3, true, &lam);
        assert!(qs.is_isolated(3));
        let mut ps = p.clone();
        ps.substitute_var(3, true, &lam);
        assert!(!ps.uses_var(3));
        for x in all_points(3) {
            let mut y = x.clone();
            bits::set(&mut y, 3, !bits::dot(&x, &lam));
            assert_eq!(qs.eval(&x), q.eval(&y));
            assert_eq!(ps.eval(&x), p.eval(&y));
        }
        // flip
        let mut qf = q.clone();
        qf.flip(1);
        for x in all_points(4) {
            let mut y = x.clone();
            bits::toggle(&mut y, 1);
            assert_eq!(qf.eval(&x), q.eval(&y));
        }
    }

    #[test]
    fn elimination_matches_brute_force() {
        let mut q = QuadForm::new(3);
        q.add_linear(0, 1);
        q.add_quad(0, 1);
        q.add_quad(0, 2);
        q.add_linear(2, 2);
        // Σ_{x0} i^{q} as a function of (x1, x2), brute force
        let brute = |q: &QuadForm, x1: bool, x2: bool| -> C64 {
            let mut s = C64::new(0.0, 0.0);
            for x0 in [false, true] {
                let x = bits::from_bools(&[x0, x1, x2]);
                s += C64::i_pow(q.eval(&x) as i64);
            }
            s
        };
        let q0 = q.clone();
        match q.eliminate(0) {
            Elimination::Factor(ph) => {
                for (x1, x2) in [(false, false), (true, false), (false, true), (true, true)] {
                    let x = bits::from_bools(&[false, x1, x2]);
                    let got = ph.to_c64() * C64::i_pow(q.eval(&x) as i64);
                    assert!((got - brute(&q0, x1, x2)).norm() < 1e-12);
                }
            }
            other => panic!("expected a factor, got {other:?}"),
        }
        // even case: constraint
        let mut q = QuadForm::new(3);
        q.add_linear(0, 2);
        q.add_quad(0, 1);
        q.add_quad(0, 2);
        let q0 = q.clone();
        match q.eliminate(0) {
            Elimination::Constraint { lambda, value } => {
                assert_eq!(bits::ones(&lambda), vec![1, 2]);
                assert!(value);
                for (x1, x2) in [(false, false), (true, false), (false, true), (true, true)] {
                    let x = bits::from_bools(&[false, x1, x2]);
                    let on = (x1 ^ x2) == value;
                    let got = if on {
                        C64::new(2.0, 0.0) * C64::i_pow(q.eval(&x) as i64)
                    } else {
                        C64::new(0.0, 0.0)
                    };
                    assert!((got - brute(&q0, x1, x2)).norm() < 1e-12);
                }
            }
            other => panic!("expected a constraint, got {other:?}"),
        }
        let mut q = QuadForm::new(1);
        q.add_linear(0, 2);
        assert_eq!(q.eliminate(0), Elimination::Zero);
        let mut q = QuadForm::new(1);
        assert_eq!(q.eliminate(0), Elimination::Factor(Phasor::new(2, 0, 0)));
    }

    #[test]
    fn exp_sums_through_ncalg() {
        // T on every qubit of |+>^3: Σ ζ8^{x0+x1+x2} = (1+ζ8)^3
        let mut p = PhasePoly::new(3, 3);
        for j in 0..3 {
            p.add_monomial(1, &[j]);
        }
        assert!(!p.is_clifford());
        let s: C64 = p.exp_sum().unwrap();
        let z = C64::root_of_unity(1, 3).unwrap();
        let want = (C64::new(1.0, 0.0) + z).powi(3);
        assert!((s - want).norm() < 1e-12);
        let e: Cyclo<8> = p.exp_sum().unwrap();
        assert!((e.to_complex() - want).norm() < 1e-12);
        assert!(p.exp_sum::<Cyclo<4>>().is_err());
        // a Clifford sum: Σ i^{x0 + 2 x0 x1} = 2 + (1+i)·... check against brute force and the phasor
        let mut q = QuadForm::new(2);
        q.add_linear(0, 1);
        q.add_quad(0, 1);
        let ph = q.gauss_sum();
        let mut brute = C64::new(0.0, 0.0);
        for x in all_points(2) {
            brute += C64::i_pow(q.eval(&x) as i64);
        }
        assert!((ph.to_c64() - brute).norm() < 1e-12);
    }
}
