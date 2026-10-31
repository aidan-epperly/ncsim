//! Dense state vectors: the reference backend, exact in any [`Amplitude`]
//! type, exponential in the number of qubits.

use super::QuantumState;
use crate::bits::{self, Bits};
use crate::error::{check_qubits, Result, SimError};
use crate::gate::{Gate, MonomialGate};
use crate::pauli::Pauli;
use crate::scalar::Amplitude;
use rand::{Rng, RngCore};
use rayon::prelude::*;
use std::fmt;

/// `2^n` amplitudes, little-endian: qubit `q` is bit `q` of the index.
#[derive(Clone, PartialEq)]
pub struct StateVector<S: Amplitude> {
    n: usize,
    amp: Vec<S>,
}

const BACKEND: &str = "StateVector";

impl<S: Amplitude> StateVector<S> {
    /// `|0…0⟩`.
    pub fn zero_state(n: usize) -> Self {
        Self::basis_state(n, &bits::zero(n))
    }
    /// `|x⟩`.
    pub fn basis_state(n: usize, x: &[u64]) -> Self {
        assert!(n <= 30, "state vector of more than 30 qubits");
        let mut amp = vec![S::zero(); 1usize << n];
        amp[bits::to_u64(x, n) as usize] = S::one();
        StateVector { n, amp }
    }
    /// `|+⟩^{⊗n}` (needs `2^{-n/2}` in `S`).
    pub fn plus_state(n: usize) -> Result<Self> {
        let c = S::sqrt2_pow(-(n as i64)).ok_or_else(|| SimError::NotRepresentable(format!("2^(-{n}/2)")))?;
        Ok(StateVector {
            n,
            amp: vec![c; 1usize << n],
        })
    }
    /// From amplitudes (length `2^n`).
    pub fn from_amplitudes(n: usize, amp: Vec<S>) -> Result<Self> {
        if amp.len() != 1usize << n {
            return Err(SimError::Invalid(format!(
                "{} amplitudes for {n} qubits",
                amp.len()
            )));
        }
        Ok(StateVector { n, amp })
    }
    /// The amplitudes.
    pub fn amplitudes(&self) -> &[S] {
        &self.amp
    }
    /// Mutable amplitudes.
    pub fn amplitudes_mut(&mut self) -> &mut [S] {
        &mut self.amp
    }
    /// The index of basis bits `x`.
    pub fn index(x: &[u64], n: usize) -> usize {
        bits::to_u64(x, n) as usize
    }

    /// Apply a dense `2^k × 2^k` matrix on `qubits`.
    pub fn apply_matrix(&mut self, m: &[S], qubits: &[usize]) -> Result<()> {
        check_qubits(qubits, self.n)?;
        let k = qubits.len();
        let gd = 1usize << k;
        assert_eq!(m.len(), gd * gd, "matrix size");
        let nz: Vec<Vec<(usize, S)>> = (0..gd)
            .map(|a| {
                (0..gd)
                    .filter(|&b| !m[a * gd + b].is_exactly_zero())
                    .map(|b| (b, m[a * gd + b].clone()))
                    .collect()
            })
            .collect();
        for_each_fiber(&mut self.amp, qubits, &|_, v: &mut [S], w: &mut [S]| {
            for (a, row) in nz.iter().enumerate() {
                let mut acc = S::zero();
                for (b, c) in row {
                    if !v[*b].is_exactly_zero() {
                        acc.add_assign_ref(&c.mul_ref(&v[*b]));
                    }
                }
                w[a] = acc;
            }
            v.clone_from_slice(w);
        });
        Ok(())
    }

    /// Apply a monomial gate on `qubits`: a permutation with phases, no matrix.
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
        let gd = 1usize << k;
        let mut images = Vec::with_capacity(gd);
        for x in 0..gd {
            let (s, y) = g.apply_to_basis(&bits::from_u64(x as u64, k));
            let c = S::from_phasor(s).ok_or_else(|| SimError::NotRepresentable(format!("phase {s}")))?;
            images.push((c, bits::to_u64(&y, k) as usize));
        }
        let diagonal = images.iter().enumerate().all(|(x, (_, y))| x == *y);
        let one = S::one();
        let phased: Vec<(usize, S)> = images
            .iter()
            .enumerate()
            .filter(|(_, (c, _))| *c != one)
            .map(|(x, (c, _))| (x, c.clone()))
            .collect();
        let unit: Vec<bool> = images.iter().map(|(c, _)| *c == one).collect();
        if diagonal {
            for_each_fiber(&mut self.amp, qubits, &|_, v: &mut [S], _w: &mut [S]| {
                for &(x, ref c) in &phased {
                    if !v[x].is_exactly_zero() {
                        v[x] = c.mul_ref(&v[x]);
                    }
                }
            });
        } else {
            for_each_fiber(&mut self.amp, qubits, &|_, v: &mut [S], w: &mut [S]| {
                for (x, (c, y)) in images.iter().enumerate() {
                    w[*y] = if unit[x] { v[x].clone() } else { c.mul_ref(&v[x]) };
                }
                v.clone_from_slice(w);
            });
        }
        Ok(())
    }

    /// `ψ ← P ψ` for a Pauli `P` (any phase), in place: `O(2^n)` with no
    /// buffer, whatever the weight of `P`.
    pub fn apply_pauli(&mut self, p: &Pauli) {
        assert_eq!(p.num_qubits(), self.n, "apply_pauli: register size");
        let (xm, zm, ph) = pauli_masks(p);
        let ipow: Vec<S> = (0..4).map(|k| S::i_pow(k)).collect();
        let k_of = |c: usize| (ph as usize + 2 * (zm & c).count_ones() as usize) & 3;
        if xm == 0 {
            self.amp.par_iter_mut().enumerate().for_each(|(c, a)| {
                let k = k_of(c);
                if k != 0 && !a.is_exactly_zero() {
                    *a = ipow[k].mul_ref(a);
                }
            });
        } else {
            for_each_pair(&mut self.amp, xm, &|c: usize, a: &mut S, b: &mut S| {
                // (Pψ)(c) = i^{k(c ⊕ x)} ψ(c ⊕ x)
                let na = ipow[k_of(c ^ xm)].mul_ref(b);
                let nb = ipow[k_of(c)].mul_ref(a);
                *a = na;
                *b = nb;
            });
        }
    }

    /// `self ⊗ other` (`self` on the low qubits): amplitude
    /// `ψ(x_a) φ(x_b)` at index `x_a + 2^{n_a} x_b`.
    pub fn kron(&self, other: &StateVector<S>) -> Result<StateVector<S>> {
        let n = self.n + other.n;
        if n > 30 {
            return Err(SimError::Invalid(format!("state vector of {n} qubits")));
        }
        let la = self.amp.len();
        let mut amp = vec![S::zero(); 1usize << n];
        amp.par_chunks_mut(la)
            .zip(other.amp.par_iter())
            .for_each(|(chunk, b)| {
                if b.is_exactly_zero() {
                    return;
                }
                for (o, a) in chunk.iter_mut().zip(&self.amp) {
                    if !a.is_exactly_zero() {
                        *o = a.mul_ref(b);
                    }
                }
            });
        Ok(StateVector { n, amp })
    }
}

/// Elements per sequential block of the dense kernels.
const BLOCK: usize = 1 << 12;

/// Call `f(base, v, scratch)` on every *fiber* of `amp` over `qubits`: the
/// `2^k` amplitudes whose indices agree off the gate qubits, in local
/// little-endian order (`v[b]` has bit `j` of `b` on `qubits[j]`); `base` is
/// the global index with the gate bits zero.  `f` leaves its result in `v`.
/// Safe, allocation-free per fiber and parallel: the gate qubits whose
/// blocks are large split the vector into independent slices first, the
/// rest is handled in blocks of `BLOCK` elements.
fn for_each_fiber<S, F>(amp: &mut [S], qubits: &[usize], f: &F)
where
    S: Amplitude,
    F: Fn(usize, &mut [S], &mut [S]) + Sync,
{
    let mut order: Vec<(usize, usize)> = qubits.iter().enumerate().map(|(j, &q)| (q, j)).collect();
    order.sort_unstable_by_key(|&(q, _)| std::cmp::Reverse(q));
    let n_high = order.iter().take_while(|(q, _)| (2usize << q) > BLOCK).count();
    split_fibers(
        vec![(0usize, amp)],
        0,
        &order[..n_high],
        &order[n_high..],
        qubits.len(),
        f,
    );
}

fn split_fibers<S, F>(
    slices: Vec<(usize, &mut [S])>,
    start: usize,
    high: &[(usize, usize)],
    low: &[(usize, usize)],
    k: usize,
    f: &F,
) where
    S: Amplitude,
    F: Fn(usize, &mut [S], &mut [S]) + Sync,
{
    let len = slices[0].1.len();
    let (chunk, rest) = match high.split_first() {
        Some((&(h, _), rest)) => (2usize << h, rest),
        None => (BLOCK.min(len), high),
    };
    let nch = len / chunk;
    let mut items: Vec<Vec<(usize, &mut [S])>> =
        (0..nch).map(|_| Vec::with_capacity(2 * slices.len())).collect();
    for (b, s) in slices {
        for (c, ch) in s.chunks_mut(chunk).enumerate() {
            match high.first() {
                Some(&(_, j)) => {
                    let (lo, hi) = ch.split_at_mut(chunk / 2);
                    items[c].push((b, lo));
                    items[c].push((b | (1 << j), hi));
                }
                None => items[c].push((b, ch)),
            }
        }
    }
    if high.is_empty() {
        items
            .into_par_iter()
            .enumerate()
            .for_each(|(c, sl)| fiber_leaf(sl, start + c * chunk, low, k, f));
    } else {
        items
            .into_par_iter()
            .enumerate()
            .for_each(|(c, sl)| split_fibers(sl, start + c * chunk, rest, low, k, f));
    }
}

fn fiber_leaf<S, F>(mut sl: Vec<(usize, &mut [S])>, start: usize, low: &[(usize, usize)], k: usize, f: &F)
where
    S: Amplitude,
    F: Fn(usize, &mut [S], &mut [S]) + Sync,
{
    let len = sl[0].1.len();
    let mask: usize = low.iter().map(|(q, _)| 1usize << q).sum();
    // offsets and local bits of the low patterns
    let pats: Vec<(usize, usize)> = (0..1usize << low.len())
        .map(|p| {
            let mut off = 0;
            let mut loc = 0;
            for (i, &(q, j)) in low.iter().enumerate() {
                if (p >> i) & 1 == 1 {
                    off |= 1 << q;
                    loc |= 1 << j;
                }
            }
            (off, loc)
        })
        .collect();
    let mut v = vec![S::zero(); 1 << k];
    let mut w = vec![S::zero(); 1 << k];
    let mut i = 0usize;
    while i < len {
        for (b1, s) in sl.iter() {
            for &(off, loc) in &pats {
                v[*b1 | loc] = s[i | off].clone();
            }
        }
        f(start + i, &mut v, &mut w);
        for (b1, s) in sl.iter_mut() {
            for &(off, loc) in &pats {
                std::mem::swap(&mut s[i | off], &mut v[*b1 | loc]);
            }
        }
        i = ((i | mask) + 1) & !mask;
    }
}

/// Call `f(c, ψ_c, ψ_{c⊕x})` on every pair `{c, c ⊕ x}` (`x ≠ 0`; `c` the
/// member with the highest bit `h` of `x` clear), in place and in parallel:
/// blocks of the lower half of each `2^{h+1}` chunk meet the blocks of the
/// upper half that `⊕ x` maps them to.
fn for_each_pair<S, F>(amp: &mut [S], x: usize, f: &F)
where
    S: Amplitude,
    F: Fn(usize, &mut S, &mut S) + Sync,
{
    debug_assert!(x != 0);
    let h = (usize::BITS - 1 - x.leading_zeros()) as usize;
    let half = 1usize << h;
    let xl = x & (half - 1);
    // xl < span ≤ b, both powers of two: i ⊕ xl stays inside its aligned block of b
    let span = if xl == 0 { 1 } else { (xl + 1).next_power_of_two() };
    let b = span.max(BLOCK.min(half));
    amp.par_chunks_mut(2 * half).enumerate().for_each(|(ci, chunk)| {
        let (lo, hi) = chunk.split_at_mut(half);
        lo.par_chunks_mut(b)
            .zip(hi.par_chunks_mut(b))
            .enumerate()
            .for_each(|(bi, (lb, hb))| {
                let base = ci * 2 * half + bi * b;
                for i in 0..lb.len() {
                    f(base + i, &mut lb[i], &mut hb[i ^ xl]);
                }
            });
    });
}

/// `Σ_{lo ≤ i < hi} f(i)`, pairwise (stable as [`Amplitude::sum`]) and parallel.
fn par_sum<S: Amplitude, F: Fn(usize) -> S + Sync>(lo: usize, hi: usize, f: &F) -> S {
    let len = hi - lo;
    if len <= 8 {
        let mut acc = S::zero();
        for i in lo..hi {
            acc.add_assign_ref(&f(i));
        }
        return acc;
    }
    let mid = lo + len / 2;
    let (a, b) = if len >= 1 << 14 {
        rayon::join(|| par_sum(lo, mid, f), || par_sum(mid, hi, f))
    } else {
        (par_sum(lo, mid, f), par_sum(mid, hi, f))
    };
    a.add_ref(&b)
}

/// `(x mask, z mask, phase)` of a Pauli on at most 64 qubits.
fn pauli_masks(p: &Pauli) -> (usize, usize, u8) {
    (p.x_bits()[0] as usize, p.z_bits()[0] as usize, p.phase())
}

impl<S: Amplitude> QuantumState<S> for StateVector<S> {
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
                if let crate::circuit::Op::Gate { gate: g, qubits: qs } = op {
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
        let a = &self.amp;
        Ok(par_sum(0, a.len(), &|i| {
            if a[i].is_exactly_zero() {
                S::zero()
            } else {
                a[i].abs_sq()
            }
        }))
    }

    fn amplitude(&self, x: &[u64]) -> Result<S> {
        Ok(self.amp[bits::to_u64(x, self.n) as usize].clone())
    }

    fn expectation(&self, p: &Pauli) -> Result<S> {
        if p.num_qubits() != self.n {
            return Err(SimError::Invalid(
                "expectation: operator on the wrong register".into(),
            ));
        }
        let (xm, zm, ph) = pauli_masks(p);
        let a = &self.amp;
        let ipow: Vec<S> = (0..4).map(|k| S::i_pow(k)).collect();
        // ⟨ψ|P|ψ⟩ = Σ_c conj ψ(c ⊕ x) i^{ph + 2|z ∧ c|} ψ(c)
        Ok(par_sum(0, a.len(), &|c| {
            if a[c].is_exactly_zero() || a[c ^ xm].is_exactly_zero() {
                return S::zero();
            }
            let k = (ph as usize + 2 * (zm & c).count_ones() as usize) & 3;
            a[c ^ xm].conj().mul_ref(&ipow[k].mul_ref(&a[c]))
        }))
    }

    fn inner(&self, other: &Self) -> Result<S> {
        if other.n != self.n {
            return Err(SimError::Invalid("inner: register sizes differ".into()));
        }
        let (a, b) = (&self.amp, &other.amp);
        Ok(par_sum(0, a.len(), &|i| {
            if a[i].is_exactly_zero() || b[i].is_exactly_zero() {
                S::zero()
            } else {
                a[i].conj().mul_ref(&b[i])
            }
        }))
    }

    fn project(&mut self, p: &Pauli, outcome: bool) -> Result<S> {
        if !p.is_hermitian() {
            return Err(SimError::Invalid(format!("projector of non-Hermitian {p}")));
        }
        let (xm, zm, ph) = pauli_masks(p);
        let half = S::one().add_ref(&S::one()).inverse();
        let sgn: Vec<S> = (0..4)
            .map(|k| {
                let c = S::i_pow(k);
                if outcome {
                    c.negated()
                } else {
                    c
                }
            })
            .collect();
        // (Pψ)(c) = i^{ph + 2|z ∧ (c ⊕ x)|} ψ(c ⊕ x): pairs {c, c ⊕ x} in place
        let k_of = |c: usize| (ph as usize + 2 * (zm & c).count_ones() as usize) & 3;
        if xm == 0 {
            self.amp.par_iter_mut().enumerate().for_each(|(c, a)| {
                if !a.is_exactly_zero() {
                    *a = a.add_ref(&sgn[k_of(c)].mul_ref(a)).mul_ref(&half);
                }
            });
        } else {
            for_each_pair(&mut self.amp, xm, &|c: usize, a: &mut S, b: &mut S| {
                let d = c ^ xm;
                let na = a.add_ref(&sgn[k_of(d)].mul_ref(b)).mul_ref(&half);
                let nb = b.add_ref(&sgn[k_of(c)].mul_ref(a)).mul_ref(&half);
                *a = na;
                *b = nb;
            });
        }
        self.norm_sq()
    }

    fn scale(&mut self, c: &S) -> Result<()> {
        self.amp.par_iter_mut().for_each(|a| {
            if !a.is_exactly_zero() {
                *a = a.mul_ref(c);
            }
        });
        Ok(())
    }

    /// Sampling in one pass over the amplitudes: block weights (in parallel),
    /// the shots' uniforms sorted, each block scanned once for the shots that
    /// land in it.  `O(2^n + shots log shots)` time, `O(2^n / BLOCK + shots)`
    /// memory (no cumulative array of `2^n` floats).
    fn sample(&self, rng: &mut dyn RngCore, shots: usize) -> Result<Vec<Bits>> {
        let a = &self.amp;
        let bl = BLOCK.min(a.len());
        let weights: Vec<f64> = a
            .par_chunks(bl)
            .map(|c| c.iter().map(|x| x.abs_sq_f64()).sum::<f64>())
            .collect();
        let total: f64 = weights.iter().sum();
        if total <= 0.0 {
            return Err(SimError::ZeroState("sample".into()));
        }
        let mut us: Vec<(f64, usize)> = (0..shots).map(|k| (rng.gen::<f64>() * total, k)).collect();
        us.sort_unstable_by(|x, y| x.0.total_cmp(&y.0));
        let mut out = vec![Bits::new(); shots];
        let mut next = 0usize;
        let mut acc = 0.0;
        for (b, w) in weights.iter().enumerate() {
            if next == shots {
                break;
            }
            let hi = acc + w;
            let last_block = b + 1 == weights.len();
            if us[next].0 < hi || last_block {
                // walk the block for the shots below `hi`
                let mut run = acc;
                let start = b * bl;
                let end = (start + bl).min(a.len());
                let mut idx = start;
                while next < shots && (us[next].0 < hi || last_block) {
                    let u = us[next].0;
                    while idx + 1 < end && run + a[idx].abs_sq_f64() <= u {
                        run += a[idx].abs_sq_f64();
                        idx += 1;
                    }
                    // never return an index of zero weight (rounding at block ends)
                    let mut pick = idx;
                    while a[pick].is_exactly_zero() && pick > start {
                        pick -= 1;
                    }
                    out[us[next].1] = bits::from_u64(pick as u64, self.n);
                    next += 1;
                }
            }
            acc = hi;
        }
        Ok(out)
    }

    fn to_vector(&self) -> Result<Vec<S>> {
        Ok(self.amp.clone())
    }

    fn tensor(&self, other: &Self) -> Result<Self> {
        self.kron(other)
    }
}

impl<S: Amplitude> fmt::Debug for StateVector<S> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "StateVector({} qubits)", self.n)?;
        if self.n <= 4 {
            for (i, a) in self.amp.iter().enumerate() {
                if !a.is_exactly_zero() {
                    write!(
                        f,
                        " |{}⟩: {}",
                        bits::to_string(&bits::from_u64(i as u64, self.n), self.n),
                        a
                    )?;
                }
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::circuit::Circuit;
    use crate::scalar::C64;
    use crate::Cyclo;
    use ncalg::Coeff;
    use rand::rngs::StdRng;
    use rand::SeedableRng;

    #[test]
    fn ghz_exact_and_float() {
        let mut c = Circuit::new(3);
        c.h(0).cx(0, 1).cx(1, 2);
        let mut e: StateVector<Cyclo<8>> = StateVector::zero_state(3);
        let mut f: StateVector<C64> = StateVector::zero_state(3);
        let mut rng = StdRng::seed_from_u64(1);
        e.run(&c, &mut rng).unwrap();
        f.run(&c, &mut rng).unwrap();
        let r = Cyclo::<8>::sqrt2_pow(-1).unwrap();
        assert_eq!(e.amplitude(&bits::from_u64(0, 3)).unwrap(), r);
        assert_eq!(e.amplitude(&bits::from_u64(7, 3)).unwrap(), r);
        assert!(e.amplitude(&bits::from_u64(1, 3)).unwrap().is_zero());
        for (a, b) in e.to_vector().unwrap().iter().zip(f.to_vector().unwrap()) {
            assert!((a.to_complex() - b).norm() < 1e-14);
        }
        // ⟨X X X⟩ = 1, ⟨Z Z I⟩ = 1, ⟨Z I I⟩ = 0
        let xxx = Pauli::parse("X0*X1*X2").unwrap();
        assert!(e.expectation(&xxx).unwrap().is_one());
        assert!(e
            .expectation(&Pauli::from_string(3, "Z0*Z1").unwrap())
            .unwrap()
            .is_one());
        assert!(e
            .expectation(&Pauli::from_string(3, "Z0").unwrap())
            .unwrap()
            .is_zero());
        // measurement statistics of Z0: half 0, half 1, and the other qubits follow
        let mut zeros = 0;
        for _ in 0..200 {
            let mut s = f.clone();
            let b = s
                .measure(&Pauli::from_string(3, "Z0").unwrap(), &mut rng)
                .unwrap();
            let b1 = s
                .measure(&Pauli::from_string(3, "Z1").unwrap(), &mut rng)
                .unwrap();
            assert_eq!(b, b1);
            if !b {
                zeros += 1;
            }
        }
        assert!((60..=140).contains(&zeros), "{zeros}");
        let samples = f.sample(&mut rng, 100).unwrap();
        assert!(samples
            .iter()
            .all(|x| bits::to_u64(x, 3) == 0 || bits::to_u64(x, 3) == 7));
        // projection and renormalization
        let mut p = e.clone();
        let w = p.project(&Pauli::from_string(3, "Z0").unwrap(), true).unwrap();
        assert_eq!(w.to_complex().re, 0.5);
        p.normalize().unwrap();
        assert!(p.amplitude(&bits::from_u64(7, 3)).unwrap().is_one());
    }

    #[test]
    fn non_clifford_matches_matrix() {
        let mut c = Circuit::new(2);
        c.h(0)
            .h(1)
            .t(0)
            .gate(Gate::Cs, &[0, 1])
            .gate(Gate::SqrtX, &[1])
            .gate(Gate::Rz(0.7), &[0]);
        let mut s: StateVector<C64> = StateVector::zero_state(2);
        let mut rng = StdRng::seed_from_u64(0);
        s.run(&c, &mut rng).unwrap();
        let u: Vec<C64> = c.unitary().unwrap();
        for i in 0..4 {
            assert!((s.amplitudes()[i] - u[i * 4]).norm() < 1e-12);
        }
        // exact scalars refuse the rotation
        let mut e: StateVector<Cyclo<8>> = StateVector::zero_state(2);
        assert!(matches!(e.run(&c, &mut rng), Err(SimError::NotRepresentable(_))));
    }
}

#[cfg(test)]
mod kernel_tests {
    use super::*;
    use crate::scalar::C64;
    use rand::rngs::StdRng;
    use rand::{Rng, SeedableRng};

    /// The obvious `O(4^k 2^n)` application, for checking the fiber kernel.
    fn naive_apply(amp: &[C64], m: &[C64], qubits: &[usize]) -> Vec<C64> {
        let k = qubits.len();
        let gd = 1usize << k;
        let local =
            |i: usize| -> usize { qubits.iter().enumerate().map(|(j, &q)| ((i >> q) & 1) << j).sum() };
        let with = |i: usize, b: usize| -> usize {
            let mut r = i;
            for (j, &q) in qubits.iter().enumerate() {
                r = (r & !(1 << q)) | (((b >> j) & 1) << q);
            }
            r
        };
        (0..amp.len())
            .map(|i| {
                let a = local(i);
                (0..gd).map(|b| m[a * gd + b] * amp[with(i, b)]).sum()
            })
            .collect()
    }

    #[test]
    fn fiber_kernel_matches_naive_on_high_and_low_qubits() {
        let mut rng = StdRng::seed_from_u64(8);
        let n = 15;
        let mut v: StateVector<C64> = StateVector::zero_state(n);
        for a in v.amplitudes_mut() {
            *a = C64::new(rng.gen::<f64>() - 0.5, rng.gen::<f64>() - 0.5);
        }
        for trial in 0..12 {
            let k = 1 + trial % 3;
            let mut qs: Vec<usize> = Vec::new();
            while qs.len() < k {
                // mix qubits above and below the block size
                let q = if rng.gen::<bool>() {
                    rng.gen_range(12..n)
                } else {
                    rng.gen_range(0..12)
                };
                if !qs.contains(&q) {
                    qs.push(q);
                }
            }
            let gd = 1usize << k;
            let m: Vec<C64> = (0..gd * gd).map(|_| C64::new(rng.gen(), rng.gen())).collect();
            let want = naive_apply(v.amplitudes(), &m, &qs);
            v.apply_matrix(&m, &qs).unwrap();
            let err: f64 = v
                .amplitudes()
                .iter()
                .zip(&want)
                .map(|(a, b)| (a - b).norm())
                .fold(0.0, f64::max);
            assert!(err < 1e-9, "trial {trial} on {qs:?}: {err}");
            let s = v.norm_sq().unwrap().re.sqrt();
            v.scale(&C64::new(1.0 / s, 0.0)).unwrap();
        }
        // Pauli expectation and projection against the dense Pauli matrix action
        for trial in 0..10 {
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
            let p = if trial % 3 == 0 { p.negated() } else { p };
            let pv = crate::clifford::apply_pauli_dense(&p, v.amplitudes());
            let want: C64 = v.amplitudes().iter().zip(&pv).map(|(a, b)| a.conj() * b).sum();
            assert!((v.expectation(&p).unwrap() - want).norm() < 1e-9, "trial {trial}");
            // the Pauli as a gate, in place
            let mut u = v.clone();
            u.apply_pauli(&p);
            assert!(
                u.amplitudes()
                    .iter()
                    .zip(&pv)
                    .all(|(a, b)| (a - b).norm() < 1e-12),
                "trial {trial}: P ψ"
            );
            let mut w = v.clone();
            let b = rng.gen::<bool>();
            w.project(&p, b).unwrap();
            let sgn = if b { -1.0 } else { 1.0 };
            for (i, x) in w.amplitudes().iter().enumerate() {
                let e = (v.amplitudes()[i] + pv[i] * sgn) * 0.5;
                assert!((x - e).norm() < 1e-12, "trial {trial} index {i}");
            }
        }
    }
}

#[cfg(test)]
mod sample_tests {
    use super::*;
    use crate::scalar::C64;
    use rand::rngs::StdRng;
    use rand::SeedableRng;

    #[test]
    fn one_pass_sampling_follows_the_born_rule_across_blocks() {
        let mut rng = StdRng::seed_from_u64(2);
        let n = 14;
        let mut v: StateVector<C64> = StateVector::zero_state(n);
        // weight concentrated on a few indices in different blocks, the rest small
        for (i, a) in v.amplitudes_mut().iter_mut().enumerate() {
            *a = C64::new(
                if i % 4099 == 7 {
                    3.0
                } else {
                    0.01 * ((i % 13) as f64)
                },
                0.0,
            );
        }
        let shots = 200_000;
        let s = v.sample(&mut rng, shots).unwrap();
        let total: f64 = v.amplitudes().iter().map(|a| a.norm_sqr()).sum();
        let mut counts = std::collections::HashMap::new();
        for x in &s {
            *counts.entry(bits::to_u64(x, n) as usize).or_insert(0usize) += 1;
        }
        for i in (7..1 << n).step_by(4099) {
            let p = v.amplitudes()[i].norm_sqr() / total;
            let f = *counts.get(&i).unwrap_or(&0) as f64 / shots as f64;
            let sd = (p * (1.0 - p) / shots as f64).sqrt();
            assert!((f - p).abs() < 6.0 * sd, "index {i}: {f} vs {p}");
        }
        // never a zero-amplitude index
        assert!(s
            .iter()
            .all(|x| bits::to_u64(x, n) % 13 != 0 || bits::to_u64(x, n) % 4099 == 7));
    }
}
