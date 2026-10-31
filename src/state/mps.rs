//! Matrix product states: approximate simulation of low-entanglement states
//! in `Complex<f64>`, with faer's thin SVD and QR.
//!
//! Site `q` holds two `χ_q × χ_{q+1}` matrices `A_q^0`, `A_q^1`
//! (`χ_0 = χ_n = 1`) and `⟨x|ψ⟩ = A_0^{x_0} A_1^{x_1} ⋯ A_{n−1}^{x_{n−1}}`.  A
//! mixed canonical form is kept: sites left of the *center* are
//! left-canonical, sites right of it right-canonical, so an SVD truncation at
//! the center is optimal for that bond.  Every multi-qubit gate goes through
//! one routine, [`Mps::apply_matrix`]: the targets are brought together with
//! nearest-neighbor swaps (themselves two-site gates), the contiguous block is
//! contracted, the dense gate applied, and the block split again by SVDs,
//! truncating by [`MpsOptions`] and accumulating the discarded weight.
//! Observables are transfer-matrix contractions, `O(n χ³)`.

use super::QuantumState;
use crate::bits::{self, Bits};
use crate::circuit::Op;
use crate::error::{check_qubits, unsupported, Result, SimError};
use crate::gate::Gate;
use crate::pauli::Pauli;
use crate::scalar::C64;
use faer::Mat;
use rand::{Rng, RngCore};
use std::fmt;

const BACKEND: &str = "Mps";

type M = Mat<C64>;

/// Truncation options.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct MpsOptions {
    /// Largest bond dimension kept (`None`: unlimited).
    pub max_bond: Option<usize>,
    /// Singular values below `cutoff · σ_max` are dropped.
    pub cutoff: f64,
}

impl Default for MpsOptions {
    fn default() -> Self {
        MpsOptions {
            max_bond: None,
            cutoff: 1e-12,
        }
    }
}

/// A matrix product state (see the [module documentation](self)).
#[derive(Clone)]
pub struct Mps {
    n: usize,
    tensors: Vec<[M; 2]>,
    center: usize,
    options: MpsOptions,
    discarded: f64,
}

fn zero_mat(r: usize, c: usize) -> M {
    Mat::zeros(r, c)
}

fn mat_mul(a: &M, b: &M) -> M {
    a * b
}

/// `[a; b]`.
fn stack_rows(a: &M, b: &M) -> M {
    assert_eq!(a.ncols(), b.ncols());
    Mat::from_fn(a.nrows() + b.nrows(), a.ncols(), |i, j| {
        if i < a.nrows() {
            a[(i, j)]
        } else {
            b[(i - a.nrows(), j)]
        }
    })
}

/// `[a | b]`.
fn concat_cols(a: &M, b: &M) -> M {
    assert_eq!(a.nrows(), b.nrows());
    Mat::from_fn(a.nrows(), a.ncols() + b.ncols(), |i, j| {
        if j < a.ncols() {
            a[(i, j)]
        } else {
            b[(i, j - a.ncols())]
        }
    })
}

/// Thin SVD `B = U diag(s) V†` with the singular values as `f64`s.
fn thin_svd(b: &M) -> Result<(M, Vec<f64>, M)> {
    let svd = b
        .thin_svd()
        .map_err(|e| SimError::Invalid(format!("SVD failed: {e:?}")))?;
    let s = svd.S().column_vector();
    let sv: Vec<f64> = (0..s.nrows()).map(|i| s[i].re).collect();
    Ok((svd.U().to_owned(), sv, svd.V().to_owned()))
}

impl Mps {
    /// `|0…0⟩`.
    pub fn zero_state(n: usize) -> Self {
        Self::basis_state(n, &bits::zero(n))
    }
    /// `|x⟩`.
    pub fn basis_state(n: usize, x: &[u64]) -> Self {
        assert!(n >= 1, "an MPS needs at least one qubit");
        let tensors = (0..n)
            .map(|q| {
                let b = bits::get(x, q);
                let one = Mat::from_fn(1, 1, |_, _| C64::new(1.0, 0.0));
                if b {
                    [zero_mat(1, 1), one]
                } else {
                    [one, zero_mat(1, 1)]
                }
            })
            .collect();
        Mps {
            n,
            tensors,
            center: 0,
            options: MpsOptions::default(),
            discarded: 0.0,
        }
    }
    /// Set the truncation options.
    pub fn with_options(mut self, options: MpsOptions) -> Self {
        self.options = options;
        self
    }
    /// The truncation options.
    pub fn options(&self) -> MpsOptions {
        self.options
    }
    /// From a dense vector (little-endian), by successive SVDs; exact up to the cutoff.
    pub fn from_vector(n: usize, v: &[C64]) -> Result<Self> {
        if v.len() != 1usize << n {
            return Err(SimError::Invalid(format!(
                "{} amplitudes for {n} qubits",
                v.len()
            )));
        }
        let mut mps = Mps::zero_state(n);
        // remaining tensor: rows (left bond), columns (the remaining qubits' bits)
        let mut rest: M = Mat::from_fn(1, v.len(), |_, j| v[j]);
        for q in 0..n {
            let l = rest.nrows();
            let cols = rest.ncols();
            if q == n - 1 {
                mps.tensors[q] = [
                    Mat::from_fn(l, 1, |i, _| rest[(i, 0)]),
                    Mat::from_fn(l, 1, |i, _| rest[(i, 1)]),
                ];
                break;
            }
            // reshape to (2l) × (cols/2): row (s, i) ↔ column index s + 2·rest_bits
            let half = cols / 2;
            let b: M = Mat::from_fn(2 * l, half, |r, c| {
                let (s, i) = (r / l, r % l);
                rest[(i, s + 2 * c)]
            });
            let (u, s, vmat) = thin_svd(&b)?;
            let k = mps.keep_count(&s);
            mps.tensors[q] = [
                Mat::from_fn(l, k, |i, j| u[(i, j)]),
                Mat::from_fn(l, k, |i, j| u[(l + i, j)]),
            ];
            rest = Mat::from_fn(k, half, |i, c| vmat[(c, i)].conj() * s[i]);
        }
        mps.center = n - 1;
        Ok(mps)
    }
    /// Number of qubits.
    pub fn num_qubits(&self) -> usize {
        self.n
    }
    /// The two matrices of site `q`.
    pub fn site(&self, q: usize) -> &[M; 2] {
        &self.tensors[q]
    }
    /// Bond dimensions `χ_1, …, χ_{n−1}`.
    pub fn bond_dimensions(&self) -> Vec<usize> {
        (0..self.n - 1).map(|q| self.tensors[q][0].ncols()).collect()
    }
    /// The largest bond dimension.
    pub fn max_bond_dimension(&self) -> usize {
        self.bond_dimensions().into_iter().max().unwrap_or(1)
    }
    /// Total squared weight discarded by truncations so far.
    pub fn discarded_weight(&self) -> f64 {
        self.discarded
    }
    /// The site holding the norm (the orthogonality center).
    pub fn center(&self) -> usize {
        self.center
    }

    /// Number of singular values to keep.
    fn keep_count(&self, s: &[f64]) -> usize {
        let smax = s.iter().cloned().fold(0.0, f64::max);
        let mut k = s
            .iter()
            .filter(|&&x| x > self.options.cutoff * smax)
            .count()
            .max(1);
        if let Some(m) = self.options.max_bond {
            k = k.min(m);
        }
        k.min(s.len()).max(1)
    }

    /// Center `c → c+1` by a thin QR of `[A_c^0; A_c^1]`.
    fn move_right(&mut self) {
        let c = self.center;
        debug_assert!(c + 1 < self.n);
        let [a0, a1] = &self.tensors[c];
        let l = a0.nrows();
        let b = stack_rows(a0, a1);
        let qr = b.qr();
        let q = qr.compute_thin_Q();
        let r = qr.thin_R().to_owned();
        let m = q.ncols();
        self.tensors[c] = [
            Mat::from_fn(l, m, |i, j| q[(i, j)]),
            Mat::from_fn(l, m, |i, j| q[(l + i, j)]),
        ];
        let [n0, n1] = &self.tensors[c + 1];
        self.tensors[c + 1] = [mat_mul(&r, n0), mat_mul(&r, n1)];
        self.center = c + 1;
    }

    /// Center `c → c−1` by an SVD of `[A_c^0 | A_c^1]` (truncating when asked)
    /// or a QR of its adjoint.
    fn move_left(&mut self, truncate: bool) -> Result<()> {
        let c = self.center;
        debug_assert!(c >= 1);
        let [a0, a1] = &self.tensors[c];
        let (l, r) = (a0.nrows(), a0.ncols());
        let b = concat_cols(a0, a1); // l × 2r
        let (left, right): (M, M) = if truncate {
            let (u, s, v) = thin_svd(&b)?;
            let k = self.keep_count(&s);
            self.discarded += s[k..].iter().map(|x| x * x).sum::<f64>();
            // A_c = V†[:k, :], carry U S to the left
            let vh = Mat::from_fn(k, 2 * r, |i, j| v[(j, i)].conj());
            let us = Mat::from_fn(l, k, |i, j| u[(i, j)] * s[j]);
            (us, vh)
        } else {
            let qr = b.adjoint().to_owned().qr();
            let q = qr.compute_thin_Q(); // 2r × m
            let rr = qr.thin_R().to_owned(); // m × l
            let m = q.ncols();
            let qh = Mat::from_fn(m, 2 * r, |i, j| q[(j, i)].conj());
            let rh = Mat::from_fn(l, m, |i, j| rr[(j, i)].conj());
            (rh, qh)
        };
        let m = right.nrows();
        self.tensors[c] = [
            Mat::from_fn(m, r, |i, j| right[(i, j)]),
            Mat::from_fn(m, r, |i, j| right[(i, r + j)]),
        ];
        let [p0, p1] = &self.tensors[c - 1];
        self.tensors[c - 1] = [mat_mul(p0, &left), mat_mul(p1, &left)];
        self.center = c - 1;
        Ok(())
    }

    /// Move the center to `q`.
    fn move_center_to(&mut self, q: usize) -> Result<()> {
        while self.center < q {
            self.move_right();
        }
        while self.center > q {
            self.move_left(false)?;
        }
        Ok(())
    }

    /// A single-qubit gate on site `q` (keeps every canonical form).
    fn apply_single(&mut self, g: &[C64], q: usize) {
        let [a0, a1] = &self.tensors[q];
        let n0 = Mat::from_fn(a0.nrows(), a0.ncols(), |i, j| {
            g[0] * a0[(i, j)] + g[1] * a1[(i, j)]
        });
        let n1 = Mat::from_fn(a0.nrows(), a0.ncols(), |i, j| {
            g[2] * a0[(i, j)] + g[3] * a1[(i, j)]
        });
        self.tensors[q] = [n0, n1];
    }

    /// A `k`-qubit gate (dense `2^k × 2^k`, little-endian on `q0, q0+1, …`)
    /// on the contiguous block starting at `q0`.
    fn apply_block(&mut self, q0: usize, k: usize, g: &[C64]) -> Result<()> {
        debug_assert!(k >= 2 && q0 + k <= self.n);
        self.move_center_to(q0)?;
        let l = self.tensors[q0][0].nrows();
        let r = self.tensors[q0 + k - 1][0].ncols();
        let dim = 1usize << k;
        // Θ^s = A_{q0}^{s_0} ⋯ A_{q0+k−1}^{s_{k−1}}
        let theta: Vec<M> = (0..dim)
            .map(|s| {
                let mut m = self.tensors[q0][s & 1].clone();
                for i in 1..k {
                    m = mat_mul(&m, &self.tensors[q0 + i][(s >> i) & 1]);
                }
                m
            })
            .collect();
        let mut out: Vec<M> = (0..dim)
            .map(|sp| {
                let mut m = zero_mat(l, r);
                for (s, th) in theta.iter().enumerate() {
                    let c = g[sp * dim + s];
                    if c != C64::new(0.0, 0.0) {
                        m += faer::Scale(c) * th;
                    }
                }
                m
            })
            .collect();
        // split from the left: rest[s_rest] holds the matrices for the remaining qubits
        let mut left = l;
        for i in 0..k - 1 {
            let remaining = k - i; // qubits still in `out`
            let half = 1usize << (remaining - 1);
            // B[(s0, a), (rest, b)] = out[s0 + 2 rest][a, b]
            let b: M = Mat::from_fn(2 * left, half * r, |row, col| {
                let (s0, a) = (row / left, row % left);
                let (rest, bb) = (col / r, col % r);
                out[s0 + 2 * rest][(a, bb)]
            });
            let (u, s, v) = thin_svd(&b)?;
            let keep = self.keep_count(&s);
            self.discarded += s[keep..].iter().map(|x| x * x).sum::<f64>();
            self.tensors[q0 + i] = [
                Mat::from_fn(left, keep, |a, j| u[(a, j)]),
                Mat::from_fn(left, keep, |a, j| u[(left + a, j)]),
            ];
            // rest matrices: (S V†) reshaped, keep × r per rest index
            out = (0..half)
                .map(|rest| Mat::from_fn(keep, r, |j, bb| s[j] * v[(rest * r + bb, j)].conj()))
                .collect();
            left = keep;
        }
        self.tensors[q0 + k - 1] = [out[0].clone(), out[1].clone()];
        self.center = q0 + k - 1;
        Ok(())
    }

    /// Swap sites `q` and `q+1`.
    fn swap_adjacent(&mut self, q: usize) -> Result<()> {
        let sw: Vec<C64> = Gate::Swap.matrix().unwrap();
        self.apply_block(q, 2, &sw)
    }

    /// Apply a dense gate (`2^k × 2^k`, little-endian on `qubits`): the
    /// targets are routed to a contiguous block with swaps, which are undone
    /// afterwards.
    pub fn apply_matrix(&mut self, g: &[C64], qubits: &[usize]) -> Result<()> {
        check_qubits(qubits, self.n)?;
        let k = qubits.len();
        if g.len() != (1usize << k) * (1usize << k) {
            return Err(SimError::Invalid("gate matrix size".into()));
        }
        if k == 1 {
            self.apply_single(g, qubits[0]);
            return Ok(());
        }
        let t0 = *qubits.iter().min().unwrap();
        let mut pos: Vec<usize> = qubits.to_vec();
        let mut swaps: Vec<usize> = Vec::new();
        for i in 0..k {
            let target = t0 + i;
            let mut cur = pos[i];
            while cur > target {
                self.swap_adjacent(cur - 1)?;
                swaps.push(cur - 1);
                for p in pos.iter_mut() {
                    if *p == cur - 1 {
                        *p = cur;
                    }
                }
                cur -= 1;
            }
            pos[i] = cur;
        }
        self.apply_block(t0, k, g)?;
        for &s in swaps.iter().rev() {
            self.swap_adjacent(s)?;
        }
        Ok(())
    }

    /// `self + other` as a direct sum of bonds (no truncation; call [`compress`](Self::compress)).
    pub fn add(&self, other: &Mps) -> Result<Mps> {
        if other.n != self.n {
            return Err(SimError::Invalid("add: register sizes differ".into()));
        }
        let n = self.n;
        let tensors: Vec<[M; 2]> = (0..n)
            .map(|q| {
                let mk = |s: usize| -> M {
                    let (a, b) = (&self.tensors[q][s], &other.tensors[q][s]);
                    if n == 1 {
                        Mat::from_fn(1, 1, |_, _| a[(0, 0)] + b[(0, 0)])
                    } else if q == 0 {
                        concat_cols(a, b)
                    } else if q == n - 1 {
                        stack_rows(a, b)
                    } else {
                        Mat::from_fn(a.nrows() + b.nrows(), a.ncols() + b.ncols(), |i, j| {
                            if i < a.nrows() && j < a.ncols() {
                                a[(i, j)]
                            } else if i >= a.nrows() && j >= a.ncols() {
                                b[(i - a.nrows(), j - a.ncols())]
                            } else {
                                C64::new(0.0, 0.0)
                            }
                        })
                    }
                };
                [mk(0), mk(1)]
            })
            .collect();
        Ok(Mps {
            n,
            tensors,
            center: 0,
            options: self.options,
            discarded: self.discarded + other.discarded,
        })
    }

    /// `self ⊗ other` (`self` on the low sites): the sites concatenated
    /// (bond dimension 1 between them), the canonical form restored by QR
    /// sweeps from `self`'s center to `other`'s; options are `self`'s.
    pub fn tensor(&self, other: &Mps) -> Result<Mps> {
        let na = self.n;
        let mut tensors = self.tensors.clone();
        tensors.extend(other.tensors.iter().cloned());
        let mut m = Mps {
            n: na + other.n,
            tensors,
            center: self.center,
            options: self.options,
            discarded: self.discarded + other.discarded,
        };
        m.move_center_to(na + other.center)?;
        Ok(m)
    }

    /// Re-establish the canonical form and truncate every bond by the options.
    pub fn compress(&mut self) -> Result<()> {
        self.center = 0;
        self.move_center_to(self.n - 1)?;
        while self.center > 0 {
            self.move_left(true)?;
        }
        Ok(())
    }

    /// `⟨self| P |other⟩` by transfer matrices (`P = None`: the identity).
    fn transfer(&self, other: &Mps, p: Option<&Pauli>) -> C64 {
        let mut e: M = Mat::from_fn(1, 1, |_, _| C64::new(1.0, 0.0));
        for q in 0..self.n {
            // local 2×2 operator entries o[s'][s]
            let o: [[C64; 2]; 2] = match p {
                None => [
                    [C64::new(1.0, 0.0), C64::new(0.0, 0.0)],
                    [C64::new(0.0, 0.0), C64::new(1.0, 0.0)],
                ],
                Some(p) => {
                    let (x, z) = (bits::get(p.x_bits(), q), bits::get(p.z_bits(), q));
                    let mut o = [[C64::new(0.0, 0.0); 2]; 2];
                    for s in 0..2 {
                        let sp = s ^ x as usize;
                        let sign = if z && s == 1 { -1.0 } else { 1.0 };
                        o[sp][s] = C64::new(sign, 0.0);
                    }
                    o
                }
            };
            let (a, b) = (&self.tensors[q], &other.tensors[q]);
            let mut next = zero_mat(a[0].ncols(), b[0].ncols());
            for sp in 0..2 {
                for s in 0..2 {
                    let c = o[sp][s];
                    if c == C64::new(0.0, 0.0) {
                        continue;
                    }
                    let t = mat_mul(&mat_mul(&a[sp].adjoint().to_owned(), &e), &b[s]);
                    next += faer::Scale(c) * &t;
                }
            }
            e = next;
        }
        let v = e[(0, 0)];
        match p {
            Some(p) => v * C64::i_pow_c(p.phase()),
            None => v,
        }
    }

    /// `ψ ← P ψ` for a Pauli (local matrices, global phase on the center).
    fn apply_pauli(&mut self, p: &Pauli) {
        for q in 0..self.n {
            let (x, z) = (bits::get(p.x_bits(), q), bits::get(p.z_bits(), q));
            if !x && !z {
                continue;
            }
            let g =
                Pauli::new(1, bits::from_u64(x as u64, 1), bits::from_u64(z as u64, 1), 0).matrix::<C64>();
            self.apply_single(&g, q);
        }
        let ph = C64::i_pow_c(p.phase());
        if ph != C64::new(1.0, 0.0) {
            self.scale_by(ph);
        }
    }

    fn scale_by(&mut self, c: C64) {
        let q = self.center;
        for s in 0..2 {
            self.tensors[q][s] = faer::Scale(c) * &self.tensors[q][s];
        }
    }
}

trait IPow {
    fn i_pow_c(k: u8) -> C64;
}
impl IPow for C64 {
    fn i_pow_c(k: u8) -> C64 {
        match k & 3 {
            0 => C64::new(1.0, 0.0),
            1 => C64::new(0.0, 1.0),
            2 => C64::new(-1.0, 0.0),
            _ => C64::new(0.0, -1.0),
        }
    }
}

impl QuantumState<C64> for Mps {
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
        if gate.arity() <= 4 {
            if let Some(m) = gate.matrix::<C64>() {
                return self.apply_matrix(&m, qubits);
            }
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
            format!("gate {gate} (no matrix of at most 4 qubits, no decomposition)"),
        ))
    }

    fn norm_sq(&self) -> Result<C64> {
        Ok(self.transfer(self, None))
    }

    fn amplitude(&self, x: &[u64]) -> Result<C64> {
        let mut v: M = Mat::from_fn(1, 1, |_, _| C64::new(1.0, 0.0));
        for q in 0..self.n {
            v = mat_mul(&v, &self.tensors[q][bits::get(x, q) as usize]);
        }
        Ok(v[(0, 0)])
    }

    fn expectation(&self, p: &Pauli) -> Result<C64> {
        if p.num_qubits() != self.n {
            return Err(SimError::Invalid(
                "expectation: operator on the wrong register".into(),
            ));
        }
        Ok(self.transfer(self, Some(p)))
    }

    fn inner(&self, other: &Self) -> Result<C64> {
        if other.n != self.n {
            return Err(SimError::Invalid("inner: register sizes differ".into()));
        }
        Ok(self.transfer(other, None))
    }

    fn project(&mut self, p: &Pauli, outcome: bool) -> Result<C64> {
        if !p.is_hermitian() {
            return Err(SimError::Invalid(format!("projector of non-Hermitian {p}")));
        }
        let mut pp = self.clone();
        pp.apply_pauli(p);
        if outcome {
            pp.scale_by(C64::new(-1.0, 0.0));
        }
        let mut sum = self.add(&pp)?;
        sum.compress()?;
        sum.scale_by(C64::new(0.5, 0.0));
        *self = sum;
        self.norm_sq()
    }

    fn scale(&mut self, c: &C64) -> Result<()> {
        self.scale_by(*c);
        Ok(())
    }

    fn tensor(&self, other: &Self) -> Result<Self> {
        Mps::tensor(self, other)
    }

    fn sample(&self, rng: &mut dyn RngCore, shots: usize) -> Result<Vec<Bits>> {
        let mut s = self.clone();
        s.move_center_to(0)?;
        let nn = s.norm_sq()?.re;
        if nn <= 0.0 {
            return Err(SimError::ZeroState("sample".into()));
        }
        // shots in parallel, each with a generator seeded from `rng`
        use rand::SeedableRng;
        use rayon::prelude::*;
        let seeds: Vec<u64> = (0..shots).map(|_| rng.next_u64()).collect();
        let s = &s;
        Ok(seeds
            .par_iter()
            .map(|&seed| {
                let mut r = rand::rngs::StdRng::seed_from_u64(seed);
                let mut v: M = Mat::from_fn(1, 1, |_, _| C64::new(1.0 / nn.sqrt(), 0.0));
                let mut x = bits::zero(s.n);
                for q in 0..s.n {
                    let v0 = mat_mul(&v, &s.tensors[q][0]);
                    let v1 = mat_mul(&v, &s.tensors[q][1]);
                    let p0 = v0.norm_l2().powi(2);
                    let p1 = v1.norm_l2().powi(2);
                    let u = r.gen::<f64>() * (p0 + p1);
                    if u < p0 {
                        v = faer::Scale(C64::new(1.0 / p0.sqrt(), 0.0)) * &v0;
                    } else {
                        bits::toggle(&mut x, q);
                        v = faer::Scale(C64::new(1.0 / p1.sqrt(), 0.0)) * &v1;
                    }
                }
                x
            })
            .collect())
    }

    fn to_vector(&self) -> Result<Vec<C64>> {
        assert!(self.n <= 20, "dense vector of more than 20 qubits");
        (0..(1u64 << self.n))
            .map(|x| self.amplitude(&bits::from_u64(x, self.n)))
            .collect()
    }
}

impl fmt::Debug for Mps {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "Mps({} qubits, bonds {:?}, center {}, discarded {:.2e})",
            self.n,
            self.bond_dimensions(),
            self.center,
            self.discarded
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::circuit::Circuit;
    use crate::state::StateVector;
    use rand::rngs::StdRng;
    use rand::SeedableRng;

    fn close(a: &[C64], b: &[C64], tol: f64) -> bool {
        a.len() == b.len() && a.iter().zip(b).all(|(x, y)| (x - y).norm() < tol)
    }

    fn random_circuit(n: usize, gates: usize, rng: &mut StdRng) -> Circuit {
        let mut c = Circuit::new(n);
        let one = [Gate::H, Gate::S, Gate::T, Gate::SqrtX, Gate::Rz(0.37), Gate::Y];
        let two = [Gate::Cx, Gate::Cz, Gate::Cs, Gate::Swap];
        for _ in 0..gates {
            let r = rng.gen::<f64>();
            if r < 0.4 {
                c.gate(one[rng.gen_range(0..one.len())].clone(), &[rng.gen_range(0..n)]);
            } else if r < 0.9 {
                let a = rng.gen_range(0..n);
                let mut b = rng.gen_range(0..n - 1);
                if b >= a {
                    b += 1;
                }
                c.gate(two[rng.gen_range(0..two.len())].clone(), &[a, b]);
            } else {
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

    #[test]
    fn random_circuits_match_the_state_vector() {
        let mut rng = StdRng::seed_from_u64(21);
        for trial in 0..8 {
            let n = 3 + trial % 4;
            let c = random_circuit(n, 24, &mut rng);
            let mut m = Mps::zero_state(n);
            let mut v: StateVector<C64> = StateVector::zero_state(n);
            m.run(&c, &mut rng).unwrap();
            v.run(&c, &mut rng).unwrap();
            assert!(
                close(&m.to_vector().unwrap(), &v.to_vector().unwrap(), 1e-9),
                "trial {trial}: {c}"
            );
            assert!((m.norm_sq().unwrap() - v.norm_sq().unwrap()).norm() < 1e-9);
            for _ in 0..4 {
                let p = crate::clifford::random_pauli(n, &mut rng, true);
                let a = m.expectation(&p).unwrap();
                let b = v.expectation(&p).unwrap();
                assert!((a - b).norm() < 1e-9, "trial {trial}: <{p}>");
            }
            let c2 = random_circuit(n, 20, &mut rng);
            let mut m2 = Mps::zero_state(n);
            let mut v2: StateVector<C64> = StateVector::zero_state(n);
            m2.run(&c2, &mut rng).unwrap();
            v2.run(&c2, &mut rng).unwrap();
            assert!((m.inner(&m2).unwrap() - v.inner(&v2).unwrap()).norm() < 1e-9);
            // projection + normalization
            let z = Pauli::z(n, rng.gen_range(0..n));
            let (mut pm, mut pv) = (m.clone(), v.clone());
            let wa = pm.project(&z, true).unwrap();
            let wb = pv.project(&z, true).unwrap();
            assert!((wa - wb).norm() < 1e-9);
            if wb.re > 1e-6 {
                pm.normalize().unwrap();
                pv.normalize().unwrap();
                assert!(close(&pm.to_vector().unwrap(), &pv.to_vector().unwrap(), 1e-8));
            }
            // from_vector round trip
            let back = Mps::from_vector(n, &v.to_vector().unwrap()).unwrap();
            assert!(close(&back.to_vector().unwrap(), &v.to_vector().unwrap(), 1e-9));
        }
    }

    #[test]
    fn ghz_on_thirty_qubits() {
        let n = 30;
        let mut c = Circuit::new(n);
        c.h(0);
        for q in 0..n - 1 {
            c.cx(q, q + 1);
        }
        let mut m = Mps::zero_state(n);
        let mut rng = StdRng::seed_from_u64(1);
        let start = std::time::Instant::now();
        m.run(&c, &mut rng).unwrap();
        assert!(m.bond_dimensions().iter().all(|&b| b == 2));
        let xs = Pauli::from_kinds(&vec![crate::pauli::PauliKind::X; n]);
        assert!((m.expectation(&xs).unwrap() - C64::new(1.0, 0.0)).norm() < 1e-10);
        let zz = Pauli::from_string(n, "Z0*Z29").unwrap();
        assert!((m.expectation(&zz).unwrap() - C64::new(1.0, 0.0)).norm() < 1e-10);
        assert!(m.expectation(&Pauli::z(n, 0)).unwrap().norm() < 1e-10);
        assert!(start.elapsed().as_secs_f64() < 5.0);
        // sampling: all zeros or all ones
        let samples = m.sample(&mut rng, 200).unwrap();
        let ones = samples.iter().filter(|x| bits::popcount(x) == n).count();
        let zeros = samples.iter().filter(|x| bits::popcount(x) == 0).count();
        assert_eq!(ones + zeros, 200);
        assert!((60..=140).contains(&ones));
        // projection onto Z0 = −1 and renormalization gives |1…1⟩
        let w = m.project(&Pauli::z(n, 0), true).unwrap();
        assert!((w.re - 0.5).abs() < 1e-10);
        m.normalize().unwrap();
        let all_ones = bits::from_indices(n, &(0..n).collect::<Vec<_>>());
        assert!((m.amplitude(&all_ones).unwrap() - C64::new(1.0, 0.0)).norm() < 1e-9);
    }

    #[test]
    fn truncation_is_exact_on_product_states_and_tracked_otherwise() {
        let mut rng = StdRng::seed_from_u64(4);
        let mut c = Circuit::new(4);
        c.h(0).t(1).gate(Gate::SqrtX, &[2]).s(3).gate(Gate::Rz(0.5), &[0]);
        let mut m = Mps::zero_state(4).with_options(MpsOptions {
            max_bond: Some(1),
            cutoff: 1e-12,
        });
        let mut v: StateVector<C64> = StateVector::zero_state(4);
        m.run(&c, &mut rng).unwrap();
        v.run(&c, &mut rng).unwrap();
        assert!(close(&m.to_vector().unwrap(), &v.to_vector().unwrap(), 1e-12));
        assert_eq!(m.discarded_weight(), 0.0);
        let mut e = Circuit::new(4);
        e.h(0).cx(0, 1).cx(1, 2).cx(2, 3).h(1).cz(0, 3);
        let mut m = Mps::zero_state(4).with_options(MpsOptions {
            max_bond: Some(1),
            cutoff: 1e-12,
        });
        m.run(&e, &mut rng).unwrap();
        assert!(m.discarded_weight() > 0.1);
        assert!(m.norm_sq().unwrap().re < 0.9);
        assert_eq!(m.max_bond_dimension(), 1);
    }
}
