//! Commuting Pauli measurements on a tensor product of stabilizer states,
//! without forming the joint tableau.
//!
//! Measuring commuting Hermitian Paulis `P_1, …, P_L` in order on
//! `ψ = φ_1 ⊗ ⋯ ⊗ φ_B` (stabilizer group `S = ⊕_b S_b`) gives these outcomes:
//! - `o_i` is deterministic iff some product `P_i Π_{j<i} P_j^{v_j}` lies in `±S`;
//!   its value is then fixed by the eigenvalue of that product and the
//!   earlier outcomes `o_j`;
//! - otherwise `o_i` is a fair coin.
//!
//! An operator `Q` lies in `±S` exactly when, in every block, its pull-back
//! `V_b Q|_b V_b†` (`φ_b = U_b|0⟩`, `V_b = U_b⁻¹`) has no `X` part.  So the
//! test is GF(2) elimination on the per-block `X` parts.  Each column keeps
//! its blocks' pull-backs with exact phases, so the eigenvalue of a
//! dependent product is read off when its `X` parts vanish.  Pivots are kept
//! per block, so the fill-in stays in the blocks a column touches.
//!
//! The cost is that of the elimination on the columns' block structure.
//! When each `P_i` touches few blocks it is far below measuring on the joint
//! tableau.

use super::Stabilizer;
use crate::bits;
use crate::error::{Result, SimError};
use crate::pauli::Pauli;
use rand::{Rng, RngCore};

/// One operator during elimination: `i^phase ⊗_b V_b-pull-backs`, and the
/// set of measured indices whose product it is.
#[derive(Clone)]
struct Column {
    phase: u8,
    /// (block, pull-back on that block), sorted by block, none the identity
    parts: Vec<(usize, Pauli)>,
    /// indices `j` of the product, sorted
    combo: Vec<u32>,
}

impl Column {
    /// The lowest `(block, position)` of an `X` bit.
    fn lead(&self) -> Option<(usize, usize)> {
        self.parts
            .iter()
            .find_map(|(b, p)| bits::lowest_one(p.x_bits()).map(|q| (*b, q)))
    }
    /// `self ← self · other` (blockwise; global operators commute, so the
    /// blockwise products represent the product exactly).
    fn mul_assign(&mut self, other: &Column) {
        self.phase = (self.phase + other.phase) & 3;
        let mut out: Vec<(usize, Pauli)> = Vec::with_capacity(self.parts.len() + other.parts.len());
        let (mut i, mut j) = (0, 0);
        let a = std::mem::take(&mut self.parts);
        while i < a.len() || j < other.parts.len() {
            let take_a = j >= other.parts.len() || (i < a.len() && a[i].0 < other.parts[j].0);
            let take_b = i >= a.len() || (j < other.parts.len() && other.parts[j].0 < a[i].0);
            if take_a {
                out.push(a[i].clone());
                i += 1;
            } else if take_b {
                out.push(other.parts[j].clone());
                j += 1;
            } else {
                let mut p = a[i].1.clone();
                p.mul_assign(&other.parts[j].1);
                out.push((a[i].0, p));
                i += 1;
                j += 1;
            }
        }
        // identity parts carry only a phase
        let mut kept = Vec::with_capacity(out.len());
        for (b, p) in out {
            if bits::is_zero(p.x_bits()) && bits::is_zero(p.z_bits()) {
                self.phase = (self.phase + p.phase()) & 3;
            } else {
                kept.push((b, p));
            }
        }
        self.parts = kept;
        // symmetric difference of the index sets
        let (x, y) = (std::mem::take(&mut self.combo), &other.combo);
        let mut c = Vec::with_capacity(x.len() + y.len());
        let (mut i, mut j) = (0, 0);
        while i < x.len() || j < y.len() {
            if j >= y.len() || (i < x.len() && x[i] < y[j]) {
                c.push(x[i]);
                i += 1;
            } else if i >= x.len() || y[j] < x[i] {
                c.push(y[j]);
                j += 1;
            } else {
                i += 1;
                j += 1;
            }
        }
        self.combo = c;
    }
}

/// Measure the commuting Hermitian Paulis `paulis` (on the register of the
/// blocks concatenated, block 0 on the lowest qubits), in order, on
/// `blocks[0] ⊗ blocks[1] ⊗ ⋯`; returns the outcomes (`true` = `−1`).
/// The blocks are not modified (the post-measurement state is not formed).
pub fn measure_commuting_on_product(
    blocks: &[Stabilizer],
    paulis: &[Pauli],
    rng: &mut dyn RngCore,
) -> Result<Vec<bool>> {
    let mut offsets = Vec::with_capacity(blocks.len() + 1);
    let mut n = 0usize;
    for b in blocks {
        offsets.push(n);
        n += b.num_qubits();
    }
    offsets.push(n);
    let block_of = |q: usize| offsets.partition_point(|&o| o <= q) - 1;
    let mut pivots: Vec<Vec<Option<u32>>> = blocks.iter().map(|b| vec![None; b.num_qubits()]).collect();
    let mut basis: Vec<Column> = Vec::new();
    let mut out = Vec::with_capacity(paulis.len());
    for (i, p) in paulis.iter().enumerate() {
        if p.num_qubits() != n {
            return Err(SimError::Invalid(format!(
                "Pauli on {} qubits, product on {n}",
                p.num_qubits()
            )));
        }
        if !p.is_hermitian() {
            return Err(SimError::Invalid(format!(
                "measured operator {p} is not Hermitian"
            )));
        }
        // restrictions (letters, phase 0) pulled back block by block
        let mut touched: Vec<usize> = p.support().into_iter().map(block_of).collect();
        touched.dedup();
        let mut parts = Vec::with_capacity(touched.len());
        for &b in &touched {
            let nb = blocks[b].num_qubits();
            let mut x = bits::zero(nb);
            let mut z = bits::zero(nb);
            for q in offsets[b]..offsets[b + 1] {
                if bits::get(p.x_bits(), q) {
                    bits::set(&mut x, q - offsets[b], true);
                }
                if bits::get(p.z_bits(), q) {
                    bits::set(&mut z, q - offsets[b], true);
                }
            }
            parts.push((b, blocks[b].pull_back(&Pauli::new(nb, x, z, 0))));
        }
        let mut col = Column {
            phase: p.phase(),
            parts,
            combo: vec![i as u32],
        };
        // eliminate the X parts against the pivots
        loop {
            match col.lead() {
                None => {
                    // P_i Π P_j^{v_j} ∈ ±S: its eigenvalue i^{phase + Σ part phases}
                    let e = col
                        .parts
                        .iter()
                        .fold(col.phase as u32, |acc, (_, q)| acc + q.phase() as u32)
                        & 3;
                    debug_assert!(e % 2 == 0, "a non-Hermitian product of commuting Paulis");
                    let mut o = e == 2;
                    for &j in &col.combo {
                        if j as usize != i {
                            o ^= out[j as usize];
                        }
                    }
                    out.push(o);
                    break;
                }
                Some((b, q)) => match pivots[b][q] {
                    Some(k) => col.mul_assign(&basis[k as usize]),
                    None => {
                        pivots[b][q] = Some(basis.len() as u32);
                        basis.push(col);
                        out.push(rng.gen::<bool>());
                        break;
                    }
                },
            }
        }
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::clifford::random_stabilizer_state;
    use rand::rngs::StdRng;
    use rand::SeedableRng;
    use std::collections::HashMap;

    /// Commuting Paulis: random products of the stabilizers of a random state,
    /// with signs.
    fn commuting(n: usize, l: usize, rng: &mut StdRng) -> Vec<Pauli> {
        let g = random_stabilizer_state(n, rng).stabilizers();
        (0..l)
            .map(|_| {
                let mut p = Pauli::identity(n);
                for h in &g {
                    if rng.gen::<bool>() {
                        p.mul_assign(h);
                    }
                }
                let p = if p.is_hermitian() { p } else { p.times_i() };
                if rng.gen::<bool>() {
                    p.negated()
                } else {
                    p
                }
            })
            .collect()
    }

    #[test]
    fn matches_sequential_measurement_on_the_joint_tableau() {
        let mut rng = StdRng::seed_from_u64(5);
        for trial in 0..40 {
            let sizes: Vec<usize> = (0..1 + trial % 4).map(|_| rng.gen_range(1..4)).collect();
            let blocks: Vec<Stabilizer> = sizes
                .iter()
                .map(|&s| random_stabilizer_state(s, &mut rng))
                .collect();
            let n: usize = sizes.iter().sum();
            let ps = commuting(n, 2 + trial % 6, &mut rng);
            let joint = Stabilizer::tensor_all(&blocks);
            let shots = 4000;
            let (mut ha, mut hb): (HashMap<Vec<bool>, usize>, HashMap<Vec<bool>, usize>) =
                (HashMap::new(), HashMap::new());
            for _ in 0..shots {
                let a = measure_commuting_on_product(&blocks, &ps, &mut rng).unwrap();
                *ha.entry(a).or_default() += 1;
                let mut j = joint.clone();
                let b: Vec<bool> = ps.iter().map(|p| j.measure_pauli(p, &mut rng).unwrap()).collect();
                *hb.entry(b).or_default() += 1;
            }
            // the same support, and both uniform on it (joint stabilizer outcomes
            // are uniform on an affine set): every count within 5σ of the mean
            assert_eq!(
                ha.keys().collect::<std::collections::BTreeSet<_>>(),
                hb.keys().collect::<std::collections::BTreeSet<_>>(),
                "trial {trial}: supports differ"
            );
            let k = ha.len() as f64;
            let mean = shots as f64 / k;
            let sd = (shots as f64 * (1.0 / k) * (1.0 - 1.0 / k)).sqrt().max(1.0);
            for h in [&ha, &hb] {
                for (o, &c) in h.iter() {
                    assert!(
                        (c as f64 - mean).abs() < 5.0 * sd,
                        "trial {trial}: outcome {o:?} {c} vs {mean}"
                    );
                }
            }
        }
    }
}
