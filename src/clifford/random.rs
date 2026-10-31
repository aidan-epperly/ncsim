//! Uniformly random Clifford unitaries, stabilizer states and Pauli operators.
//!
//! [`random_clifford`] samples a uniformly random *symplectic basis* of
//! `𝔽₂^{2n}`: `v_1` uniform among the nonzero vectors, `w_1` uniform among the
//! vectors with `⟨v_1, w_1⟩ = 1`, then the same inside the symplectic
//! complement of `span(v_1, w_1)`, and so on.  Every ordered symplectic basis
//! — equivalently every element of `Sp(2n, 𝔽₂)` — arises with probability
//! `Π_j 1 / ((2^{2j} − 1) 2^{2j−1}) = 1/|Sp(2n, 𝔽₂)|`, so the symplectic part
//! is exactly uniform; the `2n` signs are independent uniform bits.  Cost
//! `O(n³/64)` word operations.

use crate::bits::{self, Bits};
use crate::clifford::CliffordTableau;
use crate::pauli::Pauli;
use crate::state::Stabilizer;
use rand::{Rng, RngCore};

#[derive(Clone)]
struct Vec2n {
    x: Bits,
    z: Bits,
}

/// The symplectic form `x·z' + z·x'` (1 iff the Paulis anticommute).
fn symp(a: &Vec2n, b: &Vec2n) -> bool {
    bits::dot(&a.x, &b.z) ^ bits::dot(&a.z, &b.x)
}

fn xor(a: &mut Vec2n, b: &Vec2n) {
    bits::xor_into(&mut a.x, &b.x);
    bits::xor_into(&mut a.z, &b.z);
}

/// A uniformly random element of the span of `span` (a random 𝔽₂-combination
/// of a spanning set is uniform over its span).
fn random_combination(span: &[Vec2n], n: usize, rng: &mut dyn RngCore) -> Vec2n {
    let mut v = Vec2n {
        x: bits::zero(n),
        z: bits::zero(n),
    };
    for s in span {
        if rng.gen::<bool>() {
            xor(&mut v, s);
        }
    }
    v
}

/// A uniformly random `n`-qubit Clifford tableau (uniform symplectic part,
/// independent uniform signs).
pub fn random_clifford(n: usize, rng: &mut dyn RngCore) -> CliffordTableau {
    let mut span: Vec<Vec2n> = (0..n)
        .map(|q| Vec2n {
            x: bits::unit(n, q),
            z: bits::zero(n),
        })
        .chain((0..n).map(|q| Vec2n {
            x: bits::zero(n),
            z: bits::unit(n, q),
        }))
        .collect();
    let mut xs = Vec::with_capacity(n);
    let mut zs = Vec::with_capacity(n);
    for _ in 0..n {
        let v = loop {
            let c = random_combination(&span, n, rng);
            if !bits::is_zero(&c.x) || !bits::is_zero(&c.z) {
                break c;
            }
        };
        let b = span
            .iter()
            .find(|s| symp(&v, s))
            .cloned()
            .expect("the symplectic form is non-degenerate on the complement");
        let mut w = random_combination(&span, n, rng);
        if !symp(&v, &w) {
            xor(&mut w, &b);
        }
        // project the spanning set onto the symplectic complement of (v, w)
        for s in span.iter_mut() {
            let a = symp(s, &w);
            let c = symp(s, &v);
            if a {
                xor(s, &v);
            }
            if c {
                xor(s, &w);
            }
        }
        xs.push(v);
        zs.push(w);
    }
    let hermitian = |v: Vec2n, rng: &mut dyn RngCore| {
        let sign = if rng.gen::<bool>() { 2 } else { 0 };
        let ph = bits::dot(&v.x, &v.z) as u8 + sign;
        Pauli::new(n, v.x, v.z, ph)
    };
    let x_images: Vec<Pauli> = xs.into_iter().map(|v| hermitian(v, rng)).collect();
    let z_images: Vec<Pauli> = zs.into_iter().map(|v| hermitian(v, rng)).collect();
    // a symplectic basis by construction: skip the O(n³) validity check
    let mut images = crate::pauli::PauliTable::new(n);
    for p in x_images.iter().chain(z_images.iter()) {
        images.push(p);
    }
    let t = CliffordTableau::from_images_unchecked(images);
    debug_assert!(t.is_valid(), "random_clifford: not a symplectic basis");
    t
}

/// A uniformly random `n`-qubit stabilizer state (a random Clifford applied to `|0…0⟩`).
pub fn random_stabilizer_state(n: usize, rng: &mut dyn RngCore) -> Stabilizer {
    Stabilizer::from_clifford(&random_clifford(n, rng))
}

/// A uniformly random `n`-qubit Pauli operator; with `hermitian`, a uniform
/// sign on a uniform letter string, otherwise a uniform phase `i^k`.
pub fn random_pauli(n: usize, rng: &mut dyn RngCore, hermitian: bool) -> Pauli {
    let mut x = bits::zero(n);
    let mut z = bits::zero(n);
    for w in x.iter_mut().chain(z.iter_mut()) {
        *w = rng.gen::<u64>();
    }
    bits::mask_to(&mut x, n);
    bits::mask_to(&mut z, n);
    let phase = if hermitian {
        bits::dot(&x, &z) as u8 + if rng.gen::<bool>() { 2 } else { 0 }
    } else {
        rng.gen_range(0..4u8)
    };
    Pauli::new(n, x, z, phase)
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::rngs::StdRng;
    use rand::SeedableRng;
    use std::collections::HashMap;

    fn symplectic_key(t: &CliffordTableau) -> Vec<u64> {
        let m = t.symplectic();
        (0..m.rows()).flat_map(|i| m.row_words(i).to_vec()).collect()
    }

    /// Chi-square statistic of `counts` against a uniform distribution over `classes` classes.
    fn chi_square(counts: &HashMap<Vec<u64>, usize>, classes: usize, samples: usize) -> f64 {
        assert!(
            counts.len() <= classes,
            "{} classes seen, {classes} expected",
            counts.len()
        );
        let e = samples as f64 / classes as f64;
        let seen: f64 = counts.values().map(|&c| (c as f64 - e).powi(2) / e).sum();
        seen + (classes - counts.len()) as f64 * e
    }

    #[test]
    fn samples_are_valid_and_fast() {
        let mut rng = StdRng::seed_from_u64(1);
        for n in [1, 2, 3, 5, 8, 50] {
            for _ in 0..5 {
                let t = random_clifford(n, &mut rng);
                assert!(t.is_valid(), "n = {n}");
            }
        }
        let start = std::time::Instant::now();
        let t = random_clifford(200, &mut rng);
        assert!(t.is_valid());
        assert!(start.elapsed().as_secs_f64() < 5.0);
        let s = random_stabilizer_state(6, &mut rng);
        assert_eq!(s.num_qubits(), 6);
        let p = random_pauli(70, &mut rng, true);
        assert!(p.is_hermitian());
    }

    #[test]
    fn one_qubit_cliffords_are_uniform() {
        let mut rng = StdRng::seed_from_u64(2);
        let samples = 24_000;
        let mut counts: HashMap<Vec<u64>, usize> = HashMap::new();
        for _ in 0..samples {
            let t = random_clifford(1, &mut rng);
            let mut key = symplectic_key(&t);
            key.extend(t.phases().iter().map(|&p| p as u64));
            *counts.entry(key).or_default() += 1;
        }
        assert_eq!(counts.len(), 24);
        let chi = chi_square(&counts, 24, samples);
        assert!(chi < 60.0, "chi-square {chi} (df 23)");
    }

    #[test]
    fn two_qubit_symplectic_part_and_signs_are_uniform() {
        let mut rng = StdRng::seed_from_u64(3);
        let samples = 72_000;
        let mut sym: HashMap<Vec<u64>, usize> = HashMap::new();
        let mut signs: HashMap<Vec<u64>, usize> = HashMap::new();
        for _ in 0..samples {
            let t = random_clifford(2, &mut rng);
            *sym.entry(symplectic_key(&t)).or_default() += 1;
            let sk: Vec<u64> = t.phases().iter().map(|&p| (p >= 2) as u64).collect();
            *signs.entry(sk).or_default() += 1;
        }
        assert_eq!(sym.len(), 720);
        let chi = chi_square(&sym, 720, samples);
        assert!(chi < 950.0, "chi-square {chi} (df 719)");
        assert_eq!(signs.len(), 16);
        let chi_s = chi_square(&signs, 16, samples);
        assert!(chi_s < 45.0, "sign chi-square {chi_s} (df 15)");
    }
}
