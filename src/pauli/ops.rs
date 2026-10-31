//! Pauli arithmetic on packed bit slices, shared by [`Pauli`](super::Pauli)
//! (one operator) and [`PauliTable`](super::PauliTable) (many rows): the one
//! implementation of each rule.
//!
//! Convention: a Pauli operator on `n` qubits is `i^phase · ⊗_q X^{x_q} Z^{z_q}`
//! with `phase` mod 4, so `Y = i X Z` has `x = z = 1`, `phase = 1`.

use crate::bits;

/// Phase of the product `(i^a X^x Z^z)(i^b X^x' Z^z') = i^{a + b + 2 z·x'} X^{x⊕x'} Z^{z⊕z'}`.
#[inline]
pub fn product_phase(a: u8, z: &[u64], b: u8, x2: &[u64]) -> u8 {
    (a + b + 2 * bits::dot(z, x2) as u8) & 3
}

/// Do `X^x Z^z` and `X^x' Z^z'` commute?  (`(−1)^{x·z' + z·x'} = 1`.)
#[inline]
pub fn commute(x1: &[u64], z1: &[u64], x2: &[u64], z2: &[u64]) -> bool {
    bits::dot(x1, z2) == bits::dot(z1, x2)
}

/// Is `i^phase X^x Z^z` Hermitian?  Its adjoint is `i^{−phase} (−1)^{x·z} X^x Z^z`,
/// so it is Hermitian exactly when `phase ≡ x·z (mod 2)`.
#[inline]
pub fn is_hermitian(x: &[u64], z: &[u64], phase: u8) -> bool {
    (phase & 1 == 1) == bits::dot(x, z)
}

/// Phase of the adjoint `(i^phase X^x Z^z)† = i^{−phase + 2 x·z} X^x Z^z`.
#[inline]
pub fn adjoint_phase(x: &[u64], z: &[u64], phase: u8) -> u8 {
    ((4 - phase) + 2 * bits::dot(x, z) as u8) & 3
}

/// Number of qubits on which the operator is not the identity.
#[inline]
pub fn weight(x: &[u64], z: &[u64]) -> usize {
    x.iter().zip(z).map(|(a, b)| (a | b).count_ones() as usize).sum()
}

/// `i^phase X^x Z^z |v⟩ = i^{phase + 2 z·v} |v ⊕ x⟩`: the phase of the image.
#[inline]
pub fn basis_image_phase(z: &[u64], phase: u8, v: &[u64]) -> u8 {
    (phase + 2 * bits::dot(z, v) as u8) & 3
}
