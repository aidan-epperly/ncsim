//! Synthesis of a Clifford tableau as a circuit of `H`, `S`, `CX`, `SWAP`
//! and Pauli gates.
//!
//! The tableau `U` is reduced to the identity by left-multiplying gates
//! (`T ← G ∘ T`): for each qubit `q`, the image of `X_q` is brought to `X_q`
//! by a column pivot (`SWAP`, or `H` to turn a `Z` bit into an `X` bit),
//! `CX(q, j)` to clear the other `X` bits, `S_q` to clear `Z_q`, and `H_j`
//! followed by `CX(q, j)` to clear the remaining `Z` bits; the image of `Z_q`
//! is handled the same way inside an `H_q … H_q` conjugation; signs are fixed
//! with `Z_q` and `X_q`.  Qubits already processed are untouched because every
//! other image commutes with `X_q` and `Z_q`.  If `G_k ∘ ⋯ ∘ G_1 ∘ U = 1` then
//! `U = G_1† ∘ ⋯ ∘ G_k†`, i.e. the circuit applies `G_k†` first.  At most
//! `O(n²)` gates, `O(n³/64)` work.

use crate::bits;
use crate::circuit::Circuit;
use crate::clifford::CliffordTableau;
use crate::gate::Gate;
use crate::pauli::ColumnTable;

impl CliffordTableau {
    /// A circuit `c` with `c.tableau() == *self` (signs included).  The
    /// reduction runs on a column-major copy of the images
    /// ([`ColumnTable`]): each gate is `O(n/64)` words and each test one bit,
    /// so the synthesis costs `O(n³/64)`.
    pub fn to_circuit(&self) -> Circuit {
        let n = self.num_qubits();
        let mut t = ColumnTable::from_table(self.images());
        let mut applied: Vec<(Gate, Vec<usize>)> = Vec::new();
        let xb = |t: &ColumnTable, row: usize, j: usize| bits::get(t.x_col(j), row);
        let zb = |t: &ColumnTable, row: usize, j: usize| bits::get(t.z_col(j), row);
        let apply = |t: &mut ColumnTable, applied: &mut Vec<(Gate, Vec<usize>)>, g: Gate, qs: &[usize]| {
            match g {
                Gate::H => t.apply_h(qs[0]),
                Gate::S => t.apply_s_pow(qs[0], 1),
                Gate::Z => t.apply_s_pow(qs[0], 2),
                Gate::X => t.apply_pauli_1q(qs[0], true, false),
                Gate::Cx => t.apply_cx(qs[0], qs[1]),
                Gate::Swap => t.apply_swap(qs[0], qs[1]),
                _ => unreachable!("synthesis gate set"),
            }
            applied.push((g, qs.to_vec()));
        };
        for q in 0..n {
            // image of X_q (row q) → X_q
            match (q..n).find(|&j| xb(&t, q, j)) {
                Some(j) if j != q => apply(&mut t, &mut applied, Gate::Swap, &[q, j]),
                Some(_) => {}
                None => {
                    let j = (q..n)
                        .find(|&j| zb(&t, q, j))
                        .expect("valid tableau: nonzero image");
                    apply(&mut t, &mut applied, Gate::H, &[j]);
                    if j != q {
                        apply(&mut t, &mut applied, Gate::Swap, &[q, j]);
                    }
                }
            }
            for j in q + 1..n {
                if xb(&t, q, j) {
                    apply(&mut t, &mut applied, Gate::Cx, &[q, j]);
                }
            }
            if zb(&t, q, q) {
                apply(&mut t, &mut applied, Gate::S, &[q]);
            }
            for j in q + 1..n {
                if zb(&t, q, j) {
                    apply(&mut t, &mut applied, Gate::H, &[j]);
                    apply(&mut t, &mut applied, Gate::Cx, &[q, j]);
                }
            }
            // image of Z_q (row n + q) → Z_q, inside an H_q conjugation (skipped when already there)
            let r = n + q;
            let z_done = |t: &ColumnTable| (0..n).all(|j| !xb(t, r, j) && zb(t, r, j) == (j == q));
            if !z_done(&t) {
                apply(&mut t, &mut applied, Gate::H, &[q]);
                for j in q + 1..n {
                    if xb(&t, r, j) {
                        apply(&mut t, &mut applied, Gate::Cx, &[q, j]);
                    }
                }
                if zb(&t, r, q) {
                    apply(&mut t, &mut applied, Gate::S, &[q]);
                }
                for j in q + 1..n {
                    if zb(&t, r, j) {
                        apply(&mut t, &mut applied, Gate::H, &[j]);
                        apply(&mut t, &mut applied, Gate::Cx, &[q, j]);
                    }
                }
                apply(&mut t, &mut applied, Gate::H, &[q]);
            }
            // signs
            if t.phase(q) == 2 {
                apply(&mut t, &mut applied, Gate::Z, &[q]);
            }
            if t.phase(r) == 2 {
                apply(&mut t, &mut applied, Gate::X, &[q]);
            }
        }
        debug_assert_eq!(
            t,
            ColumnTable::destabilizer_basis(n),
            "reduction did not reach the identity"
        );
        let mut c = Circuit::new(n);
        for (g, qs) in applied.iter().rev() {
            c.push_gate(g.dagger().expect("named gate"), qs)
                .expect("valid gate");
        }
        c
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::clifford::random::random_clifford;
    use rand::rngs::StdRng;
    use rand::SeedableRng;
    use std::collections::HashSet;

    #[test]
    fn round_trip_random_tableaux() {
        let mut rng = StdRng::seed_from_u64(5);
        for n in [1, 2, 3, 4, 6, 8, 12] {
            for _ in 0..4 {
                let t = random_clifford(n, &mut rng);
                let c = t.to_circuit();
                assert_eq!(c.tableau().unwrap(), t, "n = {n}");
                assert!(
                    c.num_gates() <= 12 * n * n + 6 * n,
                    "n = {n}: {} gates",
                    c.num_gates()
                );
            }
        }
    }

    #[test]
    fn all_single_qubit_cliffords() {
        let mut rng = StdRng::seed_from_u64(6);
        let mut seen = HashSet::new();
        let mut tries = 0;
        while seen.len() < 24 && tries < 2000 {
            tries += 1;
            let t = random_clifford(1, &mut rng);
            let c = t.to_circuit();
            assert_eq!(c.tableau().unwrap(), t);
            seen.insert(format!("{t:?}"));
        }
        assert_eq!(seen.len(), 24);
        assert_eq!(CliffordTableau::identity(3).to_circuit().num_gates(), 0);
    }
}
