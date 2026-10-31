//! Bridge to ncalg's monomial unitaries `R|x⟩ = ω^{phase[x]} |perm[x]⟩`
//! ([`Monomial`], `ω = e^{2πi/2^bits}`), the representation of the
//! Clifford-hierarchy groups in `ncalg::pcgroup::families` (`pauli_cz`, `xs`,
//! `xt`, `x_diag`, …).
//!
//! Conventions agree: in both libraries qubit `q` is bit `q` of the basis
//! index, the phase is evaluated on the input basis state, and `a · b`
//! applies `b` first.  A monomial is a [`MonomialGate`] exactly when its
//! permutation is affine, `perm(x) = M x ⊕ b`; its phase function becomes a
//! phase polynomial by the Möbius (Reed–Muller) transform
//! `c_S = Σ_{T ⊆ S} (−1)^{|S|−|T|} phase(T)` mod `2^bits`.

use crate::bits::{self, Bits};
use crate::clifford::CliffordTableau;
use crate::error::{Result, SimError};
use crate::gate::{Gate, MonomialGate};
use crate::phasepoly::PhasePoly;
use crate::scalar::Phasor;
pub use ncalg::symmetry::perm::Monomial;

impl MonomialGate {
    /// From an ncalg monomial with phases mod `2^bits` (`1 ≤ bits ≤ 7`);
    /// the permutation must be affine on the basis bits.
    pub fn from_monomial(m: &Monomial, bits: u32) -> Result<MonomialGate> {
        let dim = m.dim();
        if dim == 0 || !dim.is_power_of_two() {
            return Err(SimError::Invalid(format!(
                "monomial of dimension {dim} is not a qubit operator"
            )));
        }
        if !(1..=7).contains(&bits) {
            return Err(SimError::Invalid("monomial phase bits must be in 1..=7".into()));
        }
        let k = dim.trailing_zeros() as usize;
        let b = m.perm[0] as usize;
        let cols: Vec<usize> = (0..k).map(|j| m.perm[1 << j] as usize ^ b).collect();
        for x in 0..dim {
            let mut y = b;
            for (j, c) in cols.iter().enumerate() {
                if (x >> j) & 1 == 1 {
                    y ^= c;
                }
            }
            if y != m.perm[x] as usize {
                return Err(SimError::Invalid("monomial permutation is not affine".into()));
            }
        }
        let rows: Vec<Bits> = (0..k)
            .map(|i| {
                let mut r = bits::zero(k);
                for (j, c) in cols.iter().enumerate() {
                    if (c >> i) & 1 == 1 {
                        bits::toggle(&mut r, j);
                    }
                }
                r
            })
            .collect();
        // Möbius transform of the phase function
        let modulus = 1i64 << bits;
        let mut c: Vec<i64> = m.phase.iter().map(|&p| p as i64).collect();
        for j in 0..k {
            for x in 0..dim {
                if (x >> j) & 1 == 1 {
                    c[x] -= c[x ^ (1 << j)];
                }
            }
        }
        let mut phase = PhasePoly::new(k, bits);
        phase.add_constant(c[0].rem_euclid(modulus));
        for s in 1..dim {
            let v = c[s].rem_euclid(modulus);
            if v != 0 {
                let vars: Vec<usize> = (0..k).filter(|j| (s >> j) & 1 == 1).collect();
                phase.add_monomial(v, &vars);
            }
        }
        MonomialGate::new(rows, bits::from_u64(b as u64, k), phase, Phasor::ONE)
    }

    /// The ncalg monomial with phases mod `2^bits`; the gate's phases must be
    /// `2^bits`-th roots of unity.
    pub fn to_monomial(&self, bits: u32) -> Result<Monomial> {
        if !(1..=7).contains(&bits) {
            return Err(SimError::Invalid("monomial phase bits must be in 1..=7".into()));
        }
        let k = self.num_qubits();
        let dim = 1usize << k;
        let mut perm = vec![0u32; dim];
        let mut phase = vec![0u8; dim];
        for x in 0..dim {
            let (s, y) = self.apply_to_basis(&bits::from_u64(x as u64, k));
            match s {
                Phasor::Value {
                    sqrt2_power: 0,
                    phase: ph,
                    bits: b,
                } if b <= bits => {
                    phase[x] = (ph << (bits - b)) as u8;
                }
                other => {
                    return Err(SimError::NotRepresentable(format!(
                        "phase {other} with {bits} bits"
                    )));
                }
            }
            perm[x] = bits::to_u64(&y, k) as u32;
        }
        Ok(Monomial { perm, phase })
    }
}

impl CliffordTableau {
    /// The tableau of a Clifford monomial (affine permutation, level ≤ 2 phases).
    pub fn from_monomial(m: &Monomial, bits: u32) -> Result<Self> {
        MonomialGate::from_monomial(m, bits)?.to_tableau()
    }
}

impl Gate {
    /// A monomial unitary as a gate.
    pub fn from_monomial(m: &Monomial, bits: u32) -> Result<Gate> {
        Ok(Gate::Monomial(MonomialGate::from_monomial(m, bits)?))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::scalar::{Amplitude, C64};
    use ncalg::pcgroup::families::{pauli_cz, xs, xt, PcFamily};
    use rand::rngs::StdRng;
    use rand::{Rng, SeedableRng};

    fn random_exps(fam: &PcFamily, rng: &mut StdRng) -> Vec<u8> {
        (0..fam.pres.names().len())
            .map(|_| rng.gen_range(0..2u8))
            .collect()
    }

    #[test]
    fn families_round_trip_and_compose() {
        let mut rng = StdRng::seed_from_u64(8);
        for fam in [pauli_cz(3).unwrap(), xs(3).unwrap(), xt(2).unwrap()] {
            let bits = fam.phase_bits();
            let dim = fam.dim();
            for _ in 0..8 {
                let e = random_exps(&fam, &mut rng);
                let f = random_exps(&fam, &mut rng);
                let me = fam.monomial(&e);
                let mf = fam.monomial(&f);
                let ge = MonomialGate::from_monomial(&me, bits).unwrap();
                let gf = MonomialGate::from_monomial(&mf, bits).unwrap();
                assert_eq!(ge.to_monomial(bits).unwrap(), me);
                // composition (b first) matches the monomial product
                let prod = me.mul_bits(&mf, bits);
                assert_eq!(ge.compose(&gf).to_monomial(bits).unwrap(), prod);
                // and the pc multiplication
                let ef = fam.pres.mul(&e, &f);
                assert_eq!(fam.monomial(&ef), prod, "{}", fam.family);
                // dense matrix agrees with the monomial on basis vectors
                let mat: Vec<C64> = ge.matrix().unwrap();
                for x in 0..dim {
                    let y = me.perm[x] as usize;
                    let want = C64::root_of_unity(me.phase[x] as i64, bits).unwrap();
                    assert!((mat[y * dim + x] - want).norm() < 1e-12);
                    for r in 0..dim {
                        if r != y {
                            assert!(mat[r * dim + x].norm() < 1e-12);
                        }
                    }
                }
                if fam.family != "XT" {
                    let t = ge.to_tableau().unwrap();
                    assert!(t.is_valid());
                    assert_eq!(t, CliffordTableau::from_monomial(&me, bits).unwrap());
                }
            }
        }
        // XT: an element with a T is not Clifford, X alone is
        let fam = xt(2).unwrap();
        let names = fam.pres.names();
        let t_idx = names.iter().position(|s| s == "T0").unwrap();
        let x_idx = names.iter().position(|s| s == "X0").unwrap();
        let mut e = vec![0u8; names.len()];
        e[t_idx] = 1;
        assert!(MonomialGate::from_monomial(&fam.monomial(&e), 3)
            .unwrap()
            .to_tableau()
            .is_err());
        let mut e = vec![0u8; names.len()];
        e[x_idx] = 1;
        assert!(MonomialGate::from_monomial(&fam.monomial(&e), 3)
            .unwrap()
            .to_tableau()
            .is_ok());
        // a non-affine permutation (Toffoli) is rejected; CNOT (x0 ^= x1) is affine
        let cnot = Monomial {
            perm: vec![0, 1, 3, 2],
            phase: vec![0; 4],
        };
        assert!(MonomialGate::from_monomial(&cnot, 3).is_ok());
        let bad = Monomial {
            perm: vec![0, 1, 2, 3, 4, 5, 7, 6],
            phase: vec![0; 8],
        };
        assert!(MonomialGate::from_monomial(&bad, 3).is_err());
    }
}
