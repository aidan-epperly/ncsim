//! Randomized cross-checks of the exact families on registers beyond one
//! word: tableau, CH-form and path sum on Clifford circuits with projections;
//! stabilizer sums with path-sum and CH-form terms on Clifford+T circuits.
//!
//! ```text
//! cargo run --release --example stress [-- trials seed]
//! ```

use ncsim::{
    ChForm, ChTerm, Circuit, Gate, PathSum, Pauli, PauliKind, QuantumState, Stabilizer, StabilizerSum, C64,
};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

fn random_pauli(n: usize, weight: usize, rng: &mut StdRng) -> Pauli {
    let mut kinds = vec![PauliKind::I; n];
    for _ in 0..weight {
        kinds[rng.gen_range(0..n)] = [PauliKind::X, PauliKind::Y, PauliKind::Z][rng.gen_range(0..3)];
    }
    let p = Pauli::from_kinds(&kinds);
    if rng.gen::<bool>() {
        p.negated()
    } else {
        p
    }
}

fn random_clifford(n: usize, gates: usize, rng: &mut StdRng) -> Circuit {
    let one = [
        Gate::H,
        Gate::S,
        Gate::Sdg,
        Gate::X,
        Gate::Y,
        Gate::Z,
        Gate::SqrtX,
        Gate::SqrtY,
    ];
    let two = [Gate::Cx, Gate::Cz, Gate::Cy, Gate::Swap];
    let mut c = Circuit::new(n);
    for _ in 0..gates {
        if rng.gen::<bool>() {
            c.gate(one[rng.gen_range(0..one.len())].clone(), &[rng.gen_range(0..n)]);
        } else {
            let a = rng.gen_range(0..n);
            let b = (a + 1 + rng.gen_range(0..n - 1)) % n;
            c.gate(two[rng.gen_range(0..two.len())].clone(), &[a, b]);
        }
    }
    c
}

fn close(a: C64, b: C64) -> bool {
    (a - b).norm() < 1e-8 * (1.0 + b.norm())
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let trials: usize = args.get(1).and_then(|s| s.parse().ok()).unwrap_or(40);
    let seed: u64 = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(1);
    let mut rng = StdRng::seed_from_u64(seed);
    let mut checks = 0usize;
    for trial in 0..trials {
        let n = 60 + rng.gen_range(0..80);
        // Clifford: three families, interleaved projections with the same outcomes
        let mut st = Stabilizer::zero_state(n);
        let mut ch = ChForm::zero_state(n);
        let mut ps: PathSum<C64> = PathSum::zero_state(n);
        for round in 0..6 {
            let c = random_clifford(n, 4 * n, &mut rng);
            for op in c.ops() {
                if let ncsim::Op::Gate { gate, qubits } = op {
                    st.apply_gate(gate, qubits).unwrap();
                    ch.apply(gate, qubits).unwrap();
                    ps.apply_gate(gate, qubits).unwrap();
                }
            }
            for _ in 0..5 {
                let p = random_pauli(n, 1 + rng.gen_range(0..6), &mut rng);
                let a: C64 = QuantumState::<C64>::expectation(&st, &p).unwrap();
                let nb: C64 = QuantumState::<C64>::norm_sq(&ch).unwrap();
                let b: C64 = QuantumState::<C64>::expectation(&ch, &p).unwrap() / nb;
                let c = ps.expectation(&p).unwrap() / ps.norm_sq().unwrap();
                assert!(
                    close(a, b) && close(a, c),
                    "trial {trial} round {round}: <{p}> {a} {b} {c}"
                );
                checks += 1;
            }
            // a projection whose outcome is possible
            let p = random_pauli(n, 1 + rng.gen_range(0..4), &mut rng);
            let p = if p.is_hermitian() { p } else { p.times_i() };
            let e: C64 = QuantumState::<C64>::expectation(&st, &p).unwrap();
            let outcome = if e.re > 0.5 {
                false
            } else if e.re < -0.5 {
                true
            } else {
                rng.gen()
            };
            let w1: C64 = QuantumState::<C64>::project(&mut st, &p, outcome).unwrap();
            let n0: C64 = QuantumState::<C64>::norm_sq(&ch).unwrap();
            let w2: C64 = QuantumState::<C64>::project(&mut ch, &p, outcome).unwrap();
            let p0 = ps.norm_sq().unwrap();
            let w3 = ps.project(&p, outcome).unwrap();
            assert!(
                close(w1, w2 / n0) && close(w1, w3 / p0),
                "trial {trial}: projection weights {w1} {} {}",
                w2 / n0,
                w3 / p0
            );
            checks += 1;
        }
        // the exact overlap of the CH-form and the path sum (same vector up to normalization)
        let o: C64 =
            QuantumState::<C64>::inner(&ChForm::from_stabilizer(&ps.to_stabilizer().unwrap()), &ch).unwrap();
        assert!(
            (o.norm() - QuantumState::<C64>::norm_sq(&ch).unwrap().re.sqrt()).abs() < 1e-8,
            "trial {trial}: overlap {o}"
        );
        checks += 1;

        // Clifford+T: stabilizer sums with the two term families
        let m = 40 + rng.gen_range(0..60);
        let mut sa: StabilizerSum<C64> = StabilizerSum::zero_state(m);
        let mut sb: StabilizerSum<C64, ChTerm<C64>> = StabilizerSum::zero_state(m);
        for _ in 0..3 {
            let c = random_clifford(m, 3 * m, &mut rng);
            for op in c.ops() {
                if let ncsim::Op::Gate { gate, qubits } = op {
                    sa.apply_gate(gate, qubits).unwrap();
                    sb.apply_gate(gate, qubits).unwrap();
                }
            }
            for _ in 0..2 {
                let q = rng.gen_range(0..m);
                let g = if rng.gen::<bool>() { Gate::T } else { Gate::Ccz };
                let qs = if g.arity() == 1 {
                    vec![q]
                } else {
                    vec![q, (q + 1) % m, (q + 2) % m]
                };
                sa.apply_gate(&g, &qs).unwrap();
                sb.apply_gate(&g, &qs).unwrap();
            }
        }
        for _ in 0..3 {
            let p = random_pauli(m, 1 + rng.gen_range(0..4), &mut rng);
            let p = if p.is_hermitian() { p } else { p.times_i() };
            let a = sa.expectation(&p).unwrap();
            let b = sb.expectation(&p).unwrap();
            assert!(close(a, b), "trial {trial}: stabilizer sums <{p}> {a} vs {b}");
            checks += 1;
        }
        let x = ncsim::bits::zero(m);
        assert!(
            close(sa.amplitude(&x).unwrap(), sb.amplitude(&x).unwrap()),
            "trial {trial}: amplitude"
        );
        checks += 1;
        if trial % 10 == 9 {
            println!("{} trials, {checks} checks passed", trial + 1);
        }
    }
    println!("all {checks} checks passed");
}
