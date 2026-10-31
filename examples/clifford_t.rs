//! Exact simulation of a Clifford+T circuit three ways, with the cost
//! parameters of each family: path variables and width of the path sum,
//! number of terms of the stabilizer-rank sum, and the dense vector as the
//! reference.
//!
//! ```text
//! cargo run --release --example clifford_t [-- n t seed]
//! ```

use ncsim::bits;
use ncsim::{Amplitude, Circuit, Cyclo, Gate, PathSum, Pauli, QuantumState, StabilizerSum, StateVector, C64};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use std::time::Instant;

fn random_clifford_t(n: usize, t_gates: usize, rng: &mut StdRng) -> Circuit {
    let mut c = Circuit::new(n);
    let cliff = [Gate::H, Gate::S, Gate::Cx, Gate::Cz];
    let mut t_left = t_gates;
    let total = 6 * n + 2 * t_gates;
    for step in 0..total {
        let put_t = t_left > 0 && (step % 3 == 2);
        if put_t {
            c.gate(Gate::T, &[rng.gen_range(0..n)]);
            t_left -= 1;
            continue;
        }
        let g = cliff[rng.gen_range(0..cliff.len())].clone();
        if g.arity() == 1 || n < 2 {
            c.gate(if g.arity() == 1 { g } else { Gate::H }, &[rng.gen_range(0..n)]);
        } else {
            let a = rng.gen_range(0..n);
            let mut b = rng.gen_range(0..n - 1);
            if b >= a {
                b += 1;
            }
            c.gate(g, &[a, b]);
        }
    }
    c
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let n: usize = args.get(1).and_then(|s| s.parse().ok()).unwrap_or(8);
    let t: usize = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(6);
    let seed: u64 = args.get(3).and_then(|s| s.parse().ok()).unwrap_or(1);
    let mut rng = StdRng::seed_from_u64(seed);
    let c = random_clifford_t(n, t, &mut rng);
    println!(
        "{n} qubits, {} gates, {} T gates\n{c}",
        c.num_gates(),
        c.non_clifford_count()
    );

    let obs = Pauli::from_string(n, "X0*Z1").unwrap();
    let x0 = bits::zero(n);

    let start = Instant::now();
    let mut p: PathSum<Cyclo<8>> = PathSum::zero_state(n);
    p.run(&c, &mut rng).unwrap();
    let plan = p.amplitude_plan(&x0);
    let a_exact = p.amplitude(&x0).unwrap();
    let e_exact = p.expectation(&obs).unwrap();
    println!(
        "path sum (exact, Q(ζ8)): {} path variables, amplitude plan {:?}\n  <0|ψ> = {}\n  <X0 Z1> = {}\n  {:?}",
        p.num_path_vars(),
        plan,
        a_exact,
        e_exact,
        start.elapsed()
    );

    let start = Instant::now();
    let mut s: StabilizerSum<C64> = StabilizerSum::zero_state(n);
    s.run(&c, &mut rng).unwrap();
    let a_sum = s.amplitude(&x0).unwrap();
    let e_sum = s.expectation(&obs).unwrap();
    println!(
        "stabilizer sum (float coefficients): {} terms\n  <0|ψ> = {a_sum}\n  <X0 Z1> = {e_sum}\n  {:?}",
        s.num_terms(),
        start.elapsed()
    );

    let start = Instant::now();
    let mut m = ncsim::Mps::zero_state(n);
    m.run(&c, &mut rng).unwrap();
    println!(
        "matrix product state: max bond {}, discarded weight {:.1e}
  <0|ψ> = {}
  <X0 Z1> = {}
  {:?}",
        m.max_bond_dimension(),
        m.discarded_weight(),
        m.amplitude(&x0).unwrap(),
        m.expectation(&obs).unwrap(),
        start.elapsed()
    );

    if n <= 22 {
        let start = Instant::now();
        let mut v: StateVector<C64> = StateVector::zero_state(n);
        v.run(&c, &mut rng).unwrap();
        let a_vec = v.amplitude(&x0).unwrap();
        let e_vec = v.expectation(&obs).unwrap();
        println!(
            "dense vector: <0|ψ> = {a_vec}, <X0 Z1> = {e_vec}, {:?}",
            start.elapsed()
        );
        println!(
            "agreement: |Δ amplitude| = {:.1e}, |Δ expectation| = {:.1e}",
            (a_exact.to_complex() - a_vec).norm(),
            (e_exact.to_complex() - e_vec).norm()
        );
        assert!((a_sum - a_vec).norm() < 1e-9 && (e_sum - e_vec).norm() < 1e-9);
    }
    let _ = Cyclo::<8>::sqrt2_pow(0);
}
