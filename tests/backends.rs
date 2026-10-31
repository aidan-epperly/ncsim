//! Cross-backend consistency on circuits given in the text format, and checks
//! against closed forms (ncalg exponential sums).

use ncalg::Coeff;
use ncsim::bits;
use ncsim::phasepoly::PhasePoly;
use ncsim::{
    Amplitude, Circuit, Cyclo, Gate, PathSum, Pauli, QuantumState, Stabilizer, StabilizerSum, StateVector,
    C64,
};
use rand::rngs::StdRng;
use rand::SeedableRng;

fn close(a: &[C64], b: &[C64]) -> bool {
    a.len() == b.len() && a.iter().zip(b).all(|(x, y)| (x - y).norm() < 1e-9)
}

const CLIFFORD_T: &str = "
QUBITS 4
H 0 1 2 3
T 0
CX 0 1
CS 1 2
T_DAG 3
CZ 2 3
H 1
CCZ 0 1 2
SQRT_X 3
S 0
CX 3 0
H 2
T 2
";

#[test]
fn all_exact_backends_agree_on_a_clifford_t_circuit() {
    let c = Circuit::parse(CLIFFORD_T).unwrap();
    let mut rng = StdRng::seed_from_u64(0);
    let mut v: StateVector<C64> = StateVector::zero_state(4);
    let mut p: PathSum<C64> = PathSum::zero_state(4);
    let mut e: PathSum<Cyclo<8>> = PathSum::zero_state(4);
    let mut s: StabilizerSum<C64> = StabilizerSum::zero_state(4);
    let mut se: StabilizerSum<Cyclo<8>> = StabilizerSum::zero_state(4);
    v.run(&c, &mut rng).unwrap();
    p.run(&c, &mut rng).unwrap();
    e.run(&c, &mut rng).unwrap();
    s.run(&c, &mut rng).unwrap();
    se.run(&c, &mut rng).unwrap();
    let vv = v.to_vector().unwrap();
    assert!(close(&p.to_vector().unwrap(), &vv));
    assert!(close(&s.to_vector().unwrap(), &vv));
    let ev: Vec<C64> = e.to_vector().unwrap().iter().map(|a| a.to_complex()).collect();
    assert!(close(&ev, &vv));
    // the two exact families agree exactly
    assert_eq!(e.to_vector().unwrap(), se.to_vector().unwrap());
    assert!(e.norm_sq().unwrap().is_one());
    assert_eq!(c.non_clifford_count(), 5);
    assert!(se.num_terms() <= 32);
    for text in ["X0*Z1", "Y2", "Z0*Z1*Z2*Z3", "X3"] {
        let obs = Pauli::from_string(4, text).unwrap();
        let want = v.expectation(&obs).unwrap();
        assert!((p.expectation(&obs).unwrap() - want).norm() < 1e-9, "{text}");
        assert!((s.expectation(&obs).unwrap() - want).norm() < 1e-9, "{text}");
        assert!(
            (e.expectation(&obs).unwrap().to_complex() - want).norm() < 1e-9,
            "{text}"
        );
    }
    // a stabilizer tableau refuses the T gate
    let mut st = Stabilizer::zero_state(4);
    assert!(matches!(
        QuantumState::<C64>::run(&mut st, &c, &mut rng),
        Err(ncsim::SimError::Unsupported { .. })
    ));
}

#[test]
fn clifford_circuit_with_measurements_on_every_backend() {
    let text = "
QUBITS 3
H 0
CX 0 1
CX 1 2
S 2
MPP Z0*Z1
MPP X0*X1*X2
M 0
";
    let c = Circuit::parse(text).unwrap();
    // deterministic outcomes: Z0Z1 = +1, X0X1X2 after S2 = ... check against the vector
    let mut rng = StdRng::seed_from_u64(3);
    let mut v: StateVector<C64> = StateVector::zero_state(3);
    let mut st = Stabilizer::zero_state(3);
    let mut p: PathSum<Cyclo<8>> = PathSum::zero_state(3);
    let rv = v.run(&c, &mut rng).unwrap();
    let rs = QuantumState::<C64>::run(&mut st, &c, &mut rng).unwrap();
    let rp = p.run(&c, &mut rng).unwrap();
    // first measurement is determined (+1): false everywhere
    assert!(!rv[0] && !rs[0] && !rp[0]);
    // after the final Z0 measurement every backend is a basis-like state consistent with its record
    let z0 = Pauli::from_string(3, "Z0").unwrap();
    let sign = |b: bool| if b { -1.0 } else { 1.0 };
    assert!((v.expectation(&z0).unwrap().re / v.norm_sq().unwrap().re - sign(rv[2])).abs() < 1e-12);
    assert!((QuantumState::<C64>::expectation(&st, &z0).unwrap().re - sign(rs[2])).abs() < 1e-12);
    let pe = p.expectation(&z0).unwrap().to_complex().re / p.norm_sq().unwrap().to_complex().re;
    assert!((pe - sign(rp[2])).abs() < 1e-12);
}

#[test]
fn diagonal_gate_expectations_are_ncalg_exponential_sums() {
    // <+^m| D_f |+^m> = 2^{-m} Σ_x ζ^{f(x)} for f = x0 + 3 x1 x2 + 4 x0 x1 x2 (mod 8)
    let m = 3;
    let mut f = PhasePoly::new(m, 3);
    f.add_monomial(1, &[0]);
    f.add_monomial(3, &[1, 2]);
    f.add_monomial(4, &[0, 1, 2]);
    let want: Cyclo<8> = f.exp_sum().unwrap();
    let scale = Cyclo::<8>::sqrt2_pow(-2 * m as i64).unwrap();
    let want = want.mul_ref(&scale);
    let mut c = Circuit::new(m);
    for q in 0..m {
        c.h(q);
    }
    c.gate(Gate::Diagonal(f.clone()), &[0, 1, 2]);
    for q in 0..m {
        c.h(q);
    }
    let mut rng = StdRng::seed_from_u64(0);
    let mut p: PathSum<Cyclo<8>> = PathSum::zero_state(m);
    p.run(&c, &mut rng).unwrap();
    assert_eq!(p.amplitude(&bits::zero(m)).unwrap(), want);
    let mut s: StabilizerSum<Cyclo<8>> = StabilizerSum::zero_state(m);
    s.run(&c, &mut rng).unwrap();
    assert_eq!(s.amplitude(&bits::zero(m)).unwrap(), want);
    assert!(s.num_terms() <= 8); // three non-Clifford factors
}

#[test]
fn sampling_statistics_agree_across_backends() {
    let text = "
QUBITS 3
H 0
CX 0 1
T 1
H 1
CX 1 2
";
    let c = Circuit::parse(text).unwrap();
    let mut rng = StdRng::seed_from_u64(42);
    let mut v: StateVector<C64> = StateVector::zero_state(3);
    let mut p: PathSum<C64> = PathSum::zero_state(3);
    v.run(&c, &mut rng).unwrap();
    p.run(&c, &mut rng).unwrap();
    let shots = 600;
    let count = |samples: &[Vec<u64>]| -> Vec<usize> {
        let mut h = vec![0usize; 8];
        for s in samples {
            h[bits::to_u64(s, 3) as usize] += 1;
        }
        h
    };
    let hv = count(&v.sample(&mut rng, shots).unwrap());
    let hp = count(&p.sample(&mut rng, shots).unwrap());
    for x in 0..8usize {
        let prob = v.probability(&bits::from_u64(x as u64, 3)).unwrap();
        let expect = prob * shots as f64;
        let tol = 4.0 * (expect.max(1.0)).sqrt() + 5.0;
        assert!(
            (hv[x] as f64 - expect).abs() < tol,
            "vector histogram at {x}: {} vs {expect}",
            hv[x]
        );
        assert!(
            (hp[x] as f64 - expect).abs() < tol,
            "path-sum histogram at {x}: {} vs {expect}",
            hp[x]
        );
    }
}

/// `ψ ⊗ φ` on every family against the Kronecker product of the dense vectors.
#[test]
fn tensor_products_agree_with_kron() {
    use ncsim::{Mps, PauliKind};
    use rand::Rng;
    let mut rng = StdRng::seed_from_u64(11);
    let text_a = "QUBITS 3\nH 0 1\nT 0\nCX 0 2\nS 2\nH 2\nCCZ 0 1 2\n";
    let text_b = "QUBITS 2\nH 0\nCX 0 1\nT_DAG 1\nH 1\n";
    let (ca, cb) = (Circuit::parse(text_a).unwrap(), Circuit::parse(text_b).unwrap());
    let mut va: StateVector<C64> = StateVector::zero_state(3);
    let mut vb: StateVector<C64> = StateVector::zero_state(2);
    va.run(&ca, &mut rng).unwrap();
    vb.run(&cb, &mut rng).unwrap();
    let want = va.kron(&vb).unwrap().to_vector().unwrap();
    // Kronecker convention: index x_a + 8 x_b
    for xa in 0..8 {
        for xb in 0..4 {
            let a = va.amplitudes()[xa] * vb.amplitudes()[xb];
            assert!((want[xa + 8 * xb] - a).norm() < 1e-12);
        }
    }
    let mut pa: PathSum<C64> = PathSum::zero_state(3);
    let mut pb: PathSum<C64> = PathSum::zero_state(2);
    pa.run(&ca, &mut rng).unwrap();
    pb.run(&cb, &mut rng).unwrap();
    assert!(close(&pa.tensor(&pb).to_vector().unwrap(), &want));
    let mut sa: StabilizerSum<C64> = StabilizerSum::zero_state(3);
    let mut sb: StabilizerSum<C64> = StabilizerSum::zero_state(2);
    sa.run(&ca, &mut rng).unwrap();
    sb.run(&cb, &mut rng).unwrap();
    assert!(close(
        &QuantumState::tensor(&sa, &sb).unwrap().to_vector().unwrap(),
        &want
    ));
    let mut ma = Mps::zero_state(3);
    let mut mb = Mps::zero_state(2);
    ma.run(&ca, &mut rng).unwrap();
    mb.run(&cb, &mut rng).unwrap();
    let mt = ma.tensor(&mb).unwrap();
    assert!(close(&mt.to_vector().unwrap(), &want));
    // the canonical form survives: a gate across the seam and an expectation
    let mut mt2 = mt.clone();
    let mut vt = va.kron(&vb).unwrap();
    mt2.apply_gate(&Gate::Cx, &[1, 4]).unwrap();
    vt.apply_gate(&Gate::Cx, &[1, 4]).unwrap();
    assert!(close(&mt2.to_vector().unwrap(), &vt.to_vector().unwrap()));

    // stabilizer tableaux: tensor, tensor_all and embed against the vectors (up to phase)
    for trial in 0..30 {
        let ns: Vec<usize> = (0..3).map(|_| rng.gen_range(1..4)).collect();
        let states: Vec<Stabilizer> = ns
            .iter()
            .map(|&n| ncsim::clifford::random_stabilizer_state(n, &mut rng))
            .collect();
        let total: usize = ns.iter().sum();
        let t = Stabilizer::tensor_all(&states);
        let t2 = states[0].tensor(&states[1]).tensor(&states[2]);
        assert_eq!(t.table(), t2.table(), "trial {trial}");
        let mut v: StateVector<C64> =
            StateVector::from_amplitudes(ns[0], QuantumState::<C64>::to_vector(&states[0]).unwrap()).unwrap();
        for s in &states[1..] {
            let w = StateVector::from_amplitudes(s.num_qubits(), QuantumState::<C64>::to_vector(s).unwrap())
                .unwrap();
            v = v.kron(&w).unwrap();
        }
        for _ in 0..10 {
            let kinds: Vec<PauliKind> = (0..total)
                .map(|_| [PauliKind::I, PauliKind::X, PauliKind::Y, PauliKind::Z][rng.gen_range(0..4)])
                .collect();
            let p = Pauli::from_kinds(&kinds);
            let a: C64 = QuantumState::<C64>::expectation(&t, &p).unwrap();
            assert!(
                (a - v.expectation(&p).unwrap()).norm() < 1e-9,
                "trial {trial}: {p}"
            );
        }
        // embed state 0 on scattered qubits of a 6-qubit register
        let n0 = ns[0];
        let qubits: Vec<usize> = [4usize, 1, 5].iter().take(n0).cloned().collect();
        let e = states[0].embed(6, &qubits).unwrap();
        for _ in 0..10 {
            let kinds: Vec<PauliKind> = (0..n0)
                .map(|_| [PauliKind::I, PauliKind::X, PauliKind::Y, PauliKind::Z][rng.gen_range(0..4)])
                .collect();
            let p = Pauli::from_kinds(&kinds);
            assert_eq!(
                states[0].expectation_phasor(&p),
                e.expectation_phasor(&p.embed(6, &qubits))
            );
        }
        let rest: Vec<usize> = (0..6).filter(|q| !qubits.contains(q)).collect();
        for &q in &rest {
            assert!(e.is_stabilized_by(&Pauli::z(6, q)));
        }
    }
}

#[test]
fn tensor_of_large_tableaux_across_word_boundaries() {
    let mut rng = StdRng::seed_from_u64(12);
    let a = ncsim::clifford::random_stabilizer_state(70, &mut rng);
    let b = ncsim::clifford::random_stabilizer_state(61, &mut rng);
    let t = a.tensor(&b);
    let (ia, ib) = (Pauli::identity(70), Pauli::identity(61));
    for i in 0..70 {
        assert!(t.is_stabilized_by(&a.stabilizer(i).tensor(&ib)));
    }
    for i in 0..61 {
        assert!(t.is_stabilized_by(&ia.tensor(&b.stabilizer(i))));
        assert!(!t.is_stabilized_by(&ia.tensor(&b.destabilizer(i))));
    }
    let mut c = t.clone();
    let mut rng2 = StdRng::seed_from_u64(1);
    // measurements on the factors commute with the tensor product
    let p = ia.tensor(&b.destabilizer(3));
    let p = if p.is_hermitian() { p } else { p.times_i() };
    c.measure(&p, &mut rng2).unwrap();
    for i in 0..70 {
        assert!(c.is_stabilized_by(&a.stabilizer(i).tensor(&ib)));
    }
}

/// The phase-tracking tableau against the exact path sum on 100 qubits
/// (beyond one word), and the eigenvalue query on tableau states.
#[test]
fn chform_matches_pathsum_on_many_qubits() {
    use ncsim::ChForm;
    use rand::Rng;
    let mut rng = StdRng::seed_from_u64(21);
    let n = 100;
    let gates = [
        Gate::H,
        Gate::S,
        Gate::Sdg,
        Gate::X,
        Gate::Y,
        Gate::SqrtX,
        Gate::Cx,
        Gate::Cz,
        Gate::Cy,
        Gate::Swap,
    ];
    let mut c = ChForm::zero_state(n);
    let mut p: PathSum<C64> = PathSum::zero_state(n);
    for _ in 0..1500 {
        let g = gates[rng.gen_range(0..gates.len())].clone();
        let qs: Vec<usize> = if g.arity() == 1 {
            vec![rng.gen_range(0..n)]
        } else {
            let a = rng.gen_range(0..n);
            vec![a, (a + 1 + rng.gen_range(0..n - 1)) % n]
        };
        c.apply(&g, &qs).unwrap();
        p.apply_gate(&g, &qs).unwrap();
    }
    let (_, af) = c.to_affine_form();
    for _ in 0..20 {
        let mut x = af.shift.clone();
        for b in &af.basis {
            if rng.gen::<bool>() {
                bits::xor_into(&mut x, b);
            }
        }
        let a: C64 = QuantumState::<C64>::amplitude(&c, &x).unwrap();
        let b = p.amplitude(&x).unwrap();
        assert!((a - b).norm() < 1e-9 * b.norm().max(1e-30), "{a} vs {b}");
    }
    // projections and an exact overlap
    let z = Pauli::from_string(n, "Z3*X70*Y99").unwrap();
    let mut c2 = c.clone();
    let mut p2 = p.clone();
    let w1: C64 = QuantumState::<C64>::project(&mut c2, &z, true).unwrap();
    let w2 = p2.project(&z, true).unwrap();
    assert!((w1 - w2).norm() < 1e-9);
    let o1: C64 = QuantumState::<C64>::inner(&c, &c2).unwrap();
    let o2 = p.inner(&p2).unwrap();
    assert!((o1 - o2).norm() < 1e-9, "{o1} vs {o2}");
}

/// The three routes of `Stabilizer::apply_tableau` (lookup table, synthesised
/// circuit, row conjugation) agree, and `from_clifford` is `U|0⟩`.
#[test]
fn apply_tableau_routes_agree() {
    use ncsim::clifford::{random_clifford, random_stabilizer_state};
    use rand::Rng;
    let mut rng = StdRng::seed_from_u64(31);
    let n = 70;
    for &k in &[2usize, 3, 7, 33, 40] {
        let s0 = random_stabilizer_state(n, &mut rng);
        let u = random_clifford(k, &mut rng);
        let mut qs: Vec<usize> = Vec::new();
        while qs.len() < k {
            let q = rng.gen_range(0..n);
            if !qs.contains(&q) {
                qs.push(q);
            }
        }
        let mut a = s0.clone();
        a.apply_tableau(&u, &qs).unwrap();
        let mut b = s0.clone();
        for op in u.to_circuit().ops() {
            if let ncsim::Op::Gate { gate, qubits } = op {
                let mapped: Vec<usize> = qubits.iter().map(|&q| qs[q]).collect();
                b.apply_gate(gate, &mapped).unwrap();
            }
        }
        for i in 0..n {
            assert!(a.is_stabilized_by(&b.stabilizer(i)), "k = {k}");
        }
    }
    let u = random_clifford(9, &mut rng);
    let mut z = Stabilizer::zero_state(9);
    z.apply_tableau(&u, &(0..9).collect::<Vec<_>>()).unwrap();
    let f = Stabilizer::from_clifford(&u);
    for i in 0..9 {
        assert!(z.is_stabilized_by(&f.stabilizer(i)));
    }
}

/// Tableau and CH-form sampling (uniform on the affine support) against the
/// dense Born probabilities.
#[test]
fn stabilizer_sampling_matches_probabilities() {
    use ncsim::ChForm;
    let mut rng = StdRng::seed_from_u64(41);
    let c = Circuit::parse("QUBITS 4\nH 0 2\nCX 0 1\nS 1\nH 1\nCZ 1 3\nH 3\nCX 2 0\n").unwrap();
    let mut v: StateVector<C64> = StateVector::zero_state(4);
    let mut s = Stabilizer::zero_state(4);
    let mut h = ChForm::zero_state(4);
    v.run(&c, &mut rng).unwrap();
    s.run(&c, &mut rng).unwrap();
    h.apply_unitary_circuit(&c).unwrap();
    let shots = 40_000;
    for samples in [
        QuantumState::<C64>::sample(&s, &mut rng, shots).unwrap(),
        QuantumState::<C64>::sample(&h, &mut rng, shots).unwrap(),
    ] {
        let mut counts = [0usize; 16];
        for x in &samples {
            counts[bits::to_u64(x, 4) as usize] += 1;
        }
        for (i, &c) in counts.iter().enumerate() {
            let p = v.amplitudes()[i].norm_sqr();
            let f = c as f64 / shots as f64;
            assert!((f - p).abs() < 0.015, "index {i}: {f} vs {p}");
        }
    }
}
