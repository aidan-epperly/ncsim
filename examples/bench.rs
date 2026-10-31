//! Timing of the hot paths of each family, for regressions and tuning.
//!
//! ```text
//! cargo run --release --example bench [-- which]
//! ```
//! `which` is one of `tableau`, `chform`, `vector`, `sparse`, `pathsum`, `stabsum`, `frames`, `mps`, `all` (default).

use ncsim::{Circuit, Gate, Mps, MpsOptions, PathSum, Pauli, QuantumState, Stabilizer, StateVector, C64};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use std::time::Instant;

fn random_clifford(n: usize, gates: usize, rng: &mut StdRng) -> Circuit {
    let mut c = Circuit::new(n);
    let one = [Gate::H, Gate::S, Gate::Sdg, Gate::X, Gate::SqrtX];
    let two = [Gate::Cx, Gate::Cz, Gate::Swap];
    for _ in 0..gates {
        if rng.gen::<bool>() {
            c.gate(one[rng.gen_range(0..one.len())].clone(), &[rng.gen_range(0..n)]);
        } else {
            let a = rng.gen_range(0..n);
            let mut b = rng.gen_range(0..n - 1);
            if b >= a {
                b += 1;
            }
            c.gate(two[rng.gen_range(0..two.len())].clone(), &[a, b]);
        }
    }
    c
}

fn time<T>(label: &str, f: impl FnOnce() -> T) -> T {
    let t = Instant::now();
    let r = f();
    println!("  {label:<44} {:>10.3} ms", t.elapsed().as_secs_f64() * 1e3);
    r
}

fn tableau() {
    let mut rng = StdRng::seed_from_u64(1);
    for &n in &[100usize, 1000, 4000] {
        println!("tableau n = {n}");
        let gates = 20 * n;
        let c = random_clifford(n, gates, &mut rng);
        let mut s = Stabilizer::zero_state(n);
        time(&format!("{gates} random gates"), || {
            for op in c.ops() {
                if let ncsim::Op::Gate { gate, qubits } = op {
                    QuantumState::<C64>::apply_gate(&mut s, gate, qubits).unwrap();
                }
            }
        });
        let ms = n.min(1000);
        let mut t = s.clone();
        time(&format!("{ms} single-qubit Z measurements"), || {
            for q in 0..ms {
                t.measure_pauli(&Pauli::z(n, q), &mut rng).unwrap();
            }
        });
        time(&format!("{ms} deterministic Z measurements"), || {
            for q in 0..ms {
                t.measure_pauli(&Pauli::z(n, q), &mut rng).unwrap();
            }
        });
        let ks: Vec<Pauli> = (0..200)
            .map(|_| {
                let kinds: Vec<ncsim::PauliKind> = (0..n)
                    .map(|_| {
                        [
                            ncsim::PauliKind::I,
                            ncsim::PauliKind::X,
                            ncsim::PauliKind::Y,
                            ncsim::PauliKind::Z,
                        ][rng.gen_range(0..4)]
                    })
                    .collect();
                Pauli::from_kinds(&kinds)
            })
            .collect();
        time("200 dense-Pauli expectations", || {
            for p in &ks {
                std::hint::black_box(s.expectation_phasor(p));
            }
        });
        if n <= 1000 {
            time("affine form", || std::hint::black_box(s.affine_form()));
            let s2 = t.clone();
            time("inner product", || std::hint::black_box(s.inner_phasor(&s2)));
        }
    }
}

fn chform() {
    let mut rng = StdRng::seed_from_u64(4);
    for &n in &[100usize, 1000, 4000] {
        println!("ch-form n = {n}");
        let gates = 20 * n;
        let c = random_clifford(n, gates, &mut rng);
        let mut s = ncsim::ChForm::zero_state(n);
        time(&format!("{gates} random gates"), || {
            for op in c.ops() {
                if let ncsim::Op::Gate { gate, qubits } = op {
                    s.apply(gate, qubits).unwrap();
                }
            }
        });
        let mut t = s.clone();
        let ms = n.min(1000);
        time(&format!("{ms} single-qubit Z measurements"), || {
            for q in 0..ms {
                t.measure_pauli(&Pauli::z(n, q), &mut rng).unwrap();
            }
        });
        if n <= 1000 {
            time("inner product", || std::hint::black_box(s.inner_phasor(&t)));
        }
    }
}

fn vector() {
    let mut rng = StdRng::seed_from_u64(2);
    for &n in &[16usize, 22] {
        println!("state vector n = {n}");
        let mut v: StateVector<C64> = StateVector::zero_state(n);
        time("H on every qubit", || {
            for q in 0..n {
                v.apply_gate(&Gate::H, &[q]).unwrap();
            }
        });
        time("T, CX, CCZ, Rx layer", || {
            for q in 0..n {
                v.apply_gate(&Gate::T, &[q]).unwrap();
                v.apply_gate(&Gate::Cx, &[q, (q + 1) % n]).unwrap();
                v.apply_gate(&Gate::Ccz, &[q, (q + 1) % n, (q + 2) % n]).unwrap();
                v.apply_gate(&Gate::Rx(0.3), &[q]).unwrap();
            }
        });
        let p = Pauli::from_string(n, "X0*Y3*Z5").unwrap();
        time("20 expectations", || {
            for _ in 0..20 {
                std::hint::black_box(v.expectation(&p).unwrap());
            }
        });
        time("10 projections", || {
            for q in 0..10 {
                let mut w = v.clone();
                w.project(&Pauli::z(n, q), rng.gen()).unwrap();
            }
        });
    }
}

fn pathsum() {
    let mut rng = StdRng::seed_from_u64(3);
    for &n in &[40usize, 200, 1000] {
        println!("path sum n = {n}");
        let c = random_clifford(n, 10 * n, &mut rng);
        let mut p: PathSum<C64> = PathSum::zero_state(n);
        time(&format!("{} random Clifford gates", 10 * n), || {
            for op in c.ops() {
                if let ncsim::Op::Gate { gate, qubits } = op {
                    p.apply_gate(gate, qubits).unwrap();
                }
            }
        });
        time("10 T gates", || {
            for q in 0..10 {
                p.apply_gate(&Gate::T, &[q]).unwrap();
            }
        });
        time("amplitude", || {
            std::hint::black_box(p.amplitude(&ncsim::bits::zero(n)).unwrap())
        });
        time("norm", || std::hint::black_box(p.norm_sq().unwrap()));
    }
}

fn stabsum_run<T: ncsim::StabilizerTerm<C64>>(label: &str, n: usize, t: usize, rng: &mut StdRng) {
    println!("stabilizer sum ({label} terms) n = {n}, {t} T gates");
    let mut s: ncsim::StabilizerSum<C64, T> = ncsim::StabilizerSum::zero_state(n);
    let c = random_clifford(n, 5 * n, rng);
    time(
        &format!("{} Clifford gates, {t} T, {} more", 5 * n, 5 * n),
        || {
            for op in c.ops() {
                if let ncsim::Op::Gate { gate, qubits } = op {
                    s.apply_gate(gate, qubits).unwrap();
                }
            }
            for q in 0..t {
                s.apply_gate(&Gate::T, &[q]).unwrap();
            }
            for op in c.ops() {
                if let ncsim::Op::Gate { gate, qubits } = op {
                    s.apply_gate(gate, qubits).unwrap();
                }
            }
        },
    );
    println!("  terms {}", s.num_terms());
    time("<Z0 Z1>", || {
        std::hint::black_box(s.expectation(&Pauli::from_string(n, "Z0*Z1").unwrap()).unwrap())
    });
}

fn gadgets() {
    // 30 qubits, 16 T gates: projector factors (2^t terms) vs magic-state gadgets (2^{t/2})
    let mut rng = StdRng::seed_from_u64(12);
    let n = 30;
    let mut c = random_clifford(n, 60, &mut rng);
    for k in 0..16 {
        c.t(k % n);
        let more = random_clifford(n, 20, &mut rng);
        c.append(&more).unwrap();
    }
    let z = Pauli::from_string(n, "Z0*Z1").unwrap();
    let mut a: ncsim::StabilizerSum<C64> = ncsim::StabilizerSum::zero_state(n);
    time("projector factors: run", || a.run(&c, &mut rng).unwrap());
    let ea = time(&format!("  <Z0 Z1> over {} terms", a.num_terms()), || {
        a.expectation(&z).unwrap() / a.norm_sq().unwrap()
    });
    let (b, t) = time("gadgets: run", || {
        ncsim::StabilizerSum::<C64, ncsim::ChTerm<C64>>::run_with_t_gadgets(&c, &mut rng).unwrap()
    });
    let zb = z.tensor(&Pauli::identity(t));
    let eb = time(&format!("  <Z0 Z1> over {} terms", b.num_terms()), || {
        b.expectation(&zb).unwrap() / b.norm_sq().unwrap()
    });
    println!("  agreement {:.1e}", (ea - eb).norm());
    let est = time("  Bravyi–Gosset estimate of <Z0 Z1>, 1000 samples", || {
        b.estimate_expectation(&zb, 1000, &mut rng).unwrap()
    });
    println!("  estimate {est:.3} vs exact {:.3}", eb.re);
}

fn stabsum() {
    let mut rng = StdRng::seed_from_u64(5);
    for &n in &[50usize, 200] {
        stabsum_run::<PathSum<C64>>("path-sum", n, 6, &mut rng);
        stabsum_run::<ncsim::ChTerm<C64>>("CH-form", n, 6, &mut rng);
    }
    stabsum_run::<ncsim::ChTerm<C64>>("CH-form", 1000, 6, &mut rng);
}

fn sparse() {
    // a ripple of Toffolis and CNOTs on 1000 qubits over 2^10 superposed inputs
    let n = 1000;
    println!("sparse vector n = {n}, 1024 nonzero amplitudes");
    let mut s: ncsim::SparseVector<C64> = ncsim::SparseVector::zero_state(n);
    for q in 0..10 {
        s.apply_gate(&Gate::H, &[q]).unwrap();
    }
    time(&format!("{} CCX + CX gates", 2 * (n - 2)), || {
        for q in 0..n - 2 {
            s.apply_gate(&Gate::Ccx, &[q, q + 1, q + 2]).unwrap();
            s.apply_gate(&Gate::Cx, &[q, q + 2]).unwrap();
        }
    });
    time("<Z0 Z999>", || {
        std::hint::black_box(s.expectation(&Pauli::from_string(n, "Z0*Z999").unwrap()).unwrap())
    });
}

fn frames() {
    // 1000 qubits: 20 rounds of random CX layers with depolarizing noise and
    // 100 Z measurements per round
    let n = 1000;
    let mut rng = StdRng::seed_from_u64(9);
    let mut c = Circuit::new(n);
    for q in 0..n / 2 {
        c.h(q);
    }
    for _ in 0..20 {
        for q in (0..n).step_by(2) {
            let b = (q + 1 + rng.gen_range(0..n - 1)) % n;
            c.cx(q, b);
            c.noise(ncsim::Noise::Depolarize2(1e-3), &[q, b]).unwrap();
        }
        for _ in 0..100 {
            c.measure_z(rng.gen_range(0..n));
        }
    }
    let fs = time("frame sampler: reference run", || {
        ncsim::FrameSampler::new(&c, 0).unwrap()
    });
    println!(
        "frames n = {n}, {} gates, {} measurements",
        c.num_gates(),
        c.num_measurements()
    );
    time("100 000 shots", || {
        std::hint::black_box(fs.sample_packed(100_000, &mut rng).unwrap())
    });
}

fn mps() {
    let n = 40;
    println!("mps n = {n}");
    let mut m = Mps::zero_state(n).with_options(MpsOptions::default());
    time("brickwork depth 8 (H, T, CX)", || {
        for layer in 0..8 {
            for q in 0..n {
                m.apply_gate(&Gate::H, &[q]).unwrap();
                m.apply_gate(&Gate::T, &[q]).unwrap();
            }
            for q in (layer % 2..n - 1).step_by(2) {
                m.apply_gate(&Gate::Cx, &[q, q + 1]).unwrap();
            }
        }
    });
    time("expectation", || {
        std::hint::black_box(m.expectation(&Pauli::from_string(n, "Z0*Z20").unwrap()).unwrap())
    });
}

fn main() {
    let which = std::env::args().nth(1).unwrap_or_else(|| "all".into());
    let all = which == "all";
    if all || which == "tableau" {
        tableau();
    }
    if all || which == "chform" {
        chform();
    }
    if all || which == "vector" {
        vector();
    }
    if all || which == "pathsum" {
        pathsum();
    }
    if all || which == "gadgets" {
        gadgets();
    }
    if all || which == "stabsum" {
        stabsum();
    }
    if all || which == "sparse" {
        sparse();
    }
    if all || which == "frames" {
        frames();
    }
    if all || which == "mps" {
        mps();
    }
}
