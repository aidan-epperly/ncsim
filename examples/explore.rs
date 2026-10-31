//! Exploratory stabilizer-simulation runs with timings and diagnostics, to
//! find where the library spends its time at scale.
//!
//! ```text
//! cargo run --release --example explore -- tableau | ops | surface | clifford | chform
//! ```

#![allow(clippy::needless_range_loop, clippy::collapsible_match)]

use ncsim::pauli::ColumnTable;
use ncsim::{bits, ChForm, Circuit, FrameSampler, Gate, Noise, Pauli, PauliKind, Stabilizer};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use std::time::Instant;

fn secs<T>(f: impl FnOnce() -> T) -> (T, f64) {
    let t = Instant::now();
    let r = f();
    (r, t.elapsed().as_secs_f64())
}

fn random_state(n: usize, depth: usize, rng: &mut StdRng) -> Stabilizer {
    let mut s = Stabilizer::zero_state(n);
    for _ in 0..depth {
        for q in 0..n {
            s.apply_gate(&Gate::H, &[q]).unwrap();
            if rng.gen::<bool>() {
                s.apply_gate(&Gate::S, &[q]).unwrap();
            }
        }
        let mut perm: Vec<usize> = (0..n).collect();
        for i in (1..n).rev() {
            perm.swap(i, rng.gen_range(0..=i));
        }
        for p in perm.chunks(2) {
            if p.len() == 2 {
                s.apply_gate(&Gate::Cx, &[p[0], p[1]]).unwrap();
            }
        }
    }
    s
}

fn random_pauli(n: usize, w: usize, rng: &mut StdRng) -> Pauli {
    let mut kinds = vec![PauliKind::I; n];
    for _ in 0..w {
        kinds[rng.gen_range(0..n)] = [PauliKind::X, PauliKind::Y, PauliKind::Z][rng.gen_range(0..3)];
    }
    let p = Pauli::from_kinds(&kinds);
    if p.is_hermitian() {
        p
    } else {
        p.times_i()
    }
}

/// Gate and measurement costs of the column-major tableau as n grows.
fn tableau() {
    let mut rng = StdRng::seed_from_u64(1);
    println!("n,bytes,prep_gates,ns_per_gate,us_rand_z,us_det_z,us_w16,us_dense,us_expect_w4,ideal_gate_ns,ideal_meas_us");
    for &n in &[1000usize, 2000, 4000, 8000, 12000, 16000, 24000] {
        let depth = 4;
        let (mut s, tp) = secs(|| random_state(n, depth, &mut rng));
        let gates = depth * (n + n / 2 + n / 2);
        let bytes = 2 * n * (2 * n).div_ceil(64) * 8;
        // random-outcome single-qubit Z measurements
        let k = 50;
        let (_, tr) = secs(|| {
            for i in 0..k {
                s.measure(&Pauli::z(n, (i * 7919) % n), &mut rng).unwrap();
            }
        });
        // the same qubits again: now deterministic
        let (_, td) = secs(|| {
            for i in 0..k {
                s.measure(&Pauli::z(n, (i * 7919) % n), &mut rng).unwrap();
            }
        });
        let ps16: Vec<Pauli> = (0..k).map(|_| random_pauli(n, 16, &mut rng)).collect();
        let (_, tw) = secs(|| {
            for p in &ps16 {
                s.measure(p, &mut rng).unwrap();
            }
        });
        let kd = 10;
        let pd: Vec<Pauli> = (0..kd).map(|_| random_pauli(n, n / 2, &mut rng)).collect();
        let (_, tdn) = secs(|| {
            for p in &pd {
                s.measure(p, &mut rng).unwrap();
            }
        });
        let pe: Vec<Pauli> = (0..200).map(|_| random_pauli(n, 4, &mut rng)).collect();
        let (_, te) = secs(|| {
            for p in &pe {
                std::hint::black_box(s.expectation_phasor(p));
            }
        });
        // ideal: a gate touches 2 columns of 2n/64 words; a measurement ~ n columns of 2n/64 words
        let words = (2 * n).div_ceil(64) as f64;
        println!(
            "{n},{bytes},{gates},{:.1},{:.1},{:.1},{:.1},{:.1},{:.2},{:.1},{:.1}",
            tp / gates as f64 * 1e9,
            tr / k as f64 * 1e6,
            td / k as f64 * 1e6,
            tw / k as f64 * 1e6,
            tdn / kd as f64 * 1e6,
            te / 200.0 * 1e6,
            2.0 * words * 0.3,
            n as f64 * words * 0.3 / 1000.0
        );
    }
}

/// Where a random-outcome measurement spends its time: the ColumnTable primitives.
fn ops() {
    let mut rng = StdRng::seed_from_u64(2);
    println!("n,us_mask_w1,us_mask_dense,us_rowsum,us_row,us_set_row,us_product,us_measure");
    for &n in &[2000usize, 4000, 8000, 16000] {
        let s = random_state(n, 3, &mut rng);
        let t: ColumnTable = ColumnTable::from_table(&s.table());
        let rows = 2 * n;
        let reps = 20;
        let z = Pauli::z(n, n / 3);
        let (m1, a) = secs(|| {
            let mut m = Vec::new();
            for _ in 0..reps {
                m = t.anticommute_mask(&z);
            }
            m
        });
        let dense = random_pauli(n, n / 2, &mut rng);
        let (_, b) = secs(|| {
            for _ in 0..reps {
                std::hint::black_box(t.anticommute_mask(&dense));
            }
        });
        let src = bits::ones(&m1).into_iter().find(|&r| r >= n).unwrap_or(n);
        let mut mask = m1.clone();
        bits::mask_to(&mut mask, rows);
        bits::set(&mut mask, src, false);
        let mut tt = t.clone();
        let (_, c) = secs(|| {
            for _ in 0..reps {
                tt.rowsum_masked(&mask, src);
            }
        });
        let (row, d) = secs(|| {
            let mut r = Pauli::identity(n);
            for _ in 0..reps {
                r = t.row(src);
            }
            r
        });
        let (_, e) = secs(|| {
            for _ in 0..reps {
                tt.set_row(src - n, &row);
            }
        });
        let mut sel = bits::zero(rows);
        for i in n..rows {
            if rng.gen::<bool>() {
                bits::set(&mut sel, i, true);
            }
        }
        let (_, f) = secs(|| {
            for _ in 0..reps {
                std::hint::black_box(t.product_masked(&sel));
            }
        });
        let mut s2 = s.clone();
        let (_, g) = secs(|| {
            for i in 0..reps {
                s2.measure(&Pauli::z(n, (i * 7919 + 13) % n), &mut rng).unwrap();
            }
        });
        let u = |x: f64| x / reps as f64 * 1e6;
        println!(
            "{n},{:.1},{:.1},{:.1},{:.1},{:.1},{:.1},{:.1}",
            u(a),
            u(b),
            u(c),
            u(d),
            u(e),
            u(f),
            u(g)
        );
    }
}

/// A rotated surface-code memory-Z experiment with circuit-level depolarizing
/// noise (Stim's conventions): data on a d×d grid, X- and Z-type ancillas.
fn surface_circuit(d: usize, rounds: usize, p: f64) -> (Circuit, usize) {
    // coordinates: data at (2i+1, 2j+1); ancilla at (2i, 2j) for 0 <= i, j <= d
    let data = |i: usize, j: usize| i * d + j;
    let mut anc: Vec<(usize, usize, bool)> = Vec::new(); // (x, y, is_x_type)
    for i in 0..=d {
        for j in 0..=d {
            let x_type = (i + j) % 2 == 0;
            // boundaries: X-type ancillas only on the top/bottom rows (j interior),
            // Z-type only on the left/right columns (i interior)
            let on_lr = i == 0 || i == d;
            let on_tb = j == 0 || j == d;
            if on_lr && on_tb {
                continue;
            }
            if on_lr && x_type {
                continue;
            }
            if on_tb && !x_type {
                continue;
            }
            anc.push((i, j, x_type));
        }
    }
    let nd = d * d;
    let n = nd + anc.len();
    // neighbor data of ancilla (i, j): data (i-1|i, j-1|j); orders chosen so X and Z checks commute
    let nb = |i: usize, j: usize, k: usize, x_type: bool| -> Option<usize> {
        // offsets (di, dj) in {0,1}: data at (i - 1 + di, j - 1 + dj)
        let order_x = [(1, 1), (0, 1), (1, 0), (0, 0)];
        let order_z = [(1, 1), (1, 0), (0, 1), (0, 0)];
        let (di, dj) = if x_type { order_x[k] } else { order_z[k] };
        let (a, b) = (i as isize - 1 + di, j as isize - 1 + dj);
        (a >= 0 && b >= 0 && (a as usize) < d && (b as usize) < d).then(|| data(a as usize, b as usize))
    };
    let mut c = Circuit::new(n);
    let dep = |c: &mut Circuit, qs: &[usize]| {
        for &q in qs {
            c.noise(Noise::Depolarize1(p), &[q]).unwrap();
        }
    };
    let all_data: Vec<usize> = (0..nd).collect();
    let ancq: Vec<usize> = (0..anc.len()).map(|a| nd + a).collect();
    let mut prev: Option<Vec<usize>> = None; // measurement index of each ancilla last round
    let mut meas_count = 0usize;
    for r in 0..rounds {
        dep(&mut c, &all_data);
        for (a, &(_, _, x)) in anc.iter().enumerate() {
            if x {
                c.h(nd + a);
            }
        }
        dep(&mut c, &ancq);
        for k in 0..4 {
            for (a, &(i, j, x)) in anc.iter().enumerate() {
                if let Some(q) = nb(i, j, k, x) {
                    let (ctl, tgt) = if x { (nd + a, q) } else { (q, nd + a) };
                    c.cx(ctl, tgt);
                    c.noise(Noise::Depolarize2(p), &[ctl, tgt]).unwrap();
                }
            }
        }
        for (a, &(_, _, x)) in anc.iter().enumerate() {
            if x {
                c.h(nd + a);
            }
        }
        dep(&mut c, &ancq);
        let mut this = Vec::with_capacity(anc.len());
        for a in 0..anc.len() {
            c.noise(Noise::XError(p), &[nd + a]).unwrap();
            c.measure_z(nd + a);
            this.push(meas_count);
            meas_count += 1;
            c.reset(nd + a);
        }
        for (a, &(_, _, x)) in anc.iter().enumerate() {
            let back = |idx: usize| meas_count - idx;
            if r == 0 {
                if !x {
                    c.push(ncsim::Op::Detector {
                        lookback: vec![back(this[a])],
                    })
                    .unwrap();
                }
            } else {
                let pr = prev.as_ref().unwrap()[a];
                c.push(ncsim::Op::Detector {
                    lookback: vec![back(this[a]), back(pr)],
                })
                .unwrap();
            }
        }
        prev = Some(this);
    }
    // final data measurement in Z; Z-plaquette detectors from data; logical Z along a column
    let mut dmeas = vec![0usize; nd];
    for q in 0..nd {
        c.noise(Noise::XError(p), &[q]).unwrap();
        c.measure_z(q);
        dmeas[q] = meas_count;
        meas_count += 1;
    }
    for (a, &(i, j, x)) in anc.iter().enumerate() {
        if x {
            continue;
        }
        let mut lb: Vec<usize> = (0..4)
            .filter_map(|k| nb(i, j, k, false))
            .map(|q| meas_count - dmeas[q])
            .collect();
        lb.push(meas_count - prev.as_ref().unwrap()[a]);
        c.push(ncsim::Op::Detector { lookback: lb }).unwrap();
    }
    let logical: Vec<usize> = (0..d).map(|i| meas_count - dmeas[data(i, 0)]).collect();
    c.push(ncsim::Op::Observable {
        index: 0,
        lookback: logical,
    })
    .unwrap();
    (c, n)
}

fn surface() {
    let shots = 100_000;
    let mut rng = StdRng::seed_from_u64(3);
    println!("d,qubits,ops,measurements,detectors,s_reference,s_100k_shots,us_per_shot,noiseless_detector_hits,detection_fraction_p1e-3,observable_flip_raw");
    for &d in &[3usize, 5, 7, 11, 15, 21, 25, 31] {
        let (c, n) = surface_circuit(d, d, 1e-3);
        let quiet = c.without_noise();
        let fq = FrameSampler::new(&quiet, 0).unwrap();
        let (dq, _) = fq.sample_detectors(256, &mut rng).unwrap();
        let hits: usize = dq.iter().map(|b| bits::popcount(b)).sum();
        let (fs, tref) = secs(|| FrameSampler::new(&c, 0).unwrap());
        let ((det, obs), ts) = secs(|| fs.sample_detectors(shots, &mut rng).unwrap());
        let ndet = det.len();
        let frac = det.iter().map(|b| bits::popcount(b)).sum::<usize>() as f64 / (ndet * shots) as f64;
        let oflip = bits::popcount(&obs[0]) as f64 / shots as f64;
        println!(
            "{d},{n},{},{},{ndet},{tref:.3},{ts:.3},{:.2},{hits},{frac:.5},{oflip:.4}",
            c.len(),
            c.num_measurements(),
            ts / shots as f64 * 1e6
        );
    }
}

/// Uniform random Cliffords: sampling, synthesis and the state from them.
fn clifford() {
    let mut rng = StdRng::seed_from_u64(4);
    println!(
        "n,s_random_clifford,s_to_circuit,gates,s_from_clifford,s_from_stabilizers,s_affine_form,s_inner"
    );
    for &n in &[500usize, 1000, 2000, 4000] {
        let (u, ta) = secs(|| ncsim::clifford::random_clifford(n, &mut rng));
        let (c, tb) = secs(|| u.to_circuit());
        let (s, tc) = secs(|| Stabilizer::from_clifford(&u));
        let (t, td) = secs(|| Stabilizer::from_stabilizers(&s.stabilizers()).unwrap());
        let (_, te) = secs(|| s.affine_form());
        let (_, tf) = secs(|| s.inner_phasor(&t));
        println!(
            "{n},{ta:.3},{tb:.3},{},{tc:.3},{td:.3},{te:.3},{tf:.3}",
            c.num_gates()
        );
    }
}

/// CH-form against the tableau on the same Clifford circuit with measurements.
fn chform() {
    let mut rng = StdRng::seed_from_u64(5);
    println!("n,s_tableau_gates,s_chform_gates,frac_H,s_tableau_meas,s_chform_meas");
    for &n in &[500usize, 1000, 2000, 4000] {
        let mut ops: Vec<(Gate, Vec<usize>)> = Vec::new();
        for _ in 0..10 * n {
            let a = rng.gen_range(0..n);
            match rng.gen_range(0..4) {
                0 => ops.push((Gate::H, vec![a])),
                1 => ops.push((Gate::S, vec![a])),
                _ => ops.push((Gate::Cx, vec![a, (a + 1 + rng.gen_range(0..n - 1)) % n])),
            }
        }
        let mut st = Stabilizer::zero_state(n);
        let (_, ta) = secs(|| {
            for (g, q) in &ops {
                st.apply_gate(g, q).unwrap();
            }
        });
        let mut ch = ChForm::zero_state(n);
        let (_, tb) = secs(|| {
            for (g, q) in &ops {
                ch.apply(g, q).unwrap();
            }
        });
        let qs: Vec<usize> = (0..200).map(|_| rng.gen_range(0..n)).collect();
        let (_, tc) = secs(|| {
            for &q in &qs {
                st.measure(&Pauli::z(n, q), &mut rng).unwrap();
            }
        });
        let (_, td) = secs(|| {
            for &q in &qs {
                ch.measure_pauli(&Pauli::z(n, q), &mut rng).unwrap();
            }
        });
        println!("{n},{ta:.3},{tb:.3},0.25,{tc:.3},{td:.3}");
    }
}

fn main() {
    match std::env::args().nth(1).as_deref() {
        Some("tableau") => tableau(),
        Some("ops") => ops(),
        Some("surface") => surface(),
        Some("clifford") => clifford(),
        Some("chform") => chform(),
        Some("framemem") => {
            // detectors of a d = 21 memory experiment for a million shots: time and record size
            let mut rng = StdRng::seed_from_u64(10);
            let (c, n) = surface_circuit(21, 21, 1e-3);
            let fs = FrameSampler::new(&c, 0).unwrap();
            let shots = 1_000_000;
            let ((det, obs), t) = secs(|| fs.sample_detectors(shots, &mut rng).unwrap());
            let raw_bytes = c.num_measurements() * shots / 8;
            let det_bytes = det.len() * shots / 8;
            println!(
                "d 21: {n} qubits, {} measurements, {} detectors, {shots} shots in {t:.2} s; raw records {:.2} GB, detectors {:.2} GB, observable flips {}",
                c.num_measurements(),
                det.len(),
                raw_bytes as f64 / 1e9,
                det_bytes as f64 / 1e9,
                bits::popcount(&obs[0])
            );
        }
        Some("cliffparts") => {
            let mut rng = StdRng::seed_from_u64(11);
            println!("n,s_random_clifford,s_is_valid,s_inverse,s_compose_identity");
            for n in [1000usize, 2000, 4000] {
                let (u, a) = secs(|| ncsim::clifford::random_clifford(n, &mut rng));
                let (_, b) = secs(|| std::hint::black_box(u.is_valid()));
                let (inv, c) = secs(|| u.inverse());
                let (_, d) = secs(|| std::hint::black_box(u.compose(&inv)));
                println!("{n},{a:.3},{b:.3},{c:.3},{d:.3}");
            }
        }
        Some("refprof") => {
            // the noiseless reference run of surface-code memory circuits, split by operation kind
            let mut rng = StdRng::seed_from_u64(12);
            println!("d,qubits,s_total,s_gates,n_rand,s_rand,n_det,s_det,us_rand,us_det");
            for d in [11usize, 21, 31, 41] {
                let (c, n) = surface_circuit(d, d, 1e-3);
                let c = c.without_noise();
                let mut st = Stabilizer::zero_state(n);
                let (mut tg, mut tr, mut td) = (0.0, 0.0, 0.0);
                let (mut nr, mut nd) = (0usize, 0usize);
                let t0 = Instant::now();
                let mut meas = |st: &mut Stabilizer, p: &Pauli, rng: &mut StdRng| -> bool {
                    let random = !st.is_deterministic(p);
                    let t = Instant::now();
                    let b = st.measure(p, rng).unwrap();
                    let e = t.elapsed().as_secs_f64();
                    if random {
                        nr += 1;
                        tr += e;
                    } else {
                        nd += 1;
                        td += e;
                    }
                    b
                };
                for op in c.ops() {
                    match op {
                        ncsim::Op::Gate { gate, qubits } => {
                            let t = Instant::now();
                            st.apply_gate(gate, qubits).unwrap();
                            tg += t.elapsed().as_secs_f64();
                        }
                        ncsim::Op::Measure { pauli } => {
                            meas(&mut st, pauli, &mut rng);
                        }
                        ncsim::Op::Reset { qubit } => {
                            if meas(&mut st, &Pauli::z(n, *qubit), &mut rng) {
                                st.apply_gate(&Gate::X, &[*qubit]).unwrap();
                            }
                        }
                        _ => {}
                    }
                }
                let tt = t0.elapsed().as_secs_f64();
                println!(
                    "{d},{n},{tt:.3},{tg:.3},{nr},{tr:.3},{nd},{td:.3},{:.1},{:.1}",
                    tr / nr.max(1) as f64 * 1e6,
                    td / nd.max(1) as f64 * 1e6
                );
            }
        }
        Some("surfcheck") => {
            let mut rng = StdRng::seed_from_u64(9);
            for d in [3usize, 5] {
                let (c, n) = surface_circuit(d, d, 1e-3);
                let fq = FrameSampler::new(&c.without_noise(), 0).unwrap();
                let (dq, oq) = fq.sample_detectors(512, &mut rng).unwrap();
                let hits: Vec<usize> = dq.iter().map(|b| bits::popcount(b)).collect();
                println!(
                    "d {d}: {n} qubits, {} detectors, noiseless hits {:?}, observable hits {}",
                    dq.len(),
                    hits.iter().sum::<usize>(),
                    bits::popcount(&oq[0])
                );
                if hits.iter().any(|&h| h > 0) {
                    println!(
                        "  firing detectors: {:?}",
                        hits.iter()
                            .enumerate()
                            .filter(|(_, &h)| h > 0)
                            .map(|(i, _)| i)
                            .collect::<Vec<_>>()
                    );
                }
            }
        }
        _ => eprintln!("usage: explore tableau|ops|surface|clifford|chform"),
    }
}
