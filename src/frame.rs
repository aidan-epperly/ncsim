//! Pauli-frame sampling of noisy Clifford circuits (the method of Stim's
//! frame simulator).
//!
//! One noiseless run on a stabilizer tableau gives a *reference* measurement
//! record.  Every shot is then that reference times a Pauli *frame*: the
//! accumulated error, conjugated through the gates.  A measurement of `P`
//! records `reference ⊕ [frame anticommutes with P]`.  The frame is randomized
//! by stabilizers of the current state so that random outcomes come out
//! random:
//! - initially by `Z_q` on every qubit;
//! - after measuring `P`, by `P`;
//! - after a reset, by `Z_q`.
//!
//! The frames of many shots are the rows of a [`ColumnTable`], so a gate costs
//! a few word operations per qubit for 64 shots at a time; batches of shots
//! run in parallel.  Noise channels inject Pauli errors into the frames, the
//! shots hit drawn by geometric skipping (`O(p · shots)` for small `p`).

use crate::bits::{self, Bits};
use crate::circuit::{Circuit, Noise, Op};
use crate::error::{Result, SimError};
use crate::gate::Gate;
use crate::pauli::ColumnTable;
use crate::state::Stabilizer;
use rand::rngs::StdRng;
use rand::{Rng, RngCore, SeedableRng};
use rayon::prelude::*;

/// Shots per parallel batch (a multiple of 64).
const BATCH: usize = 1 << 14;

/// Bulk sampler of a noisy Clifford circuit (see the [module documentation](self)).
#[derive(Clone, Debug)]
pub struct FrameSampler {
    circuit: Circuit,
    reference: Vec<bool>,
}

impl FrameSampler {
    /// Check that every gate is Clifford and compute the reference record
    /// (a noiseless run with the outcomes drawn from `reference_seed`).
    pub fn new(circuit: &Circuit, reference_seed: u64) -> Result<Self> {
        for op in circuit.ops() {
            if let Op::Gate { gate, .. } = op {
                if !gate.is_clifford() {
                    return Err(SimError::Unsupported {
                        backend: "FrameSampler",
                        what: format!("non-Clifford gate {gate}"),
                    });
                }
            }
        }
        let mut st = Stabilizer::zero_state(circuit.num_qubits());
        let mut rng = StdRng::seed_from_u64(reference_seed);
        let reference = st.run(&circuit.without_noise(), &mut rng)?;
        Ok(FrameSampler {
            circuit: circuit.clone(),
            reference,
        })
    }
    /// The circuit.
    pub fn circuit(&self) -> &Circuit {
        &self.circuit
    }
    /// The noiseless reference record.
    pub fn reference(&self) -> &[bool] {
        &self.reference
    }
    /// `shots` measurement records, packed by measurement: entry `k` holds
    /// the outcomes of measurement `k` over the shots (bit `s` = shot `s`).
    pub fn sample_packed(&self, shots: usize, rng: &mut dyn RngCore) -> Result<Vec<Bits>> {
        let m = self.reference.len();
        let batches: Vec<(usize, u64)> = (0..shots.div_ceil(BATCH))
            .map(|b| ((shots - b * BATCH).min(BATCH), rng.next_u64()))
            .collect();
        let parts: Vec<Vec<Bits>> = batches
            .par_iter()
            .map(|&(size, seed)| self.run_batch(size, seed))
            .collect::<Result<_>>()?;
        let mut out: Vec<Bits> = vec![bits::zero(shots); m];
        for (b, part) in parts.iter().enumerate() {
            let offset = b * BATCH;
            let size = batches[b].0;
            for (k, rec) in part.iter().enumerate() {
                bits::or_shifted(&mut out[k], rec, size, offset);
            }
        }
        Ok(out)
    }
    /// `shots` measurement records, one per shot (bit `k` = measurement `k`).
    pub fn sample(&self, shots: usize, rng: &mut dyn RngCore) -> Result<Vec<Bits>> {
        let packed = self.sample_packed(shots, rng)?;
        let m = self.reference.len();
        let mut out = vec![bits::zero(m); shots];
        for (k, col) in packed.iter().enumerate() {
            for s in bits::ones(col) {
                if s < shots {
                    bits::set(&mut out[s], k, true);
                }
            }
        }
        Ok(out)
    }

    fn run_batch(&self, shots: usize, seed: u64) -> Result<Vec<Bits>> {
        let n = self.circuit.num_qubits();
        let mut rng = StdRng::seed_from_u64(seed);
        let mut t = ColumnTable::new(n, shots);
        let random_mask = |rng: &mut StdRng| -> Bits {
            let mut b: Bits = (0..bits::nwords(shots)).map(|_| rng.next_u64()).collect();
            bits::mask_to(&mut b, shots);
            b
        };
        for q in 0..n {
            let r = random_mask(&mut rng);
            t.cols_mut(q).1.copy_from_slice(&r);
        }
        let mut out = Vec::with_capacity(self.reference.len());
        let mut k = 0usize;
        for op in self.circuit.ops() {
            match op {
                Op::Gate { gate, qubits } => apply_frame_gate(&mut t, gate, qubits)?,
                Op::Measure { pauli } => {
                    let mut rec = t.anticommute_mask(pauli);
                    bits::mask_to(&mut rec, shots);
                    if self.reference[k] {
                        for w in rec.iter_mut() {
                            *w = !*w;
                        }
                        bits::mask_to(&mut rec, shots);
                    }
                    out.push(rec);
                    k += 1;
                    let r = random_mask(&mut rng);
                    t.xor_pauli_masked(pauli, &r);
                }
                Op::Reset { qubit } => {
                    let r = random_mask(&mut rng);
                    let (x, z) = t.cols_mut(*qubit);
                    x.fill(0);
                    z.copy_from_slice(&r);
                }
                Op::Noise { channel, qubits } => inject(&mut t, channel, qubits, shots, &mut rng),
                Op::Tick | Op::Detector { .. } | Op::Observable { .. } => {}
            }
        }
        Ok(out)
    }

    /// Detector and observable *flips* over the shots, packed (bit `s` = shot
    /// `s`): each is the parity of its measurement results relative to the
    /// noiseless reference, so a noiseless circuit gives all zeros.
    pub fn sample_detectors(&self, shots: usize, rng: &mut dyn RngCore) -> Result<(Vec<Bits>, Vec<Bits>)> {
        // detectors are formed batch by batch: the raw records of one batch
        // (not of all shots) are held at a time
        let dets = self.circuit.detectors();
        let obss = self.circuit.observables();
        let ref_flip = |idx: &[usize]| idx.iter().filter(|&&k| self.reference[k]).count() % 2 == 1;
        let dflip: Vec<bool> = dets.iter().map(|d| ref_flip(d)).collect();
        let oflip: Vec<bool> = obss.iter().map(|o| ref_flip(o)).collect();
        let batches: Vec<(usize, u64)> = (0..shots.div_ceil(BATCH))
            .map(|b| ((shots - b * BATCH).min(BATCH), rng.next_u64()))
            .collect();
        let mut det: Vec<Bits> = vec![bits::zero(shots); dets.len()];
        let mut obs: Vec<Bits> = vec![bits::zero(shots); obss.len()];
        // groups of one batch per thread: only a group's detectors are held
        // besides the result
        let group = rayon::current_num_threads().max(1);
        for (g, chunk) in batches.chunks(group).enumerate() {
            let parts: Vec<(Vec<Bits>, Vec<Bits>)> = chunk
                .par_iter()
                .map(|&(size, seed)| {
                    let rec = self.run_batch(size, seed)?;
                    let form = |idx: &[Vec<usize>], fl: &[bool]| -> Vec<Bits> {
                        idx.iter()
                            .zip(fl)
                            .map(|(d, &f)| {
                                let mut b = parity(&rec, d);
                                if f {
                                    for w in b.iter_mut() {
                                        *w = !*w;
                                    }
                                }
                                bits::mask_to(&mut b, size);
                                b
                            })
                            .collect()
                    };
                    Ok((form(&dets, &dflip), form(&obss, &oflip)))
                })
                .collect::<Result<_>>()?;
            for (i, (pd, po)) in parts.iter().enumerate() {
                let b = g * group + i;
                let (offset, size) = (b * BATCH, batches[b].0);
                for (k, v) in pd.iter().enumerate() {
                    bits::or_shifted(&mut det[k], v, size, offset);
                }
                for (k, v) in po.iter().enumerate() {
                    bits::or_shifted(&mut obs[k], v, size, offset);
                }
            }
        }
        Ok((det, obs))
    }
}

/// Conjugate the frames by a Clifford gate (signs are irrelevant for frames).
fn apply_frame_gate(t: &mut ColumnTable, gate: &Gate, qubits: &[usize]) -> Result<()> {
    match gate {
        Gate::I | Gate::X | Gate::Y | Gate::Z | Gate::Pauli(_) => {}
        Gate::H => t.apply_h(qubits[0]),
        Gate::S | Gate::Sdg => t.apply_s_pow(qubits[0], 1),
        Gate::Cx => t.apply_cx(qubits[0], qubits[1]),
        Gate::Cz => t.apply_cz(qubits[0], qubits[1]),
        Gate::Swap => t.apply_swap(qubits[0], qubits[1]),
        _ => {
            if let Some(lut) = gate.cached_lut() {
                t.apply_lut(lut, qubits);
                return Ok(());
            }
            let tab = gate.tableau().ok_or_else(|| SimError::Unsupported {
                backend: "FrameSampler",
                what: format!("non-Clifford gate {gate}"),
            })?;
            if tab.num_qubits() <= 3 {
                t.apply_lut(&tab.local_lut(), qubits);
            } else {
                for op in tab.to_circuit().ops() {
                    if let Op::Gate { gate: g, qubits: qs } = op {
                        let mapped: Vec<usize> = qs.iter().map(|&q| qubits[q]).collect();
                        apply_frame_gate(t, g, &mapped)?;
                    }
                }
            }
        }
    }
    Ok(())
}

/// The shots hit by an event of probability `p`, by geometric skipping.
fn hits(shots: usize, p: f64, rng: &mut StdRng) -> Vec<usize> {
    if p <= 0.0 {
        return Vec::new();
    }
    if p >= 0.25 {
        return (0..shots).filter(|_| rng.gen::<f64>() < p).collect();
    }
    let l = (1.0 - p).ln();
    let mut out = Vec::new();
    let mut s = 0usize;
    loop {
        let u: f64 = rng.gen::<f64>();
        let gap = ((1.0 - u).ln() / l).floor();
        if !gap.is_finite() || gap >= (shots - s) as f64 {
            break;
        }
        s += gap as usize;
        out.push(s);
        s += 1;
        if s >= shots {
            break;
        }
    }
    out
}

/// Inject one application of a noise channel into the frames.
fn inject(t: &mut ColumnTable, ch: &Noise, qubits: &[usize], shots: usize, rng: &mut StdRng) {
    let errors = ch.errors();
    let total: f64 = errors.iter().map(|(p, _, _)| p).sum();
    for s in hits(shots, total, rng) {
        // which error, given that one happened
        let mut u = rng.gen::<f64>() * total;
        let mut pick = errors.len() - 1;
        for (i, (p, _, _)) in errors.iter().enumerate() {
            if u < *p {
                pick = i;
                break;
            }
            u -= p;
        }
        let (_, x, z) = errors[pick];
        let (w, m) = (s / 64, 1u64 << (s % 64));
        for (j, &q) in qubits.iter().enumerate() {
            let (xc, zc) = t.cols_mut(q);
            if (x >> j) & 1 == 1 {
                xc[w] ^= m;
            }
            if (z >> j) & 1 == 1 {
                zc[w] ^= m;
            }
        }
    }
}

/// The parity `⊕_k record[k]` over a set of measurement indices for packed
/// records (`sample_packed`): a detector or observable, per shot.
pub fn parity(packed: &[Bits], indices: &[usize]) -> Bits {
    let mut out = packed
        .first()
        .map(|b| vec![0u64; b.len()])
        .unwrap_or_else(|| vec![0]);
    for &k in indices {
        bits::xor_into(&mut out, &packed[k]);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::scalar::C64;
    use crate::state::QuantumState;

    /// Measurement-record histograms of the frame sampler against stabilizer
    /// trajectories (`QuantumState::run` with sampled noise).
    fn compare(text: &str, shots: usize, tol: f64) {
        let c = Circuit::parse(text).unwrap();
        let fs = FrameSampler::new(&c, 7).unwrap();
        let mut rng = StdRng::seed_from_u64(11);
        let frames = fs.sample(shots, &mut rng).unwrap();
        let m = c.num_measurements();
        let key = |b: &Bits| bits::to_u64(b, m) as usize;
        let mut hf = vec![0f64; 1 << m];
        for r in &frames {
            hf[key(r)] += 1.0 / shots as f64;
        }
        let mut ht = vec![0f64; 1 << m];
        for _ in 0..shots {
            let mut st = Stabilizer::zero_state(c.num_qubits());
            let rec = QuantumState::<C64>::run(&mut st, &c, &mut rng).unwrap();
            ht[key(&bits::from_bools(&rec))] += 1.0 / shots as f64;
        }
        let tv: f64 = hf.iter().zip(&ht).map(|(a, b)| (a - b).abs()).sum::<f64>() / 2.0;
        assert!(
            tv < tol,
            "total variation {tv} for\n{text}\nframes {hf:?}\ntrajectories {ht:?}"
        );
    }

    #[test]
    fn noiseless_records_match_trajectories() {
        compare("QUBITS 3\nH 0\nCX 0 1\nCX 1 2\nM 0 1 2\nMX 0\nM 2\n", 20000, 0.03);
        compare(
            "QUBITS 2\nH 0\nS 0\nCX 0 1\nMY 0\nMPP X0*X1 Z0*Z1\nR 0\nM 0\nH 1\nM 1\n",
            20000,
            0.03,
        );
    }

    #[test]
    fn noisy_records_match_trajectories() {
        compare(
            "QUBITS 3\nH 0\nDEPOLARIZE1(0.1) 0\nCX 0 1\nDEPOLARIZE2(0.2) 0 1\nCX 1 2\nX_ERROR(0.15) 2\nM 0 1 2\nMPP Z0*Z1\n",
            40000,
            0.03,
        );
        compare(
            "QUBITS 2\nH 0\nCX 0 1\nPAULI_CHANNEL_1(0.05,0.1,0.15) 1\nSQRT_X 0\nCY 0 1\nMX 0 1\nZ_ERROR(0.3) 0\nR 1\nMX 0\nM 1\n",
            40000,
            0.03,
        );
    }

    #[test]
    fn noise_round_trips_through_the_text_format() {
        let text = "QUBITS 3
X_ERROR(0.01) 0
DEPOLARIZE2(0.002) 1 2
PAULI_CHANNEL_1(0.1,0,0.2) 2
M 0
";
        let c = Circuit::parse(text).unwrap();
        assert_eq!(Circuit::parse(&c.to_string()).unwrap(), c);
        assert!(Circuit::parse("X_ERROR(1.5) 0").is_err());
        assert!(FrameSampler::new(
            &Circuit::parse(
                "T 0
M 0"
            )
            .unwrap(),
            0
        )
        .is_err());
    }

    #[test]
    fn repetition_code_memory_with_detectors() {
        // distance-5 repetition code, 3 rounds of syndrome extraction with
        // X errors on the data; detectors compare consecutive syndromes
        let (d, rounds, p) = (5usize, 3usize, 0.02);
        let mut text = format!("QUBITS {}\n", 2 * d - 1);
        for r in 0..rounds {
            text += &format!("X_ERROR({p})");
            for q in 0..d {
                text += &format!(" {q}");
            }
            text += "\n";
            for a in 0..d - 1 {
                let anc = d + a;
                text += &format!("CX {a} {anc}\nCX {} {anc}\nM {anc}\nR {anc}\n", a + 1);
            }
            for a in 0..d - 1 {
                if r == 0 {
                    text += &format!("DETECTOR rec[-{}]\n", d - 1 - a);
                } else {
                    text += &format!("DETECTOR rec[-{}] rec[-{}]\n", d - 1 - a, 2 * (d - 1) - a);
                }
            }
        }
        text += "M";
        for q in 0..d {
            text += &format!(" {q}");
        }
        text += "\nOBSERVABLE_INCLUDE(0) rec[-1]\n";
        let c = Circuit::parse(&text).unwrap();
        assert_eq!(Circuit::parse(&c.to_string()).unwrap(), c);
        assert_eq!(c.detectors().len(), rounds * (d - 1));
        let fs = FrameSampler::new(&c, 5).unwrap();
        // noiseless: no detector or observable fires
        let quiet = FrameSampler::new(&c.without_noise(), 5).unwrap();
        let (dq, oq) = quiet
            .sample_detectors(1000, &mut StdRng::seed_from_u64(1))
            .unwrap();
        assert!(dq.iter().chain(&oq).all(|b| bits::is_zero(b)));
        // noisy: a first-round detector fires iff exactly one of its two data qubits flipped
        let shots = 100_000;
        let (det, obs) = fs.sample_detectors(shots, &mut StdRng::seed_from_u64(2)).unwrap();
        let rate = bits::popcount(&det[0]) as f64 / shots as f64;
        assert!((rate - 2.0 * p * (1.0 - p)).abs() < 0.005, "{rate}");
        // the observable (last data qubit) flips when it flipped an odd number of times
        let q = 1.0 - (1.0 - 2.0 * p).powi(rounds as i32);
        let orate = bits::popcount(&obs[0]) as f64 / shots as f64;
        assert!((orate - q / 2.0).abs() < 0.005, "{orate} vs {}", q / 2.0);
    }

    #[test]
    fn repetition_code_logical_error_rate() {
        // distance-3 bit-flip code, one round of independent X errors at p:
        // majority vote fails with probability 3p²(1−p) + p³
        let p = 0.1;
        let c = Circuit::parse(&format!("QUBITS 3\nX_ERROR({p}) 0 1 2\nM 0 1 2\n")).unwrap();
        let fs = FrameSampler::new(&c, 0).unwrap();
        let shots = 200_000;
        let packed = fs.sample_packed(shots, &mut StdRng::seed_from_u64(3)).unwrap();
        let mut fails = 0usize;
        for s in 0..shots {
            let votes = (0..3).filter(|&k| bits::get(&packed[k], s)).count();
            fails += (votes >= 2) as usize;
        }
        let want = 3.0 * p * p * (1.0 - p) + p * p * p;
        let got = fails as f64 / shots as f64;
        assert!(
            (got - want).abs() < 4.0 * (want / shots as f64).sqrt(),
            "{got} vs {want}"
        );
        // parity of two measurements: a detector
        let d = parity(&packed, &[0, 1]);
        let rate = bits::popcount(&d) as f64 / shots as f64;
        assert!((rate - 2.0 * p * (1.0 - p)).abs() < 0.01);
    }
}
