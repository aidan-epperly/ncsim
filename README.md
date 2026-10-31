# ncsim — quantum circuit simulation on ncalg

A library in safe Rust (`#![forbid(unsafe_code)]`) for simulating quantum
circuits exactly or numerically, built on the algebra of
[ncalg](https://github.com/aidan-epperly/ncalg): bit-packed GF(2) linear algebra, exact
exponential sums of phase polynomials, cyclotomic scalars and the
Clifford-hierarchy groups.  It is a library only; add it to a crate with
`ncsim = { git = "https://github.com/aidan-epperly/ncsim" }` (it fetches ncalg
the same way).

## Contents

- [Quick start](#quick-start)
- [State families](#state-families)
- [Gates and circuits](#gates-and-circuits)
- [Noise and Pauli frames](#noise-and-pauli-frames)
- [Scalars](#scalars)
- [Pauli operators and Clifford tableaux](#pauli-operators-and-clifford-tableaux)
- [Phase polynomials](#phase-polynomials)
- [ncalg bridge](#ncalg-bridge)
- [Examples and tests](#examples-and-tests)
- [Performance](#performance)
- [Conventions](#conventions)
- [Bugs and improvements](#bugs-and-improvements)

## Quick start

```rust
use ncsim::{Circuit, Cyclo, Pauli, PathSum, QuantumState, Stabilizer, StateVector, C64};
use rand::{rngs::StdRng, SeedableRng};

let c = Circuit::parse("H 0\nCX 0 1\nT 1\nCX 1 2\nM 2").unwrap();
let mut rng = StdRng::seed_from_u64(1);

// exact: a path sum with amplitudes in Q(ζ8)
let mut psi: PathSum<Cyclo<8>> = PathSum::zero_state(3);
let record = psi.run(&c, &mut rng).unwrap();          // measurement outcomes
let a = psi.amplitude(&ncsim::bits::from_u64(0b011, 3)).unwrap();
let e = psi.expectation(&Pauli::from_string(3, "X0*X1").unwrap()).unwrap();

// numerical reference
let mut v: StateVector<C64> = StateVector::zero_state(3);
v.run(&c, &mut rng).unwrap();

// Clifford circuits at scale: a stabilizer tableau
let mut s = Stabilizer::zero_state(200);
```

Every family implements the same [`QuantumState<S>`](src/state/mod.rs) trait:
`apply_gate`, `run` (a whole circuit with measurements), `amplitude`,
`expectation` (of a Pauli; `expectation_sum` of a `PauliSum`), `inner`,
`norm_sq`, `project`, `measure`, `sample`, `to_vector`, `normalize`.  A family
declines an operation it cannot do with `SimError::Unsupported` (a tableau and
a `T` gate, an exact scalar type and a rotation), so the same circuit runs on
whichever backend fits.

## State families

| family | scalars | what it does | cost |
|---|---|---|---|
| `StateVector<S>` | any | dense reference, every gate | `2^n` |
| `SparseVector<S>` | any | the nonzero amplitudes only, sorted `(basis state, amplitude)` pairs on any number of qubits: reversible arithmetic on superposed inputs, permutation-heavy circuits; monomial gates map entries one to one, dense gates merge by a parallel sort | `O(N log N)` per gate for `N` nonzero amplitudes |
| `Stabilizer` | — | stabilizer tableau stored by the inverse Clifford (Stim's layout): Clifford gates, Pauli measurements, exact `⟨P⟩`; amplitudes and inner products through the affine form `ψ(c⊕Bu) = 2^{-k/2} i^{Q(u)}` (up to a global phase); `measure_commuting_on_product` for commuting Paulis on a tensor product without the joint tableau | `O(k n/64)` per gate, `O(|P| n/64)` per deterministic and `O(n²/64)` per random measurement |
| `ChForm` | — | Stabilizer state *with its global phase* (Bravyi et al.'s CH-form `ω U_C U_H \|s⟩`): exact amplitudes, `⟨P⟩`, inner products and overlaps `⟨ψ\|U\|ψ⟩` of Clifford circuits; `Stabilizer::circuit_expectation` / `circuit_eigenvalue` use it on rays | `O(n/64)` per `S`, `CZ`, `CX`, Pauli; `O(n²/64)` per `H` and measurement; `O(n³/64)` inner product |
| `PathSum<S>` | any | exact sum over paths `scalar·Σ_y i^{Q(y)} ζ^{g(y)} |c⊕Ay⟩`; monomial gates by substitution, `H` adds a path variable; Clifford variables are summed out at once (Gauss sums), so Clifford circuits stay in affine normal form with phase and norm; observables are ncalg exponential sums | polynomial for Clifford, `O(2^width)` of the non-Clifford core (`amplitude_plan`) |
| `StabilizerSum<S, T>` | any | `Σ_j φ_j` of stabilizer terms, `T = PathSum<S>` (default) or `ChTerm<S>` (CH-form with a coefficient; fast Clifford gates on large registers): non-Clifford diagonal gates as projector factors `I + (ζ^c − 1) Π P_λ` (rank ×2 per `T`, `CS`, `CCZ`, …), Clifford gates term by term in parallel, observables as `O(χ²)` exact inner products, `merge_duplicates` | `2^t` terms for `t` non-Clifford factors |
| `Mps` | `C64` | matrix product state on faer SVD/QR: any dense gate of ≤ 4 qubits (swap routing for distant targets), truncation by `MpsOptions { max_bond, cutoff }`, `discarded_weight`, transfer-matrix observables, `O(nχ²)` sampling | `O(n χ³)` |

`PathSum::from_stabilizer`, `to_stabilizer`, `StabilizerSum::from_stabilizer`
and `Mps::from_vector` convert between them.  With `S = Cyclo<8>` every
Clifford+T computation is exact; with `S = Complex<f64>` the exact sums are
converted at the end.

The exact families are checked against the dense vector on random
Clifford+T+CS+CCZ circuits (`cargo test`); `examples/clifford_t.rs` shows the
three of them agreeing to `1e-17` with their cost parameters.

## Gates and circuits

`Gate` is data with *views* that backends ask for: `tableau()` (Clifford),
`monomial()` (an affine permutation times a diagonal phase, `U|x⟩ = s ζ^{f(x)}|Mx⊕b⟩`),
`matrix::<S>()` (dense, exact when representable), `decomposition()` (into
simpler gates), `dagger()`.  Named gates: `I X Y Z H S Sdg SqrtX SqrtXdg SqrtY
SqrtYdg T Tdg Cx Cy Cz Swap Cs Csdg Ccz Ccx`, `Phase{k, bits}` =
`diag(1, ζ_{2^bits}^k)`, `CPhase`, `Diagonal(PhasePoly)`, `Monomial`, `Pauli`,
`Clifford(tableau)` (any Clifford; synthesised into `H, S, CX` by
`to_circuit` when a backend needs gates), `Rx/Ry/Rz(θ)`, `Unitary(dense)`,
and `Custom(Arc<dyn CustomGate>)` for user gates that provide whichever views
they have.

`Circuit` holds gates, Pauli measurements (`Op::Measure`), resets and ticks,
with a Stim-like text format:

```text
QUBITS 4          # optional; otherwise the largest index + 1
H 0 1
CX 0 1 2 3        # groups of the gate's arity
T 2
PHASE(3,4) 1      # diag(1, ζ_16^3)
CCZ 0 1 2
RZ(0.5) 0
M 0               # Z measurement (MX, MY for X, Y)
MPP X0*Z1 Y2*Y3   # Pauli product measurements
R 3               # reset
```

`Circuit::tableau()` is the Clifford of a Clifford circuit, `unitary::<S>()`
its dense matrix, `dagger()` the adjoint circuit, `non_clifford_count()` the
number of non-Clifford gates.

## Noise and Pauli frames

Circuits carry Pauli noise channels: `Op::Noise` with a `Noise`, written `X_ERROR(p)`, `Y_ERROR(p)`,
`Z_ERROR(p)`, `PAULI_CHANNEL_1(px,py,pz)`, `DEPOLARIZE1(p)`, `DEPOLARIZE2(p)` in the text format.
Every backend runs them as one sampled trajectory: `run` applies a random Pauli drawn from the
channel. `Circuit::without_noise` strips them.

`FrameSampler` samples noisy Clifford circuits in bulk, as Stim's frame simulator does.
- **Reference:** one noiseless tableau run gives a reference record.
- **Frames:** each shot's frame (its accumulated Pauli error) is a row of a `ColumnTable`, so gates
  cost a few word operations per qubit for 64 shots at a time.
- **Parallelism:** batches of 16 384 shots run in parallel.
- **Output:** `sample_packed` returns the records by measurement. `DETECTOR rec[-k] …` and
  `OBSERVABLE_INCLUDE(i) rec[-k] …` mark parities, and `sample_detectors` returns their flips relative
  to the noiseless reference, as Stim reports them.

```rust
let c = Circuit::parse("QUBITS 3
X_ERROR(0.1) 0 1 2
M 0 1 2
")?;
let fs = FrameSampler::new(&c, 0)?;                 // reference seed
let records = fs.sample_packed(1_000_000, &mut rng)?; // records[k]: measurement k over the shots
```

## Scalars

`Amplitude` (`src/scalar.rs`) is implemented for `Complex<f64>` and for
ncalg's exact `Cyclo<N>` (`4 | N`): roots of unity `ζ_{2^r}^k`, `√2^k`,
conversion from cyclotomic numbers, pairwise sums for stability.  `Phasor` is
the exact scalar `√2^p ζ_{2^b}^k` (or `0`) of stabilizer computations:
amplitudes, inner products, expectation values of `Stabilizer` states and
Clifford-class Gauss sums come out as phasors and convert into any `S` that
can hold them.

## Pauli operators and Clifford tableaux

- `Pauli`: bit-packed `i^phase X^x Z^z` on `n` qubits; products, commutation,
  adjoint, restriction and embedding, dense matrix, text in letter form
  (`-i*X0*Y2`, `+XZ_Y`).  `PauliTable`: rows in one allocation (tableaux).
  `PauliSum<S>`: linear combinations (Hamiltonians), with `parse` through
  ncalg's parser.
- `CliffordTableau`: images of `X_q`, `Z_q`; `conjugate` (any Pauli),
  `compose`, `inverse`, `apply_gate` / `prepend_gate` (local, with a `4^k`
  lookup for gates on ≤ 3 qubits), `unitary`, `to_circuit`, `is_valid`.
- `clifford::random_clifford` draws exactly uniformly from the Clifford group
  (a uniformly random symplectic basis, independent signs; verified against
  the 24 and 11 520 elements for `n = 1, 2`); `random_stabilizer_state`,
  `random_pauli`.
- `clifford::monomial`: ncalg's `Monomial` (the representation of
  `pcgroup::families`: Pauli-CZ, XS, XT, `x_diag`) ⇄ `MonomialGate`, and its
  tableau when it is Clifford.

## Phase polynomials

`PhasePoly` is ncalg's parity-product form `c + Σ_T c_T Π_{λ∈T}[λ·x]` mod
`2^r` with the operations simulation needs (substitution, single-variable
substitution, extension, Clifford/non-Clifford split, `exp_sum::<S>()` through
ncalg); `QuadForm` is the ℤ₄ quadratic form of the Clifford class with
single-variable Gauss-sum elimination and a closed-form sum.  Both are
candidates for ncalg and are written to become re-exports.

## ncalg bridge

`bridge` turns polynomials over `ncalg::relations::pauli(n)` (generators
`X0, Y0, Z0, X1, …`) into `PauliSum`s and back (`from_polynomial`,
`to_polynomial`, `parse`), Majorana polynomials through Jordan–Wigner
(`from_majorana_polynomial`, `majorana_pauli`), and evaluates the moment
functional `L(w) = ⟨ψ|w|ψ⟩` of any state on the words of a relaxation
(`moments`, `expectation_polynomial`), with exact coefficient conversions
(`IntoAmplitude`, `FromAmplitude`) between ncalg's coefficient types and the
scalar types here.

## Examples and tests

```
cargo test                                   # ~70 tests, a few seconds
cargo run --release --example clifford_t     # exact path sum, stabilizer sum, MPS and dense vector on a random Clifford+T circuit
cargo run --release --example bench [-- tableau|chform|vector|sparse|pathsum|gadgets|stabsum|frames|mps]   # timings of the hot paths
cargo run --release --example stress [-- trials seed]   # randomized cross-checks of the exact families on 60–140 qubits
cargo run --release --example explore -- tableau|ops|surface|clifford|chform   # stabilizer runs at scale with timings and diagnostics
```

Tests compare every family against the dense vector on random circuits
(amplitudes, expectation values, inner products, projections, sampling
statistics), check the uniformity of random Cliffords, the synthesis round
trip, the ncalg monomial groups, and closed forms (GHZ, magic states, ncalg
exponential sums).

## Performance

`examples/bench.rs` on a 16-thread desktop (release build, `Complex<f64>`):

| family | workload | time |
|---|---|---|
| `Stabilizer` | n = 4000: 80 000 random 1- and 2-qubit Clifford gates | 10 ms |
| | n = 4000: 1000 random-outcome / deterministic `Z` measurements | 185 / 65 ms |
| `ChForm` | n = 1000: 20 000 random Clifford gates (exact phase) | 0.16 s |
| | n = 1000: exact inner product | 43 ms |
| `StateVector` | n = 22: layer of `T`, `CX`, `CCZ`, `Rx` on every qubit (88 gates) | 0.10 s |
| | n = 22: one Pauli expectation | 1.4 ms |
| `PathSum` | n = 1000: 10 000 random Clifford gates | 61 ms |
| `StabilizerSum<_, ChTerm>` | n = 1000: 10 000 Clifford gates and 6 `T` (32 terms) | 0.35 s |
| `SparseVector` | n = 1000: Toffoli/CNOT ripple (1996 gates) over 1024 superposed inputs | 7 ms |
| `StabilizerSum::run_with_t_gadgets` | n = 30, 16 `T`: 256 terms (projector factors: 3072); `⟨Z₀Z₁⟩` | 46 ms (17.5 s) |
| `FrameSampler` | n = 1000: 10 500 noisy gates, 2000 measurements, 100 000 shots | 31 ms (+ 51 ms reference) |

Where the time goes:
- **Tableaux** are stored by the inverse Clifford, as in Stim. A deterministic measurement is one pull-back, `O(|P| n/64)`, and a random collapse reads only the nonzero words of its fold.
- **Commuting measurements on product states** (`measure_commuting_on_product`) eliminate per block and never form the joint tableau.
- **Anticommutation** with a Pauli `P` costs `O(|P| n/64)`, which keeps sparse measurements cheap.
- **Dense kernels** run in parallel over independent fibers and use pairwise parallel sums; they allocate nothing per amplitude.
- **Path sums** keep a rank cache, a left inverse and left kernel of their output matrix, so an `H` needs no nullspace.
- **Sampling** from stabilizer states draws from the affine support; dense sampling makes one pass over the amplitudes.

## Conventions

- Qubit `q` is bit `q` of basis indices and packed bit vectors
  (little-endian); dense matrices are row-major on this order; gate targets
  list controls first.
- `P = i^phase · ⊗_q X^{x_q} Z^{z_q}`; the printed letter form `i^k ⊗ σ_q`
  has `k = phase − #Y`.
- Measurement outcomes are `bool`s, `true` = eigenvalue `−1`.
- States are not kept normalized; `project` returns the branch weight.  The
  tableau family is a ray: it stays normalized and returns the probability.
- Everything is deterministic for a seeded `rand` RNG; parallel sections
  (stabilizer sums) never change results.

Design notes: [docs/DESIGN.md](docs/DESIGN.md).

## Bugs and improvements

One file per issue in [`issues/`](issues/README.md), filed with
`python scripts/issues.py new bug|improvement|docs "<title>" --area <area>`
(areas: `pauli clifford phasepoly gate circuit state pathsum stabsum mps bridge
examples docs build`); see [CONTRIBUTING.md](CONTRIBUTING.md).  Problems in
ncalg itself go to ncalg's tracker.

## License

Copyright (C) 2026 Aidan Epperly.  Licensed under the GNU General Public
License, version 3 or (at your option) any later version; see [LICENSE](LICENSE).
