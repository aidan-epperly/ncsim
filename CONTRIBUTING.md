# Contributing to ncsim

The rules are [ncalg's](https://github.com/aidan-epperly/ncalg/blob/main/CONTRIBUTING.md); the short version:

- **Library only.**  No binaries and no result files.  Code
  that uses ncsim lives in its own crate, which depends on ncsim as a path or
  git dependency.
- **Safe and documented.**  `#![forbid(unsafe_code)]`, `#![warn(missing_docs)]`
  (every public item has a doc comment that says what it computes), `cargo fmt`,
  `cargo clippy --all-targets` clean, `cargo doc --no-deps` without warnings.
- **One implementation per algorithm.**  Reuse ncalg (`ff::BitMat`,
  `BitEchelon`, `cyclo::expsum`, `pcgroup`) and faer (products, SVD, QR,
  eigendecompositions); never a private copy.  A helper that two modules need
  lives in one shared module (`bits`, `pauli::ops`, `phasepoly`).  If a
  duplicate is justified (a hot-path specialization), say why in a comment.
- **Exact first, stable second.**  Phases are integers mod `2^r`, stabilizer
  data is bits, exact scalars go through `Phasor` / `Cyclo<N>`; floating
  sums are pairwise; truncations go through SVD; no single-pass Gram–Schmidt.
- **Families are plug-ins.**  A new state family implements `QuantumState<S>`
  in its own module under `state/` and declines what it cannot do with
  `SimError::Unsupported`; a new gate kind is a `Gate` variant with its views
  or a `CustomGate`.  Existing families need no change.
- **Tests against the reference.**  Every family is tested against
  `StateVector` on random circuits (amplitudes, expectation values, inner
  products, projections); closed forms (GHZ, magic states, ncalg exponential
  sums) are checked exactly.

## Issues

One Markdown file per issue in `issues/`, filed and closed with
`scripts/issues.py`:

```bash
python scripts/issues.py new bug "<what goes wrong>" --area pathsum --severity 1
python scripts/issues.py list
python scripts/issues.py close <id> --commit <hash> --note "..."
```

Areas: `pauli clifford phasepoly gate circuit state pathsum stabsum mps bridge
examples docs build`.  Severity 1 is a wrong result without an error, 2 a
panic or error on valid input, 3 performance or usability.  A missing feature
or a bug in ncalg is filed in [ncalg's tracker](https://github.com/aidan-epperly/ncalg/issues),
not worked around here.  Parts of `phasepoly` and `bits` may move into ncalg.

## Layout

```
src/scalar.rs      Amplitude, Phasor
src/bits.rs        packed GF(2) vectors (ncalg layout)
src/pauli/         Pauli, PauliTable, ColumnTable, PauliSum, ops
src/clifford/      CliffordTableau; random, decompose, monomial
src/phasepoly.rs   PhasePoly, QuadForm
src/gate.rs        Gate, MonomialGate, CustomGate
src/circuit.rs     Circuit, Op, text format
src/state/         QuantumState and the families: vector, sparse, stabilizer, product,
                   chform, pathsum, stabsum, mps
src/frame.rs       FrameSampler (Pauli-frame sampling of noisy Clifford circuits)
src/bridge.rs      ncalg polynomials ⇄ PauliSum, moments
tests/             cross-backend and bridge integration tests
examples/          clifford_t, bench, stress, explore
docs/DESIGN.md     design notes
```
