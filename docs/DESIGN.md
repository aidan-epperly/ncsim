# ncsim design

Quantum circuit simulation on ncalg, as a library.  This document records the
decisions behind the layout so that new families of states and gates can be
added without touching the existing ones, and so that the pieces that belong
in ncalg can be moved there later.

## Goals and ground rules

- A **library**: no binaries and no result files.  Downstream crates depend on
  it as a git dependency (`ncsim = { git = "https://github.com/aidan-epperly/ncsim" }`).
- The ncalg rules apply: `#![forbid(unsafe_code)]`, one implementation per
  algorithm (reuse ncalg and faer; never a private copy of an existing
  routine), numerical stability before speed (pairwise sums, SVD truncation,
  MGS×2 or Householder if a Gram–Schmidt ever appears), documented public API.
- **Exact when possible.**  Phases are integers mod `2^r`; stabilizer data is
  bits; amplitudes are computed in a scalar type `S: Amplitude` that is either
  `Complex<f64>` or the exact cyclotomic field `Cyclo<N>` (`Q(ζ_N)`), and the
  exact value is converted to the float at the very end.
- **Room for new families.**  Every state family implements one trait and
  declines operations it cannot do; every gate is data with *views* that
  backends ask for.  Adding a family is a new module; adding a gate kind is a
  new `Gate` variant or a `CustomGate` implementation.
- Qubits only (a design decision); traits avoid baking in
  `d = 2` where it is free, so a qudit family could be a separate module.

## Layers

```
scalar      Amplitude (Complex<f64> | Cyclo<N>), Phasor = √2^p ζ_{2^b}^k | 0
bits        packed 𝔽₂ vectors (ncalg layout); GF(2) inverse, echelon+transform
pauli       Pauli (i^k X^x Z^z), PauliTable (packed rows), ColumnTable (packed
            columns, bit-sliced phases), PauliSum<S>
clifford    CliffordTableau (images of X_q, Z_q): conjugate, compose, inverse,
            local gate application with a 4^k lookup for k ≤ 3; random, decompose,
            monomial bridge
phasepoly   PhasePoly (parity-product form mod 2^r) and QuadForm (ℤ₄ quadratic
            form): substitution, single-variable Gauss-sum elimination, exp sums via ncalg
gate        Gate (named gates, Phase(k,r), Diagonal, Monomial, Pauli, Clifford,
            rotations, dense, Custom) with views: tableau(), monomial(), matrix::<S>(),
            decomposition(), dagger()
circuit     Circuit/Op, Stim-like text format, tableau() and unitary() of a circuit
state       QuantumState<S> trait and the families:
              StateVector<S>   dense reference, any S
              SparseVector<S>  nonzero amplitudes as sorted (basis state, amplitude) pairs, any n
              Stabilizer       inverse tableau V = U⁻¹ (ψ = U|0⟩, rows V X V†, V Z V†, Stim's
                               layout): O(k n/64) gates, O(|P| n/64) deterministic and O(n²/64)
                               random measurements; generators recovered on demand;
                               affine form for amplitudes / inner products
              product          measure_commuting_on_product: commuting Paulis on ⊗ of tableaux,
                               per-block elimination of pull-backs, no joint tableau
              PathSum<S>       exact sum over paths: monomial gates by substitution,
                               H adds a variable, Clifford normal form by elimination,
                               observables by ncalg exponential sums
              ChForm           CH-form ω U_C U_H |s⟩ (Bravyi et al.): stabilizer vectors with
                               exact global phase; O(n/64) C-type gates, O(n²/64) H
              StabilizerSum<S, T> Σ of stabilizer terms T: StabilizerTerm (Clifford-class
                               PathSums by default, or ChTerm = coefficient × ChForm): non-Clifford diagonal
                               gates as projector factors (rank ×2 per factor)
              Mps              matrix product states, Complex<f64>, faer SVD
frame       FrameSampler: noiseless tableau reference + Pauli frames (ColumnTable rows =
            shots), Pauli noise channels by geometric skipping, parallel batches
bridge      ncalg polynomials ⇄ PauliSum, moments L(w) of states
```

## Conventions

- **Pauli convention.**  `P = i^phase · ⊗_q X^{x_q} Z^{z_q}`, phase mod 4,
  so `Y = i X Z` is `x = z = 1, phase = 1`.  Products: phase
  `a + b + 2 z·x'`.  Hermitian iff `phase ≡ x·z (mod 2)`.  The *letter* form
  `i^k ⊗ σ_q` printed and parsed has `k = phase − #Y` (`display_phase`).
- **Basis order.**  Little-endian: qubit `q` is bit `q` of a basis index or of
  a packed `Bits` vector.  Dense matrices are row-major on this order; gate
  targets are listed controls first.
- **Measurement outcomes.**  `bool`, `true` = eigenvalue `−1` = bit `1`.
- **Norms.**  States are not kept normalized; `project` returns the branch
  weight `⟨ψ|Π|ψ⟩` and leaves `Πψ` (the tableau family, which is a ray, keeps
  the state normalized and returns the probability).
- **Phase polynomials.**  `f = c + Σ_T c_T Π_{λ∈T} [λ·x]` mod `2^r`
  (ncalg's representation).  Term level `s + r − v₂(c) − 1`; level ≤ 2 is the
  Clifford class and is kept as a `QuadForm` `i^{q(x)}`.
- **Phasors.**  Exact scalars `√2^p ζ_{2^b}^k` (normalized, so equality is
  structural) for everything a stabilizer computation produces; converted to
  `S` by `Amplitude::from_phasor` (`None` when `S` lacks `√2` or the root).

## The state trait

```rust
trait QuantumState<S: Amplitude> {
    const BACKEND: &'static str;
    fn num_qubits(&self) -> usize;
    fn apply_gate(&mut self, gate: &Gate, qubits: &[usize]) -> Result<()>;
    fn norm_sq(&self) -> Result<S>;
    fn amplitude(&self, x: &[u64]) -> Result<S>;
    fn expectation(&self, p: &Pauli) -> Result<S>;
    fn inner(&self, other: &Self) -> Result<S>;
    fn project(&mut self, p: &Pauli, outcome: bool) -> Result<S>;
    fn scale(&mut self, c: &S) -> Result<()>;
    // provided: expectation_sum, probability, measure, apply_op, run, sample, to_vector, normalize
}
```

The provided methods are correct for every family (measurement from
`expectation` + `project`, sampling by sequential projections, …); a family
overrides them when it has something faster (the tableau draws a random bit
instead of computing a probability; the dense vector samples from the
probability vector).

## Gate views

A backend applies a gate by asking for the view it understands, in order of
preference, and returns `SimError::Unsupported` if none applies:

| backend       | views used                                                   |
|---------------|--------------------------------------------------------------|
| StateVector   | `monomial()` (permutation + phases), `matrix::<S>()`, `decomposition()` |
| SparseVector  | `Pauli` in place, `monomial()`, `matrix::<S>()`, `decomposition()` |
| Stabilizer    | `tableau()`                                                  |
| ChForm        | named fast paths, `monomial()` (Clifford: `s·X^b·C`), `decomposition()`, `tableau()` (phase of `CliffordTableau::unitary`) |
| PathSum       | `H` natively, `monomial()`, `decomposition()`                |
| StabilizerSum | `tableau()` per term, `monomial()` (non-Clifford terms as projector factors), `decomposition()` |
| Mps           | `matrix::<C64>()`, `decomposition()`                         |

`MonomialGate` (`U|x⟩ = s ζ^{f(x)} |Mx ⊕ b⟩`) is the common generalization of
`X`, `CX`, `SWAP`, every diagonal gate, Paulis and the monomial representations
of ncalg's Clifford-hierarchy groups (`pcgroup::families`); it is closed under
composition and inverse and knows when it is Clifford (`to_tableau`).

## Path sums and the Clifford normal form

`|ψ⟩ = scalar · Σ_{y∈𝔽₂^m} i^{Q(y)} ζ^{g(y)} |shift ⊕ A y⟩`.  After every gate,
`reduce()`:

1. sums out every path variable that appears in no output row and in no
   non-Clifford term, by `QuadForm::eliminate`: odd linear coefficient gives a
   factor `(1 ± i)` and a rank-one update (Amy's `[ω]` rule); even coefficient
   gives an affine constraint that is substituted (`[HH]`), or a factor 2, or
   zero;
2. changes variables along kernel directions of `A` among the Clifford-only
   variables so that such a variable appears (`[Elim]`), and repeats.

Step 2 does not compute a nullspace each time: a rank cache (a left inverse
`L` and a basis `W` of the left kernel of the Clifford columns `A_C`) is kept
through monomial gates (`L ← L Ĝ⁻¹`, `W ← Ĝ⁻ᵀ W`), `H` (a row of `A` replaced:
redundant iff some `w ∈ W` has `w_q = 1`, otherwise `L e_q` is the only new
kernel direction), projections (a new column), Gauss-sum constraints and
variables leaving `C`; other operations drop it and the next `reduce`
rebuilds it by one elimination.

So a Clifford circuit keeps `m = rank A`: the state is the affine form of a
stabilizer state with its global phase and norm in `scalar`, and the terms of a
`StabilizerSum` are exact vectors.  Non-Clifford variables stay; the cost of an
observable is `O(terms · 2^width)` where the width is reported by
`amplitude_plan` (ncalg's `exp_sum_plan`): polynomial for product-type and
bounded-treewidth non-Clifford structure, `2^d` for `d` non-Clifford parities
in general (by the Cai–Lu–Xia dichotomy, exact evaluation for larger classes
of phase types is #P-hard).

## What belongs in ncalg

Candidates for ncalg: term access, variable embedding and single-variable
elimination on `PhasePolynomial`, a public `QuadForm`, `BitMat::inverse`,
`echelon_with_transform`, and the packed-bit helpers.  `ncsim::phasepoly` and
`ncsim::bits` are written to become re-exports if they move there (same
semantics, conversion `to_ncalg()` at the boundary).

## Planned extensions (same representations)

- Stabilizer-rank sums: lower-rank magic-state decompositions (`|T⟩^{⊗6}`
  with rank 7, `|CCZ⟩` gadgets).
- Path sums: Amy's rewrite rules beyond the Clifford ones (`[ω]` with
  non-Clifford neighbors), a float evaluation path for `S = Complex<f64>`
  that skips the cyclotomic arithmetic.
- Tableaux: Pauli-based computation, Stim circuit import.
- Qudit families as separate modules on `ncalg::ff` (`Gf`).
