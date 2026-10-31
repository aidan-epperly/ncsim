//! The ncalg bridge: polynomials of the Pauli and Majorana algebras as Pauli
//! sums, moment functionals of states.

use ncalg::relations;
use ncalg::{
    Coeff, Complex, ComplexCoeff, FreeAlgebra, GroebnerBasis, GroebnerOptions, HybridRational, Polynomial,
    Word,
};
use ncsim::bridge::{expectation_polynomial, majorana_pauli, moments};
use ncsim::{Amplitude, Circuit, Cyclo, PauliSum, QuantumState, Stabilizer, StateVector, C64};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

type Q = Complex<HybridRational>;

/// A random polynomial with `terms` terms, words of length at most 3 in any
/// order of the letters, small Gaussian rational coefficients.
fn random_poly(alg: &FreeAlgebra, rng: &mut StdRng, terms: usize) -> Polynomial<Q> {
    let nv = alg.num_vars() as u32;
    let mut ts = Vec::with_capacity(terms);
    for _ in 0..terms {
        let len = rng.gen_range(0..=3);
        let w = Word::from_vec((0..len).map(|_| rng.gen_range(0..nv)).collect());
        let c = Complex::new(
            HybridRational::new(rng.gen_range(-3..=3), rng.gen_range(1..=3)),
            HybridRational::new(rng.gen_range(-3..=3), rng.gen_range(1..=3)),
        );
        ts.push((w, c));
    }
    Polynomial::from_terms(alg.clone(), ts)
}

#[test]
fn products_of_polynomials_are_products_of_sums() {
    let (alg, _) = relations::pauli::<Q>(3);
    let mut rng = StdRng::seed_from_u64(11);
    for _ in 0..20 {
        let p = random_poly(&alg, &mut rng, 6);
        let q = random_poly(&alg, &mut rng, 6);
        let pq = &p * &q;
        // exact
        let hp: PauliSum<Cyclo<4>> = PauliSum::from_polynomial(&alg, &p, 3).unwrap();
        let hq: PauliSum<Cyclo<4>> = PauliSum::from_polynomial(&alg, &q, 3).unwrap();
        let hpq: PauliSum<Cyclo<4>> = PauliSum::from_polynomial(&alg, &pq, 3).unwrap();
        assert_eq!(hp.mul(&hq), hpq, "p = {p}, q = {q}");
        // floating
        let fp: PauliSum<C64> = PauliSum::from_polynomial(&alg, &p, 3).unwrap();
        let fq: PauliSum<C64> = PauliSum::from_polynomial(&alg, &q, 3).unwrap();
        let fpq: PauliSum<C64> = PauliSum::from_polynomial(&alg, &pq, 3).unwrap();
        assert!(
            fp.mul(&fq).sub(&fpq).max_coefficient() < 1e-12,
            "p = {p}, q = {q}"
        );
        // the floating sum is the exact one, rounded
        let rounded: PauliSum<C64> = hpq.map_coeffs(|c| c.to_complex());
        assert!(rounded.sub(&fpq).max_coefficient() < 1e-12);
    }
}

#[test]
fn to_polynomial_is_the_normal_form() {
    let (alg, rels) = relations::pauli::<Q>(3);
    let gb = GroebnerBasis::compute(rels, &GroebnerOptions::default());
    assert!(gb.is_complete());
    let mut rng = StdRng::seed_from_u64(5);
    for _ in 0..20 {
        let p = random_poly(&alg, &mut rng, 8);
        let reduced = gb.reduce(&p);
        let h: PauliSum<Cyclo<4>> = PauliSum::from_polynomial(&alg, &p, 3).unwrap();
        let back: Polynomial<Q> = h.to_polynomial(&alg).unwrap();
        assert_eq!(back, reduced, "p = {p}");
        for w in back.words() {
            assert!(gb.is_normal_word(w));
        }
        // floating coefficients: the same up to rounding
        let hf: PauliSum<C64> = PauliSum::from_polynomial(&alg, &p, 3).unwrap();
        let backf: Polynomial<Complex<f64>> = hf.to_polynomial(&alg).unwrap();
        assert!(backf.approx_eq(&reduced.convert(), 1e-12), "p = {p}");
    }
}

/// The singlet `(|01⟩ − |10⟩)/√2` on two qubits.
fn singlet_circuit() -> Circuit {
    let mut c = Circuit::new(2);
    c.x(1).h(0).cx(0, 1).z(0);
    c
}

#[test]
fn heisenberg_on_the_singlet() {
    let c = singlet_circuit();
    let mut rng = StdRng::seed_from_u64(0);
    let mut sv: StateVector<C64> = StateVector::zero_state(2);
    sv.run(&c, &mut rng).unwrap();
    // check the state: index bit q is qubit q, so |q0 q1⟩ = |01⟩ is index 0b10
    let v = sv.to_vector().unwrap();
    let r = std::f64::consts::FRAC_1_SQRT_2;
    assert!((v[0b10] - C64::new(r, 0.0)).norm() < 1e-14, "{v:?}");
    assert!((v[0b01] - C64::new(-r, 0.0)).norm() < 1e-14, "{v:?}");
    assert!(v[0b00].norm() < 1e-14 && v[0b11].norm() < 1e-14);
    let h: PauliSum<C64> = PauliSum::parse(2, "X0*X1 + Y0*Y1 + Z0*Z1").unwrap();
    assert!(h.is_hermitian());
    let e = sv.expectation_sum(&h).unwrap();
    assert!((e - C64::new(-3.0, 0.0)).norm() < 1e-12, "{e}");
    // the same through the polynomial, and exactly on exact backends
    let (alg, _) = relations::pauli::<Q>(2);
    let p: Polynomial<Q> = alg.parse("X0*X1 + Y0*Y1 + Z0*Z1").unwrap();
    let e2: C64 = expectation_polynomial(&sv, &alg, &p).unwrap();
    assert!((e2 - C64::new(-3.0, 0.0)).norm() < 1e-12);
    let mut ev: StateVector<Cyclo<8>> = StateVector::zero_state(2);
    ev.run(&c, &mut rng).unwrap();
    let e3: Cyclo<8> = expectation_polynomial(&ev, &alg, &p).unwrap();
    assert_eq!(e3, Cyclo::<8>::from_i64(-3));
    let mut st = Stabilizer::zero_state(2);
    QuantumState::<Cyclo<4>>::run(&mut st, &c, &mut rng).unwrap();
    let e4: Cyclo<4> = expectation_polynomial(&st, &alg, &p).unwrap();
    assert_eq!(e4, Cyclo::<4>::from_i64(-3));
}

#[test]
fn ghz_moments_agree_between_backends() {
    let mut c = Circuit::new(3);
    c.h(0).cx(0, 1).cx(1, 2);
    let mut rng = StdRng::seed_from_u64(3);
    let mut st = Stabilizer::zero_state(3);
    QuantumState::<Cyclo<4>>::run(&mut st, &c, &mut rng).unwrap();
    let mut sv: StateVector<Cyclo<8>> = StateVector::zero_state(3);
    sv.run(&c, &mut rng).unwrap();
    let (alg, _) = relations::pauli::<Q>(3);
    let mut words: Vec<Word> = Vec::new();
    for d in 0..=2 {
        words.extend(alg.words_of_length(d));
    }
    assert_eq!(words.len(), 1 + 9 + 81);
    let m_st: Vec<Cyclo<4>> = moments(&st, &alg, &words).unwrap();
    let m_sv: Vec<Cyclo<8>> = moments(&sv, &alg, &words).unwrap();
    for ((w, a), b) in words.iter().zip(&m_st).zip(&m_sv) {
        let lifted: Cyclo<8> = Cyclo::from_cyclo(a).unwrap();
        assert_eq!(lifted, *b, "L({})", alg.format_word(w));
    }
    // spot checks: ⟨Z0 Z1⟩ = 1, ⟨X0 X1⟩ = 0, ⟨X0 Z0⟩ = ⟨−i Y0⟩ = 0, ⟨Z0 Z0⟩ = 1
    let at = |s: &str| -> Cyclo<4> {
        let p: Polynomial<Q> = alg.parse(s).unwrap();
        let w = p.leading_word().unwrap().clone();
        m_st[words.iter().position(|x| *x == w).unwrap()].clone()
    };
    assert!(at("Z0*Z1").is_one());
    assert!(Coeff::is_zero(&at("X0*X1")));
    assert!(Coeff::is_zero(&at("X0*Z0")));
    assert!(at("Z0*Z0").is_one());
    // and the floating backend agrees
    let mut fv: StateVector<C64> = StateVector::zero_state(3);
    fv.run(&c, &mut rng).unwrap();
    let m_f: Vec<C64> = moments(&fv, &alg, &words).unwrap();
    for (a, b) in m_sv.iter().zip(&m_f) {
        assert!((a.to_complex() - b).norm() < 1e-12);
    }
}

#[test]
fn majorana_algebra_and_quadratic_hamiltonian() {
    let n = 3; // qubits, 6 Majoranas
    let gamma: Vec<PauliSum<Cyclo<4>>> = (0..2 * n)
        .map(|k| PauliSum::from_pauli(&majorana_pauli(k, n)))
        .collect();
    for k in 0..2 * n {
        for l in 0..2 * n {
            let anti = gamma[k].mul(&gamma[l]).add(&gamma[l].mul(&gamma[k]));
            let expected = if k == l {
                PauliSum::scalar(n, Cyclo::<4>::from_i64(2))
            } else {
                PauliSum::zero(n)
            };
            assert_eq!(anti, expected, "{{γ{k}, γ{l}}}");
        }
    }
    // H = i Σ_{k<l} a_kl γ_k γ_l with rational a_kl
    let (alg, rels) = relations::majorana::<Q>(2 * n);
    let gb = GroebnerBasis::compute(rels, &GroebnerOptions::default());
    let mut rng = StdRng::seed_from_u64(9);
    let mut h = alg.zero::<Q>();
    let mut by_hand: PauliSum<Cyclo<4>> = PauliSum::zero(n);
    let i4 = Cyclo::<4>::imaginary_unit();
    for k in 0..2 * n {
        for l in (k + 1)..2 * n {
            let a = HybridRational::new(rng.gen_range(-4..=4), rng.gen_range(1..=3));
            let coeff = Complex::new(HybridRational::new(0, 1), a.clone());
            h += alg.term(coeff, Word::from_vec(vec![k as u32, l as u32]));
            let ac: Cyclo<4> = Cyclo::from_rational(a.to_big());
            by_hand.add_scaled(&gamma[k].mul(&gamma[l]), &i4.mul_ref(&ac));
        }
    }
    // Hermitian modulo the relations (the adjoint of a normal word is not normal)
    assert_eq!(gb.reduce(&h.adjoint()), gb.reduce(&h));
    let from_poly: PauliSum<Cyclo<4>> = PauliSum::from_majorana_polynomial(&alg, &h, n).unwrap();
    assert_eq!(from_poly, by_hand);
    assert!(from_poly.is_hermitian());
    // non-normal words multiply out: γ_1 γ_0 γ_1 = −γ_0
    let w: Polynomial<Q> = alg.parse("g1*g0*g1 + g0").unwrap();
    let z: PauliSum<Cyclo<4>> = PauliSum::from_majorana_polynomial(&alg, &w, n).unwrap();
    assert!(z.is_empty());
    assert!(gb.reduce(&w).is_zero());
}
