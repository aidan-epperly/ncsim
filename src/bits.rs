//! Bit vectors over `𝔽₂` packed in `u64` words: bit `j` is bit `j % 64` of
//! word `j / 64`, the layout of `ncalg::ff::BitMat` rows and `BitEchelon`.
//!
//! ncalg keeps equivalent helpers private in `cyclo::expsum`; this module is
//! the one copy in ncsim and a candidate for a public `ncalg::ff::bits`, after
//! which it would become re-exports.

use ncalg::ff::BitMat;

/// A packed bit vector.
pub type Bits = Vec<u64>;

/// Words needed for `n` bits (at least one).
#[inline]
pub fn nwords(n: usize) -> usize {
    n.div_ceil(64).max(1)
}

/// The zero vector on `n` bits.
#[inline]
pub fn zero(n: usize) -> Bits {
    vec![0; nwords(n)]
}

/// Bit `i`.
#[inline]
pub fn get(b: &[u64], i: usize) -> bool {
    (b[i / 64] >> (i % 64)) & 1 == 1
}

/// Set bit `i` to `v`.
#[inline]
pub fn set(b: &mut [u64], i: usize, v: bool) {
    if v {
        b[i / 64] |= 1u64 << (i % 64);
    } else {
        b[i / 64] &= !(1u64 << (i % 64));
    }
}

/// Flip bit `i`.
#[inline]
pub fn toggle(b: &mut [u64], i: usize) {
    b[i / 64] ^= 1u64 << (i % 64);
}

/// `a ^= b` (over the common length).
#[inline]
pub fn xor_into(a: &mut [u64], b: &[u64]) {
    for (x, y) in a.iter_mut().zip(b) {
        *x ^= *y;
    }
}

/// The parity of `a · b` (number of common ones mod 2).
#[inline]
pub fn dot(a: &[u64], b: &[u64]) -> bool {
    a.iter()
        .zip(b)
        .fold(0u32, |acc, (x, y)| acc ^ (x & y).count_ones())
        & 1
        == 1
}

/// Number of ones.
#[inline]
pub fn popcount(a: &[u64]) -> usize {
    a.iter().map(|w| w.count_ones() as usize).sum()
}

/// Number of positions where both are one.
#[inline]
pub fn and_count(a: &[u64], b: &[u64]) -> usize {
    a.iter().zip(b).map(|(x, y)| (x & y).count_ones() as usize).sum()
}

/// All bits zero?
#[inline]
pub fn is_zero(a: &[u64]) -> bool {
    a.iter().all(|&w| w == 0)
}

/// Positions of the ones, increasing.
pub fn ones(a: &[u64]) -> Vec<usize> {
    let mut v = Vec::new();
    for (w, &x) in a.iter().enumerate() {
        let mut x = x;
        while x != 0 {
            v.push(w * 64 + x.trailing_zeros() as usize);
            x &= x - 1;
        }
    }
    v
}

/// The lowest set bit, if any.
pub fn lowest_one(a: &[u64]) -> Option<usize> {
    a.iter()
        .enumerate()
        .find(|(_, &x)| x != 0)
        .map(|(w, &x)| w * 64 + x.trailing_zeros() as usize)
}

/// The unit vector `e_i` on `n` bits.
pub fn unit(n: usize, i: usize) -> Bits {
    let mut b = zero(n);
    toggle(&mut b, i);
    b
}

/// From a list of set positions.
pub fn from_indices(n: usize, idx: &[usize]) -> Bits {
    let mut b = zero(n);
    for &i in idx {
        set(&mut b, i, true);
    }
    b
}

/// From `bool`s.
pub fn from_bools(v: &[bool]) -> Bits {
    let mut b = zero(v.len());
    for (i, &x) in v.iter().enumerate() {
        if x {
            toggle(&mut b, i);
        }
    }
    b
}

/// From `u8`s (`0`/`1`, as ncalg's exponential sums take them).
pub fn from_u8s(v: &[u8]) -> Bits {
    let mut b = zero(v.len());
    for (i, &x) in v.iter().enumerate() {
        if x & 1 == 1 {
            toggle(&mut b, i);
        }
    }
    b
}

/// The first `n` bits as `bool`s.
pub fn to_bools(b: &[u64], n: usize) -> Vec<bool> {
    (0..n).map(|i| get(b, i)).collect()
}

/// The first `n` bits as `u8`s.
pub fn to_u8s(b: &[u64], n: usize) -> Vec<u8> {
    (0..n).map(|i| get(b, i) as u8).collect()
}

/// Parse a string of `0`/`1` characters (left to right = increasing index).
pub fn parse(s: &str) -> Option<Bits> {
    let mut b = zero(s.len());
    for (i, c) in s.chars().enumerate() {
        match c {
            '0' => {}
            '1' => toggle(&mut b, i),
            _ => return None,
        }
    }
    Some(b)
}

/// The first `n` bits as a string of `0`/`1`.
pub fn to_string(b: &[u64], n: usize) -> String {
    (0..n).map(|i| if get(b, i) { '1' } else { '0' }).collect()
}

/// Make sure `b` can hold `n` bits (zero-extending); never shrinks.
pub fn ensure_len(b: &mut Bits, n: usize) {
    let w = nwords(n);
    if b.len() < w {
        b.resize(w, 0);
    }
}

/// Clear every bit at position `≥ n`.
pub fn mask_to(b: &mut [u64], n: usize) {
    for (w, x) in b.iter_mut().enumerate() {
        let lo = w * 64;
        if lo >= n {
            *x = 0;
        } else if lo + 64 > n {
            *x &= (1u64 << (n - lo)) - 1;
        }
    }
}

/// `a` (its first `na` bits) followed by `b` (its first `nb` bits): bit
/// `na + j` of the result is bit `j` of `b`.
pub fn concat(a: &[u64], na: usize, b: &[u64], nb: usize) -> Bits {
    let mut out = zero(na + nb);
    let wa = na.div_ceil(64);
    out[..wa].copy_from_slice(&a[..wa]);
    mask_to(&mut out, na);
    or_shifted(&mut out, b, nb, na);
    out
}

/// `out |= b << offset` for the first `nb` bits of `b` (`out` long enough).
pub fn or_shifted(out: &mut [u64], b: &[u64], nb: usize, offset: usize) {
    let (w, s) = (offset / 64, offset % 64);
    let wb = nb.div_ceil(64);
    for (i, &x) in b[..wb].iter().enumerate() {
        let x = if i + 1 == wb && nb % 64 != 0 {
            x & ((1u64 << (nb % 64)) - 1)
        } else {
            x
        };
        if x == 0 {
            continue;
        }
        out[w + i] |= x << s;
        if s != 0 && w + i + 1 < out.len() {
            out[w + i + 1] |= x >> (64 - s);
        }
    }
}

/// Delete bit `j`: the bits above it move down by one (the top bit becomes 0).
pub fn remove_bit(b: &mut [u64], j: usize) {
    let w = j / 64;
    if w >= b.len() {
        return;
    }
    let o = j % 64;
    let low_mask = if o == 0 { 0 } else { (1u64 << o) - 1 };
    let mut cur = (b[w] & low_mask) | ((b[w] >> 1) & !low_mask);
    for k in w + 1..b.len() {
        let next = b[k];
        b[k - 1] = cur | ((next & 1) << 63);
        cur = next >> 1;
    }
    let last = b.len() - 1;
    b[last] = cur;
}

/// All solutions of `A y = rhs` over GF(2) for `A` given by its rows (each
/// with `m` columns): `(particular solution, kernel basis)`, or `None` if
/// the system is inconsistent.  One Gauss–Jordan pass on the packed rows
/// (the row XORs in parallel for large systems), no bit-by-bit matrix.
pub fn solve_affine(rows: &[Bits], m: usize, rhs: &[u64]) -> Option<(Bits, Vec<Bits>)> {
    use rayon::prelude::*;
    let n = rows.len();
    let w = nwords(m + 1);
    let mut a: Vec<Bits> = rows
        .iter()
        .enumerate()
        .map(|(i, r)| {
            let mut v = vec![0u64; w];
            for (d, s) in v.iter_mut().zip(r) {
                *d = *s;
            }
            mask_to(&mut v, m);
            if get(rhs, i) {
                set(&mut v, m, true);
            }
            v
        })
        .collect();
    let mut pivots = Vec::new();
    let mut r = 0;
    for c in 0..m {
        if r == n {
            break;
        }
        let Some(p) = (r..n).find(|&i| get(&a[i], c)) else {
            continue;
        };
        a.swap(r, p);
        let pivot = a[r].clone();
        let (wc, mc) = (c / 64, 1u64 << (c % 64));
        let body = |(i, row): (usize, &mut Bits)| {
            if i != r && row[wc] & mc != 0 {
                for (x, y) in row.iter_mut().zip(&pivot) {
                    *x ^= *y;
                }
            }
        };
        if n * w >= 1 << 16 {
            a.par_iter_mut().enumerate().for_each(body);
        } else {
            a.iter_mut().enumerate().for_each(body);
        }
        pivots.push(c);
        r += 1;
    }
    if a[r..].iter().any(|row| get(row, m)) {
        return None;
    }
    let mut x = zero(m);
    for (k, &c) in pivots.iter().enumerate() {
        if get(&a[k], m) {
            set(&mut x, c, true);
        }
    }
    let mut is_pivot = vec![false; m];
    for &c in &pivots {
        is_pivot[c] = true;
    }
    let kernel = (0..m)
        .filter(|&f| !is_pivot[f])
        .map(|f| {
            let mut v = unit(m, f);
            for (k, &c) in pivots.iter().enumerate() {
                if get(&a[k], f) {
                    set(&mut v, c, true);
                }
            }
            v
        })
        .collect();
    Some((x, kernel))
}

/// Rows as a [`BitMat`] with `cols` columns.
pub fn to_bitmat(rows: &[Bits], cols: usize) -> BitMat {
    let mut m = BitMat::zeros(rows.len(), cols);
    for (i, r) in rows.iter().enumerate() {
        for j in ones(r) {
            if j < cols {
                m.set(i, j, true);
            }
        }
    }
    m
}

/// The rows of a [`BitMat`] as packed vectors.
pub fn from_bitmat(m: &BitMat) -> Vec<Bits> {
    (0..m.rows()).map(|i| m.row_words(i).to_vec()).collect()
}

/// The inverse of the `k × k` matrix with the given rows over GF(2), by row
/// reduction of `[M | I]` (ncalg's `BitMat` has `rref`, `solve` and
/// `nullspace` but no inverse; candidate for `BitMat::inverse`).
pub fn inverse(rows: &[Bits], k: usize) -> Option<Vec<Bits>> {
    assert_eq!(rows.len(), k);
    let mut aug = BitMat::zeros(k, 2 * k);
    for (i, r) in rows.iter().enumerate() {
        for j in ones(r) {
            if j < k {
                aug.set(i, j, true);
            }
        }
        aug.set(i, k + i, true);
    }
    let pivots = aug.rref();
    if pivots.len() != k || pivots.iter().any(|&p| p >= k) {
        return None;
    }
    // after rref the left half is the identity (pivots 0..k in order)
    let mut inv = Vec::with_capacity(k);
    for i in 0..k {
        let mut r = zero(k);
        for j in 0..k {
            if aug.get(i, k + j) {
                toggle(&mut r, j);
            }
        }
        inv.push(r);
    }
    Some(inv)
}

/// Reduced row echelon form of `k` independent rows on `n` bits, with the
/// transformation: returns `(reduced rows, pivot columns, T)` where
/// `reduced_j = Σ_i T_ji rows_i` (each `T_j` a `k`-bit vector).  `None` if the
/// rows are dependent.
pub fn echelon_with_transform(rows: &[Bits], n: usize) -> Option<(Vec<Bits>, Vec<usize>, Vec<Bits>)> {
    let k = rows.len();
    let mut aug = BitMat::zeros(k, n + k);
    for (i, r) in rows.iter().enumerate() {
        for j in ones(r) {
            if j < n {
                aug.set(i, j, true);
            }
        }
        aug.set(i, n + i, true);
    }
    let pivots = aug.rref();
    if pivots.len() != k || pivots.iter().any(|&p| p >= n) {
        return None;
    }
    let mut red = Vec::with_capacity(k);
    let mut t = Vec::with_capacity(k);
    for i in 0..k {
        let mut r = zero(n);
        for j in 0..n {
            if aug.get(i, j) {
                toggle(&mut r, j);
            }
        }
        let mut c = zero(k);
        for j in 0..k {
            if aug.get(i, n + j) {
                toggle(&mut c, j);
            }
        }
        red.push(r);
        t.push(c);
    }
    Some((red, pivots, t))
}

/// Integer value of the first `n` bits (`n ≤ 64`), bit `i` weighted `2^i`.
pub fn to_u64(b: &[u64], n: usize) -> u64 {
    assert!(n <= 64);
    if n == 64 {
        b[0]
    } else {
        b[0] & ((1u64 << n) - 1)
    }
}

/// Bits of an integer on `n` bits (`n ≤ 64`).
pub fn from_u64(v: u64, n: usize) -> Bits {
    assert!(n <= 64);
    let mut b = zero(n);
    b[0] = if n == 64 { v } else { v & ((1u64 << n) - 1) };
    b
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn solve_affine_matches_bitmat() {
        use rand::rngs::StdRng;
        use rand::{Rng, SeedableRng};
        let mut rng = StdRng::seed_from_u64(3);
        for trial in 0..200 {
            let (n, m) = (1 + trial % 13, 1 + (trial * 7) % 90);
            let rows: Vec<Bits> = (0..n)
                .map(|_| {
                    let mut r = zero(m);
                    for j in 0..m {
                        if rng.gen::<f64>() < 0.3 {
                            set(&mut r, j, true);
                        }
                    }
                    r
                })
                .collect();
            let rhs = from_bools(&(0..n).map(|_| rng.gen::<bool>()).collect::<Vec<_>>());
            let bm = to_bitmat(&rows, m);
            let want = bm.solve(&to_bools(&rhs, n));
            match solve_affine(&rows, m, &rhs) {
                None => assert!(want.is_none(), "trial {trial}"),
                Some((x, ker)) => {
                    assert!(want.is_some(), "trial {trial}");
                    for (i, r) in rows.iter().enumerate() {
                        assert_eq!(dot(r, &x), get(&rhs, i), "trial {trial}");
                        for v in &ker {
                            assert!(!dot(r, v), "trial {trial}");
                        }
                    }
                    assert_eq!(ker.len(), m - bm.rank(), "trial {trial}");
                }
            }
        }
    }

    #[test]
    fn remove_bit_shifts_the_rest_down() {
        for (len, j) in [(10, 3), (64, 0), (64, 63), (130, 64), (130, 5), (200, 129)] {
            let ones: Vec<usize> = (0..len).filter(|i| i % 3 != 1).collect();
            let mut b = from_indices(len, &ones);
            remove_bit(&mut b, j);
            let want: Vec<usize> = ones
                .iter()
                .filter(|&&i| i != j)
                .map(|&i| if i > j { i - 1 } else { i })
                .collect();
            assert_eq!(super::ones(&b), want, "len {len} j {j}");
        }
    }

    #[test]
    fn dot_and_ones() {
        let a = from_indices(130, &[0, 65, 129]);
        let b = from_indices(130, &[65, 129, 3]);
        assert!(!dot(&a, &b));
        let c = from_indices(130, &[65]);
        assert!(dot(&a, &c));
        assert_eq!(ones(&a), vec![0, 65, 129]);
        assert_eq!(popcount(&a), 3);
        assert_eq!(lowest_one(&b), Some(3));
        let mut m = a.clone();
        mask_to(&mut m, 66);
        assert_eq!(ones(&m), vec![0, 65]);
        assert_eq!(parse("0110").unwrap(), from_indices(4, &[1, 2]));
        assert_eq!(to_string(&from_indices(4, &[1, 2]), 4), "0110");
        let m = vec![
            from_indices(3, &[0, 1]),
            from_indices(3, &[1]),
            from_indices(3, &[1, 2]),
        ];
        let inv = inverse(&m, 3).unwrap();
        // M · M⁻¹ = I: row i of M dotted with column j of M⁻¹
        for i in 0..3 {
            for j in 0..3 {
                let col: Vec<bool> = (0..3).map(|r| get(&inv[r], j)).collect();
                let v = from_bools(&col);
                assert_eq!(dot(&m[i], &v), i == j);
            }
        }
        assert!(inverse(&[from_indices(2, &[0]), from_indices(2, &[0])], 2).is_none());
        for (na, nb) in [(0, 5), (3, 70), (64, 64), (63, 129), (100, 0)] {
            let a = from_indices(na.max(1), &(0..na).filter(|i| i % 3 == 0).collect::<Vec<_>>());
            let b = from_indices(nb.max(1), &(0..nb).filter(|i| i % 5 != 1).collect::<Vec<_>>());
            let c = concat(&a, na, &b, nb);
            assert_eq!(c.len(), nwords(na + nb));
            for i in 0..na + nb {
                let want = if i < na { get(&a, i) } else { get(&b, i - na) };
                assert_eq!(get(&c, i), want, "{na} {nb} {i}");
            }
        }
    }
}
