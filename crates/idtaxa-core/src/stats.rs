//! Small numerical helpers reproducing R semantics where it matters.

/// R's default (type 7) quantile of `x` at probability `p`.
pub fn quantile_type7(x: &[f64], p: f64) -> f64 {
    assert!(!x.is_empty());
    let mut v: Vec<f64> = x.to_vec();
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let n = v.len();
    let h = (n as f64 - 1.0) * p;
    let lo = h.floor();
    let lo_i = lo as usize;
    let hi_i = (lo_i + 1).min(n - 1);
    // R: qs <- x[lo]; qs + (h - lo) * (x[hi] - x[lo]) (with fuzz for exactness)
    let frac = h - lo;
    if frac.abs() < 4.0 * f64::EPSILON {
        v[lo_i]
    } else {
        v[lo_i] + frac * (v[hi_i] - v[lo_i])
    }
}

/// log of the binomial coefficient via lgamma-free summation (stable for
/// moderate n).
fn ln_choose(n: u64, k: u64) -> f64 {
    let k = k.min(n - k);
    let mut s = 0.0;
    for i in 0..k {
        s += ((n - i) as f64).ln() - ((i + 1) as f64).ln();
    }
    s
}

/// Binomial CDF P(X <= q) for X ~ Bin(n, p) (R's `pbinom(q, n, p)`).
pub fn pbinom(q: f64, n: u64, p: f64) -> f64 {
    if q < 0.0 {
        return 0.0;
    }
    let q = q.floor() as u64;
    if q >= n {
        return 1.0;
    }
    if p <= 0.0 {
        return 1.0;
    }
    if p >= 1.0 {
        return 0.0;
    }
    let lp = p.ln();
    let lq = (-p).ln_1p();
    let mut s = 0.0;
    for k in 0..=q {
        s += (ln_choose(n, k) + k as f64 * lp + (n - k) as f64 * lq).exp();
    }
    s.min(1.0)
}

/// Upper tail P(X > q) computed directly to avoid cancellation.
pub fn pbinom_upper(q: f64, n: u64, p: f64) -> f64 {
    if q < 0.0 {
        return 1.0;
    }
    let q = q.floor() as u64;
    if q >= n || p <= 0.0 {
        return 0.0;
    }
    if p >= 1.0 {
        return 1.0;
    }
    if q == 0 {
        // 1 - (1 - p)^n without cancellation
        return -((n as f64) * (-p).ln_1p()).exp_m1();
    }
    if n > 2000 {
        return (1.0 - pbinom(q as f64, n, p)).max(0.0);
    }
    let lp = p.ln();
    let lq = (-p).ln_1p();
    let mut s = 0.0;
    for k in (q + 1)..=n {
        s += (ln_choose(n, k) + k as f64 * lp + (n - k) as f64 * lq).exp();
    }
    s.min(1.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn quantiles_match_r() {
        // quantile(c(1,2,3,4,10), c(0.9, 0.99, 0.5)) = 7.6, 9.76, 3
        let x = [1.0, 2.0, 3.0, 4.0, 10.0];
        assert!((quantile_type7(&x, 0.9) - 7.6).abs() < 1e-12);
        assert!((quantile_type7(&x, 0.99) - 9.76).abs() < 1e-12);
        assert_eq!(quantile_type7(&x, 0.5), 3.0);
        assert_eq!(quantile_type7(&[5.0], 0.3), 5.0);
    }

    #[test]
    fn pbinom_matches_r() {
        // pbinom(3, 10, 0.3) = 0.6496107184
        assert!((pbinom(3.0, 10, 0.3) - 0.6496107184).abs() < 1e-9);
        assert!((pbinom_upper(3.0, 10, 0.3) - (1.0 - 0.6496107184)).abs() < 1e-9);
        assert_eq!(pbinom(-1.0, 10, 0.3), 0.0);
        assert_eq!(pbinom(10.0, 10, 0.3), 1.0);
        // 1 - pbinom(0, 5000, 1e-6) = 1 - (1-1e-6)^5000
        let v = pbinom_upper(0.0, 5000, 1e-6);
        assert!((v - (1.0 - (1.0f64 - 1e-6).powi(5000))).abs() < 1e-12);
    }
}
