//! Held-out evaluation metrics (Phase 5).
//!
//! For each held-out sequence and rank depth `d` (1 = first rank below
//! Root), the outcome is one of:
//! * `TP` – assigned at `d` and the assigned taxon equals the truth;
//! * `MC` – misclassified: assigned, wrong, truth taxon present in reference;
//! * `OC` – overclassified: assigned although the truth taxon at `d` is
//!   absent from the reference;
//! * `UC` – underclassified: not assigned although the truth is present;
//! * `TN` – not assigned and the truth taxon is absent;
//! * `NA` – the truth lineage stops above `d`; excluded from the counts at
//!   `d` (an unknown truth is neither right nor wrong).
//!
//! Following Murali et al. (2018), taxa absent from the reference make any
//! assignment at that rank an overclassification.

use std::collections::HashSet;

use rand::{Rng, SeedableRng};
use rand_xoshiro::Xoshiro256PlusPlus;
use serde::Serialize;

use crate::model::Model;
use crate::taxonomy::clean_lineage;

/// Outcome at one rank.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
pub enum Outcome {
    TP,
    MC,
    OC,
    UC,
    TN,
    /// The truth lineage has no name at this rank (e.g. a label truncated
    /// at a placeholder): correctness is unknown, so the sequence is left
    /// out of the counts at this rank.
    NA,
}

/// One held-out sequence's truth and prediction.
#[derive(Clone, Debug)]
pub struct EvalRecord {
    /// Truth names below Root.
    pub truth: Vec<String>,
    /// Predicted lineage names below Root (full, regardless of threshold).
    pub pred: Vec<String>,
    /// Confidence per predicted name (percent).
    pub conf: Vec<f64>,
}

/// Set of lineage paths (`"A;B;"`) present in a model.
pub fn present_paths(model: &Model) -> HashSet<String> {
    model
        .taxonomy
        .iter()
        .map(|t| t.strip_prefix("Root;").unwrap_or(t).to_string())
        .collect()
}

/// Split a raw lineage into names below Root.
pub fn truth_names(raw: &str) -> Vec<String> {
    match clean_lineage(raw) {
        Ok(c) => c
            .trim_end_matches(';')
            .split(';')
            .map(str::to_string)
            .collect(),
        Err(_) => Vec::new(),
    }
}

fn path(names: &[String], d: usize) -> String {
    let mut s = String::new();
    for n in &names[..d] {
        s.push_str(n);
        s.push(';');
    }
    s
}

/// Outcome of a record at depth `d` given per-depth thresholds
/// (`thresholds[d-1]`); assignments are truncated at the first failing rank.
pub fn outcome(r: &EvalRecord, d: usize, thresholds: &[f64], present: &HashSet<String>) -> Outcome {
    let mut assigned_depth = 0;
    for (i, &c) in r.conf.iter().enumerate() {
        let t = thresholds.get(i).copied().unwrap_or(f64::INFINITY);
        if c >= t {
            assigned_depth = i + 1;
        } else {
            break;
        }
    }
    if r.truth.len() < d {
        return Outcome::NA;
    }
    let assigned = assigned_depth >= d;
    let truth_present = present.contains(&path(&r.truth, d));
    match (assigned, truth_present) {
        (true, true) => {
            if path(&r.pred, d) == path(&r.truth, d) {
                Outcome::TP
            } else {
                Outcome::MC
            }
        }
        (true, false) => Outcome::OC,
        (false, true) => Outcome::UC,
        (false, false) => Outcome::TN,
    }
}

/// Aggregated metrics at one rank and threshold.
#[derive(Clone, Debug, Default, Serialize)]
pub struct RankMetrics {
    pub depth: usize,
    pub threshold: f64,
    pub n: usize,
    pub tp: usize,
    pub mc: usize,
    pub oc: usize,
    pub uc: usize,
    pub tn: usize,
}

impl RankMetrics {
    pub fn n_present(&self) -> usize {
        self.tp + self.mc + self.uc
    }
    pub fn n_absent(&self) -> usize {
        self.oc + self.tn
    }
    /// TP / present.
    pub fn accuracy(&self) -> f64 {
        ratio(self.tp, self.n_present())
    }
    /// MC / present.
    pub fn misclassification_rate(&self) -> f64 {
        ratio(self.mc, self.n_present())
    }
    /// UC / present.
    pub fn underclassification_rate(&self) -> f64 {
        ratio(self.uc, self.n_present())
    }
    /// OC / absent.
    pub fn overclassification_rate(&self) -> f64 {
        ratio(self.oc, self.n_absent())
    }
    /// (MC + OC) / all.
    pub fn error_rate(&self) -> f64 {
        ratio(self.mc + self.oc, self.n)
    }
    /// (MC + OC) / assigned.
    pub fn error_among_assigned(&self) -> f64 {
        ratio(self.mc + self.oc, self.tp + self.mc + self.oc)
    }
}

fn ratio(a: usize, b: usize) -> f64 {
    if b == 0 {
        f64::NAN
    } else {
        a as f64 / b as f64
    }
}

/// Metrics at `depth` for a uniform threshold.
pub fn metrics(
    records: &[EvalRecord],
    depth: usize,
    thresholds: &[f64],
    present: &HashSet<String>,
) -> RankMetrics {
    let mut m = RankMetrics {
        depth,
        threshold: thresholds.get(depth - 1).copied().unwrap_or(f64::NAN),
        ..Default::default()
    };
    for r in records {
        match outcome(r, depth, thresholds, present) {
            Outcome::TP => m.tp += 1,
            Outcome::MC => m.mc += 1,
            Outcome::OC => m.oc += 1,
            Outcome::UC => m.uc += 1,
            Outcome::TN => m.tn += 1,
            Outcome::NA => continue,
        }
        m.n += 1;
    }
    m
}

/// Paired comparison of two configurations at one depth: difference
/// (B - A) of a per-sequence metric with a percentile bootstrap CI.
#[derive(Clone, Debug, Serialize)]
pub struct Comparison {
    pub depth: usize,
    pub metric: String,
    pub a: f64,
    pub b: f64,
    pub diff: f64,
    pub ci_low: f64,
    pub ci_high: f64,
}

/// Indicator per sequence for a metric name: `correct` (TP over all),
/// `error` (MC+OC over all), `overclassified` (OC over all),
/// `underclassified` (UC over all).
pub fn indicator(o: Outcome, metric: &str) -> f64 {
    let v = match metric {
        "correct" => o == Outcome::TP,
        "error" => matches!(o, Outcome::MC | Outcome::OC),
        "overclassified" => o == Outcome::OC,
        "misclassified" => o == Outcome::MC,
        "underclassified" => o == Outcome::UC,
        _ => false,
    };
    if v {
        1.0
    } else {
        0.0
    }
}

/// Paired bootstrap CI of mean(b) - mean(a).
pub fn paired_bootstrap(a: &[f64], b: &[f64], reps: usize, seed: u64) -> (f64, f64, f64) {
    let n = a.len();
    let diff: Vec<f64> = a.iter().zip(b).map(|(x, y)| y - x).collect();
    let mean = diff.iter().sum::<f64>() / n.max(1) as f64;
    if n == 0 {
        return (f64::NAN, f64::NAN, f64::NAN);
    }
    let mut rng = Xoshiro256PlusPlus::seed_from_u64(seed);
    let mut means: Vec<f64> = (0..reps)
        .map(|_| {
            let mut s = 0.0;
            for _ in 0..n {
                s += diff[rng.gen_range(0..n)];
            }
            s / n as f64
        })
        .collect();
    means.sort_by(|x, y| x.partial_cmp(y).unwrap());
    let lo = means[((reps as f64) * 0.025) as usize];
    let hi = means[(((reps as f64) * 0.975) as usize).min(reps - 1)];
    (mean, lo, hi)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn outcomes() {
        let present: HashSet<String> = ["A;", "A;B;", "A;C;"]
            .iter()
            .map(|s| s.to_string())
            .collect();
        let r = EvalRecord {
            truth: vec!["A".into(), "B".into()],
            pred: vec!["A".into(), "C".into()],
            conf: vec![99.0, 70.0],
        };
        assert_eq!(outcome(&r, 1, &[60.0, 60.0], &present), Outcome::TP);
        assert_eq!(outcome(&r, 2, &[60.0, 60.0], &present), Outcome::MC);
        assert_eq!(outcome(&r, 2, &[60.0, 80.0], &present), Outcome::UC);
        let novel = EvalRecord {
            truth: vec!["A".into(), "Z".into()],
            ..r.clone()
        };
        assert_eq!(outcome(&novel, 2, &[60.0, 60.0], &present), Outcome::OC);
        assert_eq!(outcome(&novel, 2, &[60.0, 80.0], &present), Outcome::TN);
        let m = metrics(&[r, novel], 2, &[60.0, 60.0], &present);
        assert_eq!((m.mc, m.oc), (1, 1));
        assert_eq!(m.error_rate(), 1.0);
        // a truth label truncated above the rank is unknown, not an
        // overclassification
        let shallow = EvalRecord {
            truth: vec!["A".into()],
            pred: vec!["A".into(), "B".into()],
            conf: vec![99.0, 99.0],
        };
        assert_eq!(outcome(&shallow, 2, &[60.0, 60.0], &present), Outcome::NA);
        assert_eq!(outcome(&shallow, 1, &[60.0, 60.0], &present), Outcome::TP);
        let m2 = metrics(&[shallow], 2, &[60.0, 60.0], &present);
        assert_eq!((m2.n, m2.oc), (0, 0));
        let (d, lo, hi) = paired_bootstrap(&[0.0, 0.0, 1.0, 1.0], &[1.0, 1.0, 1.0, 1.0], 200, 1);
        assert_eq!(d, 0.5);
        assert!(lo <= d && d <= hi);
    }
}
