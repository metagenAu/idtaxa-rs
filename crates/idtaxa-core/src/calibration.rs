//! Per-rank confidence calibration (Phase 6.3) and novelty cut-offs (6.4).

use serde::{Deserialize, Serialize};

use crate::classify::{Classification, SimilarityProfile};
use crate::error::{Error, Result};

/// Calibration of one rank (depth below Root, 1-based).
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct RankCalibration {
    pub depth: usize,
    pub rank: String,
    /// Number of held-out assignments used.
    pub n: usize,
    /// Isotonic fit as piecewise-linear knots: raw confidence (percent) →
    /// probability correct (0..1).
    pub x: Vec<f64>,
    pub y: Vec<f64>,
    /// Raw-confidence threshold achieving the target error rate.
    pub threshold: f64,
    /// (target error rate, raw threshold) pairs.
    pub grid: Vec<(f64, f64)>,
}

impl RankCalibration {
    /// Calibrated probability (0..1) for a raw confidence (percent).
    pub fn apply(&self, conf: f64) -> f64 {
        interp(&self.x, &self.y, conf)
    }
}

/// Novelty cut-off for one rank.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct NoveltyCutoff {
    pub depth: usize,
    pub rank: String,
    /// Minimum similarity to the assigned taxon for it to count as known.
    pub cutoff: f64,
    pub n_known: usize,
    pub n_novel: usize,
    /// Fraction of known taxa kept / novel taxa flagged at the cut-off.
    pub sensitivity_known: f64,
    pub specificity_novel: f64,
}

/// Calibration file content (`calibration.json`).
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct Calibration {
    pub format: String,
    pub version: u32,
    pub model_id: String,
    pub target_error: f64,
    pub ranks: Vec<RankCalibration>,
    #[serde(default)]
    pub novelty: Option<Vec<NoveltyCutoff>>,
}

impl Calibration {
    pub fn load(path: &std::path::Path) -> Result<Self> {
        let b = std::fs::read(path).map_err(|e| Error::io(path, e))?;
        let c: Calibration = serde_json::from_slice(&b)?;
        if c.format != "idtaxa-calibration" {
            return Err(Error::InvalidParameter(format!(
                "{} is not a calibration file",
                path.display()
            )));
        }
        Ok(c)
    }

    /// Refuse calibration produced for another model.
    pub fn check_model(&self, model_id: &str) -> Result<()> {
        if self.model_id != model_id {
            return Err(Error::InvalidParameter(format!(
                "calibration was made for model {} but the model is {model_id}; recalibrate",
                self.model_id
            )));
        }
        Ok(())
    }

    /// Calibrated confidences (percent) for a lineage; Root keeps its raw
    /// value, ranks without a calibration are copied unchanged.
    pub fn calibrate(&self, cls: &Classification) -> Vec<f64> {
        cls.confidences
            .iter()
            .enumerate()
            .map(|(d, &c)| match self.ranks.iter().find(|r| r.depth == d) {
                Some(r) if d > 0 => 100.0 * r.apply(c),
                _ => c,
            })
            .collect()
    }

    /// Per-depth raw thresholds (index 0 = first rank); ranks without a
    /// calibration use `default`.
    pub fn thresholds(&self, max_depth: usize, default: f64) -> Vec<f64> {
        (1..=max_depth)
            .map(|d| {
                self.ranks
                    .iter()
                    .find(|r| r.depth == d)
                    .map(|r| r.threshold)
                    .unwrap_or(default)
            })
            .collect()
    }

    /// Deepest depth consistent with a known taxon (0 = novel at the first
    /// rank); `None` without novelty cut-offs.
    pub fn novelty_depth(&self, prof: &SimilarityProfile) -> Option<usize> {
        let cuts = self.novelty.as_ref()?;
        let mut depth = 0;
        for d in 1..prof.per_lineage.len() {
            match cuts.iter().find(|c| c.depth == d) {
                Some(c) if prof.per_lineage[d] < c.cutoff => break,
                _ => depth = d,
            }
        }
        Some(depth)
    }
}

/// Piecewise-linear interpolation with constant extrapolation.
pub fn interp(x: &[f64], y: &[f64], v: f64) -> f64 {
    if x.is_empty() {
        return v / 100.0;
    }
    if v <= x[0] {
        return y[0];
    }
    if v >= x[x.len() - 1] {
        return y[y.len() - 1];
    }
    let i = x.partition_point(|&a| a <= v);
    let (x0, x1, y0, y1) = (x[i - 1], x[i], y[i - 1], y[i]);
    if x1 == x0 {
        y1
    } else {
        y0 + (y1 - y0) * (v - x0) / (x1 - x0)
    }
}

/// Isotonic (non-decreasing) regression of `correct` on `conf` by the
/// pool-adjacent-violators algorithm; returns piecewise-linear knots
/// (block start and end x, block mean y).
pub fn isotonic(points: &[(f64, bool)]) -> (Vec<f64>, Vec<f64>) {
    let mut p: Vec<(f64, f64)> = points
        .iter()
        .map(|&(x, c)| (x, if c { 1.0 } else { 0.0 }))
        .collect();
    p.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
    // blocks: (xmin, xmax, sum, count); tied x values start in one block
    let mut blocks: Vec<(f64, f64, f64, f64)> = Vec::new();
    for (x, y) in p {
        match blocks.last_mut() {
            Some(b) if b.1 == x => {
                b.2 += y;
                b.3 += 1.0;
            }
            _ => blocks.push((x, x, y, 1.0)),
        }
        while blocks.len() >= 2 {
            let n = blocks.len();
            let (a, b) = (blocks[n - 2], blocks[n - 1]);
            if a.2 / a.3 > b.2 / b.3 {
                blocks[n - 2] = (a.0, b.1, a.2 + b.2, a.3 + b.3);
                blocks.pop();
            } else {
                break;
            }
        }
    }
    let mut xs = Vec::new();
    let mut ys = Vec::new();
    for (lo, hi, s, n) in blocks {
        let m = s / n;
        xs.push(lo);
        ys.push(m);
        if hi > lo {
            xs.push(hi);
            ys.push(m);
        }
    }
    (xs, ys)
}

/// Smallest raw threshold whose error rate among assignments with
/// confidence >= threshold is at most `target`; 100.1 if none.
pub fn threshold_for_error(points: &[(f64, bool)], target: f64) -> f64 {
    let mut p: Vec<(f64, bool)> = points.to_vec();
    p.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap());
    // sweep from high to low confidence; record error at each distinct x
    let (mut n, mut err) = (0usize, 0usize);
    let mut best = 100.1;
    let mut i = 0;
    while i < p.len() {
        let x = p[i].0;
        while i < p.len() && p[i].0 == x {
            n += 1;
            if !p[i].1 {
                err += 1;
            }
            i += 1;
        }
        if (err as f64) <= target * n as f64 {
            best = x;
        }
    }
    best
}

/// Choose the similarity cut-off maximising balanced accuracy between
/// known (should be >= cut-off) and novel (should be < cut-off) queries.
pub fn novelty_cutoff(known: &[f64], novel: &[f64]) -> (f64, f64, f64) {
    if known.is_empty() || novel.is_empty() {
        return (0.0, 1.0, 0.0);
    }
    let mut cands: Vec<f64> = known.iter().chain(novel.iter()).cloned().collect();
    cands.sort_by(|a, b| a.partial_cmp(b).unwrap());
    cands.dedup();
    let mut best = (0.0, -1.0, 0.0, 0.0);
    for &c in &cands {
        let sens = known.iter().filter(|&&v| v >= c).count() as f64 / known.len() as f64;
        let spec = novel.iter().filter(|&&v| v < c).count() as f64 / novel.len() as f64;
        let ba = 0.5 * (sens + spec);
        if ba > best.1 {
            best = (c, ba, sens, spec);
        }
    }
    (best.0, best.2, best.3)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pav_monotone() {
        let pts = vec![
            (10.0, false),
            (20.0, true),
            (30.0, false),
            (40.0, true),
            (90.0, true),
        ];
        let (x, y) = isotonic(&pts);
        assert!(y.windows(2).all(|w| w[0] <= w[1]));
        assert_eq!(interp(&x, &y, 5.0), 0.0);
        assert_eq!(interp(&x, &y, 95.0), 1.0);
        assert!((interp(&x, &y, 25.0) - 0.5).abs() < 1e-12);
    }

    #[test]
    fn threshold_selection() {
        let pts = vec![
            (95.0, true),
            (90.0, true),
            (80.0, false),
            (70.0, true),
            (50.0, false),
        ];
        assert_eq!(threshold_for_error(&pts, 0.0), 90.0);
        assert_eq!(threshold_for_error(&pts, 0.25), 70.0);
        assert_eq!(threshold_for_error(&pts, 0.5), 50.0);
        assert_eq!(threshold_for_error(&[(10.0, false)], 0.1), 100.1);
    }

    #[test]
    fn novelty_cut() {
        let (c, s, sp) = novelty_cutoff(&[0.9, 0.95, 0.8], &[0.5, 0.6, 0.85]);
        // 0.8 and 0.9 tie on balanced accuracy; the lower cut-off wins
        assert_eq!(c, 0.8);
        assert!(s == 1.0 && (sp - 2.0 / 3.0).abs() < 1e-12);
    }
}
