//! Hold-out cross-validation of a reference database (optionally trimmed to
//! a primer region). Reports per-rank accuracy, under-classification and
//! over-classification, so you can compare e.g. a full-length classifier
//! against a primer-specific one, or SILVA against your in-house database.

use crate::classify::{classify_seq, ClassifyParams};
use crate::model::ModelMeta;
use crate::train::{mix_seed, train, TrainParams, TrainSeq};
use anyhow::{bail, Result};
use rand::rngs::SmallRng;
use rand::seq::SliceRandom;
use rand::SeedableRng;
use rayon::prelude::*;
use std::io::Write;

#[derive(Clone, Debug, Default)]
pub struct LevelStats {
    pub depth: usize,
    pub rank: String,
    /// Test sequences whose taxon at this level is present in training.
    pub n_known: usize,
    pub correct: usize,
    pub misclassified: usize,
    pub underclassified: usize,
    /// Test sequences whose taxon at this level is absent from training.
    pub n_novel: usize,
    pub novel_unassigned: usize,
    pub overclassified: usize,
}

pub struct EvalReport {
    pub n_train: usize,
    pub n_test: usize,
    pub levels: Vec<LevelStats>,
}

/// Hold out `holdout` (0–1) of the sequences, train on the rest and classify
/// the held-out ones.
pub fn evaluate(mut seqs: Vec<TrainSeq>, holdout: f64, tp: &TrainParams, cp: &ClassifyParams) -> Result<EvalReport> {
    if !(0.0..1.0).contains(&holdout) || holdout == 0.0 {
        bail!("holdout must be between 0 and 1");
    }
    let mut rng = SmallRng::seed_from_u64(tp.seed);
    seqs.shuffle(&mut rng);
    let n_test = ((seqs.len() as f64) * holdout).round().max(1.0) as usize;
    if n_test >= seqs.len() {
        bail!("not enough sequences for a hold-out split");
    }
    let test: Vec<TrainSeq> = seqs.drain(..n_test).collect();
    let n_train = seqs.len();
    let (model, _) = train(seqs, tp, ModelMeta::default())?;

    let preds: Vec<Vec<String>> = test
        .par_iter()
        .enumerate()
        .map(|(i, t)| {
            let mut r = SmallRng::seed_from_u64(mix_seed(cp.seed, i as u64, 7));
            let a = classify_seq(&model, &t.seq, cp, &mut r);
            a.classified().iter().map(|&(n, _)| model.taxonomy.nodes[n].name.clone()).collect()
        })
        .collect();

    let max_depth = test.iter().map(|t| t.lineage.len()).max().unwrap_or(0);
    let mut levels: Vec<LevelStats> = (1..=max_depth).map(|d| LevelStats { depth: d, ..Default::default() }).collect();
    for (t, pred) in test.iter().zip(&preds) {
        for d in 1..=t.lineage.len() {
            let st = &mut levels[d - 1];
            if st.rank.is_empty() {
                if let Some(r) = &t.lineage.ranks[d - 1] {
                    st.rank = r.clone();
                }
            }
            let truth = &t.lineage.names[..d];
            let known = model.taxonomy.find(truth).is_some();
            let assigned = pred.len() >= d;
            if known {
                st.n_known += 1;
                if !assigned {
                    st.underclassified += 1;
                } else if pred[..d] == *truth {
                    st.correct += 1;
                } else {
                    st.misclassified += 1;
                }
            } else {
                st.n_novel += 1;
                if assigned {
                    st.overclassified += 1;
                } else {
                    st.novel_unassigned += 1;
                }
            }
        }
    }
    Ok(EvalReport { n_train, n_test, levels })
}

impl EvalReport {
    pub fn write_tsv(&self, w: &mut dyn Write) -> std::io::Result<()> {
        writeln!(w, "level\trank\tn_known\tcorrect\tmisclassified\tunderclassified\tpct_correct\tpct_misclassified\tn_novel\tnovel_unassigned\toverclassified\tpct_overclassified")?;
        let pct = |a: usize, b: usize| if b == 0 { f64::NAN } else { 100.0 * a as f64 / b as f64 };
        for l in &self.levels {
            writeln!(
                w,
                "{}\t{}\t{}\t{}\t{}\t{}\t{:.1}\t{:.1}\t{}\t{}\t{}\t{:.1}",
                l.depth,
                if l.rank.is_empty() { "NA" } else { &l.rank },
                l.n_known,
                l.correct,
                l.misclassified,
                l.underclassified,
                pct(l.correct, l.n_known),
                pct(l.misclassified, l.n_known),
                l.n_novel,
                l.novel_unassigned,
                l.overclassified,
                pct(l.overclassified, l.n_novel),
            )?;
        }
        Ok(())
    }
}
