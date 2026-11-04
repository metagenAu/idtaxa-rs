//! Reference curation: find training sequences whose label conflicts with a
//! confident leave-one-out classification. These are usually mislabelled,
//! contaminated or chimeric references (or taxa that the marker cannot
//! resolve), and are worth reviewing before building a production database.

use crate::classify::{classify_leave_one_out, ClassifyParams, Strand};
use crate::model::Model;
use crate::train::mix_seed;
use rand::rngs::SmallRng;
use rand::SeedableRng;
use rayon::prelude::*;

/// A reference whose label conflicts with its leave-one-out classification.
#[derive(Clone, Debug)]
pub struct Conflict {
    /// Index of the training sequence in the model.
    pub index: usize,
    pub id: String,
    pub lineage: String,
    /// Confidently predicted lineage (down to the first conflicting rank).
    pub predicted: String,
    /// Depth (1 = first rank below root) of the first conflicting rank.
    pub depth: usize,
    pub rank: String,
    /// Confidence of the conflicting assignment.
    pub confidence: f32,
}

/// Classify every training sequence against the others and report
/// confident disagreements (at `p.threshold`).
pub fn find_conflicts(model: &Model, p: &ClassifyParams) -> Vec<Conflict> {
    let params = ClassifyParams { strand: Strand::Top, ..p.clone() };
    let tax = &model.taxonomy;
    let mut out: Vec<Conflict> = (0..model.seq_kmers.len())
        .into_par_iter()
        .filter_map(|s| {
            let mut rng = SmallRng::seed_from_u64(mix_seed(params.seed, s as u64, 99));
            let a = classify_leave_one_out(model, s, &params, &mut rng);
            let truth = tax.path(model.seq_leaf[s]);
            let cl = a.classified();
            for (d, &(n, c)) in cl.iter().enumerate() {
                let depth = d + 1;
                if depth >= truth.len() {
                    break; // predicted deeper than the reference's own label
                }
                if n != truth[depth] {
                    return Some(Conflict {
                        index: s,
                        id: model.seq_ids[s].clone(),
                        lineage: tax.lineage_string(model.seq_leaf[s]),
                        predicted: tax.lineage_string(n),
                        depth,
                        rank: tax.nodes[n].rank.clone().unwrap_or_else(|| "NA".into()),
                        confidence: c,
                    });
                }
            }
            None
        })
        .collect();
    out.sort_by_key(|c| (c.depth, c.index));
    out
}
