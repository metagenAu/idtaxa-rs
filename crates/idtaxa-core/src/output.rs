//! Output formatting (native TSV and DECIPHER-compatible).

use crate::classify::{Classification, SimilarityProfile};
use crate::model::Model;

/// Output format.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Format {
    /// Native TSV (`id seq_hash strand taxonomy confidences ...`).
    Tsv,
    /// DECIPHER `IdTaxa(type = "collapsed")` strings: `id<TAB>Root [rank, 100.0%]; ...`.
    Decipher,
}

/// Which optional columns to write.
#[derive(Clone, Debug, Default)]
pub struct Columns {
    pub full_confidences: bool,
    pub calibrated: bool,
    pub novelty: bool,
}

/// Everything known about one query's result.
#[derive(Clone, Debug)]
pub struct ResultRow {
    pub cls: Classification,
    /// Calibrated confidence per lineage entry (percent), if calibrated.
    pub calibrated: Option<Vec<f64>>,
    pub similarity: Option<SimilarityProfile>,
    /// Depth (1 = first rank) of the deepest rank consistent with a known
    /// taxon; 0 = novel at the first rank; `None` when not computed.
    pub novelty_depth: Option<usize>,
    /// Number of lineage entries (incl. Root) assigned after thresholds and
    /// optional novelty truncation.
    pub n_assigned: usize,
}

/// Header line for the native TSV.
pub fn header(cols: &Columns) -> String {
    let mut h = vec!["id", "seq_hash", "strand", "taxonomy", "confidences"];
    if cols.full_confidences {
        h.push("full_taxonomy");
        h.push("full_confidences");
    }
    if cols.calibrated {
        h.push("calibrated_confidences");
    }
    if cols.novelty {
        h.push("max_similarity");
        h.push("novelty_rank");
    }
    h.push("bootstraps_used");
    h.join("\t")
}

fn fmt1(x: f64) -> String {
    format!("{x:.1}")
}

/// Native TSV row (without trailing newline).
pub fn tsv_row(model: &Model, id: &str, seq_hash: u64, row: &ResultRow, cols: &Columns) -> String {
    let c = &row.cls;
    let n = row.n_assigned.min(c.lineage.len());
    // Root (index 0) is not printed; ranks below it are
    let names: Vec<&str> = c.lineage[1..n.max(1)]
        .iter()
        .map(|&x| model.taxon(x))
        .collect();
    let confs: Vec<String> = c.confidences[1..n.max(1)]
        .iter()
        .map(|&x| fmt1(x))
        .collect();
    let mut out = format!(
        "{id}\t{seq_hash:016x}\t{}\t{}\t{}",
        c.orientation.as_str(),
        names.join(";"),
        confs.join(";")
    );
    if cols.full_confidences {
        let fnames: Vec<&str> = c.lineage[1..].iter().map(|&x| model.taxon(x)).collect();
        let fconfs: Vec<String> = c.confidences[1..].iter().map(|&x| fmt1(x)).collect();
        out.push('\t');
        out.push_str(&fnames.join(";"));
        out.push('\t');
        out.push_str(&fconfs.join(";"));
    }
    if cols.calibrated {
        out.push('\t');
        if let Some(cal) = &row.calibrated {
            let v: Vec<String> = cal[1..].iter().map(|&x| fmt1(x)).collect();
            out.push_str(&v.join(";"));
        }
    }
    if cols.novelty {
        out.push('\t');
        if let Some(s) = &row.similarity {
            out.push_str(&format!("{:.4}", s.max_reference));
        }
        out.push('\t');
        match row.novelty_depth {
            Some(d) => {
                // name the rank: deepest consistent rank of the lineage
                let label = if d == 0 {
                    "none".to_string()
                } else if d + 1 >= c.lineage.len() {
                    "known".to_string()
                } else {
                    rank_label(model, c.lineage[d])
                };
                out.push_str(&label);
            }
            None => {}
        }
    }
    out.push('\t');
    out.push_str(&c.bootstraps_used.to_string());
    out
}

fn rank_label(model: &Model, node: u32) -> String {
    match model.rank(node) {
        Some(r) => r.to_string(),
        None => format!("level{}", model.level(node) - 1),
    }
}

/// DECIPHER `type = "collapsed"` string for a result, reproducing
/// `IdTaxa()`'s formatting (R/IdTaxa.R): assigned taxa with confidences
/// and, when not every rank passes, a final `unclassified_<taxon>` entry.
pub fn decipher_collapsed(model: &Model, row: &ResultRow) -> String {
    let c = &row.cls;
    let has_ranks = model.meta.ranks.is_some();
    let entry = |name: &str, rank_node: Option<u32>, conf: f64| -> String {
        if has_ranks {
            let r = rank_node.and_then(|n| model.rank(n)).unwrap_or("NA");
            format!("{name} [{r}, {conf:.1}%]")
        } else {
            format!("{name} [{conf:.1}%]")
        }
    };
    if c.orientation == crate::classify::Orientation::None {
        // literal default of IdTaxa for unclassifiable queries
        return if has_ranks {
            format!(
                "Root [{}, 0%]; unclassified_Root [{}, 0%]",
                model.rank(0).unwrap_or("NA"),
                model
                    .meta
                    .ranks
                    .as_ref()
                    .and_then(|r| r.get(1))
                    .map(String::as_str)
                    .unwrap_or("NA")
            )
        } else {
            "Root [0%]; unclassified_Root [0%]".to_string()
        };
    }
    let full = c.lineage.len();
    let n = row.n_assigned.clamp(1, full);
    let mut parts: Vec<String> = c.lineage[..n]
        .iter()
        .zip(&c.confidences[..n])
        .map(|(&node, &cf)| entry(model.taxon(node), Some(node), cf))
        .collect();
    if n < full {
        let last = c.lineage[n - 1];
        // rank of the next lineage element (Root's second rank when the
        // query was not classified at all)
        let next = Some(c.lineage[n]);
        parts.push(entry(
            &format!("unclassified_{}", model.taxon(last)),
            next,
            c.confidences[n - 1],
        ));
    }
    parts.join("; ")
}
