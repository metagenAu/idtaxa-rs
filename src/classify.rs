//! Classification of query sequences.
//!
//! 1. Tree descent with decision k-mers narrows the candidate training
//!    sequences to a subtree.
//! 2. The training sequences under the stop node are ranked by weighted
//!    k-mer overlap with the query (via an inverted index) and the top
//!    `max_candidates` are kept. `bootstraps` replicates each sample
//!    `ceil(L^0.47)` of the query's L k-mers (with replacement). For every candidate group the best
//!    weighted k-mer match fraction is computed; the top group receives a
//!    vote equal to that similarity (so distant matches give low
//!    confidence — this is what makes IDTAXA conservative with novel taxa).
//! 3. A taxon's confidence is the summed votes of the groups beneath it
//!    (for ranks fixed by descent, at least the descent bootstrap support).
//!    The lineage is reported down to the last rank whose confidence meets
//!    the threshold.

use crate::kmer;
use crate::model::{DescentStep, Model};
use crate::seq;
use crate::taxonomy::{prefix_for_rank, ROOT};
use rand::rngs::SmallRng;
use rand::Rng;
use std::cmp::Ordering;
use std::collections::HashMap;
use std::io::{self, Write};

#[derive(Clone, Copy, Debug, PartialEq, Eq, clap::ValueEnum)]
pub enum Strand {
    /// Query orientation as given.
    Top,
    /// Reverse complement only.
    Bottom,
    /// Try both orientations, keep the better-supported one.
    Both,
}

#[derive(Clone, Debug)]
pub struct ClassifyParams {
    /// Confidence threshold in [0, 1].
    pub threshold: f32,
    pub bootstraps: usize,
    pub min_descend: f32,
    pub sample_exponent: f64,
    pub strand: Strand,
    pub seed: u64,
    /// Only the training sequences with the highest weighted k-mer overlap
    /// with the query are bootstrapped.
    pub max_candidates: usize,
    /// Narrow candidates by decision-k-mer tree descent first (original
    /// IDTAXA). When false, candidates come from the whole reference and the
    /// upper ranks are fixed by nearest-neighbour bootstrap support.
    pub tree_descent: bool,
    /// Fraction of query k-mers sampled per replicate when estimating
    /// nearest-neighbour support for the upper ranks.
    pub support_fraction: f32,
}

impl Default for ClassifyParams {
    fn default() -> Self {
        ClassifyParams {
            threshold: 0.6,
            bootstraps: 100,
            min_descend: 0.98,
            sample_exponent: 0.47,
            strand: Strand::Both,
            seed: 1,
            max_candidates: 500,
            tree_descent: false,
            support_fraction: 0.2,
        }
    }
}

/// Classification result for one query.
#[derive(Clone, Debug)]
pub struct Assignment {
    /// Best path from the root (inclusive) with confidences, regardless of threshold.
    pub path: Vec<(usize, f32)>,
    /// Number of entries of `path` (including root) meeting the threshold.
    pub n_classified: usize,
    pub reverse: bool,
    /// Total vote mass (mean best similarity); used to choose strand.
    pub support: f32,
}

impl Assignment {
    /// Classified lineage excluding the root.
    pub fn classified(&self) -> &[(usize, f32)] {
        &self.path[1..self.n_classified]
    }
}

/// Classify one sequence.
pub fn classify_seq(model: &Model, raw: &[u8], p: &ClassifyParams, rng: &mut SmallRng) -> Assignment {
    let clean = seq::clean(raw);
    let fwd = |rng: &mut SmallRng| classify_kmers(model, &kmer::unique_kmers(&clean, model.k), p, rng);
    let rev = |rng: &mut SmallRng| classify_kmers(model, &kmer::unique_kmers(&seq::revcomp(&clean), model.k), p, rng);
    match p.strand {
        Strand::Top => fwd(rng),
        Strand::Bottom => {
            let mut a = rev(rng);
            a.reverse = true;
            a
        }
        Strand::Both => {
            let a = fwd(rng);
            let mut b = rev(rng);
            b.reverse = true;
            if b.support > a.support {
                b
            } else {
                a
            }
        }
    }
}

/// Classify a query given its sorted unique k-mers.
pub fn classify_kmers(model: &Model, q: &[u32], p: &ClassifyParams, rng: &mut SmallRng) -> Assignment {
    classify_core(model, q, p, rng, None)
}

/// Classify training sequence `s` against the rest of the reference
/// (leave-one-out), used to detect mislabelled references.
pub fn classify_leave_one_out(model: &Model, s: usize, p: &ClassifyParams, rng: &mut SmallRng) -> Assignment {
    classify_core(model, &model.seq_kmers[s], p, rng, Some(s))
}

fn classify_core(
    model: &Model,
    q: &[u32],
    p: &ClassifyParams,
    rng: &mut SmallRng,
    exclude: Option<usize>,
) -> Assignment {
    let unclassified = Assignment { path: vec![(ROOT, 1.0)], n_classified: 1, reverse: false, support: 0.0 };
    if q.is_empty() {
        return unclassified;
    }
    // Candidate subtree: either from decision-k-mer tree descent, or the whole tree.
    let path: Vec<DescentStep> =
        if p.tree_descent { model.descend(q, rng, p.bootstraps, p.min_descend) } else { vec![(ROOT, Some(1.0))] };
    let stop = path.last().unwrap().0;
    let l = q.len();
    let words = l.div_ceil(64);

    // weighted k-mer overlap of every training sequence under `stop` (inverted index)
    let range = model.members(stop);
    let (lo, hi) = (range.start, range.end);
    let mut overlap = vec![0f32; hi - lo];
    for &x in q {
        let w = model.weight(x);
        let post = model.index.postings(x);
        let a = post.partition_point(|&s| (s as usize) < lo);
        for &s in &post[a..] {
            let s = s as usize;
            if s >= hi {
                break;
            }
            overlap[s - lo] += w;
        }
    }
    if let Some(x) = exclude {
        if (lo..hi).contains(&x) {
            overlap[x - lo] = 0.0;
        }
    }
    let mut top: Vec<u32> = (0..(hi - lo) as u32).filter(|&i| overlap[i as usize] > 0.0).collect();
    if top.len() > p.max_candidates.max(1) {
        let m = p.max_candidates.max(1);
        top.select_nth_unstable_by(m - 1, |a, b| {
            overlap[*b as usize].partial_cmp(&overlap[*a as usize]).unwrap().then(a.cmp(b))
        });
        top.truncate(m);
    }
    drop(overlap);

    // candidate sequences as bitsets over the query's k-mers
    let mut group_index: HashMap<usize, usize> = HashMap::new();
    let mut groups: Vec<usize> = Vec::new();
    let mut cands: Vec<(usize, Vec<u64>)> = Vec::with_capacity(top.len());
    for &i in &top {
        let s = lo + i as usize;
        let sk = &model.seq_kmers[s];
        let mut bits = vec![0u64; words];
        let (mut i, mut j) = (0, 0);
        while i < l && j < sk.len() {
            match q[i].cmp(&sk[j]) {
                Ordering::Less => i += 1,
                Ordering::Greater => j += 1,
                Ordering::Equal => {
                    bits[i >> 6] |= 1u64 << (i & 63);
                    i += 1;
                    j += 1;
                }
            }
        }
        let leaf = model.seq_leaf[s];
        let gi = *group_index.entry(leaf).or_insert_with(|| {
            groups.push(leaf);
            groups.len() - 1
        });
        cands.push((gi, bits));
    }

    // Bootstrap: in each replicate the most similar group wins. IDTAXA votes
    // are similarity-weighted with ceil(L^0.47) k-mers sampled; nearest-
    // neighbour support (unweighted wins) uses a larger sample so that it is
    // stable enough to apply the min_descend threshold.
    let n_samp = ((l as f64).powf(p.sample_exponent).ceil() as usize).max(1);
    let (votes, _) = bootstrap(model, q, &cands, groups.len(), n_samp, p.bootstraps, rng);
    let wins = if p.tree_descent {
        vec![0f32; groups.len()]
    } else {
        let n_support = ((p.support_fraction as f64 * l as f64).ceil() as usize).max(n_samp);
        bootstrap(model, q, &cands, groups.len(), n_support, p.bootstraps, rng).1
    };
    let support: f32 = votes.iter().sum();

    // accumulate to ancestors up to the candidate subtree root
    let mut conf: HashMap<usize, (f32, f32)> = HashMap::new();
    for (gi, &leaf) in groups.iter().enumerate() {
        if votes[gi] <= 0.0 && wins[gi] <= 0.0 {
            continue;
        }
        let mut n = leaf;
        loop {
            let e = conf.entry(n).or_insert((0.0, 0.0));
            e.0 += votes[gi];
            e.1 += wins[gi];
            if n == stop {
                break;
            }
            n = model.taxonomy.parent(n).expect("candidate lies under stop node");
        }
    }

    // ranks fixed by decision-k-mer descent
    let mut out: Vec<(usize, f32)> = Vec::with_capacity(path.len() + 8);
    let mut cummin = 1f32;
    for &(n, dc) in &path {
        let c = match dc {
            Some(d) => {
                cummin = cummin.min(d);
                cummin.max(support)
            }
            None => support,
        };
        out.push((n, c));
    }
    // below: follow the best-supported child. While the unweighted bootstrap
    // support stays ≥ min_descend the rank is reported with that support
    // (nearest-neighbour "descent"); afterwards confidence is the
    // similarity-weighted vote share.
    let mut descending = !p.tree_descent;
    let mut cur = stop;
    loop {
        let mut best: Option<(usize, (f32, f32))> = None;
        for &c in &model.taxonomy.nodes[cur].children {
            if let Some(&v) = conf.get(&c) {
                if best.map(|b| (v.1, v.0) > (b.1 .1, b.1 .0)).unwrap_or(true) {
                    best = Some((c, v));
                }
            }
        }
        match best {
            Some((c, (weighted, unweighted))) => {
                descending = descending && unweighted >= p.min_descend;
                out.push((c, if descending { unweighted.max(weighted) } else { weighted }));
                cur = c;
            }
            None => break,
        }
    }
    // confidences must not increase with depth
    for i in (0..out.len().saturating_sub(1)).rev() {
        if out[i + 1].1 > out[i].1 {
            out[i].1 = out[i + 1].1;
        }
    }
    out[0].1 = 1.0;
    let n_classified = 1 + out[1..].iter().take_while(|x| x.1 >= p.threshold).count();
    Assignment { path: out, n_classified, reverse: false, support }
}

/// Run `bootstraps` replicates sampling `n_samp` query k-mers each. Returns
/// per-group (similarity-weighted votes, unweighted wins), both divided by
/// the number of replicates.
fn bootstrap(
    model: &Model,
    q: &[u32],
    cands: &[(usize, Vec<u64>)],
    n_groups: usize,
    n_samp: usize,
    bootstraps: usize,
    rng: &mut SmallRng,
) -> (Vec<f32>, Vec<f32>) {
    let l = q.len();
    let mut votes = vec![0f32; n_groups];
    let mut wins = vec![0f32; n_groups];
    if cands.is_empty() || bootstraps == 0 {
        return (votes, wins);
    }
    let mut best = vec![0f32; n_groups];
    let mut pos = vec![0usize; n_samp];
    let mut w = vec![0f32; n_samp];
    for _ in 0..bootstraps {
        let mut denom = 0f32;
        for j in 0..n_samp {
            let x = rng.gen_range(0..l);
            pos[j] = x;
            w[j] = model.weight(q[x]);
            denom += w[j];
        }
        if denom <= 0.0 {
            continue;
        }
        best.iter_mut().for_each(|b| *b = 0.0);
        for (gi, bits) in cands {
            let mut sc = 0f32;
            for j in 0..n_samp {
                let x = pos[j];
                if (bits[x >> 6] >> (x & 63)) & 1 == 1 {
                    sc += w[j];
                }
            }
            if sc > best[*gi] {
                best[*gi] = sc;
            }
        }
        let m = best.iter().cloned().fold(0f32, f32::max);
        if m <= 0.0 {
            continue;
        }
        let cutoff = m * (1.0 - 1e-6);
        let n_ties = best.iter().filter(|&&b| b >= cutoff).count() as f32;
        for (gi, &b) in best.iter().enumerate() {
            if b >= cutoff {
                votes[gi] += (m / denom) / n_ties;
                wins[gi] += 1.0 / n_ties;
            }
        }
    }
    let inv = 1.0 / bootstraps as f32;
    votes.iter_mut().for_each(|v| *v *= inv);
    wins.iter_mut().for_each(|v| *v *= inv);
    (votes, wins)
}

/// Output formats for classification results.
#[derive(Clone, Copy, Debug, PartialEq, Eq, clap::ValueEnum)]
pub enum OutputFormat {
    /// Tab-separated table with lineage, per-level confidences and one column per rank.
    Tsv,
    /// QIIME 2 taxonomy format (Feature ID / Taxon / Confidence).
    Qiime,
}

fn rank_of(model: &Model, n: usize) -> String {
    model.taxonomy.nodes[n].rank.clone().unwrap_or_else(|| "NA".to_string())
}

/// Write assignments in the requested format.
pub fn write_assignments(
    w: &mut dyn Write,
    model: &Model,
    ids: &[String],
    assignments: &[Assignment],
    format: OutputFormat,
) -> io::Result<()> {
    let tax = &model.taxonomy;
    match format {
        OutputFormat::Tsv => {
            let ranks = &model.meta.ranks;
            write!(w, "id\tstrand\ttaxon\trank\tconfidence\tlineage\tlineage_ranks\tconfidences\tbest_path")?;
            for r in ranks {
                write!(w, "\t{r}")?;
            }
            writeln!(w)?;
            for (id, a) in ids.iter().zip(assignments) {
                let cl = a.classified();
                let (taxon, rank, c) = match cl.last() {
                    Some(&(n, c)) => (tax.nodes[n].name.clone(), rank_of(model, n), c),
                    None => ("Unclassified".to_string(), "NA".to_string(), 0.0),
                };
                let names: Vec<&str> = cl.iter().map(|&(n, _)| tax.nodes[n].name.as_str()).collect();
                let rks: Vec<String> = cl.iter().map(|&(n, _)| rank_of(model, n)).collect();
                let confs: Vec<String> = cl.iter().map(|&(_, c)| format!("{:.1}", c * 100.0)).collect();
                let best: Vec<String> =
                    a.path[1..].iter().map(|&(n, c)| format!("{}({:.1})", tax.nodes[n].name, c * 100.0)).collect();
                write!(
                    w,
                    "{id}\t{}\t{taxon}\t{rank}\t{:.1}\t{}\t{}\t{}\t{}",
                    if a.reverse { "-" } else { "+" },
                    c * 100.0,
                    names.join(";"),
                    rks.join(";"),
                    confs.join(";"),
                    best.join(";")
                )?;
                for r in ranks {
                    let name = cl
                        .iter()
                        .find(|&&(n, _)| tax.nodes[n].rank.as_deref() == Some(r.as_str()))
                        .map(|&(n, _)| tax.nodes[n].name.as_str())
                        .unwrap_or("");
                    write!(w, "\t{name}")?;
                }
                writeln!(w)?;
            }
        }
        OutputFormat::Qiime => {
            writeln!(w, "Feature ID\tTaxon\tConfidence")?;
            for (id, a) in ids.iter().zip(assignments) {
                let cl = a.classified();
                if cl.is_empty() {
                    writeln!(w, "{id}\tUnassigned\t{:.4}", 1.0 - a.support.min(1.0))?;
                    continue;
                }
                let parts: Vec<String> = cl
                    .iter()
                    .map(|&(n, _)| {
                        let node = &tax.nodes[n];
                        match node.rank.as_deref().and_then(prefix_for_rank) {
                            Some(p) => format!("{p}__{}", node.name),
                            None => node.name.clone(),
                        }
                    })
                    .collect();
                writeln!(w, "{id}\t{}\t{:.4}", parts.join("; "), cl.last().unwrap().1)?;
            }
        }
    }
    Ok(())
}
