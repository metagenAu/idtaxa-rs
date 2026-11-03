//! Native training equivalent to DECIPHER's `LearnTaxa()` (R/LearnTaxa.R).

use std::collections::{HashMap, HashSet};

use rand::{Rng, SeedableRng};
use rand_xoshiro::Xoshiro256PlusPlus;
use rayon::prelude::*;
use xxhash_rust::xxh3::Xxh3;

use crate::error::{Error, Result};
use crate::kmer::KmerEncoder;
use crate::model::{Model, ProblemSequence, RawModel, TrainParams};
use crate::stats::quantile_type7;
use crate::taxonomy::{TaxonomyTree, NO_PARENT};

/// Options for [`train`].
#[derive(Clone, Debug)]
pub struct TrainOptions {
    /// K; `None` = DECIPHER's automatic choice.
    pub k: Option<u32>,
    /// Spaced-seed pattern (Phase 6.8); its weight is K.
    pub seed_pattern: Option<String>,
    /// DECIPHER `N = 500` (expected number of k-mer matches by chance).
    pub n: f64,
    pub min_fraction: f64,
    pub max_fraction: f64,
    pub max_iterations: u32,
    pub multiplier: f64,
    pub max_children: u32,
    /// Rank names for levels below Root (optional).
    pub rank_names: Option<Vec<String>>,
    /// RNG seed for the fraction-learning bootstrap.
    pub seed: u64,
    /// Sequences per deterministic parallel block in fraction learning.
    pub block_size: usize,
    /// Optional cap on training sequences per leaf group (Phase 6.7).
    pub max_group_size: Option<u32>,
    pub verbose: bool,
}

impl Default for TrainOptions {
    fn default() -> Self {
        Self {
            k: None,
            seed_pattern: None,
            n: 500.0,
            min_fraction: 0.01,
            max_fraction: 0.06,
            max_iterations: 10,
            multiplier: 100.0,
            max_children: 200,
            rank_names: None,
            seed: 42,
            block_size: 1024,
            max_group_size: None,
            verbose: false,
        }
    }
}

/// Effective alphabet size: exp(Shannon entropy) of A/C/G/T frequencies
/// (DECIPHER `alphabetSize`, src/EnumerateSequence.c).
pub fn alphabet_size<S: AsRef<[u8]>>(seqs: &[S]) -> f64 {
    let mut d = [0f64; 4];
    for s in seqs {
        for &c in s.as_ref() {
            let b = crate::kmer::BASE_CODE[c as usize];
            if b < 4 {
                d[b as usize] += 1.0;
            }
        }
    }
    let sum: f64 = d.iter().sum();
    let mut h = 0.0;
    for &x in &d {
        let p = x / sum;
        if p > 0.0 {
            h -= p * p.ln();
        }
    }
    h.exp()
}

/// DECIPHER's automatic K: `floor(log(N * quantile(width, 0.99), alphabetSize))`,
/// clamped to [1, 13]. Returns (K, clamped?).
pub fn auto_k<S: AsRef<[u8]>>(seqs: &[S], n: f64) -> (u32, bool) {
    let widths: Vec<f64> = seqs.iter().map(|s| s.as_ref().len() as f64).collect();
    let quant = quantile_type7(&widths, 0.99);
    let a = alphabet_size(seqs);
    let k = ((n * quant).ln() / a.ln()).floor();
    if k < 1.0 {
        (1, false)
    } else if k > 13.0 {
        (13, true)
    } else {
        (k as u32, false)
    }
}

/// Training result summary (besides the model).
#[derive(Clone, Debug)]
pub struct TrainReport {
    pub k: u32,
    pub k_clamped: bool,
    pub iterations_run: u32,
    pub n_problem_sequences: usize,
    pub n_problem_groups: usize,
    pub dropped_by_group_cap: usize,
}

/// Sparse k-mer profile: sorted (kmer, frequency).
type Profile = Vec<(u32, f64)>;

fn leaf_profile(seq_ids: &[usize], kmers: &[Vec<u32>]) -> Profile {
    let mut all: Vec<u32> = Vec::with_capacity(seq_ids.iter().map(|&i| kmers[i].len()).sum());
    for &i in seq_ids {
        all.extend_from_slice(&kmers[i]);
    }
    all.sort_unstable();
    let total = all.len() as f64;
    let mut out: Profile = Vec::new();
    let mut i = 0;
    while i < all.len() {
        let mut j = i;
        while j < all.len() && all[j] == all[i] {
            j += 1;
        }
        out.push((all[i], (j - i) as f64 / total));
        i = j;
    }
    out
}

/// Per-child ordering iterator reproducing
/// `order(-profile * log(q), decreasing = TRUE)` with R's stable ties and
/// NaN (q = 0) last.
struct OrderIter<'a> {
    positive: Vec<(f64, u32)>,
    pos_i: usize,
    positive_set: HashSet<u32>,
    q_support: &'a [u32],
    zero_i: usize,
    nan_next: u64,
    n_kmers: u64,
}

impl<'a> OrderIter<'a> {
    fn new(profile: &Profile, q: &'a Profile, q_support: &'a [u32], n_kmers: u64) -> Self {
        let mut positive: Vec<(f64, u32)> = Vec::with_capacity(profile.len());
        let mut qi = 0usize;
        for &(k, p) in profile {
            while q[qi].0 < k {
                qi += 1;
            }
            let h = -p * q[qi].1.ln();
            if h > 0.0 {
                positive.push((h, k));
            }
        }
        // decreasing H; ties by ascending k-mer index (stable order)
        positive.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap().then(a.1.cmp(&b.1)));
        let positive_set = positive.iter().map(|x| x.1).collect();
        Self {
            positive,
            pos_i: 0,
            positive_set,
            q_support,
            zero_i: 0,
            nan_next: 0,
            n_kmers,
        }
    }

    fn next_kmer(&mut self) -> Option<u32> {
        if self.pos_i < self.positive.len() {
            self.pos_i += 1;
            return Some(self.positive[self.pos_i - 1].1);
        }
        while self.zero_i < self.q_support.len() {
            let k = self.q_support[self.zero_i];
            self.zero_i += 1;
            if !self.positive_set.contains(&k) {
                return Some(k);
            }
        }
        while self.nan_next < self.n_kmers {
            let k = self.nan_next as u32;
            self.nan_next += 1;
            if self.q_support.binary_search(&k).is_err() {
                return Some(k);
            }
        }
        None
    }
}

struct TreeBuilder<'a> {
    tree: &'a TaxonomyTree,
    kmers: &'a [Vec<u32>],
    subtree_seqs: &'a [Vec<usize>],
    max_children: usize,
    n_kmers: u64,
    decision: Vec<Option<(Vec<u32>, Vec<f64>)>>,
}

impl TreeBuilder<'_> {
    /// `.createTree(k)`: returns (profile, descendants)
    fn create(&mut self, k: usize) -> (Profile, f64) {
        let ch = self.tree.children[k].clone();
        let l = ch.len();
        if l > 0 && l <= self.max_children {
            let mut profiles = Vec::with_capacity(l);
            let mut desc = Vec::with_capacity(l);
            for &c in &ch {
                let (p, d) = self.create(c as usize);
                profiles.push(p);
                desc.push(d);
            }
            let dsum: f64 = desc.iter().sum();
            // q = colSums(t(profile) * descendants) / sum(descendants)
            let mut acc: HashMap<u32, f64> = HashMap::new();
            let mut support: Vec<u32> = Vec::new();
            for (j, p) in profiles.iter().enumerate() {
                for &(km, v) in p {
                    let e = acc.entry(km).or_insert_with(|| {
                        support.push(km);
                        0.0
                    });
                    *e += v * desc[j];
                }
            }
            support.sort_unstable();
            let q: Profile = support.iter().map(|&km| (km, acc[&km] / dsum)).collect();
            let q_support: Vec<u32> = q.iter().map(|x| x.0).collect();

            let max_nnz = profiles
                .iter()
                .map(|p| p.iter().filter(|x| x.1 > 0.0).count())
                .max()
                .unwrap_or(0);
            let record_kmers = (max_nnz as f64) * 0.10;
            let mut iters: Vec<OrderIter> = profiles
                .iter()
                .map(|p| OrderIter::new(p, &q, &q_support, self.n_kmers))
                .collect();
            let mut keep: HashSet<u32> = HashSet::new();
            let mut count = 0usize;
            'outer: while (count as f64) < record_kmers {
                for it in iters.iter_mut() {
                    match it.next_kmer() {
                        Some(km) => {
                            if keep.insert(km) {
                                count += 1;
                                if (count as f64) >= record_kmers {
                                    break 'outer;
                                }
                            }
                        }
                        None => break 'outer,
                    }
                }
            }
            drop(iters);
            let mut keep: Vec<u32> = keep.into_iter().collect();
            keep.sort_unstable();
            let nk = keep.len();
            let mut w = vec![0.0f64; l * nk];
            for (j, p) in profiles.iter().enumerate() {
                for (t, km) in keep.iter().enumerate() {
                    if let Ok(pos) = p.binary_search_by_key(km, |x| x.0) {
                        w[j * nk + t] = p[pos].1;
                    }
                }
            }
            self.decision[k] = Some((keep, w));
            (q, dsum)
        } else {
            (leaf_profile(&self.subtree_seqs[k], self.kmers), 1.0)
        }
    }
}

/// Train a model from sequences and raw lineage strings.
pub fn train<S: AsRef<[u8]> + Sync>(
    seqs: &[S],
    lineages: &[String],
    names: &[String],
    opts: &TrainOptions,
) -> Result<(Model, TrainReport)> {
    let n_in = seqs.len();
    if n_in < 2 {
        return Err(Error::InvalidParameter(
            "at least two training sequences are required".into(),
        ));
    }
    if lineages.len() != n_in || names.len() != n_in {
        return Err(Error::InvalidParameter(
            "sequences, lineages and names must have equal length".into(),
        ));
    }
    if !(opts.min_fraction > 0.0)
        || opts.max_fraction <= opts.min_fraction
        || opts.max_fraction >= 1.0
    {
        return Err(Error::InvalidParameter(
            "require 0 < minFraction < maxFraction < 1".into(),
        ));
    }
    if opts.max_iterations < 1 || opts.max_children < 1 || !(opts.multiplier > 0.0) {
        return Err(Error::InvalidParameter(
            "maxIterations, maxChildren and multiplier must be positive".into(),
        ));
    }

    // optional group-size cap (Phase 6.7, balanced mode)
    let mut selected: Vec<usize> = (0..n_in).collect();
    let mut dropped = 0;
    if let Some(cap) = opts.max_group_size {
        let tmp_tree = TaxonomyTree::build(lineages)?;
        let pre = KmerEncoder::from_model(
            opts.k.unwrap_or_else(|| auto_k(seqs, opts.n).0) as usize,
            opts.seed_pattern.as_deref(),
        )?;
        let km: Vec<Vec<u32>> = seqs
            .par_iter()
            .map(|s| pre.encode(s.as_ref()).unique)
            .collect();
        selected = crate::balance::cap_groups(&tmp_tree.cross_index, &km, cap as usize);
        dropped = n_in - selected.len();
    }
    let seqs_sel: Vec<&[u8]> = selected.iter().map(|&i| seqs[i].as_ref()).collect();
    let lin_sel: Vec<String> = selected.iter().map(|&i| lineages[i].clone()).collect();
    let names_sel: Vec<String> = selected.iter().map(|&i| names[i].clone()).collect();
    let l = seqs_sel.len();

    let tree = TaxonomyTree::build(&lin_sel)?;
    let (k, k_clamped, k_auto) = match (&opts.seed_pattern, opts.k) {
        (Some(p), _) => (KmerEncoder::spaced(p)?.k() as u32, false, false),
        (None, Some(k)) => {
            if !(1..=15).contains(&k) {
                return Err(Error::InvalidParameter(
                    "K can be at most 15 (and at least 1)".into(),
                ));
            }
            (k, false, false)
        }
        (None, None) => {
            let (k, c) = auto_k(&seqs_sel, opts.n);
            (k, c, true)
        }
    };
    let encoder = KmerEncoder::from_model(k as usize, opts.seed_pattern.as_deref())?;
    let n_kmers = encoder.n_kmers();
    if opts.verbose {
        eprintln!(
            "[train] {l} sequences, {} taxonomy nodes, K = {k}",
            tree.len()
        );
    }

    let kmers: Vec<Vec<u32>> = seqs_sel
        .par_iter()
        .map(|s| encoder.encode(s).unique)
        .collect();
    if let Some(i) = kmers.iter().position(|k| k.is_empty()) {
        return Err(Error::InvalidParameter(format!(
            "all training sequences must have at least one k-mer (sequence '{}' has none)",
            names_sel[i]
        )));
    }

    // sequences under each node (DECIPHER `sequences`), ascending index
    let n_nodes = tree.len();
    let mut subtree_seqs: Vec<Vec<usize>> = vec![Vec::new(); n_nodes];
    for (i, &c) in tree.cross_index.iter().enumerate() {
        let mut p = c;
        while p != NO_PARENT {
            subtree_seqs[p as usize].push(i);
            p = tree.parents[p as usize];
        }
    }
    let n_seqs_node: Vec<usize> = subtree_seqs.iter().map(|v| v.len()).collect();

    // decision k-mers
    let mut tb = TreeBuilder {
        tree: &tree,
        kmers: &kmers,
        subtree_seqs: &subtree_seqs,
        max_children: opts.max_children as usize,
        n_kmers,
        decision: vec![None; n_nodes],
    };
    tb.create(0);
    let decision = tb.decision;
    if opts.verbose {
        eprintln!("[train] decision k-mers computed");
    }

    // ancestor test via subtree membership of each node's own path
    let in_subtree = |node: u32, seq: usize| -> bool {
        let mut p = tree.cross_index[seq];
        while p != NO_PARENT {
            if p == node {
                return true;
            }
            p = tree.parents[p as usize];
        }
        false
    };

    // learn the sampling fraction
    let mut fraction = vec![opts.max_fraction; n_nodes];
    // Some(true) = incorrect, Some(false) = correct, None = NA (given up)
    let mut incorrect: Vec<Option<bool>> = vec![Some(true); l];
    let mut predicted: Vec<String> = vec![String::new(); l];
    let delta = (opts.max_fraction - opts.min_fraction) * opts.multiplier;
    let b = 100usize;
    let mut iterations_run = 0;
    for it in 0..opts.max_iterations {
        let remaining: Vec<usize> = (0..l).filter(|&i| incorrect[i] == Some(true)).collect();
        if remaining.is_empty() {
            break;
        }
        iterations_run = it + 1;
        for block in remaining.chunks(opts.block_size.max(1)) {
            // classify the block in parallel with the current fractions
            let outcomes: Vec<(usize, Option<(u32, u32)>)> = block
                .par_iter()
                .map(|&i| {
                    let mut h = Xxh3::new();
                    h.update(&(i as u64).to_le_bytes());
                    h.update(&(it as u64).to_le_bytes());
                    h.update(&opts.seed.to_le_bytes());
                    let mut rng = Xoshiro256PlusPlus::seed_from_u64(h.digest());
                    (
                        i,
                        descend_train(
                            &tree,
                            &decision,
                            &fraction,
                            &kmers[i],
                            i,
                            opts.min_fraction,
                            b,
                            &mut rng,
                            &in_subtree,
                        ),
                    )
                })
                .collect();
            // apply updates in sequence order
            for (i, out) in outcomes {
                match out {
                    None => incorrect[i] = Some(false),
                    Some((k_node, wrong_child)) => {
                        predicted[i] = tree.taxonomy[wrong_child as usize].clone();
                        let kn = k_node as usize;
                        if fraction[kn].is_nan() {
                            incorrect[i] = None;
                        } else {
                            fraction[kn] -= delta / n_seqs_node[kn] as f64;
                            if fraction[kn] <= opts.min_fraction {
                                incorrect[i] = None;
                                fraction[kn] = f64::NAN;
                            }
                        }
                    }
                }
            }
        }
        if opts.verbose {
            let n_bad = incorrect.iter().filter(|x| **x != Some(false)).count();
            eprintln!(
                "[train] iteration {}: {} sequences not yet correct",
                it + 1,
                n_bad
            );
        }
    }

    let problem_sequences: Vec<ProblemSequence> = (0..l)
        .filter(|&i| incorrect[i] != Some(false))
        .map(|i| ProblemSequence {
            index: selected[i] as u32,
            expected: format!("Root;{}", tree.classes[i]),
            predicted: predicted[i].clone(),
        })
        .collect();
    let problem_groups: Vec<String> = (0..n_nodes)
        .filter(|&i| fraction[i].is_nan())
        .map(|i| tree.taxonomy[i].clone())
        .collect();

    let ranks = opts.rank_names.as_ref().map(|names| {
        tree.levels
            .iter()
            .map(|&lv| {
                if lv == 1 {
                    "rootrank".to_string()
                } else {
                    names
                        .get(lv as usize - 2)
                        .cloned()
                        .unwrap_or_else(|| format!("level{}", lv - 1))
                }
            })
            .collect()
    });

    // reference checksum over (lineage, sequence) pairs in order
    let mut h = Xxh3::new();
    for i in 0..l {
        h.update(lin_sel[i].as_bytes());
        h.update(b"\t");
        h.update(seqs_sel[i]);
        h.update(b"\n");
    }
    let report = TrainReport {
        k,
        k_clamped,
        iterations_run,
        n_problem_sequences: problem_sequences.len(),
        n_problem_groups: problem_groups.len(),
        dropped_by_group_cap: dropped,
    };
    let raw = RawModel {
        k,
        seed_pattern: encoder.pattern(),
        taxa: tree.taxa.clone(),
        parents: tree.parents.clone(),
        children: tree.children.clone(),
        ranks,
        fraction,
        seq_kmers: kmers,
        seq_names: names_sel,
        cross_index: tree.cross_index.clone(),
        decision,
        idf_dense: None,
        problem_sequences,
        problem_groups,
        train_params: TrainParams {
            source: "idtaxa-rs train".into(),
            n: opts.n,
            min_fraction: opts.min_fraction,
            max_fraction: opts.max_fraction,
            max_iterations: opts.max_iterations,
            multiplier: opts.multiplier,
            max_children: opts.max_children,
            k_auto,
            max_group_size: opts.max_group_size,
            seed: Some(opts.seed),
        },
        reference_checksum: format!("{:032x}", h.digest128()),
    };
    let model = Model::from_raw(raw)?;
    Ok((model, report))
}

/// LearnTaxa's per-sequence descent. Returns `None` if the sequence is
/// classified correctly (or stops early), else `(node k, wrongly chosen
/// child)`.
#[allow(clippy::too_many_arguments)]
fn descend_train(
    tree: &TaxonomyTree,
    decision: &[Option<(Vec<u32>, Vec<f64>)>],
    fraction: &[f64],
    kmers: &[u32],
    seq: usize,
    min_fraction: f64,
    b: usize,
    rng: &mut Xoshiro256PlusPlus,
    in_subtree: &dyn Fn(u32, usize) -> bool,
) -> Option<(u32, u32)> {
    let mut k = 0usize;
    let mut samp: Vec<u32> = Vec::new();
    loop {
        let subtrees = &tree.children[k];
        let Some((dk, w)) = &decision[k] else {
            return None; // no decision k-mers
        };
        let n = dk.len();
        if n == 0 {
            return None;
        }
        let chosen;
        if subtrees.len() > 1 {
            let f = if fraction[k].is_nan() {
                min_fraction
            } else {
                fraction[k]
            };
            let s = (n as f64 * f).ceil() as usize;
            let mut matches = vec![false; n];
            let mut j = 0usize;
            for (i, &d) in dk.iter().enumerate() {
                j += kmers[j..].partition_point(|&x| x < d);
                if j < kmers.len() && kmers[j] == d {
                    matches[i] = true;
                }
            }
            let nc = subtrees.len();
            let mut hits = vec![0u32; nc];
            let mut vals = vec![0.0f64; nc];
            for _ in 0..b {
                samp.clear();
                for _ in 0..s {
                    samp.push(rng.gen_range(0..n as u32));
                }
                let mut maxv = f64::NEG_INFINITY;
                for c in 0..nc {
                    let row = &w[c * n..(c + 1) * n];
                    let (mut cur, mut tot) = (0.0, 0.0);
                    for &idx in &samp {
                        let wv = row[idx as usize];
                        tot += wv;
                        if matches[idx as usize] {
                            cur += wv;
                        }
                    }
                    vals[c] = if tot > 0.0 { cur / tot } else { 0.0 };
                    if vals[c] > maxv {
                        maxv = vals[c];
                    }
                }
                if maxv > 0.0 {
                    for c in 0..nc {
                        if vals[c] == maxv {
                            hits[c] += 1;
                        }
                    }
                }
            }
            // which.max: first maximum
            let mut wi = 0;
            for c in 1..nc {
                if hits[c] > hits[wi] {
                    wi = c;
                }
            }
            if (hits[wi] as f64) < b as f64 * 0.8 {
                return None; // require 80% confidence to descend
            }
            chosen = subtrees[wi];
        } else {
            chosen = subtrees[0];
        }
        if !in_subtree(chosen, seq) {
            return Some((k as u32, chosen));
        }
        if tree.children[chosen as usize].is_empty() {
            return None;
        }
        k = chosen as usize;
    }
}
