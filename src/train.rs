//! Training an IDTAXA classifier.
//!
//! Steps:
//! 1. Build the taxonomy from reference lineages; de-replicate identical
//!    sequences with identical lineages; optionally cap sequences per taxon.
//! 2. Compute unique k-mers per sequence and inverse-frequency k-mer weights
//!    (each sequence weighted by 1 / size of its group so that heavily
//!    sampled taxa do not dominate).
//! 3. For every node with ≥2 children select *decision k-mers* — k-mers whose
//!    frequency in one child most exceeds their mean frequency in the other
//!    children — chosen round-robin across children, and store naive-Bayes
//!    log-likelihoods of each decision k-mer in each child.
//! 4. *Fraction learning*: classify training sequences by tree descent and
//!    reduce the sampling fraction at nodes where more than `max_error_rate`
//!    of sequences are misdirected, repeating for misdirected sequences only. Sequences still misdirected at
//!    the end are reported as *problem sequences* (often mislabelled
//!    references — useful for curating an in-house database).

use crate::kmer;
use crate::model::{Decision, Model, ModelMeta, NodeRule};
use crate::taxonomy::{Lineage, Taxonomy, ROOT};
use anyhow::{bail, Result};
use rand::rngs::SmallRng;
use rand::seq::SliceRandom;
use rand::SeedableRng;
use rayon::prelude::*;
use std::collections::hash_map::DefaultHasher;
use std::collections::{HashMap, HashSet};
use std::hash::{Hash, Hasher};

/// A reference sequence with its lineage.
#[derive(Clone, Debug)]
pub struct TrainSeq {
    pub id: String,
    pub seq: Vec<u8>,
    pub lineage: Lineage,
}

#[derive(Clone, Debug)]
pub struct TrainParams {
    /// k-mer length (default: chosen from sequence lengths).
    pub k: Option<usize>,
    /// Target number of decision k-mers per node.
    pub decision_kmers: usize,
    /// Initial sampling fraction for tree descent.
    pub max_fraction: f32,
    /// Lower bound for the sampling fraction.
    pub min_fraction: f32,
    /// Multiplicative decrease applied to a node's fraction per error round.
    pub fraction_decay: f32,
    /// A node's fraction is only lowered when more than this proportion of
    /// the training sequences beneath it are misdirected there. Rare
    /// misdirections are reported as problem sequences instead, so a few
    /// ambiguous references cannot throttle descent for a whole subtree.
    pub max_error_rate: f32,
    pub max_iterations: usize,
    /// Bootstrap replicates used during fraction learning.
    pub train_bootstraps: usize,
    pub min_descend: f32,
    /// Keep at most this many (randomly chosen) sequences per taxon.
    pub max_per_group: Option<usize>,
    /// Minimum number of k-mers for a sequence to be used.
    pub min_kmers: usize,
    pub seed: u64,
}

impl Default for TrainParams {
    fn default() -> Self {
        TrainParams {
            k: None,
            decision_kmers: 1000,
            max_fraction: 0.20,
            min_fraction: 0.01,
            fraction_decay: 0.75,
            max_error_rate: 0.01,
            max_iterations: 10,
            train_bootstraps: 20,
            min_descend: 0.98,
            max_per_group: None,
            min_kmers: 10,
            seed: 1,
        }
    }
}

/// A training sequence misdirected by decision-k-mer tree descent during
/// fraction learning (used by `--tree-descent` classification). For
/// reference curation use [`crate::curate::find_conflicts`], which is more
/// reliable.
#[derive(Clone, Debug)]
pub struct ProblemSeq {
    /// Index of the sequence in the input vector passed to [`train`].
    pub input_index: usize,
    pub id: String,
    pub lineage: String,
    pub misdirected_at: String,
    pub went_to: String,
}

#[derive(Clone, Debug, Default)]
pub struct TrainReport {
    pub n_input: usize,
    pub n_duplicates: usize,
    pub n_subsampled_out: usize,
    pub n_too_short: usize,
    pub n_used: usize,
    pub k: usize,
    pub iterations: usize,
    pub problem_sequences: Vec<ProblemSeq>,
    /// Lineages of nodes whose sampling fraction reached the minimum.
    pub problem_groups: Vec<String>,
}

pub(crate) fn mix_seed(seed: u64, a: u64, b: u64) -> u64 {
    let mut x = seed ^ a.wrapping_mul(0x9E37_79B9_7F4A_7C15) ^ b.wrapping_mul(0xC2B2_AE3D_27D4_EB4F);
    x ^= x >> 33;
    x = x.wrapping_mul(0xFF51_AFD7_ED55_8CCD);
    x ^= x >> 33;
    x
}

/// Train a classifier.
pub fn train(seqs: Vec<TrainSeq>, params: &TrainParams, mut meta: ModelMeta) -> Result<(Model, TrainReport)> {
    let mut report = TrainReport { n_input: seqs.len(), ..Default::default() };
    if seqs.is_empty() {
        bail!("no training sequences");
    }

    // 1. taxonomy + de-replication
    let mut taxonomy = Taxonomy::new();
    let leaves: Vec<usize> = seqs.iter().map(|s| taxonomy.add_lineage(&s.lineage)).collect();
    let mut seen = HashSet::new();
    let mut keep: Vec<usize> = Vec::with_capacity(seqs.len());
    for (i, s) in seqs.iter().enumerate() {
        let mut h = DefaultHasher::new();
        s.seq.hash(&mut h);
        leaves[i].hash(&mut h);
        if seen.insert(h.finish()) {
            keep.push(i);
        }
    }
    report.n_duplicates = seqs.len() - keep.len();

    if let Some(cap) = params.max_per_group {
        let cap = cap.max(1);
        let mut by_group: HashMap<usize, Vec<usize>> = HashMap::new();
        for &i in &keep {
            by_group.entry(leaves[i]).or_default().push(i);
        }
        let mut rng = SmallRng::seed_from_u64(params.seed);
        let mut groups: Vec<_> = by_group.into_iter().collect();
        groups.sort_by_key(|g| g.0);
        let mut kept = Vec::new();
        for (_, mut members) in groups {
            if members.len() > cap {
                members.shuffle(&mut rng);
                members.truncate(cap);
            }
            kept.extend(members);
        }
        kept.sort_unstable();
        report.n_subsampled_out = keep.len() - kept.len();
        keep = kept;
    }

    // 2. k-mers
    let k = match params.k {
        Some(k) => {
            if !(kmer::MIN_K..=kmer::MAX_K).contains(&k) {
                bail!("k must be between {} and {}", kmer::MIN_K, kmer::MAX_K);
            }
            k
        }
        None => {
            let mut lens: Vec<usize> = keep.iter().map(|&i| seqs[i].seq.len()).collect();
            kmer::default_k(&mut lens)
        }
    };
    report.k = k;
    let kmers: Vec<Vec<u32>> = keep.par_iter().map(|&i| kmer::unique_kmers(&seqs[i].seq, k)).collect();

    let mut seq_ids = Vec::new();
    let mut seq_kmers = Vec::new();
    let mut seq_leaf = Vec::new();
    let mut seq_src: Vec<usize> = Vec::new();
    for (&i, km) in keep.iter().zip(kmers) {
        if km.len() < params.min_kmers {
            report.n_too_short += 1;
            continue;
        }
        seq_src.push(i);
        seq_ids.push(seqs[i].id.clone());
        seq_kmers.push(km);
        seq_leaf.push(leaves[i]);
    }
    drop(seqs);
    let n_seq = seq_kmers.len();
    if n_seq == 0 {
        bail!("no training sequences remain after filtering (sequences too short for k = {k}?)");
    }
    report.n_used = n_seq;

    // Order sequences depth-first so that every subtree is a contiguous range.
    let n_nodes = taxonomy.len();
    let mut direct: Vec<Vec<usize>> = vec![Vec::new(); n_nodes];
    for (s, &leaf) in seq_leaf.iter().enumerate() {
        direct[leaf].push(s);
    }
    let mut perm: Vec<usize> = Vec::with_capacity(n_seq);
    let mut node_range = vec![(0u32, 0u32); n_nodes];
    perm.extend(&direct[ROOT]);
    let mut stack: Vec<(usize, usize)> = vec![(ROOT, 0)];
    while let Some(&mut (n, ref mut ci)) = stack.last_mut() {
        if let Some(&c) = taxonomy.nodes[n].children.get(*ci) {
            *ci += 1;
            node_range[c].0 = perm.len() as u32;
            perm.extend(&direct[c]);
            stack.push((c, 0));
        } else {
            node_range[n].1 = perm.len() as u32;
            stack.pop();
        }
    }
    debug_assert_eq!(perm.len(), n_seq);
    let mut old_kmers = std::mem::take(&mut seq_kmers);
    let mut old_ids = std::mem::take(&mut seq_ids);
    seq_kmers = perm.iter().map(|&i| std::mem::take(&mut old_kmers[i])).collect();
    seq_ids = perm.iter().map(|&i| std::mem::take(&mut old_ids[i])).collect();
    seq_leaf = perm.iter().map(|&i| seq_leaf[i]).collect();
    seq_src = perm.iter().map(|&i| seq_src[i]).collect();
    drop((old_kmers, old_ids, direct, perm));
    let size_of = |n: usize| (node_range[n].1 - node_range[n].0) as usize;

    // weights
    let mut group_size: HashMap<usize, usize> = HashMap::new();
    for &l in &seq_leaf {
        *group_size.entry(l).or_insert(0) += 1;
    }
    let n_groups = group_size.len();
    let table = 1usize << (2 * k);
    let mut counts = vec![0f64; table];
    for (s, km) in seq_kmers.iter().enumerate() {
        let inc = 1.0 / group_size[&seq_leaf[s]] as f64;
        for &x in km {
            counts[x as usize] += inc;
        }
    }
    let g = n_groups as f64;
    let mut weights = vec![0f32; table];
    let mut max_w = 0f32;
    for (w, &c) in weights.iter_mut().zip(&counts) {
        if c > 0.0 {
            *w = (1.0 + g / c).ln() as f32;
            max_w = max_w.max(*w);
        }
    }
    for (w, &c) in weights.iter_mut().zip(&counts) {
        if c == 0.0 {
            *w = max_w;
        }
    }
    drop(counts);

    // 3. decision k-mers
    let rules: Vec<NodeRule> = (0..n_nodes)
        .into_par_iter()
        .map(|n| {
            let kids: Vec<usize> = taxonomy.nodes[n].children.iter().copied().filter(|&c| size_of(c) > 0).collect();
            match kids.len() {
                0 => NodeRule::Terminal,
                1 => NodeRule::Single(kids[0]),
                _ => NodeRule::Split(build_decision(&kids, &node_range, &seq_kmers, params.decision_kmers, k)),
            }
        })
        .collect();

    meta.k = k;
    meta.n_training_sequences = n_seq;
    meta.n_groups = n_groups;
    meta.n_taxa = n_nodes - 1;
    meta.ranks = taxonomy.ranks();
    if meta.n_input_sequences == 0 {
        meta.n_input_sequences = report.n_input;
    }
    let mut model = Model {
        meta,
        k,
        taxonomy,
        seq_ids,
        seq_kmers,
        seq_leaf,
        node_range,
        weights,
        rules,
        fraction: vec![params.max_fraction; n_nodes],
        index: Default::default(),
    };

    // 4. fraction learning
    let mut to_check: Vec<u32> = (0..n_seq as u32).collect();
    let mut wrong_final: Vec<(u32, usize, usize)> = Vec::new();
    // errors per node in the previous round, and nodes whose errors did not
    // respond to a lower fraction (e.g. identical sequences with conflicting
    // labels) — those are left alone and reported as problem sequences.
    let mut prev_errors: HashMap<usize, usize> = HashMap::new();
    let mut frozen: HashSet<usize> = HashSet::new();
    for iter in 0..=params.max_iterations {
        let m = &model;
        let wrong: Vec<(u32, usize, usize)> = to_check
            .par_iter()
            .filter_map(|&s| {
                let mut rng = SmallRng::seed_from_u64(mix_seed(params.seed, s as u64, iter as u64));
                m.check_descent(s as usize, &mut rng, params.train_bootstraps, params.min_descend)
                    .map(|(n, c)| (s, n, c))
            })
            .collect();
        report.iterations = iter + 1;
        if wrong.is_empty() || iter == params.max_iterations {
            wrong_final = wrong;
            break;
        }
        let mut per_node: HashMap<usize, usize> = HashMap::new();
        for w in &wrong {
            *per_node.entry(w.1).or_insert(0) += 1;
        }
        let mut changed = false;
        for (&n, &errors) in &per_node {
            if frozen.contains(&n) {
                continue;
            }
            if let Some(&prev) = prev_errors.get(&n) {
                if errors as f32 > 0.9 * prev as f32 {
                    frozen.insert(n);
                    continue;
                }
            }
            let size = model.members(n).len().max(1);
            if (errors as f32) / (size as f32) <= params.max_error_rate {
                continue;
            }
            let f = &mut model.fraction[n];
            let nf = (*f * params.fraction_decay).max(params.min_fraction);
            if nf < *f {
                *f = nf;
                changed = true;
            }
        }
        prev_errors = per_node;
        to_check = wrong.iter().map(|w| w.0).collect();
        if !changed {
            wrong_final = wrong;
            break;
        }
    }

    report.problem_sequences = wrong_final
        .iter()
        .map(|&(s, n, c)| ProblemSeq {
            input_index: seq_src[s as usize],
            id: model.seq_ids[s as usize].clone(),
            lineage: model.taxonomy.lineage_string(model.seq_leaf[s as usize]),
            misdirected_at: display_lineage(&model.taxonomy, n),
            went_to: display_lineage(&model.taxonomy, c),
        })
        .collect();
    report.problem_groups = (0..n_nodes)
        .filter(|&n| matches!(model.rules[n], NodeRule::Split(_)) && model.fraction[n] <= params.min_fraction)
        .map(|n| display_lineage(&model.taxonomy, n))
        .collect();
    model.index = crate::model::KmerIndex::build(&model.seq_kmers, k);
    Ok((model, report))
}

fn display_lineage(t: &Taxonomy, n: usize) -> String {
    if n == crate::taxonomy::ROOT {
        "Root".to_string()
    } else {
        t.lineage_string(n)
    }
}

/// Fraction of a child's sequences containing each k-mer, as a sorted sparse vector.
fn child_frequencies(members: std::ops::Range<usize>, seq_kmers: &[Vec<u32>], table: usize) -> Vec<(u32, f32)> {
    let mut counts = vec![0u32; table];
    for km in &seq_kmers[members.clone()] {
        for &x in km {
            counts[x as usize] += 1;
        }
    }
    let inv = 1.0 / members.len() as f32;
    counts.iter().enumerate().filter(|(_, &c)| c > 0).map(|(k, &c)| (k as u32, c as f32 * inv)).collect()
}

#[inline]
fn sparse_get(v: &[(u32, f32)], k: u32) -> f32 {
    v.binary_search_by_key(&k, |x| x.0).map(|i| v[i].1).unwrap_or(0.0)
}

fn build_decision(
    kids: &[usize],
    node_range: &[(u32, u32)],
    seq_kmers: &[Vec<u32>],
    target: usize,
    k: usize,
) -> Decision {
    let nc = kids.len();
    let table = 1usize << (2 * k);
    let freqs: Vec<Vec<(u32, f32)>> = kids
        .par_iter()
        .map(|&c| child_frequencies(node_range[c].0 as usize..node_range[c].1 as usize, seq_kmers, table))
        .collect();
    let mut total = vec![0f32; table];
    for f in &freqs {
        for &(k, v) in f {
            total[k as usize] += v;
        }
    }
    let per_child = (target / nc).max(10);
    let cap = (per_child * nc).min(50_000);
    let denom = (nc - 1) as f32;
    let ranked: Vec<Vec<u32>> = freqs
        .par_iter()
        .map(|f| {
            let mut v: Vec<(u32, f32)> =
                f.iter().map(|&(k, q)| (k, q - (total[k as usize] - q) / denom)).filter(|&(_, s)| s > 0.0).collect();
            v.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap().then(a.0.cmp(&b.0)));
            v.truncate(per_child * 2);
            v.into_iter().map(|x| x.0).collect()
        })
        .collect();
    drop(total);
    let mut chosen: HashSet<u32> = HashSet::new();
    let mut order: Vec<u32> = Vec::new();
    let mut r = 0;
    'outer: loop {
        let mut progressed = false;
        for list in &ranked {
            if let Some(&k) = list.get(r) {
                progressed = true;
                if chosen.insert(k) {
                    order.push(k);
                    if order.len() >= cap {
                        break 'outer;
                    }
                }
            }
        }
        if !progressed {
            break;
        }
        r += 1;
    }
    order.sort_unstable();
    // Naive-Bayes (RDP-style) log-likelihood of observing each k-mer in a
    // member of each child: ln((n_c + p_k) / (N_c + 1)), where p_k is the
    // pooled frequency of the k-mer. Unlike plain frequency sums this
    // penalises a child for *lacking* the query's k-mers, so small
    // homogeneous children cannot out-score large diverse ones.
    let sizes: Vec<f32> = kids.iter().map(|&c| (node_range[c].1 - node_range[c].0) as f32).collect();
    let n_total: f32 = sizes.iter().sum();
    let mut profiles = vec![0f32; order.len() * nc];
    for (i, &km) in order.iter().enumerate() {
        let row = &mut profiles[i * nc..(i + 1) * nc];
        let mut pooled = 0f32;
        for (c, f) in freqs.iter().enumerate() {
            row[c] = sparse_get(f, km) * sizes[c];
            pooled += row[c];
        }
        let p_k = (pooled + 0.5) / (n_total + 1.0);
        for (c, x) in row.iter_mut().enumerate() {
            *x = ((*x + p_k) / (sizes[c] + 1.0)).ln();
        }
    }
    Decision { children: kids.to_vec(), kmers: order, profiles }
}
