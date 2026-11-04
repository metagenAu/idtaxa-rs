//! Classification engine equivalent to DECIPHER's `IdTaxa()`
//! (R/IdTaxa.R; see `docs/algorithm-notes.md` for a line-by-line mapping).
//!
//! The engine differs from DECIPHER only in *how* the same quantities are
//! computed:
//! * the leaf phase uses the inverted index restricted to the subtree ID
//!   ranges of the selected groups instead of matching every training
//!   sequence's k-mer list;
//! * the per-sequence total score (`sumHits`) is computed from the total
//!   sample count of each query k-mer across replicates, and the
//!   per-replicate hits matrix is only materialised for the top sequence of
//!   each group (which is all DECIPHER uses from it);
//! * random numbers come from a per-query Xoshiro256++ stream seeded by
//!   `hash(sequence) ^ seed`, so results are independent of threading and
//!   batching (but not bit-identical to R's RNG).

use std::ops::Range;

use rand::Rng;
use rand::SeedableRng;
use rand_xoshiro::Xoshiro256PlusPlus;
use serde::{Deserialize, Serialize};

use crate::error::{Error, Result};
use crate::kmer::{normalize_sequence, KmerEncoder};
use crate::model::Model;
use crate::stats::{pbinom_upper, quantile_type7};
use crate::taxonomy::NO_PARENT;

/// Which strand(s) to classify.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "lowercase")]
pub enum Strand {
    /// Classify both orientations; keep the one with higher similarity.
    #[default]
    Both,
    /// Input orientation only.
    Top,
    /// Reverse complement only.
    Bottom,
}

impl std::str::FromStr for Strand {
    type Err = Error;
    fn from_str(s: &str) -> Result<Self> {
        match s.to_ascii_lowercase().as_str() {
            "both" => Ok(Strand::Both),
            "top" => Ok(Strand::Top),
            "bottom" => Ok(Strand::Bottom),
            _ => Err(Error::InvalidParameter(format!(
                "strand must be both, top or bottom (got '{s}')"
            ))),
        }
    }
}

/// Orientation used for the reported result.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum Orientation {
    #[serde(rename = "+")]
    Forward,
    #[serde(rename = "-")]
    Reverse,
    /// Not classified (too few k-mers).
    #[serde(rename = "NA")]
    None,
}

impl Orientation {
    pub fn as_str(&self) -> &'static str {
        match self {
            Orientation::Forward => "+",
            Orientation::Reverse => "-",
            Orientation::None => "NA",
        }
    }
}

/// Parameters of `IdTaxa()`; defaults are DECIPHER's.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct ClassifyParams {
    /// Confidence threshold (percent), DECIPHER `threshold = 60`.
    pub threshold: f64,
    /// Maximum bootstrap replicates, DECIPHER `bootstraps = 100`.
    pub bootstraps: u32,
    pub strand: Strand,
    /// DECIPHER `minDescend = 0.98`.
    pub min_descend: f64,
    /// Exponent of the default `samples = L^0.47`.
    pub sample_exponent: f64,
    /// DECIPHER `fullLength` as explicit (lower, upper) fold bounds on the
    /// ratio of reference to query k-mer counts; `None` = `fullLength = 0`.
    pub full_length: Option<(f64, f64)>,
    /// Global seed mixed into every per-query RNG.
    pub seed: u64,
    /// Group-size normalisation exponent for balanced scoring (Phase 6.7);
    /// 0 = standard IDTAXA.
    pub group_size_exponent: f64,
}

impl Default for ClassifyParams {
    fn default() -> Self {
        Self {
            threshold: 60.0,
            bootstraps: 100,
            strand: Strand::Both,
            min_descend: 0.98,
            sample_exponent: 0.47,
            full_length: None,
            seed: 42,
            group_size_exponent: 0.0,
        }
    }
}

impl ClassifyParams {
    /// Validate ranges as `IdTaxa()` does.
    pub fn validate(&self) -> Result<()> {
        if !(0.0..=100.0).contains(&self.threshold) {
            return Err(Error::InvalidParameter(
                "threshold must be between 0 and 100 (inclusive)".into(),
            ));
        }
        if self.bootstraps < 1 {
            return Err(Error::InvalidParameter(
                "bootstraps must be at least one".into(),
            ));
        }
        if !(0.5..=1.0).contains(&self.min_descend) {
            return Err(Error::InvalidParameter(
                "minDescend must be between 0.5 and 1 (inclusive)".into(),
            ));
        }
        if !(self.sample_exponent > 0.0 && self.sample_exponent <= 1.0) {
            return Err(Error::InvalidParameter(
                "sample exponent must be in (0, 1]".into(),
            ));
        }
        if let Some((lo, hi)) = self.full_length {
            if !(lo >= 0.0 && hi > lo) {
                return Err(Error::InvalidParameter(
                    "fullLength bounds must satisfy 0 <= lo < hi".into(),
                ));
            }
        }
        Ok(())
    }

    /// Stable string identifying all result-affecting parameters (for the
    /// persistent cache key).
    pub fn fingerprint(&self) -> String {
        format!(
            "b={};s={:?};md={};se={};fl={:?};seed={};gse={}",
            self.bootstraps,
            self.strand,
            self.min_descend,
            self.sample_exponent,
            self.full_length,
            self.seed,
            self.group_size_exponent
        )
    }
}

/// Result for one query (before thresholding).
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct Classification {
    /// Node path from Root to the selected group.
    pub lineage: Vec<u32>,
    /// Confidence (percent) per lineage node.
    pub confidences: Vec<f64>,
    pub orientation: Orientation,
    /// Number of bootstrap replicates used (DECIPHER's per-query `B`).
    pub bootstraps_used: u32,
    /// Weighted k-mer similarity of the selected group's top sequence over
    /// the bootstrap samples, divided by `B` (0..1).
    pub similarity: f64,
}

impl Classification {
    /// The "not enough k-mers" result: Root [0%]; unclassified_Root [0%].
    pub fn unclassified() -> Self {
        Self {
            lineage: vec![0],
            confidences: vec![0.0],
            orientation: Orientation::None,
            bootstraps_used: 0,
            similarity: 0.0,
        }
    }

    /// Number of lineage entries (from Root) whose confidence is at least
    /// `threshold`; DECIPHER assigns at least Root.
    pub fn n_above(&self, threshold: f64) -> usize {
        let n = self
            .confidences
            .iter()
            .take_while(|&&c| c >= threshold)
            .count();
        n.max(1)
    }

    /// Number of lineage entries passing per-depth thresholds (index = depth
    /// below Root, i.e. `thresholds[0]` applies to the first rank).
    pub fn n_above_per_rank(&self, conf: &[f64], thresholds: &[f64]) -> usize {
        let mut n = 1;
        for (d, c) in conf.iter().enumerate().skip(1) {
            let t = thresholds.get(d - 1).copied().unwrap_or(f64::INFINITY);
            if *c >= t {
                n = d + 1;
            } else {
                break;
            }
        }
        n
    }
}

/// Per-thread scratch buffers (no allocation in the hot loop once warm).
pub struct Scratch {
    acc: Vec<SeqAcc>,
    epoch: u32,
    touched: Vec<u32>,
    group_stamp: Vec<u32>,
    group_slot: Vec<u32>,
    counts: Vec<u32>,
    samp: Vec<u32>,
    qidf: Vec<f64>,
    qslot: Vec<u32>,
    matches: Vec<bool>,
    hits_desc: Vec<f64>,
    groups: Vec<GroupTop>,
    hmat: Vec<f64>,
    pair_off: Vec<u32>,
    neg_kmers: Vec<u32>,
    pair_ti: Vec<u32>,
    top_of_touched: Vec<u32>,
    shared: Vec<u64>,
    occ_mask: Vec<u64>,
    posw: Vec<u64>,
    best: Vec<f64>,
    winners: Vec<Vec<u32>>,
    ranges: Vec<Range<u32>>,
}

/// Per-training-sequence accumulator (one cache line access per posting).
#[derive(Clone, Copy, Debug, Default)]
struct SeqAcc {
    stamp: u32,
    top: u32,
    ti: u32,
    score: f64,
}

#[derive(Clone, Copy, Debug)]
struct GroupTop {
    group: u32,
    seq: u32,
    score: f64,
    orig: u32,
}

impl Scratch {
    pub fn new(model: &Model) -> Self {
        Self {
            acc: vec![SeqAcc::default(); model.n_seqs()],
            epoch: 0,
            touched: Vec::new(),
            group_stamp: vec![0; model.n_nodes()],
            group_slot: vec![0; model.n_nodes()],
            counts: Vec::new(),
            samp: Vec::new(),
            qidf: Vec::new(),
            qslot: Vec::new(),
            matches: Vec::new(),
            hits_desc: Vec::new(),
            groups: Vec::new(),
            hmat: Vec::new(),
            pair_off: Vec::new(),
            neg_kmers: Vec::new(),
            pair_ti: Vec::new(),
            top_of_touched: Vec::new(),
            shared: Vec::new(),
            occ_mask: Vec::new(),
            posw: Vec::new(),
            best: Vec::new(),
            winners: Vec::new(),
            ranges: Vec::new(),
        }
    }

    fn next_epoch(&mut self) -> u32 {
        self.epoch = self.epoch.wrapping_add(1);
        if self.epoch == 0 {
            self.acc.iter_mut().for_each(|x| x.stamp = 0);
            self.group_stamp.iter_mut().for_each(|x| *x = 0);
            self.epoch = 1;
        }
        self.epoch
    }
}

/// Output of one leaf-phase run on one orientation.
struct RunResult {
    selected: u32,
    tot_hits: Vec<(u32, f64)>,
    b: u32,
    sim: f64,
}

/// Similarity summary for novelty detection.
#[derive(Clone, Debug, Default, Serialize, Deserialize, PartialEq)]
pub struct SimilarityProfile {
    /// Max IDF-weighted containment of the query's k-mers in any reference.
    pub max_reference: f64,
    /// Max similarity within each lineage node's subtree (same order as the
    /// classification lineage).
    pub per_lineage: Vec<f64>,
}

/// IDTAXA classifier bound to a model.
pub struct Classifier<'m> {
    model: &'m Model,
    params: ClassifyParams,
    encoder: KmerEncoder,
    min_s: usize,
    /// Number of k-mers per training sequence (native order) for fullLength.
    lens: Vec<u32>,
    /// group sizes (number of training sequences per leaf group)
    group_size: Vec<u32>,
    /// use the unpruned reference path (testing)
    pub force_full: bool,
}

/// DECIPHER's minimum sample size: smallest even `minS` in 2..=100 such that
/// the probability that half of the sampled k-mers match some training
/// sequence by chance is < 1% (R/IdTaxa.R, "require a minimum sample size").
pub fn minimum_sample_size(model: &Model) -> usize {
    let ls: Vec<f64> = (0..model.n_seqs() as u32)
        .map(|i| model.seq_kmers(i).len() as f64)
        .collect();
    let l = quantile_type7(&ls, 0.9);
    let n_kmers = (1u64 << (2 * model.k())) as f64;
    let n_seqs = model.n_seqs() as u64;
    let mut min_s = 2usize;
    for m in (2..=100).step_by(2) {
        min_s = m;
        let p = pbinom_upper(m as f64 * 0.5 - 1.0, m as u64, l / n_kmers);
        let p = pbinom_upper(0.0, n_seqs, p);
        if p < 0.01 {
            break;
        }
    }
    min_s
}

/// Seed for a query's RNG stream.
pub fn query_seed(normalized_seq: &[u8], seed: u64) -> u64 {
    xxhash_rust::xxh3::xxh3_64(normalized_seq) ^ seed
}

impl<'m> Classifier<'m> {
    pub fn new(model: &'m Model, params: ClassifyParams) -> Result<Self> {
        params.validate()?;
        let lens: Vec<u32> = (0..model.n_seqs() as u32)
            .map(|i| model.seq_kmers(i).len() as u32)
            .collect();
        let mut group_size = vec![0u32; model.n_nodes()];
        for i in 0..model.n_seqs() as u32 {
            group_size[model.seq_group(i) as usize] += 1;
        }
        Ok(Self {
            model,
            encoder: model.encoder(),
            min_s: minimum_sample_size(model),
            params,
            lens,
            group_size,
            force_full: false,
        })
    }

    pub fn model(&self) -> &Model {
        self.model
    }

    pub fn params(&self) -> &ClassifyParams {
        &self.params
    }

    /// DECIPHER's minimum number of k-mers sampled per replicate.
    pub fn min_s(&self) -> usize {
        self.min_s
    }

    pub fn new_scratch(&self) -> Scratch {
        Scratch::new(self.model)
    }

    /// Classify one sequence with its own deterministic RNG stream.
    pub fn classify(&self, seq: &[u8], scratch: &mut Scratch) -> Classification {
        let norm = normalize_sequence(seq);
        let seed = query_seed(&norm, self.params.seed);
        self.classify_seeded(&norm, seed, scratch)
    }

    /// Classify with an explicit RNG seed (used for adaptive reruns).
    pub fn classify_seeded(&self, seq: &[u8], seed: u64, scratch: &mut Scratch) -> Classification {
        self.classify_with_bootstraps(seq, seed, self.params.bootstraps, scratch)
    }

    /// Classify with an explicit seed and bootstrap cap.
    pub fn classify_with_bootstraps(
        &self,
        seq: &[u8],
        seed: u64,
        bootstraps: u32,
        scratch: &mut Scratch,
    ) -> Classification {
        let mut rng = Xoshiro256PlusPlus::seed_from_u64(seed);
        let (fwd, rev) = match self.params.strand {
            Strand::Both => {
                let (f, r) = self.encoder.encode_both(seq);
                (Some(f), Some(r))
            }
            Strand::Top => (Some(self.encoder.encode(seq)), None),
            Strand::Bottom => {
                let (_, r) = self.encoder.encode_both(seq);
                (None, Some(r))
            }
        };
        let first = fwd.as_ref().or(rev.as_ref()).unwrap();
        // samples = ceiling(L^0.47) with L = number of non-NA k-mers
        let mut s = (first.n_valid as f64)
            .powf(self.params.sample_exponent)
            .ceil() as usize;
        if s < self.min_s {
            s = self.min_s;
        }
        let n_unique = first.unique.len();
        if n_unique <= s {
            return Classification::unclassified();
        }
        // B = as.integer(min(5 * length(kmers) / S, bootstraps))
        // (adaptive reruns ask for more replicates than the cap allows and
        // bypass the 5 * L / S limit)
        let b = if bootstraps > self.params.bootstraps {
            bootstraps
        } else {
            ((5.0 * n_unique as f64 / s as f64).min(bootstraps as f64) as u32).max(1)
        };

        let mut best: Option<(RunResult, Orientation)> = None;
        if let Some(f) = &fwd {
            let r = self.run(&f.unique, s, b, &mut rng, scratch);
            best = r.map(|r| (r, Orientation::Forward));
        }
        if let Some(rv) = &rev {
            let r = self.run(&rv.unique, s, b, &mut rng, scratch);
            if let Some(r) = r {
                match &best {
                    Some((f, _)) if self.params.strand == Strand::Both => {
                        // DECIPHER: keep the reverse result only if its
                        // similarity is strictly higher
                        if r.sim > f.sim {
                            best = Some((r, Orientation::Reverse));
                        }
                    }
                    _ => best = Some((r, Orientation::Reverse)),
                }
            }
        }
        let Some((res, orientation)) = best else {
            return Classification::unclassified();
        };
        self.confidences(res, orientation)
    }

    fn confidences(&self, res: RunResult, orientation: Orientation) -> Classification {
        let m = self.model;
        let lineage = m.lineage(res.selected);
        let bf = res.b as f64;
        let sel_hits = res
            .tot_hits
            .iter()
            .find(|(g, _)| *g == res.selected)
            .map(|x| x.1)
            .unwrap_or(0.0);
        let mut conf = vec![sel_hits / bf * 100.0; lineage.len()];
        for &(g, h) in &res.tot_hits {
            if g == res.selected || h <= 0.0 {
                continue;
            }
            let mut p = m.parent(g);
            while p != NO_PARENT {
                if let Some(pos) = lineage.iter().position(|&x| x == p) {
                    conf[pos] += h / bf * 100.0;
                }
                p = m.parent(p);
            }
        }
        Classification {
            lineage,
            confidences: conf,
            orientation,
            bootstraps_used: res.b,
            similarity: res.sim / bf,
        }
    }

    #[inline(never)]
    /// Tree descent: returns the children (of the stopping node) whose
    /// subtrees are scored in the leaf phase.
    fn descend(
        &self,
        kmers: &[u32],
        b: u32,
        rng: &mut Xoshiro256PlusPlus,
        sc: &mut Scratch,
    ) -> Vec<u32> {
        let m = self.model;
        let mut k = 0u32;
        loop {
            let subtrees = m.children(k);
            let (dk, w) = m.decision(k);
            let n = dk.len();
            let frac = m.fraction(k);
            if n == 0 || frac.is_nan() {
                return subtrees.to_vec();
            }
            let chosen: u32;
            if subtrees.len() > 1 {
                let s = (n as f64 * frac).ceil() as usize;
                // matches = decision k-mers present in the query
                sc.matches.clear();
                sc.matches.resize(n, false);
                let mut j = 0usize;
                for (i, &d) in dk.iter().enumerate() {
                    j += kmers[j..].partition_point(|&x| x < d);
                    if j < kmers.len() && kmers[j] == d {
                        sc.matches[i] = true;
                    }
                }
                let nc = subtrees.len();
                let mut hits = vec![0u32; nc];
                sc.samp.clear();
                sc.hits_desc.clear();
                sc.hits_desc.resize(nc, 0.0);
                for _ in 0..b {
                    sc.samp.clear();
                    for _ in 0..s {
                        sc.samp.push(rng.gen_range(0..n as u32));
                    }
                    let mut maxv = f64::NEG_INFINITY;
                    for c in 0..nc {
                        let row = &w[c * n..(c + 1) * n];
                        let mut cur = 0.0;
                        let mut tot = 0.0;
                        for &idx in &sc.samp {
                            let wv = row[idx as usize];
                            tot += wv;
                            if sc.matches[idx as usize] {
                                cur += wv;
                            }
                        }
                        let v = if tot > 0.0 { cur / tot } else { 0.0 };
                        sc.hits_desc[c] = v;
                        if v > maxv {
                            maxv = v;
                        }
                    }
                    if maxv > 0.0 {
                        for c in 0..nc {
                            if sc.hits_desc[c] == maxv {
                                hits[c] += 1;
                            }
                        }
                    }
                }
                let bf = b as f64;
                let strong: Vec<usize> = (0..nc)
                    .filter(|&c| hits[c] as f64 >= self.params.min_descend * bf)
                    .collect();
                if strong.len() != 1 {
                    let half = (0..nc).any(|c| hits[c] as f64 >= bf * 0.5);
                    if !half {
                        return subtrees.to_vec();
                    }
                    let any: Vec<u32> = (0..nc)
                        .filter(|&c| hits[c] > 0)
                        .map(|c| subtrees[c])
                        .collect();
                    if any.is_empty() {
                        return subtrees.to_vec();
                    }
                    return any;
                }
                chosen = subtrees[strong[0]];
            } else {
                chosen = subtrees[0];
            }
            if m.children(chosen).is_empty() {
                return vec![chosen];
            }
            k = chosen;
        }
    }

    /// Descent + leaf phase for one orientation. `None` if no reference
    /// sequences remain (fullLength filter).
    fn run(
        &self,
        kmers: &[u32],
        s: usize,
        b: u32,
        rng: &mut Xoshiro256PlusPlus,
        sc: &mut Scratch,
    ) -> Option<RunResult> {
        let m = self.model;
        let keep_nodes = self.descend(kmers, b, rng, sc);
        sc.ranges.clear();
        for &c in &keep_nodes {
            let r = m.subtree(c);
            if !r.is_empty() {
                sc.ranges.push(r);
            }
        }
        sc.ranges.sort_by_key(|r| r.start);
        let nq = kmers.len();
        let (lo_len, hi_len) = match self.params.full_length {
            Some((lo, hi)) => (lo * nq as f64, hi * nq as f64),
            None => (f64::NEG_INFINITY, f64::INFINITY),
        };
        let filter_len = self.params.full_length.is_some();
        if filter_len {
            let any = sc.ranges.iter().any(|r| {
                r.clone().any(|i| {
                    let l = self.lens[i as usize] as f64;
                    l >= lo_len && l <= hi_len
                })
            });
            if !any {
                return None;
            }
        }

        // per-query k-mer IDF and index slots
        sc.qidf.clear();
        sc.qslot.clear();
        for &x in kmers {
            match m.slot(x) {
                Some(slot) => {
                    sc.qslot.push(slot as u32);
                    sc.qidf.push(m.slot_idf(slot));
                }
                None => {
                    sc.qslot.push(u32::MAX);
                    sc.qidf.push(m.meta.idf_default);
                }
            }
        }

        // bootstrap sampling (with replacement) of query k-mers
        sc.counts.clear();
        sc.counts.resize(nq, 0);
        sc.samp.clear();
        let total = s * b as usize;
        let mut wsum = 0.0;
        for _ in 0..total {
            let i = rng.gen_range(0..nq as u32);
            sc.samp.push(i);
            sc.counts[i as usize] += 1;
            wsum += sc.qidf[i as usize];
        }
        let davg = wsum / b as f64;

        // sumHits over the kept ranges via the inverted index. K-mers stored
        // as complement lists (present in > half of the training set) add
        // their weight to every kept sequence and subtract it again for the
        // sequences lacking them.
        let epoch = sc.next_epoch();
        sc.touched.clear();
        sc.pair_off.clear();
        sc.pair_ti.clear();
        sc.pair_off.push(0);
        sc.neg_kmers.clear();
        let mut base = 0.0;
        for i in 0..nq {
            if sc.counts[i] > 0
                && sc.qslot[i] != u32::MAX
                && m.slot_is_complement(sc.qslot[i] as usize)
            {
                sc.neg_kmers.push(i as u32);
                base += sc.counts[i] as f64 * sc.qidf[i];
            }
        }
        let in_len = |id: usize| -> bool {
            if !filter_len {
                return true;
            }
            let l = self.lens[id] as f64;
            l >= lo_len && l <= hi_len
        };
        if !sc.neg_kmers.is_empty() {
            for r in &sc.ranges {
                for id in r.clone() {
                    let iu = id as usize;
                    if !in_len(iu) {
                        continue;
                    }
                    let acc = &mut sc.acc[iu];
                    acc.stamp = epoch;
                    acc.score = base;
                    acc.ti = sc.touched.len() as u32;
                    sc.touched.push(id);
                }
            }
        }
        let whole = |r: &Range<u32>| r.start == 0 && r.end as usize == m.n_seqs();
        for i in 0..nq {
            let c = sc.counts[i];
            if c == 0 || sc.qslot[i] == u32::MAX {
                sc.pair_off.push(sc.pair_ti.len() as u32);
                continue;
            }
            let slot = sc.qslot[i] as usize;
            let add = c as f64 * sc.qidf[i];
            let post = m.postings(slot);
            let neg = m.slot_is_complement(slot);
            for r in &sc.ranges {
                let (a, z) = if whole(r) {
                    (0, post.len())
                } else {
                    let a = post.partition_point(|&x| x < r.start);
                    (a, a + post[a..].partition_point(|&x| x < r.end))
                };
                for &id in &post[a..z] {
                    let id_u = id as usize;
                    let acc = &mut sc.acc[id_u];
                    if neg {
                        if acc.stamp == epoch {
                            acc.score -= add;
                        }
                        continue;
                    }
                    if acc.stamp != epoch {
                        if !in_len(id_u) {
                            continue;
                        }
                        acc.stamp = epoch;
                        acc.score = 0.0;
                        acc.ti = sc.touched.len() as u32;
                        sc.touched.push(id);
                    }
                    acc.score += add;
                    sc.pair_ti.push(acc.ti);
                }
            }
            sc.pair_off.push(sc.pair_ti.len() as u32);
        }

        // top sequence per group: max score, ties -> lowest original index
        sc.groups.clear();
        for &id in &sc.touched {
            let sco = sc.acc[id as usize].score;
            if sco == 0.0 {
                // shares no sampled k-mer (complement lists touch every
                // kept sequence); can never receive a replicate
                continue;
            }
            let g = m.seq_group(id);
            let orig = m.seq_orig(id);
            if sc.group_stamp[g as usize] != epoch {
                sc.group_stamp[g as usize] = epoch;
                sc.group_slot[g as usize] = sc.groups.len() as u32;
                sc.groups.push(GroupTop {
                    group: g,
                    seq: id,
                    score: sco,
                    orig,
                });
            } else {
                let gt = &mut sc.groups[sc.group_slot[g as usize] as usize];
                if sco > gt.score || (sco == gt.score && orig < gt.orig) {
                    gt.seq = id;
                    gt.score = sco;
                    gt.orig = orig;
                }
            }
        }
        // DECIPHER orders groups by taxonomy index
        sc.groups.sort_by_key(|g| g.group);
        let ng = sc.groups.len();

        if ng == 0 || sc.groups.iter().all(|g| g.score <= 0.0) {
            // no hits at all: DECIPHER samples a random group among all
            // groups of the kept sequences; all confidences are zero.
            let mut all: Vec<u32> = Vec::new();
            for r in &sc.ranges {
                for i in r.clone() {
                    if filter_len {
                        let l = self.lens[i as usize] as f64;
                        if l < lo_len || l > hi_len {
                            continue;
                        }
                    }
                    all.push(m.seq_group(i));
                }
            }
            all.sort_unstable();
            all.dedup();
            let sel = all[rng.gen_range(0..all.len())];
            return Some(RunResult {
                selected: sel,
                tot_hits: Vec::new(),
                b,
                sim: 0.0,
            });
        }

        let tot = if self.params.group_size_exponent == 0.0 && !self.force_full {
            self.winners_pruned(kmers, s, b, ng, epoch, davg, rng, sc)
        } else {
            self.winners_full(kmers, s, b, ng, epoch, davg, rng, sc)
        };
        let maxt = tot.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let ties: Vec<usize> = (0..ng).filter(|&g| tot[g] == maxt).collect();
        let sel = if ties.len() > 1 {
            ties[rng.gen_range(0..ties.len())]
        } else {
            ties[0]
        };
        // sims = sum_b hits[b, top_selected] / davg = sumHits(top) / davg
        let sim = sc.groups[sel].score / davg;
        let tot_hits = (0..ng)
            .filter(|&g| tot[g] > 0.0 || g == sel)
            .map(|g| (sc.groups[g].group, tot[g]))
            .collect();
        Some(RunResult {
            selected: sc.groups[sel].group,
            tot_hits,
            b,
            sim,
        })
    }

    /// Pick the per-replicate winner among tied groups (R's `max.col`
    /// with `ties.method = "random"`) and credit it with `best / davg`.
    fn credit(
        best: f64,
        winners: &mut [u32],
        davg: f64,
        tot: &mut [f64],
        rng: &mut Xoshiro256PlusPlus,
    ) {
        if best <= 0.0 || winners.is_empty() {
            return;
        }
        let pick = if winners.len() > 1 {
            winners.sort_unstable();
            winners[rng.gen_range(0..winners.len())]
        } else {
            winners[0]
        };
        tot[pick as usize] += best / davg;
    }

    /// Reference implementation: full hits matrix of every group's top
    /// sequence (also used for group-size-normalised scoring).
    #[allow(clippy::too_many_arguments)]
    fn winners_full(
        &self,
        kmers: &[u32],
        s: usize,
        b: u32,
        ng: usize,
        epoch: u32,
        davg: f64,
        rng: &mut Xoshiro256PlusPlus,
        sc: &mut Scratch,
    ) -> Vec<f64> {
        let m = self.model;
        let nq = kmers.len();
        let bu = b as usize;
        let _ = epoch;
        sc.hmat.clear();
        sc.hmat.resize(ng * bu, 0.0);
        let mut member = vec![false; nq];
        for gi in 0..ng {
            member.iter_mut().for_each(|x| *x = false);
            let tk = m.seq_kmers(sc.groups[gi].seq);
            let (mut x, mut y) = (0usize, 0usize);
            while x < tk.len() && y < nq {
                match tk[x].cmp(&kmers[y]) {
                    std::cmp::Ordering::Less => x += 1,
                    std::cmp::Ordering::Greater => y += 1,
                    std::cmp::Ordering::Equal => {
                        member[y] = true;
                        x += 1;
                        y += 1;
                    }
                }
            }
            for rep in 0..bu {
                let mut h = 0.0;
                for &qi in &sc.samp[rep * s..(rep + 1) * s] {
                    if member[qi as usize] {
                        h += sc.qidf[qi as usize];
                    }
                }
                sc.hmat[gi * bu + rep] = h;
            }
        }
        let gexp = self.params.group_size_exponent;
        if gexp != 0.0 {
            for (gi, g) in sc.groups.iter().enumerate() {
                let scale = (self.group_size[g.group as usize].max(1) as f64).powf(-gexp);
                sc.hmat[gi * bu..(gi + 1) * bu]
                    .iter_mut()
                    .for_each(|x| *x *= scale);
            }
        }
        let mut tot = vec![0.0f64; ng];
        let mut winners: Vec<u32> = Vec::new();
        for rep in 0..bu {
            let mut best = f64::NEG_INFINITY;
            winners.clear();
            for gi in 0..ng {
                let v = sc.hmat[gi * bu + rep];
                if v > best {
                    best = v;
                    winners.clear();
                    winners.push(gi as u32);
                } else if v == best {
                    winners.push(gi as u32);
                }
            }
            Self::credit(best, &mut winners, davg, &mut tot, rng);
        }
        tot
    }

    /// Exact per-replicate winners with dominance pruning against the
    /// leading group `L` (maximum total score). For a group `g`, let `gain`
    /// be the query k-mers that `g`'s top sequence shares and `L`'s lacks
    /// (plus k-mers `L` has with non-positive weight) and `loss` the
    /// positive-weight k-mers `L` has and `g` lacks. In a replicate where no
    /// `gain` k-mer was sampled but some `loss` k-mer was, `g` scores
    /// strictly less than `L`, so it can neither win nor tie; only the
    /// remaining replicates are scored exactly.
    #[allow(clippy::too_many_arguments)]
    #[inline(never)]
    fn winners_pruned(
        &self,
        _kmers: &[u32],
        s: usize,
        b: u32,
        ng: usize,
        epoch: u32,
        davg: f64,
        rng: &mut Xoshiro256PlusPlus,
        sc: &mut Scratch,
    ) -> Vec<f64> {
        let m = self.model;
        let nq = sc.counts.len();
        let bu = b as usize;
        let qw = nq.div_ceil(64);
        let bw = bu.div_ceil(64);
        // shared-k-mer bitsets of each group's top sequence
        for &id in &sc.touched {
            sc.acc[id as usize].top = u32::MAX;
        }
        for (gi, g) in sc.groups.iter().enumerate() {
            sc.acc[g.seq as usize].top = gi as u32;
        }
        sc.shared.clear();
        sc.shared.resize(ng * qw, 0);
        sc.top_of_touched.clear();
        sc.top_of_touched.resize(sc.touched.len(), u32::MAX);
        for (gi, g) in sc.groups.iter().enumerate() {
            sc.top_of_touched[sc.acc[g.seq as usize].ti as usize] = gi as u32;
        }
        for i in 0..nq {
            let (word, bit) = (i / 64, 1u64 << (i % 64));
            for &ti in &sc.pair_ti[sc.pair_off[i] as usize..sc.pair_off[i + 1] as usize] {
                let gi = sc.top_of_touched[ti as usize];
                if gi != u32::MAX {
                    sc.shared[gi as usize * qw + word] |= bit;
                }
            }
        }
        // complement-stored k-mers: every top has them except those listed
        if !sc.neg_kmers.is_empty() {
            let mut mask = vec![0u64; qw];
            for &i in &sc.neg_kmers {
                mask[i as usize / 64] |= 1u64 << (i % 64);
            }
            for gi in 0..ng {
                for w in 0..qw {
                    sc.shared[gi * qw + w] |= mask[w];
                }
            }
            for &i in &sc.neg_kmers {
                let i = i as usize;
                let (word, bit) = (i / 64, 1u64 << (i % 64));
                let post = m.postings(sc.qslot[i] as usize);
                for r in &sc.ranges {
                    let a = post.partition_point(|&x| x < r.start);
                    let z = a + post[a..].partition_point(|&x| x < r.end);
                    for &id in &post[a..z] {
                        let acc = sc.acc[id as usize];
                        if acc.stamp == epoch {
                            let gi = sc.top_of_touched[acc.ti as usize];
                            if gi != u32::MAX {
                                sc.shared[gi as usize * qw + word] &= !bit;
                            }
                        }
                    }
                }
            }
        }
        // replicate masks per query k-mer, positive-weight mask
        sc.occ_mask.clear();
        sc.occ_mask.resize(nq * bw, 0);
        for (t, &qi) in sc.samp.iter().enumerate() {
            let rep = t / s;
            sc.occ_mask[qi as usize * bw + rep / 64] |= 1u64 << (rep % 64);
        }
        sc.posw.clear();
        sc.posw.resize(qw, 0);
        for i in 0..nq {
            if sc.qidf[i] > 0.0 {
                sc.posw[i / 64] |= 1u64 << (i % 64);
            }
        }
        let full_last = if bu.is_multiple_of(64) {
            u64::MAX
        } else {
            (1u64 << (bu % 64)) - 1
        };
        // exact replicate score from a bitset
        let exact = |sc: &Scratch, gi: usize, rep: usize| -> f64 {
            let bits = &sc.shared[gi * qw..(gi + 1) * qw];
            let mut h = 0.0;
            for &qi in &sc.samp[rep * s..(rep + 1) * s] {
                let q = qi as usize;
                if bits[q / 64] >> (q % 64) & 1 == 1 {
                    h += sc.qidf[q];
                }
            }
            h
        };
        // leader: maximum total score (first in group order on ties)
        let mut lead = 0usize;
        for gi in 1..ng {
            if sc.groups[gi].score > sc.groups[lead].score {
                lead = gi;
            }
        }
        sc.best.clear();
        sc.winners.resize_with(bu, Vec::new);
        for rep in 0..bu {
            let v = exact(sc, lead, rep);
            sc.best.push(v);
            sc.winners[rep].clear();
            sc.winners[rep].push(lead as u32);
        }
        let mut cand = vec![0u64; bw];
        let mut lossr = vec![0u64; bw];
        for gi in 0..ng {
            if gi == lead {
                continue;
            }
            cand.iter_mut().for_each(|x| *x = 0);
            lossr.iter_mut().for_each(|x| *x = 0);
            let (gs, ls) = (gi * qw, lead * qw);
            let mut loss_full = false;
            for w in 0..qw {
                let g = sc.shared[gs + w];
                let l = sc.shared[ls + w];
                let pw = sc.posw[w];
                let mut gain = (g & !l & pw) | (l & !g & !pw);
                while gain != 0 {
                    let q = w * 64 + gain.trailing_zeros() as usize;
                    gain &= gain - 1;
                    for k in 0..bw {
                        cand[k] |= sc.occ_mask[q * bw + k];
                    }
                }
                if !loss_full {
                    let mut loss = l & !g & pw;
                    while loss != 0 {
                        let q = w * 64 + loss.trailing_zeros() as usize;
                        loss &= loss - 1;
                        for k in 0..bw {
                            lossr[k] |= sc.occ_mask[q * bw + k];
                        }
                    }
                    loss_full = lossr[..bw - 1].iter().all(|&x| x == u64::MAX)
                        && lossr[bw - 1] & full_last == full_last;
                }
            }
            // candidate replicates: some gain sampled, or no loss sampled
            for k in 0..bw {
                let mut c = cand[k] | !lossr[k];
                if k == bw - 1 {
                    c &= full_last;
                }
                while c != 0 {
                    let rep = k * 64 + c.trailing_zeros() as usize;
                    c &= c - 1;
                    let v = exact(sc, gi, rep);
                    if v > sc.best[rep] {
                        sc.best[rep] = v;
                        sc.winners[rep].clear();
                        sc.winners[rep].push(gi as u32);
                    } else if v == sc.best[rep] {
                        sc.winners[rep].push(gi as u32);
                    }
                }
            }
        }
        let mut tot = vec![0.0f64; ng];
        for rep in 0..bu {
            let best = sc.best[rep];
            let mut w = std::mem::take(&mut sc.winners[rep]);
            Self::credit(best, &mut w, davg, &mut tot, rng);
            sc.winners[rep] = w;
        }
        tot
    }

    /// IDF-weighted containment similarity of the query (best orientation
    /// given by `orientation`) to all reference sequences; returns the
    /// maximum over the whole reference and within each lineage node.
    pub fn similarity_profile(
        &self,
        seq: &[u8],
        cls: &Classification,
        sc: &mut Scratch,
    ) -> SimilarityProfile {
        let m = self.model;
        let norm = normalize_sequence(seq);
        let (f, r) = self.encoder.encode_both(&norm);
        let kmers = match cls.orientation {
            Orientation::Reverse => r.unique,
            _ => f.unique,
        };
        let mut denom = 0.0;
        let epoch = sc.next_epoch();
        sc.touched.clear();
        let mut base = 0.0;
        let mut negs: Vec<(usize, f64)> = Vec::new();
        let mut pos: Vec<(usize, f64)> = Vec::new();
        for &x in &kmers {
            match m.slot(x) {
                Some(s) => {
                    let w = m.slot_idf(s);
                    denom += w;
                    if m.slot_is_complement(s) {
                        base += w;
                        negs.push((s, w));
                    } else {
                        pos.push((s, w));
                    }
                }
                None => denom += m.meta.idf_default,
            }
        }
        if !negs.is_empty() {
            for id in 0..m.n_seqs() as u32 {
                let a = &mut sc.acc[id as usize];
                a.stamp = epoch;
                a.score = base;
                sc.touched.push(id);
            }
            for &(slot, w) in &negs {
                for &id in m.postings(slot) {
                    sc.acc[id as usize].score -= w;
                }
            }
        }
        for &(slot, w) in &pos {
            for &id in m.postings(slot) {
                let a = &mut sc.acc[id as usize];
                if a.stamp != epoch {
                    a.stamp = epoch;
                    a.score = 0.0;
                    sc.touched.push(id);
                }
                a.score += w;
            }
        }
        let mut prof = SimilarityProfile {
            max_reference: 0.0,
            per_lineage: vec![0.0; cls.lineage.len()],
        };
        if denom <= 0.0 {
            return prof;
        }
        let ranges: Vec<Range<u32>> = cls.lineage.iter().map(|&n| m.subtree(n)).collect();
        for &id in &sc.touched {
            let v = sc.acc[id as usize].score / denom;
            if v > prof.max_reference {
                prof.max_reference = v;
            }
            for (d, r) in ranges.iter().enumerate() {
                if r.contains(&id) && v > prof.per_lineage[d] {
                    prof.per_lineage[d] = v;
                }
            }
        }
        prof
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn n_above() {
        let c = Classification {
            lineage: vec![0, 1, 2, 3],
            confidences: vec![100.0, 90.0, 61.0, 40.0],
            orientation: Orientation::Forward,
            bootstraps_used: 100,
            similarity: 0.5,
        };
        assert_eq!(c.n_above(60.0), 3);
        assert_eq!(c.n_above(95.0), 1);
        assert_eq!(c.n_above(0.0), 4);
        assert_eq!(c.n_above_per_rank(&c.confidences, &[95.0, 50.0, 30.0]), 1);
        assert_eq!(c.n_above_per_rank(&c.confidences, &[80.0, 50.0, 30.0]), 4);
    }

    #[test]
    fn strand_parse() {
        assert_eq!("both".parse::<Strand>().unwrap(), Strand::Both);
        assert_eq!("TOP".parse::<Strand>().unwrap(), Strand::Top);
        assert!("x".parse::<Strand>().is_err());
    }
}
