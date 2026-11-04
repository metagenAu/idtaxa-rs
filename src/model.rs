//! The trained classifier and its tree-descent routine.

use crate::taxonomy::{Taxonomy, ROOT};
use anyhow::{bail, Context, Result};
use flate2::read::MultiGzDecoder;
use flate2::write::GzEncoder;
use flate2::Compression;
use rand::Rng;
use serde::{Deserialize, Serialize};
use std::fs::File;
use std::io::{BufReader, BufWriter, Read, Write};
use std::path::Path;

const MAGIC: &[u8; 8] = b"IDTXRS01";

/// Decision k-mers for a node with two or more children. `profiles` is a
/// row-major `kmers.len() × children.len()` matrix of naive-Bayes
/// log-likelihoods `ln((n_c(k) + p_k) / (N_c + 1))` of each k-mer in each child.
#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct Decision {
    pub children: Vec<usize>,
    pub kmers: Vec<u32>,
    pub profiles: Vec<f32>,
}

/// How tree descent proceeds from a node.
#[derive(Serialize, Deserialize, Clone, Debug)]
pub enum NodeRule {
    /// No children with training sequences.
    Terminal,
    /// Exactly one child: descend automatically (no confidence gained).
    Single(usize),
    /// Choose among children using decision k-mers.
    Split(Decision),
}

/// Provenance and summary information stored with a model.
#[derive(Serialize, Deserialize, Clone, Debug, Default)]
pub struct ModelMeta {
    pub name: String,
    pub tool_version: String,
    pub created_unix: u64,
    pub reference: String,
    pub primer_pair: Option<String>,
    pub forward_primer: Option<String>,
    pub reverse_primer: Option<String>,
    pub k: usize,
    pub n_input_sequences: usize,
    pub n_training_sequences: usize,
    pub n_groups: usize,
    pub n_taxa: usize,
    /// References flagged by the leave-one-out check (if it was run).
    pub n_problem_sequences: usize,
    pub ranks: Vec<String>,
    pub notes: Vec<String>,
}

/// A trained IDTAXA classifier.
#[derive(Serialize, Deserialize)]
pub struct Model {
    pub meta: ModelMeta,
    pub k: usize,
    pub taxonomy: Taxonomy,
    pub seq_ids: Vec<String>,
    /// Sorted unique k-mers of each training sequence.
    pub seq_kmers: Vec<Vec<u32>>,
    /// Deepest taxon (group) of each training sequence.
    pub seq_leaf: Vec<usize>,
    /// Training sequences are stored in depth-first taxonomic order, so the
    /// sequences in the subtree of node `n` are `node_range[n].0..node_range[n].1`.
    pub node_range: Vec<(u32, u32)>,
    /// Inverse-frequency weight of every possible k-mer (dense, 4^k).
    pub weights: Vec<f32>,
    pub rules: Vec<NodeRule>,
    /// Fraction of query k-mers sampled per bootstrap at each node during
    /// tree descent (learned during training).
    pub fraction: Vec<f32>,
    /// Inverted k-mer index (rebuilt on load, not serialised).
    #[serde(skip)]
    pub index: KmerIndex,
}

/// Inverted index: for each k-mer, the (ascending) training sequences containing it.
#[derive(Clone, Debug, Default)]
pub struct KmerIndex {
    offsets: Vec<u32>,
    postings: Vec<u32>,
}

impl KmerIndex {
    pub fn build(seq_kmers: &[Vec<u32>], k: usize) -> Self {
        let table = 1usize << (2 * k);
        let mut offsets = vec![0u32; table + 1];
        for km in seq_kmers {
            for &x in km {
                offsets[x as usize + 1] += 1;
            }
        }
        for i in 0..table {
            offsets[i + 1] += offsets[i];
        }
        let mut fill = offsets.clone();
        let mut postings = vec![0u32; offsets[table] as usize];
        for (s, km) in seq_kmers.iter().enumerate() {
            for &x in km {
                let f = &mut fill[x as usize];
                postings[*f as usize] = s as u32;
                *f += 1;
            }
        }
        KmerIndex { offsets, postings }
    }

    #[inline]
    pub fn postings(&self, kmer: u32) -> &[u32] {
        let k = kmer as usize;
        &self.postings[self.offsets[k] as usize..self.offsets[k + 1] as usize]
    }
}

/// One step of a tree descent: the node reached and the bootstrap support
/// for choosing it (`None` when descent was automatic through a single child).
pub type DescentStep = (usize, Option<f32>);

impl Model {
    pub fn save(&self, path: &Path) -> Result<()> {
        let f = File::create(path).with_context(|| format!("cannot create {}", path.display()))?;
        let mut w = BufWriter::new(f);
        w.write_all(MAGIC)?;
        let mut enc = BufWriter::with_capacity(1 << 20, GzEncoder::new(w, Compression::fast()));
        bincode::serialize_into(&mut enc, self).context("serialising model")?;
        let gz = enc.into_inner().map_err(|e| e.into_error())?;
        gz.finish()?.flush()?;
        Ok(())
    }

    pub fn load(path: &Path) -> Result<Model> {
        let f = File::open(path).with_context(|| format!("cannot open model {}", path.display()))?;
        let mut r = BufReader::new(f);
        let mut magic = [0u8; 8];
        r.read_exact(&mut magic).context("reading model header")?;
        if &magic != MAGIC {
            bail!("{} is not an idtaxa-rs model (bad magic)", path.display());
        }
        let dec = BufReader::with_capacity(1 << 20, MultiGzDecoder::new(r));
        let mut m: Model = bincode::deserialize_from(dec).context("deserialising model")?;
        m.taxonomy.rebuild_lookup();
        m.index = KmerIndex::build(&m.seq_kmers, m.k);
        Ok(m)
    }

    /// Indices of training sequences in the subtree of `node`.
    #[inline]
    pub fn members(&self, node: usize) -> std::ops::Range<usize> {
        let (a, b) = self.node_range[node];
        a as usize..b as usize
    }

    #[inline]
    pub fn weight(&self, kmer: u32) -> f32 {
        self.weights[kmer as usize]
    }

    /// Descend the taxonomy using decision k-mers. At each split, `bootstraps`
    /// replicates each sample `fraction[node] × |query|` k-mers with
    /// replacement from the query k-mers that are decision k-mers of the node;
    /// the child with the highest summed log-likelihood wins the replicate.
    /// Descent continues while one child wins at least `min_descend` of the
    /// replicates.
    pub fn descend<R: Rng>(&self, query: &[u32], rng: &mut R, bootstraps: usize, min_descend: f32) -> Vec<DescentStep> {
        let mut path: Vec<DescentStep> = vec![(ROOT, Some(1.0))];
        let n_q = query.len();
        if n_q == 0 || bootstraps == 0 {
            return path;
        }
        let mut cur = ROOT;
        let mut scores: Vec<f32> = Vec::new();
        let mut votes: Vec<u32> = Vec::new();
        loop {
            match &self.rules[cur] {
                NodeRule::Terminal => break,
                NodeRule::Single(c) => {
                    cur = *c;
                    path.push((cur, None));
                }
                NodeRule::Split(d) => {
                    let nc = d.children.len();
                    // rows of the decision matrix for query k-mers that are decision k-mers
                    let hits: Vec<usize> = query.iter().filter_map(|k| d.kmers.binary_search(k).ok()).collect();
                    if hits.is_empty() {
                        break;
                    }
                    let s = ((self.fraction[cur] as f64 * n_q as f64).ceil() as usize).max(1);
                    scores.clear();
                    scores.resize(nc, 0.0);
                    votes.clear();
                    votes.resize(nc, 0);
                    for _ in 0..bootstraps {
                        scores.iter_mut().for_each(|x| *x = 0.0);
                        for _ in 0..s {
                            let h = hits[rng.gen_range(0..hits.len())];
                            let row = &d.profiles[h * nc..(h + 1) * nc];
                            for (sc, &p) in scores.iter_mut().zip(row) {
                                *sc += p;
                            }
                        }
                        let mut best = 0;
                        let mut tie = false;
                        for c in 1..nc {
                            if scores[c] > scores[best] {
                                best = c;
                                tie = false;
                            } else if scores[c] == scores[best] {
                                tie = true;
                            }
                        }
                        if !tie {
                            votes[best] += 1;
                        }
                    }
                    let (best, &bv) = votes.iter().enumerate().max_by_key(|x| *x.1).unwrap();
                    let conf = bv as f32 / bootstraps as f32;
                    if conf >= min_descend {
                        cur = d.children[best];
                        path.push((cur, Some(conf)));
                    } else {
                        break;
                    }
                }
            }
        }
        path
    }

    /// Training check: returns `(node, wrongly chosen child)` if descent for
    /// training sequence `s` leaves its own lineage.
    pub fn check_descent<R: Rng>(
        &self,
        s: usize,
        rng: &mut R,
        bootstraps: usize,
        min_descend: f32,
    ) -> Option<(usize, usize)> {
        let leaf = self.seq_leaf[s];
        let path = self.descend(&self.seq_kmers[s], rng, bootstraps, min_descend);
        for w in path.windows(2) {
            let (n, _) = w[0];
            let (c, _) = w[1];
            if n == leaf {
                return None;
            }
            if !self.taxonomy.is_ancestor_or_self(c, leaf) {
                return Some((n, c));
            }
        }
        None
    }
}
