//! Trained model: in-memory representation and the native, memory-mappable
//! on-disk format.
//!
//! # File layout (format version [`FORMAT_VERSION`])
//!
//! ```text
//! offset 0   : magic  b"IDTAXARS"                       (8 bytes)
//!          8 : format version, u32 LE
//!         12 : endianness marker 0x01020304, u32 LE
//!         16 : metadata offset, u64 LE
//!         24 : metadata length, u64 LE
//!         32 : reserved (zero) up to byte 64
//! 64 ...     : flat arrays, each starting at a 64-byte aligned offset
//! meta_off   : JSON metadata ([`ModelMeta`]) including the section table
//! ```
//!
//! Arrays are plain little-endian `u32`/`u64`/`f64` slices that are accessed
//! zero-copy through `memmap2`, so loading a model is near-instant and the
//! page cache shares it between processes on a node.

use std::fs::File;
use std::io::{BufWriter, Write};
use std::ops::{Deref, Range};
use std::path::Path;
use std::sync::Arc;

use bytemuck::Pod;
use memmap2::Mmap;
use serde::{Deserialize, Serialize};
use xxhash_rust::xxh3::Xxh3;

use crate::error::{Error, Result};
use crate::kmer::KmerEncoder;
use crate::taxonomy::{full_paths, NO_PARENT};

/// Native model format version.
pub const FORMAT_VERSION: u32 = 1;
const MAGIC: &[u8; 8] = b"IDTAXARS";
const HEADER_LEN: usize = 64;
const ALIGN: usize = 64;
/// Largest K for which a dense k-mer → index-slot table is stored.
pub const DENSE_MAX_K: u32 = 12;
/// Marker for "k-mer absent" in the dense slot table.
pub const NO_SLOT: u32 = u32::MAX;

/// Array storage: either owned or a window into a shared memory map.
pub enum Arr<T: Pod> {
    Owned(Vec<T>),
    Mapped {
        map: Arc<Mmap>,
        offset: usize,
        len: usize,
    },
}

impl<T: Pod> Deref for Arr<T> {
    type Target = [T];
    fn deref(&self) -> &[T] {
        match self {
            Arr::Owned(v) => v,
            Arr::Mapped { map, offset, len } => {
                let bytes = &map[*offset..*offset + len * std::mem::size_of::<T>()];
                bytemuck::cast_slice(bytes)
            }
        }
    }
}

impl<T: Pod> Default for Arr<T> {
    fn default() -> Self {
        Arr::Owned(Vec::new())
    }
}

impl<T: Pod> From<Vec<T>> for Arr<T> {
    fn from(v: Vec<T>) -> Self {
        Arr::Owned(v)
    }
}

impl<T: Pod + std::fmt::Debug> std::fmt::Debug for Arr<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Arr(len={})", self.len())
    }
}

/// A sequence flagged during training (DECIPHER `problemSequences`).
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct ProblemSequence {
    /// 0-based index of the training sequence (original input order).
    pub index: u32,
    pub expected: String,
    pub predicted: String,
}

/// Parameters a model was trained with.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct TrainParams {
    pub source: String,
    pub n: f64,
    pub min_fraction: f64,
    pub max_fraction: f64,
    pub max_iterations: u32,
    pub multiplier: f64,
    pub max_children: u32,
    pub k_auto: bool,
    #[serde(default)]
    pub max_group_size: Option<u32>,
    #[serde(default)]
    pub seed: Option<u64>,
}

impl Default for TrainParams {
    fn default() -> Self {
        Self {
            source: "idtaxa-rs".into(),
            n: 500.0,
            min_fraction: 0.01,
            max_fraction: 0.06,
            max_iterations: 10,
            multiplier: 100.0,
            max_children: 200,
            k_auto: true,
            max_group_size: None,
            seed: None,
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
struct Section {
    name: String,
    dtype: String,
    offset: u64,
    len: u64,
}

/// JSON metadata stored in the model file.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ModelMeta {
    pub format_version: u32,
    pub tool_version: String,
    pub model_id: String,
    pub reference_checksum: String,
    pub k: u32,
    #[serde(default)]
    pub seed_pattern: Option<String>,
    pub taxa: Vec<String>,
    pub ranks: Option<Vec<String>>,
    /// Training sequence names in original input order.
    pub seq_names: Vec<String>,
    pub n_classes: u64,
    pub idf_default: f64,
    pub train_params: TrainParams,
    pub problem_sequences: Vec<ProblemSequence>,
    pub problem_groups: Vec<String>,
    sections: Vec<Section>,
}

/// Everything needed to build a [`Model`]; produced by the DECIPHER
/// importer and by native training. All indices are 0-based and sequences
/// are in original training order.
#[derive(Clone, Debug, Default)]
pub struct RawModel {
    pub k: u32,
    pub seed_pattern: Option<String>,
    pub taxa: Vec<String>,
    pub parents: Vec<u32>,
    pub children: Vec<Vec<u32>>,
    pub ranks: Option<Vec<String>>,
    /// Per node; NaN encodes DECIPHER's NA.
    pub fraction: Vec<f64>,
    /// Sorted unique 0-based k-mer codes per training sequence.
    pub seq_kmers: Vec<Vec<u32>>,
    pub seq_names: Vec<String>,
    /// Node of each training sequence.
    pub cross_index: Vec<u32>,
    /// Per node: decision k-mers (sorted, 0-based) and weights stored
    /// child-major (`weights[child * n_kmers + j]`).
    pub decision: Vec<Option<(Vec<u32>, Vec<f64>)>>,
    /// IDF weights: if `None` they are computed with DECIPHER's formula.
    pub idf_dense: Option<Vec<f64>>,
    pub problem_sequences: Vec<ProblemSequence>,
    pub problem_groups: Vec<String>,
    pub train_params: TrainParams,
    pub reference_checksum: String,
}

/// A trained IDTAXA model with an inverted k-mer index.
#[derive(Debug, Default)]
pub struct Model {
    pub meta: ModelMeta,
    /// Full path strings per node (derived, e.g. `"Root;A;B;"`).
    pub taxonomy: Vec<String>,
    // --- per node ---
    parents: Arr<u32>,
    levels: Arr<u32>,
    child_off: Arr<u32>,
    child: Arr<u32>,
    fraction: Arr<f64>,
    sub_start: Arr<u32>,
    sub_end: Arr<u32>,
    dk_off: Arr<u64>,
    dk_kmers: Arr<u32>,
    dkw_off: Arr<u64>,
    dk_w: Arr<f64>,
    // --- per training sequence, native (subtree-contiguous) order ---
    seq_off: Arr<u64>,
    seq_kmers: Arr<u32>,
    seq_group: Arr<u32>,
    seq_orig: Arr<u32>,
    // --- inverted index ---
    idx_keys: Arr<u32>,
    idx_off: Arr<u64>,
    idx_post: Arr<u32>,
    idx_idf: Arr<f64>,
    dense_slot: Arr<u32>,
}

impl Default for ModelMeta {
    fn default() -> Self {
        Self {
            format_version: FORMAT_VERSION,
            tool_version: env!("CARGO_PKG_VERSION").into(),
            model_id: String::new(),
            reference_checksum: String::new(),
            k: 1,
            seed_pattern: None,
            taxa: vec![],
            ranks: None,
            seq_names: vec![],
            n_classes: 0,
            idf_default: 0.0,
            train_params: TrainParams::default(),
            problem_sequences: vec![],
            problem_groups: vec![],
            sections: vec![],
        }
    }
}

fn csr<T: Copy>(lists: &[Vec<T>]) -> (Vec<u64>, Vec<T>) {
    let mut off = Vec::with_capacity(lists.len() + 1);
    let mut data = Vec::with_capacity(lists.iter().map(|l| l.len()).sum());
    off.push(0u64);
    for l in lists {
        data.extend_from_slice(l);
        off.push(data.len() as u64);
    }
    (off, data)
}

/// Compute DECIPHER IDF weights for the k-mers in `keys`:
/// `log(nClasses / (1 + sum_{seqs containing kmer} 1/size(class)))`
/// (R/LearnTaxa.R, "compute Inverse Document Frequency (IDF) weights").
/// Returns (weights for keys, default weight for absent k-mers, nClasses).
pub fn compute_idf(
    seq_kmers: &[Vec<u32>],
    cross_index: &[u32],
    keys: &[u32],
) -> (Vec<f64>, f64, u64) {
    let mut class_size: std::collections::HashMap<u32, u64> = Default::default();
    for &c in cross_index {
        *class_size.entry(c).or_default() += 1;
    }
    let n_classes = class_size.len() as u64;
    let mut counts = vec![0.0f64; keys.len()];
    for (i, km) in seq_kmers.iter().enumerate() {
        let w = 1.0 / class_size[&cross_index[i]] as f64;
        // both sorted: merge walk
        let mut j = 0usize;
        for &x in km {
            j += keys[j..].partition_point(|&k| k < x);
            if j < keys.len() && keys[j] == x {
                counts[j] += w;
            }
        }
    }
    let nc = n_classes as f64;
    let idf = counts.iter().map(|&c| (nc / (1.0 + c)).ln()).collect();
    (idf, nc.ln(), n_classes)
}

impl Model {
    /// Build a model (inverted index, subtree ranges) from raw parts.
    pub fn from_raw(raw: RawModel) -> Result<Model> {
        let n_nodes = raw.taxa.len();
        let n_seqs = raw.seq_kmers.len();
        if raw.parents.len() != n_nodes
            || raw.children.len() != n_nodes
            || raw.fraction.len() != n_nodes
            || raw.decision.len() != n_nodes
        {
            return Err(Error::InvalidModel("per-node arrays have inconsistent lengths".into()));
        }
        if raw.cross_index.len() != n_seqs || raw.seq_names.len() != n_seqs {
            return Err(Error::InvalidModel("per-sequence arrays have inconsistent lengths".into()));
        }
        if n_seqs >= u32::MAX as usize {
            return Err(Error::InvalidModel("too many training sequences".into()));
        }
        let encoder = KmerEncoder::from_model(raw.k as usize, raw.seed_pattern.as_deref())?;
        let n_kmers = encoder.n_kmers();
        for km in &raw.seq_kmers {
            if km.windows(2).any(|w| w[0] >= w[1]) {
                return Err(Error::InvalidModel("training k-mers must be sorted and unique".into()));
            }
            if km.last().is_some_and(|&x| (x as u64) >= n_kmers) {
                return Err(Error::InvalidModel("k-mer code out of range for K".into()));
            }
        }
        for &c in &raw.cross_index {
            if c as usize >= n_nodes {
                return Err(Error::InvalidModel("crossIndex out of range".into()));
            }
        }
        // levels
        let mut levels = vec![0u32; n_nodes];
        for i in 0..n_nodes {
            let p = raw.parents[i];
            levels[i] = if p == NO_PARENT {
                1
            } else {
                if p as usize >= i {
                    return Err(Error::InvalidModel("parents must precede children".into()));
                }
                levels[p as usize] + 1
            };
        }

        // native order: DFS preorder; at each node its own sequences
        // (ascending original index) precede its children's subtrees.
        let mut own: Vec<Vec<u32>> = vec![Vec::new(); n_nodes];
        for (s, &c) in raw.cross_index.iter().enumerate() {
            own[c as usize].push(s as u32);
        }
        let mut order: Vec<u32> = Vec::with_capacity(n_seqs);
        let mut sub_start = vec![0u32; n_nodes];
        let mut sub_end = vec![0u32; n_nodes];
        // iterative DFS with explicit enter/exit
        let mut stack: Vec<(u32, bool)> = vec![(0, false)];
        while let Some((node, exit)) = stack.pop() {
            let n = node as usize;
            if exit {
                sub_end[n] = order.len() as u32;
                continue;
            }
            sub_start[n] = order.len() as u32;
            order.extend_from_slice(&own[n]);
            stack.push((node, true));
            for &c in raw.children[n].iter().rev() {
                stack.push((c, false));
            }
        }
        if order.len() != n_seqs {
            return Err(Error::InvalidModel(
                "some training sequences are not reachable from Root".into(),
            ));
        }

        let seq_lists: Vec<Vec<u32>> =
            order.iter().map(|&o| raw.seq_kmers[o as usize].clone()).collect();
        let seq_group: Vec<u32> = order.iter().map(|&o| raw.cross_index[o as usize]).collect();
        let (seq_off, seq_kmers) = csr(&seq_lists);
        drop(seq_lists);

        // inverted index
        let (keys, idx_off, idx_post) = build_index(&seq_off, &seq_kmers, raw.k)?;

        // IDF
        let (idx_idf, idf_default, n_classes) = match &raw.idf_dense {
            Some(dense) => {
                if dense.len() as u64 != n_kmers {
                    return Err(Error::InvalidModel(format!(
                        "IDF weight vector has length {}, expected 4^K = {n_kmers}",
                        dense.len()
                    )));
                }
                let mut classes: Vec<u32> = raw.cross_index.clone();
                classes.sort_unstable();
                classes.dedup();
                let nc = classes.len() as u64;
                let default = (nc as f64).ln();
                // verify absent k-mers carry the default weight
                let mut ki = 0usize;
                for (code, &w) in dense.iter().enumerate() {
                    if ki < keys.len() && keys[ki] as usize == code {
                        ki += 1;
                        continue;
                    }
                    if (w - default).abs() > 1e-9 * default.abs().max(1.0) {
                        return Err(Error::InvalidModel(format!(
                            "IDF weight of absent k-mer {code} is {w}, expected log(nClasses) = {default}"
                        )));
                    }
                }
                let idf: Vec<f64> = keys.iter().map(|&k| dense[k as usize]).collect();
                (idf, default, nc)
            }
            None => compute_idf(&raw.seq_kmers, &raw.cross_index, &keys),
        };

        let dense_slot = if raw.k <= DENSE_MAX_K {
            let mut d = vec![NO_SLOT; n_kmers as usize];
            for (i, &k) in keys.iter().enumerate() {
                d[k as usize] = i as u32;
            }
            d
        } else {
            Vec::new()
        };

        let (child_off, child) = csr(&raw.children);
        let child_off: Vec<u32> = child_off.into_iter().map(|x| x as u32).collect();
        let mut dk_lists = Vec::with_capacity(n_nodes);
        let mut dkw_lists = Vec::with_capacity(n_nodes);
        for (i, d) in raw.decision.iter().enumerate() {
            match d {
                Some((km, w)) => {
                    if km.windows(2).any(|w| w[0] >= w[1]) {
                        return Err(Error::InvalidModel(format!(
                            "decision k-mers of node {i} are not sorted/unique"
                        )));
                    }
                    if w.len() != km.len() * raw.children[i].len() {
                        return Err(Error::InvalidModel(format!(
                            "decision weights of node {i} have length {}, expected {} x {}",
                            w.len(),
                            raw.children[i].len(),
                            km.len()
                        )));
                    }
                    dk_lists.push(km.clone());
                    dkw_lists.push(w.clone());
                }
                None => {
                    dk_lists.push(Vec::new());
                    dkw_lists.push(Vec::new());
                }
            }
        }
        let (dk_off, dk_kmers) = csr(&dk_lists);
        let (dkw_off, dk_w) = csr(&dkw_lists);

        let meta = ModelMeta {
            format_version: FORMAT_VERSION,
            tool_version: env!("CARGO_PKG_VERSION").into(),
            model_id: String::new(),
            reference_checksum: raw.reference_checksum,
            k: raw.k,
            seed_pattern: raw.seed_pattern,
            taxa: Vec::new(),
            ranks: raw.ranks,
            seq_names: raw.seq_names,
            n_classes,
            idf_default,
            train_params: raw.train_params,
            problem_sequences: raw.problem_sequences,
            problem_groups: raw.problem_groups,
            sections: vec![],
        };
        let mut model = Model {
            taxonomy: full_paths(&raw.taxa, &raw.parents),
            meta: ModelMeta { taxa: raw.taxa, ..meta },
            parents: raw.parents.into(),
            levels: levels.into(),
            child_off: child_off.into(),
            child: child.into(),
            fraction: raw.fraction.into(),
            sub_start: sub_start.into(),
            sub_end: sub_end.into(),
            dk_off: dk_off.into(),
            dk_kmers: dk_kmers.into(),
            dkw_off: dkw_off.into(),
            dk_w: dk_w.into(),
            seq_off: seq_off.into(),
            seq_kmers: seq_kmers.into(),
            seq_group: seq_group.into(),
            seq_orig: order.into(),
            idx_keys: keys.into(),
            idx_off: idx_off.into(),
            idx_post: idx_post.into(),
            idx_idf: idx_idf.into(),
            dense_slot: dense_slot.into(),
        };
        model.meta.model_id = model.compute_id();
        Ok(model)
    }

    fn arrays(&self) -> Vec<(&'static str, &'static str, &[u8])> {
        fn b<T: Pod>(a: &Arr<T>) -> &[u8] {
            bytemuck::cast_slice(a)
        }
        vec![
            ("parents", "u32", b(&self.parents)),
            ("levels", "u32", b(&self.levels)),
            ("child_off", "u32", b(&self.child_off)),
            ("child", "u32", b(&self.child)),
            ("fraction", "f64", b(&self.fraction)),
            ("sub_start", "u32", b(&self.sub_start)),
            ("sub_end", "u32", b(&self.sub_end)),
            ("dk_off", "u64", b(&self.dk_off)),
            ("dk_kmers", "u32", b(&self.dk_kmers)),
            ("dkw_off", "u64", b(&self.dkw_off)),
            ("dk_w", "f64", b(&self.dk_w)),
            ("seq_off", "u64", b(&self.seq_off)),
            ("seq_kmers", "u32", b(&self.seq_kmers)),
            ("seq_group", "u32", b(&self.seq_group)),
            ("seq_orig", "u32", b(&self.seq_orig)),
            ("idx_keys", "u32", b(&self.idx_keys)),
            ("idx_off", "u64", b(&self.idx_off)),
            ("idx_post", "u32", b(&self.idx_post)),
            ("idx_idf", "f64", b(&self.idx_idf)),
            ("dense_slot", "u32", b(&self.dense_slot)),
        ]
    }

    fn compute_id(&self) -> String {
        let mut h = Xxh3::new();
        h.update(&self.meta.k.to_le_bytes());
        if let Some(p) = &self.meta.seed_pattern {
            h.update(p.as_bytes());
        }
        for t in &self.meta.taxa {
            h.update(t.as_bytes());
            h.update(b"\0");
        }
        for (name, _, bytes) in self.arrays() {
            if name == "dense_slot" {
                continue; // derived
            }
            h.update(name.as_bytes());
            h.update(bytes);
        }
        format!("{:032x}", h.digest128())
    }

    /// Write the native model file.
    pub fn save(&self, path: &Path) -> Result<()> {
        let f = File::create(path).map_err(|e| Error::io(path, e))?;
        let mut w = BufWriter::with_capacity(1 << 20, f);
        let io = |e| Error::io(path, e);
        w.write_all(&[0u8; HEADER_LEN]).map_err(io)?;
        let mut pos = HEADER_LEN;
        let mut sections = Vec::new();
        for (name, dtype, bytes) in self.arrays() {
            let pad = (ALIGN - pos % ALIGN) % ALIGN;
            w.write_all(&vec![0u8; pad]).map_err(io)?;
            pos += pad;
            let size = match dtype {
                "u64" | "f64" => 8,
                _ => 4,
            };
            sections.push(Section {
                name: name.into(),
                dtype: dtype.into(),
                offset: pos as u64,
                len: (bytes.len() / size) as u64,
            });
            w.write_all(bytes).map_err(io)?;
            pos += bytes.len();
        }
        let mut meta = self.meta.clone();
        meta.sections = sections;
        meta.format_version = FORMAT_VERSION;
        let json = serde_json::to_vec(&meta)?;
        let meta_off = pos as u64;
        w.write_all(&json).map_err(io)?;
        let mut f = w.into_inner().map_err(|e| Error::io(path, e.into_error()))?;
        use std::io::{Seek, SeekFrom};
        let mut header = [0u8; HEADER_LEN];
        header[0..8].copy_from_slice(MAGIC);
        header[8..12].copy_from_slice(&FORMAT_VERSION.to_le_bytes());
        header[12..16].copy_from_slice(&0x0102_0304u32.to_le_bytes());
        header[16..24].copy_from_slice(&meta_off.to_le_bytes());
        header[24..32].copy_from_slice(&(json.len() as u64).to_le_bytes());
        f.seek(SeekFrom::Start(0)).map_err(io)?;
        f.write_all(&header).map_err(io)?;
        f.sync_all().ok();
        Ok(())
    }

    /// Memory-map a native model file.
    pub fn load(path: &Path) -> Result<Model> {
        if cfg!(target_endian = "big") {
            return Err(Error::InvalidModel("big-endian platforms are not supported".into()));
        }
        let f = File::open(path).map_err(|e| Error::io(path, e))?;
        // SAFETY: the file is opened read-only; concurrent modification of a
        // model file while in use is not supported (documented).
        let map = unsafe { Mmap::map(&f) }.map_err(|e| Error::io(path, e))?;
        if map.len() < HEADER_LEN || &map[0..8] != MAGIC {
            return Err(Error::InvalidModel(format!(
                "{} is not an idtaxa-rs model (bad magic)",
                path.display()
            )));
        }
        let version = u32::from_le_bytes(map[8..12].try_into().unwrap());
        if version != FORMAT_VERSION {
            return Err(Error::ModelVersion {
                found: version,
                expected: FORMAT_VERSION,
            });
        }
        if u32::from_le_bytes(map[12..16].try_into().unwrap()) != 0x0102_0304 {
            return Err(Error::InvalidModel("endianness marker mismatch".into()));
        }
        let meta_off = u64::from_le_bytes(map[16..24].try_into().unwrap()) as usize;
        let meta_len = u64::from_le_bytes(map[24..32].try_into().unwrap()) as usize;
        if meta_off.checked_add(meta_len).is_none_or(|e| e > map.len()) {
            return Err(Error::InvalidModel("truncated model file".into()));
        }
        let meta: ModelMeta = serde_json::from_slice(&map[meta_off..meta_off + meta_len])?;
        let map = Arc::new(map);
        let get = |name: &str, size: usize| -> Result<(usize, usize)> {
            let s = meta
                .sections
                .iter()
                .find(|s| s.name == name)
                .ok_or_else(|| Error::InvalidModel(format!("missing section {name}")))?;
            let off = s.offset as usize;
            let len = s.len as usize;
            if off % ALIGN != 0 || off + len * size > meta_off {
                return Err(Error::InvalidModel(format!("corrupt section {name}")));
            }
            Ok((off, len))
        };
        macro_rules! arr {
            ($name:literal, $t:ty) => {{
                let (offset, len) = get($name, std::mem::size_of::<$t>())?;
                Arr::<$t>::Mapped {
                    map: map.clone(),
                    offset,
                    len,
                }
            }};
        }
        let mut model = Model {
            taxonomy: Vec::new(),
            parents: arr!("parents", u32),
            levels: arr!("levels", u32),
            child_off: arr!("child_off", u32),
            child: arr!("child", u32),
            fraction: arr!("fraction", f64),
            sub_start: arr!("sub_start", u32),
            sub_end: arr!("sub_end", u32),
            dk_off: arr!("dk_off", u64),
            dk_kmers: arr!("dk_kmers", u32),
            dkw_off: arr!("dkw_off", u64),
            dk_w: arr!("dk_w", f64),
            seq_off: arr!("seq_off", u64),
            seq_kmers: arr!("seq_kmers", u32),
            seq_group: arr!("seq_group", u32),
            seq_orig: arr!("seq_orig", u32),
            idx_keys: arr!("idx_keys", u32),
            idx_off: arr!("idx_off", u64),
            idx_post: arr!("idx_post", u32),
            idx_idf: arr!("idx_idf", f64),
            dense_slot: arr!("dense_slot", u32),
            meta,
        };
        model.validate()?;
        model.taxonomy = full_paths(&model.meta.taxa, &model.parents);
        Ok(model)
    }

    fn validate(&self) -> Result<()> {
        let n = self.meta.taxa.len();
        let ns = self.seq_group.len();
        let ok = self.parents.len() == n
            && self.levels.len() == n
            && self.child_off.len() == n + 1
            && self.fraction.len() == n
            && self.sub_start.len() == n
            && self.sub_end.len() == n
            && self.dk_off.len() == n + 1
            && self.dkw_off.len() == n + 1
            && self.seq_off.len() == ns + 1
            && self.seq_orig.len() == ns
            && self.meta.seq_names.len() == ns
            && self.idx_off.len() == self.idx_keys.len() + 1
            && self.idx_idf.len() == self.idx_keys.len()
            && *self.child_off.last().unwrap() as usize == self.child.len()
            && *self.seq_off.last().unwrap() as usize == self.seq_kmers.len()
            && *self.idx_off.last().unwrap() as usize == self.idx_post.len()
            && *self.dk_off.last().unwrap() as usize == self.dk_kmers.len()
            && *self.dkw_off.last().unwrap() as usize == self.dk_w.len()
            && (self.dense_slot.is_empty() || self.dense_slot.len() as u64 == 1u64 << (2 * self.meta.k));
        if !ok {
            return Err(Error::InvalidModel("inconsistent array sizes".into()));
        }
        Ok(())
    }

    // ----- accessors -----

    /// K (k-mer weight).
    pub fn k(&self) -> usize {
        self.meta.k as usize
    }

    /// K-mer encoder matching this model (contiguous or spaced).
    pub fn encoder(&self) -> KmerEncoder {
        KmerEncoder::from_model(self.k(), self.meta.seed_pattern.as_deref())
            .expect("validated at build time")
    }

    /// Number of taxonomy nodes.
    pub fn n_nodes(&self) -> usize {
        self.meta.taxa.len()
    }

    /// Number of training sequences.
    pub fn n_seqs(&self) -> usize {
        self.seq_group.len()
    }

    /// Number of distinct k-mers present in the training set.
    pub fn n_index_kmers(&self) -> usize {
        self.idx_keys.len()
    }

    /// Total postings in the inverted index.
    pub fn n_postings(&self) -> usize {
        self.idx_post.len()
    }

    /// Node name.
    pub fn taxon(&self, node: u32) -> &str {
        &self.meta.taxa[node as usize]
    }

    /// Rank name of a node, if the model has ranks.
    pub fn rank(&self, node: u32) -> Option<&str> {
        self.meta.ranks.as_ref().map(|r| r[node as usize].as_str())
    }

    /// Parent node (`NO_PARENT` for Root).
    pub fn parent(&self, node: u32) -> u32 {
        self.parents[node as usize]
    }

    /// DECIPHER level (Root = 1).
    pub fn level(&self, node: u32) -> u32 {
        self.levels[node as usize]
    }

    /// Immediate children.
    pub fn children(&self, node: u32) -> &[u32] {
        let n = node as usize;
        &self.child[self.child_off[n] as usize..self.child_off[n + 1] as usize]
    }

    /// Learned sampling fraction (NaN = NA / problem group).
    pub fn fraction(&self, node: u32) -> f64 {
        self.fraction[node as usize]
    }

    /// Decision k-mers and child-major weight matrix of a node.
    pub fn decision(&self, node: u32) -> (&[u32], &[f64]) {
        let n = node as usize;
        (
            &self.dk_kmers[self.dk_off[n] as usize..self.dk_off[n + 1] as usize],
            &self.dk_w[self.dkw_off[n] as usize..self.dkw_off[n + 1] as usize],
        )
    }

    /// Native ID range of the training sequences in a node's subtree.
    pub fn subtree(&self, node: u32) -> Range<u32> {
        self.sub_start[node as usize]..self.sub_end[node as usize]
    }

    /// Sorted unique k-mers of a training sequence (native ID).
    pub fn seq_kmers(&self, id: u32) -> &[u32] {
        let i = id as usize;
        &self.seq_kmers[self.seq_off[i] as usize..self.seq_off[i + 1] as usize]
    }

    /// Group (node) of a training sequence (native ID).
    pub fn seq_group(&self, id: u32) -> u32 {
        self.seq_group[id as usize]
    }

    /// Original training index of a native ID.
    pub fn seq_orig(&self, id: u32) -> u32 {
        self.seq_orig[id as usize]
    }

    /// Native IDs in original-index order (inverse of `seq_orig`).
    pub fn orig_to_native(&self) -> Vec<u32> {
        let mut v = vec![0u32; self.n_seqs()];
        for (nat, &o) in self.seq_orig.iter().enumerate() {
            v[o as usize] = nat as u32;
        }
        v
    }

    /// Index slot of a k-mer, if present in the training set.
    #[inline]
    pub fn slot(&self, kmer: u32) -> Option<usize> {
        if !self.dense_slot.is_empty() {
            let s = self.dense_slot[kmer as usize];
            (s != NO_SLOT).then_some(s as usize)
        } else {
            self.idx_keys.binary_search(&kmer).ok()
        }
    }

    /// Postings (sorted native IDs) of an index slot.
    #[inline]
    pub fn postings(&self, slot: usize) -> &[u32] {
        &self.idx_post[self.idx_off[slot] as usize..self.idx_off[slot + 1] as usize]
    }

    /// IDF weight of an index slot.
    #[inline]
    pub fn slot_idf(&self, slot: usize) -> f64 {
        self.idx_idf[slot]
    }

    /// IDF weight of any k-mer.
    #[inline]
    pub fn idf(&self, kmer: u32) -> f64 {
        match self.slot(kmer) {
            Some(s) => self.idx_idf[s],
            None => self.meta.idf_default,
        }
    }

    /// Dense IDF vector over all 4^K k-mers (for export/comparison).
    pub fn idf_dense(&self) -> Vec<f64> {
        let n = 1usize << (2 * self.k());
        let mut v = vec![self.meta.idf_default; n];
        for (i, &k) in self.idx_keys.iter().enumerate() {
            v[k as usize] = self.idx_idf[i];
        }
        v
    }

    /// Per-sequence k-mer lists in original order (for export/retraining).
    pub fn seq_kmers_original(&self) -> Vec<Vec<u32>> {
        let o2n = self.orig_to_native();
        o2n.iter().map(|&n| self.seq_kmers(n).to_vec()).collect()
    }

    /// Lineage (node path from Root) of a node.
    pub fn lineage(&self, node: u32) -> Vec<u32> {
        let mut v = vec![node];
        let mut p = self.parent(node);
        while p != NO_PARENT {
            v.push(p);
            p = self.parent(p);
        }
        v.reverse();
        v
    }

    /// Whether `ancestor` is on the path from Root to `node` (inclusive).
    pub fn is_ancestor_or_self(&self, ancestor: u32, node: u32) -> bool {
        let r = self.subtree(ancestor);
        let s = self.subtree(node);
        // subtree ranges are nested intervals; empty ranges need care
        if ancestor == node {
            return true;
        }
        let mut p = self.parent(node);
        if !s.is_empty() && !(r.start <= s.start && s.end <= r.end) {
            return false;
        }
        while p != NO_PARENT {
            if p == ancestor {
                return true;
            }
            p = self.parent(p);
        }
        false
    }

    /// Sum of per-node subtree sizes for reporting.
    pub fn leaf_groups(&self) -> usize {
        let mut v: Vec<u32> = self.seq_group.to_vec();
        v.sort_unstable();
        v.dedup();
        v.len()
    }

    /// Raw parts of the model (original sequence order), used to retrain or
    /// re-export.
    pub fn to_raw(&self) -> RawModel {
        let n = self.n_nodes();
        let decision = (0..n as u32)
            .map(|i| {
                let (k, w) = self.decision(i);
                if k.is_empty() {
                    None
                } else {
                    Some((k.to_vec(), w.to_vec()))
                }
            })
            .collect();
        let o2n = self.orig_to_native();
        RawModel {
            k: self.meta.k,
            seed_pattern: self.meta.seed_pattern.clone(),
            taxa: self.meta.taxa.clone(),
            parents: self.parents.to_vec(),
            children: (0..n as u32).map(|i| self.children(i).to_vec()).collect(),
            ranks: self.meta.ranks.clone(),
            fraction: self.fraction.to_vec(),
            seq_kmers: o2n.iter().map(|&x| self.seq_kmers(x).to_vec()).collect(),
            seq_names: self.meta.seq_names.clone(),
            cross_index: o2n.iter().map(|&x| self.seq_group(x)).collect(),
            decision,
            idf_dense: None,
            problem_sequences: self.meta.problem_sequences.clone(),
            problem_groups: self.meta.problem_groups.clone(),
            train_params: self.meta.train_params.clone(),
            reference_checksum: self.meta.reference_checksum.clone(),
        }
    }
}

/// Build CSR inverted index (sorted keys, offsets, postings) from CSR
/// per-sequence k-mer lists.
fn build_index(seq_off: &[u64], seq_kmers: &[u32], k: u32) -> Result<(Vec<u32>, Vec<u64>, Vec<u32>)> {
    let n_seqs = seq_off.len() - 1;
    if k <= DENSE_MAX_K {
        let n = 1usize << (2 * k);
        let mut counts = vec![0u64; n + 1];
        for &x in seq_kmers {
            counts[x as usize + 1] += 1;
        }
        let keys: Vec<u32> = (0..n as u32).filter(|&x| counts[x as usize + 1] > 0).collect();
        for i in 1..=n {
            counts[i] += counts[i - 1];
        }
        let mut cursor = counts.clone();
        let mut post = vec![0u32; seq_kmers.len()];
        for s in 0..n_seqs {
            for &x in &seq_kmers[seq_off[s] as usize..seq_off[s + 1] as usize] {
                let c = &mut cursor[x as usize];
                post[*c as usize] = s as u32;
                *c += 1;
            }
        }
        let mut off = Vec::with_capacity(keys.len() + 1);
        off.push(0);
        for &k in &keys {
            off.push(counts[k as usize + 1]);
        }
        Ok((keys, off, post))
    } else {
        let mut pairs: Vec<u64> = Vec::with_capacity(seq_kmers.len());
        for s in 0..n_seqs {
            for &x in &seq_kmers[seq_off[s] as usize..seq_off[s + 1] as usize] {
                pairs.push(((x as u64) << 32) | s as u64);
            }
        }
        pairs.sort_unstable();
        let mut keys = Vec::new();
        let mut off = vec![0u64];
        let mut post = Vec::with_capacity(pairs.len());
        for p in pairs {
            let key = (p >> 32) as u32;
            if keys.last() != Some(&key) {
                if !keys.is_empty() {
                    off.push(post.len() as u64);
                }
                keys.push(key);
            }
            post.push(p as u32);
        }
        if !keys.is_empty() {
            off.push(post.len() as u64);
        }
        Ok((keys, off, post))
    }
}
