//! Import of a DECIPHER `Taxa`/`Train` object exported by
//! `scripts/export_decipher_model.R`.
//!
//! # Neutral export format (version 1)
//!
//! A directory containing:
//!
//! | file | content |
//! |------|---------|
//! | `meta.json` | `format` = `"idtaxa-decipher-export"`, `version`, `K`, `taxonomy`, `taxa`, `ranks` (or null), `levels`, `parents` (1-based, Root = 0), `fraction` (null = NA), `crossIndex` (1-based), `children` (list of 1-based vectors), `problemSequences` (`Index`, `Expected`, `Predicted`), `problemGroups`, `names` (or null), `alphabet` (must be null: nucleotide models only), `decipher_version` |
//! | `kmer_lengths.bin` | int32 LE, number of k-mers per training sequence |
//! | `kmers.bin` | int32 LE, concatenated sorted 1-based k-mer codes |
//! | `idf.bin` | float64 LE, `IDFweights` (length 4^K) |
//! | `decision_lengths.bin` | int32 LE per node, number of decision k-mers (0 = NULL) |
//! | `decision_kmers.bin` | int32 LE, concatenated 1-based decision k-mers |
//! | `decision_weights.bin` | float64 LE, per node the children × k-mers matrix in row-major order |

use std::fs;
use std::path::Path;

use serde::Deserialize;

use crate::error::{Error, Result};
use crate::model::{Model, ProblemSequence, RawModel, TrainParams};
use crate::taxonomy::NO_PARENT;

#[derive(Deserialize)]
#[allow(non_snake_case)]
struct ProblemSeqs {
    #[serde(default)]
    Index: Vec<u32>,
    #[serde(default)]
    Expected: Vec<String>,
    #[serde(default)]
    Predicted: Vec<String>,
}

#[derive(Deserialize)]
#[allow(non_snake_case)]
struct ExportMeta {
    format: String,
    version: u32,
    K: u32,
    taxonomy: Vec<String>,
    taxa: Vec<String>,
    ranks: Option<Vec<String>>,
    levels: Vec<u32>,
    parents: Vec<u32>,
    fraction: Vec<Option<f64>>,
    crossIndex: Vec<u32>,
    children: Vec<Vec<u32>>,
    problemSequences: ProblemSeqs,
    problemGroups: Vec<String>,
    names: Option<Vec<Option<String>>>,
    alphabet: Option<serde_json::Value>,
    #[serde(default)]
    decipher_version: Option<String>,
}

fn read_i32(path: &Path) -> Result<Vec<i32>> {
    let b = fs::read(path).map_err(|e| Error::io(path, e))?;
    if b.len() % 4 != 0 {
        return Err(Error::Import(format!(
            "{} is not a multiple of 4 bytes",
            path.display()
        )));
    }
    Ok(b.chunks_exact(4)
        .map(|c| i32::from_le_bytes(c.try_into().unwrap()))
        .collect())
}

fn read_f64(path: &Path) -> Result<Vec<f64>> {
    let b = fs::read(path).map_err(|e| Error::io(path, e))?;
    if b.len() % 8 != 0 {
        return Err(Error::Import(format!(
            "{} is not a multiple of 8 bytes",
            path.display()
        )));
    }
    Ok(b.chunks_exact(8)
        .map(|c| f64::from_le_bytes(c.try_into().unwrap()))
        .collect())
}

/// Read a DECIPHER export directory into [`RawModel`] (0-based indices).
pub fn read_decipher_export(dir: &Path) -> Result<RawModel> {
    let meta_path = dir.join("meta.json");
    let meta_bytes = fs::read(&meta_path).map_err(|e| Error::io(&meta_path, e))?;
    let m: ExportMeta = serde_json::from_slice(&meta_bytes)?;
    if m.format != "idtaxa-decipher-export" {
        return Err(Error::Import(format!("unexpected format '{}'", m.format)));
    }
    if m.version != 1 {
        return Err(Error::Import(format!(
            "unsupported export version {}",
            m.version
        )));
    }
    if m.alphabet.as_ref().is_some_and(|a| !a.is_null()) {
        return Err(Error::Import(
            "amino-acid (reduced alphabet) models are not supported; nucleotide models only".into(),
        ));
    }
    let n = m.taxa.len();
    if m.taxonomy.len() != n
        || m.parents.len() != n
        || m.fraction.len() != n
        || m.children.len() != n
        || m.levels.len() != n
    {
        return Err(Error::Import(
            "per-node vectors have inconsistent lengths".into(),
        ));
    }
    if m.taxonomy.first().map(String::as_str) != Some("Root;") {
        return Err(Error::Import("first taxonomy entry must be 'Root;'".into()));
    }
    let parents: Vec<u32> = m
        .parents
        .iter()
        .map(|&p| if p == 0 { NO_PARENT } else { p - 1 })
        .collect();
    let children: Vec<Vec<u32>> = m
        .children
        .iter()
        .map(|c| c.iter().map(|&x| x - 1).collect())
        .collect();
    // consistency of children vs parents
    for (i, ch) in children.iter().enumerate() {
        for &c in ch {
            if c as usize >= n || parents[c as usize] != i as u32 {
                return Err(Error::Import(format!(
                    "children/parents mismatch at node {}",
                    i + 1
                )));
            }
        }
    }
    let k = m.K;
    let n_kmers: i64 = 1i64 << (2 * k);

    let lens = read_i32(&dir.join("kmer_lengths.bin"))?;
    let flat = read_i32(&dir.join("kmers.bin"))?;
    if lens.len() != m.crossIndex.len() {
        return Err(Error::Import(
            "kmer_lengths.bin and crossIndex differ in length".into(),
        ));
    }
    if lens.iter().map(|&x| x as i64).sum::<i64>() != flat.len() as i64 {
        return Err(Error::Import(
            "kmers.bin length does not match kmer_lengths.bin".into(),
        ));
    }
    let mut seq_kmers = Vec::with_capacity(lens.len());
    let mut pos = 0usize;
    for &l in &lens {
        let v: Vec<u32> = flat[pos..pos + l as usize]
            .iter()
            .map(|&x| {
                if x < 1 || x as i64 > n_kmers {
                    Err(Error::Import(format!(
                        "k-mer code {x} out of range for K = {k}"
                    )))
                } else {
                    Ok((x - 1) as u32)
                }
            })
            .collect::<Result<_>>()?;
        pos += l as usize;
        seq_kmers.push(v);
    }

    let idf = read_f64(&dir.join("idf.bin"))?;
    let dl = read_i32(&dir.join("decision_lengths.bin"))?;
    let dkm = read_i32(&dir.join("decision_kmers.bin"))?;
    let dw = read_f64(&dir.join("decision_weights.bin"))?;
    if dl.len() != n {
        return Err(Error::Import(
            "decision_lengths.bin must have one entry per node".into(),
        ));
    }
    let mut decision = Vec::with_capacity(n);
    let (mut kp, mut wp) = (0usize, 0usize);
    for (i, &len) in dl.iter().enumerate() {
        let len = len as usize;
        if len == 0 {
            decision.push(None);
            continue;
        }
        let nc = children[i].len();
        if kp + len > dkm.len() || wp + len * nc > dw.len() {
            return Err(Error::Import("decision k-mer files are truncated".into()));
        }
        let km: Vec<u32> = dkm[kp..kp + len].iter().map(|&x| (x - 1) as u32).collect();
        let w = dw[wp..wp + len * nc].to_vec();
        kp += len;
        wp += len * nc;
        decision.push(Some((km, w)));
    }
    if kp != dkm.len() || wp != dw.len() {
        return Err(Error::Import(
            "decision k-mer files have trailing data".into(),
        ));
    }

    let fraction: Vec<f64> = m.fraction.iter().map(|f| f.unwrap_or(f64::NAN)).collect();
    let cross_index: Vec<u32> = m.crossIndex.iter().map(|&c| c - 1).collect();
    let names: Vec<String> = match m.names {
        Some(v) if v.len() == lens.len() => v
            .into_iter()
            .enumerate()
            .map(|(i, s)| s.unwrap_or_else(|| format!("train_{}", i + 1)))
            .collect(),
        _ => (1..=lens.len()).map(|i| format!("train_{i}")).collect(),
    };
    let ps = &m.problemSequences;
    let problem_sequences = (0..ps.Index.len())
        .map(|i| ProblemSequence {
            index: ps.Index[i] - 1,
            expected: ps.Expected.get(i).cloned().unwrap_or_default(),
            predicted: ps.Predicted.get(i).cloned().unwrap_or_default(),
        })
        .collect();

    // reference checksum over the imported content
    let mut h = xxhash_rust::xxh3::Xxh3::new();
    h.update(&meta_bytes);
    h.update(bytemuck::cast_slice(&flat));
    Ok(RawModel {
        k,
        seed_pattern: None,
        taxa: m.taxa,
        parents,
        children,
        ranks: m.ranks,
        fraction,
        seq_kmers,
        seq_names: names,
        cross_index,
        decision,
        idf_dense: Some(idf),
        problem_sequences,
        problem_groups: m.problemGroups,
        train_params: TrainParams {
            source: format!(
                "DECIPHER {} (import-decipher)",
                m.decipher_version.unwrap_or_else(|| "unknown".into())
            ),
            k_auto: false,
            ..TrainParams::default()
        },
        reference_checksum: format!("{:032x}", h.digest128()),
    })
}

/// Import a DECIPHER export directory and build a native model.
pub fn import_decipher(dir: &Path) -> Result<Model> {
    let raw = read_decipher_export(dir)?;
    let tax_check = crate::taxonomy::full_paths(&raw.taxa, &raw.parents);
    let meta_path = dir.join("meta.json");
    let meta: serde_json::Value =
        serde_json::from_slice(&fs::read(&meta_path).map_err(|e| Error::io(&meta_path, e))?)?;
    if let Some(t) = meta.get("taxonomy").and_then(|t| t.as_array()) {
        for (i, v) in t.iter().enumerate() {
            if v.as_str() != Some(tax_check[i].as_str()) {
                return Err(Error::Import(format!(
                    "taxonomy[{}] = {:?} does not match taxa/parents ({})",
                    i + 1,
                    v,
                    tax_check[i]
                )));
            }
        }
    }
    Model::from_raw(raw)
}
