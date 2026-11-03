//! `idtaxa classify`: streaming, dereplicated, parallel, cached, sharded.

use std::collections::HashMap;
use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};
use std::time::Instant;

use anyhow::{bail, Context, Result};
use idtaxa_core::calibration::Calibration;
use idtaxa_core::classify::query_seed;
use idtaxa_core::io::{SeqReader, SeqRecord};
use idtaxa_core::kmer::normalize_sequence;
use idtaxa_core::output::{self, Columns, Format, ResultRow};
use idtaxa_core::{Classification, Classifier, ClassifyParams, Model};
use rayon::prelude::*;

use crate::cache::Cache;

/// Seed perturbation for adaptive reruns (an independent stream).
const ADAPTIVE_SEED_XOR: u64 = 0x9e37_79b9_7f4a_7c15;

/// Options of the classify command (already parsed).
pub struct ClassifyOptions {
    pub model: PathBuf,
    pub input: PathBuf,
    pub output: PathBuf,
    pub params: ClassifyParams,
    pub batch_size: usize,
    pub cache: Option<PathBuf>,
    pub chunk: Option<(usize, usize)>,
    pub full_confidences: bool,
    pub format: Format,
    pub calibration: Option<PathBuf>,
    pub novelty_flag: bool,
    pub truncate_novel: bool,
    pub adaptive_bootstraps: Option<u32>,
    pub adaptive_window: f64,
    pub quiet: bool,
}

/// Parse `i/n` (1-based shard index).
pub fn parse_chunk(s: &str) -> Result<(usize, usize)> {
    let (a, b) = s
        .split_once('/')
        .context("--chunk must look like i/n, e.g. 3/10")?;
    let i: usize = a.trim().parse().context("invalid chunk index")?;
    let n: usize = b.trim().parse().context("invalid chunk count")?;
    if n == 0 || i == 0 || i > n {
        bail!("--chunk i/n requires 1 <= i <= n (got {s})");
    }
    Ok((i, n))
}

/// Cached/computed per-unique-sequence result.
#[derive(Clone, serde::Serialize, serde::Deserialize)]
pub struct Computed {
    pub cls: Classification,
    pub sim: Option<idtaxa_core::classify::SimilarityProfile>,
    pub rerun: bool,
}

pub struct Engine<'m> {
    pub classifier: Classifier<'m>,
    pub calibration: Option<Calibration>,
    pub thresholds: Vec<f64>,
    pub novelty: bool,
    pub truncate_novel: bool,
    pub adaptive: Option<(u32, f64)>,
}

impl Engine<'_> {
    pub fn compute(&self, norm: &[u8], sc: &mut idtaxa_core::Scratch) -> Computed {
        let seed = query_seed(norm, self.classifier.params().seed);
        let mut cls = self.classifier.classify_seeded(norm, seed, sc);
        let mut rerun = false;
        if let Some((n2, window)) = self.adaptive {
            let near = cls.confidences.iter().enumerate().skip(1).any(|(d, &c)| {
                let t = self
                    .thresholds
                    .get(d - 1)
                    .copied()
                    .unwrap_or(self.classifier.params().threshold);
                (c - t).abs() <= window
            });
            if near && cls.bootstraps_used > 0 {
                cls = self.classifier.classify_with_bootstraps(
                    norm,
                    seed ^ ADAPTIVE_SEED_XOR,
                    n2,
                    sc,
                );
                rerun = true;
            }
        }
        let sim = if self.novelty {
            Some(self.classifier.similarity_profile(norm, &cls, sc))
        } else {
            None
        };
        Computed { cls, sim, rerun }
    }

    pub fn row(&self, c: &Computed) -> ResultRow {
        let calibrated = self.calibration.as_ref().map(|cal| cal.calibrate(&c.cls));
        let mut n = c.cls.n_above_per_rank(&c.cls.confidences, &self.thresholds);
        let novelty_depth = match (&self.calibration, &c.sim) {
            (Some(cal), Some(s)) => cal.novelty_depth(s),
            _ => None,
        };
        if self.truncate_novel {
            if let Some(d) = novelty_depth {
                n = n.min(d + 1);
            }
        }
        ResultRow {
            cls: c.cls.clone(),
            calibrated,
            similarity: c.sim.clone(),
            novelty_depth,
            n_assigned: n,
        }
    }
}

pub fn run(opts: ClassifyOptions) -> Result<()> {
    let t0 = Instant::now();
    let model = Model::load(&opts.model)
        .with_context(|| format!("loading model {}", opts.model.display()))?;
    let classifier = Classifier::new(&model, opts.params.clone())?;
    let calibration = match &opts.calibration {
        Some(p) => {
            let c = Calibration::load(p)?;
            c.check_model(&model.meta.model_id)?;
            Some(c)
        }
        None => None,
    };
    let max_depth = model
        .taxonomy
        .iter()
        .map(|t| t.matches(';').count())
        .max()
        .unwrap_or(1);
    let thresholds = match &calibration {
        Some(c) => c.thresholds(max_depth, opts.params.threshold),
        None => vec![opts.params.threshold; max_depth],
    };
    if opts.novelty_flag && calibration.as_ref().is_none_or(|c| c.novelty.is_none()) && !opts.quiet
    {
        eprintln!("[classify] note: --novelty-flag without novelty cut-offs in --calibration reports max_similarity only");
    }
    let engine = Engine {
        classifier,
        calibration,
        thresholds,
        novelty: opts.novelty_flag,
        truncate_novel: opts.truncate_novel,
        adaptive: opts.adaptive_bootstraps.map(|n| (n, opts.adaptive_window)),
    };
    let cols = Columns {
        full_confidences: opts.full_confidences,
        calibrated: engine.calibration.is_some(),
        novelty: opts.novelty_flag,
    };
    let cache = match &opts.cache {
        Some(p) => Some(Cache::open(
            p,
            &model.meta.model_id,
            &cache_fingerprint(&opts),
        )?),
        None => None,
    };

    let out: Box<dyn Write> = if opts.output.as_os_str() == "-" {
        Box::new(std::io::stdout().lock())
    } else {
        Box::new(
            File::create(&opts.output)
                .with_context(|| format!("creating {}", opts.output.display()))?,
        )
    };
    let mut out = BufWriter::with_capacity(1 << 20, out);
    if let Some((i, n)) = opts.chunk {
        writeln!(
            out,
            "#idtaxa-rs chunk={i}/{n} model={}",
            model.meta.model_id
        )?;
    }
    if opts.format == Format::Tsv {
        writeln!(out, "{}", output::header(&cols))?;
    }

    let mut reader = SeqReader::open(&opts.input)?;
    let mut record_index: usize = 0;
    let (mut n_total, mut n_unique_total, mut n_cached, mut n_rerun) =
        (0usize, 0usize, 0usize, 0usize);
    let mut batch: Vec<SeqRecord> = Vec::with_capacity(opts.batch_size);
    loop {
        batch.clear();
        // fill a batch with this shard's records
        let mut more = true;
        while batch.len() < opts.batch_size {
            match reader.next() {
                Some(r) => {
                    let r = r?;
                    let take = match opts.chunk {
                        Some((i, n)) => record_index % n == i - 1,
                        None => true,
                    };
                    record_index += 1;
                    if take {
                        batch.push(r);
                    }
                }
                None => {
                    more = false;
                    break;
                }
            }
        }
        if batch.is_empty() && !more {
            break;
        }
        // dereplicate
        let norms: Vec<Vec<u8>> = batch
            .par_iter()
            .map(|r| normalize_sequence(&r.seq))
            .collect();
        let mut first: HashMap<&[u8], usize> = HashMap::with_capacity(norms.len());
        let mut map = Vec::with_capacity(norms.len());
        let mut uniq: Vec<usize> = Vec::new();
        for (i, s) in norms.iter().enumerate() {
            let u = *first.entry(s.as_slice()).or_insert_with(|| {
                uniq.push(i);
                uniq.len() - 1
            });
            map.push(u);
        }
        let hashes: Vec<u64> = uniq
            .par_iter()
            .map(|&i| xxhash_rust::xxh3::xxh3_64(&norms[i]))
            .collect();
        // cache lookup
        let mut results: Vec<Option<Computed>> = match &cache {
            Some(c) => c.get_many(uniq.iter().map(|&i| norms[i].as_slice()))?,
            None => vec![None; uniq.len()],
        };
        n_cached += results.iter().filter(|r| r.is_some()).count();
        let todo: Vec<usize> = (0..uniq.len()).filter(|&u| results[u].is_none()).collect();
        let computed: Vec<(usize, Computed)> = todo
            .par_iter()
            .map_init(
                || engine.classifier.new_scratch(),
                |sc, &u| (u, engine.compute(&norms[uniq[u]], sc)),
            )
            .collect();
        if let Some(c) = &cache {
            c.put_many(
                computed
                    .iter()
                    .map(|(u, r)| (norms[uniq[*u]].as_slice(), r)),
            )?;
        }
        for (u, r) in computed {
            results[u] = Some(r);
        }
        n_rerun += results.iter().flatten().filter(|r| r.rerun).count();
        let rows: Vec<ResultRow> = results
            .iter()
            .map(|r| engine.row(r.as_ref().unwrap()))
            .collect();
        for (i, rec) in batch.iter().enumerate() {
            let u = map[i];
            match opts.format {
                Format::Tsv => writeln!(
                    out,
                    "{}",
                    output::tsv_row(&model, &rec.id, hashes[u], &rows[u], &cols)
                )?,
                Format::Decipher => writeln!(
                    out,
                    "{}\t{}",
                    rec.id,
                    output::decipher_collapsed(&model, &rows[u])
                )?,
            }
        }
        n_total += batch.len();
        n_unique_total += uniq.len();
        if !opts.quiet {
            eprintln!(
                "[classify] {n_total} sequences ({n_unique_total} unique in batches, {n_cached} cached) in {:.1}s",
                t0.elapsed().as_secs_f64()
            );
        }
        if !more {
            break;
        }
    }
    out.flush()?;
    if !opts.quiet {
        let secs = t0.elapsed().as_secs_f64();
        eprintln!(
            "[classify] done: {n_total} sequences, {n_unique_total} classified/looked up, {n_cached} from cache, {n_rerun} adaptive reruns, {secs:.1}s ({:.0} seq/s)",
            n_total as f64 / secs.max(1e-9)
        );
    }
    Ok(())
}

fn cache_fingerprint(opts: &ClassifyOptions) -> String {
    format!(
        "{};adaptive={:?}/{};novelty={}",
        opts.params.fingerprint(),
        opts.adaptive_bootstraps,
        opts.adaptive_window,
        opts.novelty_flag
    )
}

/// `idtaxa merge`: combine chunk outputs. Chunked files (with a
/// `#idtaxa-rs chunk=i/n` line) are interleaved back into input order;
/// other files are concatenated.
pub fn merge(inputs: &[PathBuf], output: &Path) -> Result<()> {
    use std::io::BufRead;
    struct Part {
        chunk: Option<(usize, usize)>,
        model: Option<String>,
        header: Option<String>,
        lines: std::vec::IntoIter<String>,
    }
    let mut parts = Vec::new();
    for p in inputs {
        let f = File::open(p).with_context(|| format!("opening {}", p.display()))?;
        let mut chunk = None;
        let mut model = None;
        let mut header = None;
        let mut lines = Vec::new();
        for line in std::io::BufReader::new(f).lines() {
            let line = line?;
            if let Some(rest) = line.strip_prefix("#idtaxa-rs ") {
                for kv in rest.split_whitespace() {
                    if let Some(v) = kv.strip_prefix("chunk=") {
                        chunk = Some(parse_chunk(v)?);
                    } else if let Some(v) = kv.strip_prefix("model=") {
                        model = Some(v.to_string());
                    }
                }
            } else if header.is_none() && line.starts_with("id\t") {
                header = Some(line);
            } else if !line.is_empty() {
                lines.push(line);
            }
        }
        parts.push(Part {
            chunk,
            model,
            header,
            lines: lines.into_iter(),
        });
    }
    if parts.is_empty() {
        bail!("no inputs");
    }
    let h0 = parts[0].header.clone();
    if parts.iter().any(|p| p.header != h0) {
        bail!("inputs have different column headers");
    }
    let m0 = parts[0].model.clone();
    if parts.iter().any(|p| p.model != m0) {
        bail!("inputs were produced with different models");
    }
    let mut out = BufWriter::new(
        File::create(output).with_context(|| format!("creating {}", output.display()))?,
    );
    if let Some(h) = &h0 {
        writeln!(out, "{h}")?;
    }
    let chunked = parts.iter().all(|p| p.chunk.is_some());
    if chunked {
        let n = parts[0].chunk.unwrap().1;
        if parts.iter().any(|p| p.chunk.unwrap().1 != n) {
            bail!("inputs come from different --chunk totals");
        }
        let mut by_index: Vec<Option<Part>> = (0..n).map(|_| None).collect();
        for p in parts {
            let i = p.chunk.unwrap().0;
            if by_index[i - 1].is_some() {
                bail!("chunk {i}/{n} given twice");
            }
            by_index[i - 1] = Some(p);
        }
        if let Some(missing) = by_index.iter().position(|p| p.is_none()) {
            bail!("chunk {}/{n} is missing", missing + 1);
        }
        let mut ps: Vec<Part> = by_index.into_iter().map(|p| p.unwrap()).collect();
        // round-robin: record r went to chunk (r mod n) + 1
        'outer: loop {
            for (k, p) in ps.iter_mut().enumerate() {
                match p.lines.next() {
                    Some(l) => writeln!(out, "{l}")?,
                    None => {
                        // all later chunks must be exhausted too
                        let _ = k;
                        break 'outer;
                    }
                }
            }
        }
        for p in ps.iter_mut() {
            if p.lines.next().is_some() {
                bail!("chunk files have inconsistent record counts");
            }
        }
    } else {
        for p in parts {
            for l in p.lines {
                writeln!(out, "{l}")?;
            }
        }
    }
    out.flush()?;
    Ok(())
}
