//! `idtaxa` command-line interface.

#![allow(clippy::needless_range_loop)]

mod cache;
mod classify_cmd;
mod tools;

use std::path::PathBuf;

use anyhow::{bail, Context, Result};
use clap::{Args, Parser, Subcommand};
use idtaxa_core::classify::minimum_sample_size;
use idtaxa_core::output::Format;
use idtaxa_core::train::{train, TrainOptions};
use idtaxa_core::{ClassifyParams, Model, Strand, FORMAT_VERSION};

fn long_version() -> &'static str {
    Box::leak(
        format!(
            "{} (native model format version {})",
            env!("CARGO_PKG_VERSION"),
            FORMAT_VERSION
        )
        .into_boxed_str(),
    )
}

#[derive(Parser)]
#[command(
    name = "idtaxa",
    version = long_version(),
    about = "Scalable Rust reimplementation of the IDTAXA taxonomic classifier (DECIPHER LearnTaxa/IdTaxa)"
)]
struct Cli {
    /// Number of worker threads (default: all cores)
    #[arg(long, global = true)]
    threads: Option<usize>,
    /// Suppress progress messages
    #[arg(long, short, global = true)]
    quiet: bool,
    #[command(subcommand)]
    cmd: Cmd,
}

#[derive(Subcommand)]
enum Cmd {
    /// Import a DECIPHER training set exported by scripts/export_decipher_model.R
    ImportDecipher {
        #[arg(long)]
        input: PathBuf,
        #[arg(long)]
        output: PathBuf,
    },
    /// Train a model (equivalent to DECIPHER LearnTaxa)
    Train(TrainArgs),
    /// Classify sequences (equivalent to DECIPHER IdTaxa)
    Classify(ClassifyArgs),
    /// Merge outputs of `classify --chunk i/n` runs
    Merge {
        #[arg(long, num_args = 1.., required = true)]
        inputs: Vec<PathBuf>,
        #[arg(long)]
        output: PathBuf,
    },
    /// Evaluate a model on held-out sequences
    Validate(tools::ValidateArgs),
    /// Fit per-rank confidence calibration and thresholds from held-out data
    Calibrate(tools::CalibrateArgs),
    /// Split a reference into training and held-out sets (random or clade exclusion)
    Split(tools::SplitArgs),
    /// Print model information
    Info {
        #[arg(long)]
        model: PathBuf,
        /// Also print per-rank group counts
        #[arg(long)]
        detailed: bool,
    },
    /// Verify that a model's stored k-mers equal those computed from its reference
    KmerCheck {
        #[arg(long)]
        model: PathBuf,
        #[arg(long = "ref")]
        reference: PathBuf,
    },
    /// Compare two models built from the same reference (e.g. native vs DECIPHER-trained)
    CompareModels {
        #[arg(long)]
        a: PathBuf,
        #[arg(long)]
        b: PathBuf,
    },
    /// In-silico PCR: extract the amplicon region from reference sequences
    ExtractRegion(tools::ExtractArgs),
    /// Clean reference taxonomy labels and flag problem sequences
    CleanRef(tools::CleanArgs),
    /// Consensus of IDTAXA and phylogenetic placement (gappa) assignments
    Consensus(tools::ConsensusArgs),
}

#[derive(Args)]
struct TrainArgs {
    /// Reference FASTA (optionally gzip-compressed)
    #[arg(long = "ref")]
    reference: PathBuf,
    /// Taxonomy TSV (id<TAB>lineage); if omitted, lineages are read from FASTA headers
    #[arg(long)]
    taxonomy: Option<PathBuf>,
    #[arg(long)]
    output: PathBuf,
    /// K-mer length (default: DECIPHER's automatic choice)
    #[arg(long)]
    k: Option<u32>,
    #[arg(long, default_value_t = 10)]
    max_iterations: u32,
    #[arg(long, default_value_t = 0.01)]
    min_fraction: f64,
    #[arg(long, default_value_t = 0.06)]
    max_fraction: f64,
    #[arg(long, default_value_t = 100.0)]
    multiplier: f64,
    #[arg(long, default_value_t = 200)]
    max_children: u32,
    /// DECIPHER N used for the automatic K
    #[arg(long, default_value_t = 500.0)]
    n: f64,
    /// Comma-separated rank names for the levels below Root
    #[arg(long, value_delimiter = ',')]
    rank_names: Option<Vec<String>>,
    #[arg(long, default_value_t = 42)]
    seed: u64,
    /// Training mode: standard | balanced | spaced
    #[arg(long, default_value = "standard")]
    mode: String,
    /// Spaced-seed pattern for --mode spaced (default 11011011011011011, weight from K)
    #[arg(long)]
    seed_pattern: Option<String>,
    /// Maximum training sequences per leaf group for --mode balanced
    #[arg(long, default_value_t = 20)]
    max_group_size: u32,
}

#[derive(Args)]
struct ClassifyArgs {
    #[arg(long)]
    model: PathBuf,
    /// FASTA/FASTQ input (gzip ok; '-' = stdin)
    #[arg(long)]
    input: PathBuf,
    /// Output TSV ('-' = stdout)
    #[arg(long)]
    output: PathBuf,
    #[arg(long, default_value_t = 60.0)]
    threshold: f64,
    #[arg(long, default_value_t = 100)]
    bootstraps: u32,
    #[arg(long, default_value = "both")]
    strand: String,
    #[arg(long, default_value_t = 42)]
    seed: u64,
    #[arg(long, default_value_t = 50_000)]
    batch_size: usize,
    /// Persistent result cache (redb file)
    #[arg(long)]
    cache: Option<PathBuf>,
    /// Process only shard i of n (1-based), e.g. 3/10
    #[arg(long)]
    chunk: Option<String>,
    /// Also write the full lineage with all confidences
    #[arg(long)]
    full_confidences: bool,
    /// Output format: tsv | decipher (IdTaxa type="collapsed" strings)
    #[arg(long, default_value = "tsv")]
    format: String,
    /// DECIPHER minDescend
    #[arg(long, default_value_t = 0.98)]
    min_descend: f64,
    /// Exponent of samples = L^x
    #[arg(long, default_value_t = 0.47)]
    sample_exponent: f64,
    /// DECIPHER fullLength as a fold difference (e.g. 1.5); 0 = off
    #[arg(long, default_value_t = 0.0)]
    full_length: f64,
    /// Calibration file from `idtaxa calibrate`
    #[arg(long)]
    calibration: Option<PathBuf>,
    /// Report similarity and novelty rank
    #[arg(long)]
    novelty_flag: bool,
    /// Truncate assignments below the novelty rank
    #[arg(long)]
    truncate_novel: bool,
    /// Rerun queries near a threshold with this many replicates
    #[arg(long)]
    adaptive_bootstraps: Option<u32>,
    #[arg(long, default_value_t = 10.0)]
    adaptive_window: f64,
    /// standard | balanced (group-size normalised scoring)
    #[arg(long, default_value = "standard")]
    mode: String,
    /// Exponent for --mode balanced scoring normalisation
    #[arg(long, default_value_t = 0.1)]
    balance_exponent: f64,
}

#[allow(clippy::too_many_arguments)]
pub fn classify_params_from(
    threshold: f64,
    bootstraps: u32,
    strand: &str,
    seed: u64,
    min_descend: f64,
    sample_exponent: f64,
    full_length: f64,
    mode: &str,
    balance_exponent: f64,
) -> Result<ClassifyParams> {
    let full_length = if full_length == 0.0 {
        None
    } else if full_length >= 1.0 {
        Some((1.0 / full_length, full_length))
    } else {
        bail!("--full-length must be 0 (off) or a fold difference >= 1");
    };
    let group_size_exponent = match mode {
        "standard" | "spaced" => 0.0,
        "balanced" => balance_exponent,
        _ => bail!("--mode must be standard, balanced or spaced"),
    };
    let p = ClassifyParams {
        threshold,
        bootstraps,
        strand: strand.parse::<Strand>()?,
        min_descend,
        sample_exponent,
        full_length,
        seed,
        group_size_exponent,
    };
    p.validate()?;
    Ok(p)
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    if let Some(t) = cli.threads {
        rayon::ThreadPoolBuilder::new()
            .num_threads(t)
            .build_global()
            .context("configuring thread pool")?;
    }
    let quiet = cli.quiet;
    match cli.cmd {
        Cmd::ImportDecipher { input, output } => {
            let model = idtaxa_core::import::import_decipher(&input)
                .with_context(|| format!("importing {}", input.display()))?;
            model.save(&output)?;
            if !quiet {
                eprintln!(
                    "imported {} training sequences, {} taxa (K = {}) -> {}",
                    model.n_seqs(),
                    model.n_nodes(),
                    model.k(),
                    output.display()
                );
            }
        }
        Cmd::Train(a) => run_train(a, quiet)?,
        Cmd::Classify(a) => {
            let params = classify_params_from(
                a.threshold,
                a.bootstraps,
                &a.strand,
                a.seed,
                a.min_descend,
                a.sample_exponent,
                a.full_length,
                &a.mode,
                a.balance_exponent,
            )?;
            let format = match a.format.as_str() {
                "tsv" => Format::Tsv,
                "decipher" => Format::Decipher,
                f => bail!("unknown --format {f} (tsv or decipher)"),
            };
            if a.batch_size == 0 {
                bail!("--batch-size must be positive");
            }
            classify_cmd::run(classify_cmd::ClassifyOptions {
                model: a.model,
                input: a.input,
                output: a.output,
                params,
                batch_size: a.batch_size,
                cache: a.cache,
                chunk: a
                    .chunk
                    .as_deref()
                    .map(classify_cmd::parse_chunk)
                    .transpose()?,
                full_confidences: a.full_confidences,
                format,
                calibration: a.calibration,
                novelty_flag: a.novelty_flag || a.truncate_novel,
                truncate_novel: a.truncate_novel,
                adaptive_bootstraps: a.adaptive_bootstraps,
                adaptive_window: a.adaptive_window,
                quiet,
            })?;
        }
        Cmd::Merge { inputs, output } => classify_cmd::merge(&inputs, &output)?,
        Cmd::Validate(a) => tools::validate(a, quiet)?,
        Cmd::Calibrate(a) => tools::calibrate(a, quiet)?,
        Cmd::Split(a) => tools::split(a, quiet)?,
        Cmd::Info { model, detailed } => info(&model, detailed)?,
        Cmd::KmerCheck { model, reference } => kmer_check(&model, &reference)?,
        Cmd::CompareModels { a, b } => {
            let (ma, mb) = (Model::load(&a)?, Model::load(&b)?);
            let c = idtaxa_core::model::compare_models(&ma, &mb);
            println!("{}", serde_json::to_string_pretty(&c)?);
        }
        Cmd::ExtractRegion(a) => tools::extract_region(a, quiet)?,
        Cmd::CleanRef(a) => tools::clean_ref(a, quiet)?,
        Cmd::Consensus(a) => tools::consensus(a, quiet)?,
    }
    Ok(())
}

fn run_train(a: TrainArgs, quiet: bool) -> Result<()> {
    let (recs, lineages) = idtaxa_core::io::read_reference(&a.reference, a.taxonomy.as_deref())?;
    let seqs: Vec<&[u8]> = recs.iter().map(|r| r.seq.as_slice()).collect();
    let names: Vec<String> = recs.iter().map(|r| r.id.clone()).collect();
    let mut opts = TrainOptions {
        k: a.k,
        n: a.n,
        min_fraction: a.min_fraction,
        max_fraction: a.max_fraction,
        max_iterations: a.max_iterations,
        multiplier: a.multiplier,
        max_children: a.max_children,
        rank_names: a.rank_names,
        seed: a.seed,
        verbose: !quiet,
        ..Default::default()
    };
    match a.mode.as_str() {
        "standard" => {}
        "balanced" => opts.max_group_size = Some(a.max_group_size),
        "spaced" => {
            let pat = match a.seed_pattern {
                Some(p) => p,
                None => {
                    let k =
                        a.k.unwrap_or_else(|| idtaxa_core::train::auto_k(&seqs, a.n).0);
                    default_spaced_pattern(k as usize)
                }
            };
            opts.seed_pattern = Some(pat);
        }
        m => bail!("unknown --mode {m} (standard, balanced or spaced)"),
    }
    let (model, report) = train(&seqs, &lineages, &names, &opts)?;
    model.save(&a.output)?;
    if !quiet {
        eprintln!(
            "trained model: K = {}{}, {} sequences, {} taxa, {} iterations, {} problem sequences, {} problem groups{} -> {}",
            report.k,
            if report.k_clamped { " (clamped; K may be smaller than ideal)" } else { "" },
            model.n_seqs(),
            model.n_nodes(),
            report.iterations_run,
            report.n_problem_sequences,
            report.n_problem_groups,
            if report.dropped_by_group_cap > 0 {
                format!(", {} sequences dropped by group cap", report.dropped_by_group_cap)
            } else {
                String::new()
            },
            a.output.display()
        );
    }
    Ok(())
}

/// Default spaced seed of weight k: repeat "110" and trim to end on '1'.
pub fn default_spaced_pattern(k: usize) -> String {
    let mut s = String::new();
    let mut w = 0;
    let mut i = 0;
    while w < k {
        if i % 3 == 2 {
            s.push('0');
        } else {
            s.push('1');
            w += 1;
        }
        i += 1;
    }
    s
}

fn info(path: &std::path::Path, detailed: bool) -> Result<()> {
    let m = Model::load(path)?;
    let size = std::fs::metadata(path)?.len();
    println!("file\t{}", path.display());
    println!("file_size_bytes\t{size}");
    println!("format_version\t{}", m.meta.format_version);
    println!("built_with\tidtaxa-rs {}", m.meta.tool_version);
    println!("model_id\t{}", m.meta.model_id);
    println!("reference_checksum\t{}", m.meta.reference_checksum);
    println!("source\t{}", m.meta.train_params.source);
    println!("K\t{}", m.k());
    println!(
        "seed_pattern\t{}",
        m.meta.seed_pattern.as_deref().unwrap_or("contiguous")
    );
    println!("training_sequences\t{}", m.n_seqs());
    println!("taxa\t{}", m.n_nodes());
    println!("groups\t{}", m.leaf_groups());
    println!(
        "rank_levels\t{}",
        m.taxonomy
            .iter()
            .map(|t| t.matches(';').count())
            .max()
            .unwrap_or(0)
    );
    println!("indexed_kmers\t{}", m.n_index_kmers());
    println!("postings\t{}", m.n_postings());
    println!("problem_groups\t{}", m.meta.problem_groups.len());
    println!("problem_sequences\t{}", m.meta.problem_sequences.len());
    println!("min_sample_size\t{}", minimum_sample_size(&m));
    println!(
        "train_params\t{}",
        serde_json::to_string(&m.meta.train_params)?
    );
    if detailed {
        let mut by_level: std::collections::BTreeMap<u32, usize> = Default::default();
        for i in 0..m.n_nodes() as u32 {
            *by_level.entry(m.level(i)).or_default() += 1;
        }
        for (lv, n) in by_level {
            let rank = (0..m.n_nodes() as u32)
                .find(|&i| m.level(i) == lv)
                .and_then(|i| m.rank(i))
                .unwrap_or("-");
            println!("level_{}\t{}\t{}", lv - 1, rank, n);
        }
    }
    Ok(())
}

fn kmer_check(model: &std::path::Path, reference: &std::path::Path) -> Result<()> {
    let m = Model::load(model)?;
    let recs = idtaxa_core::io::read_all(reference)?;
    let enc = m.encoder();
    let o2n = m.orig_to_native();
    let by_name: std::collections::HashMap<&str, usize> = m
        .meta
        .seq_names
        .iter()
        .enumerate()
        .map(|(i, n)| (n.as_str(), i))
        .collect();
    let (mut checked, mut mismatched, mut missing) = (0, 0, 0);
    for r in &recs {
        let Some(&orig) = by_name.get(r.id.as_str()) else {
            missing += 1;
            continue;
        };
        checked += 1;
        let ours = enc.encode(&r.seq).unique;
        if ours.as_slice() != m.seq_kmers(o2n[orig]) {
            mismatched += 1;
            if mismatched <= 5 {
                eprintln!("mismatch: {}", r.id);
            }
        }
    }
    println!("checked\t{checked}\nmismatched\t{mismatched}\nnot_in_model\t{missing}");
    if mismatched > 0 {
        bail!("{mismatched} sequences have k-mer sets that differ from the model");
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    #[test]
    fn spaced_default() {
        assert_eq!(super::default_spaced_pattern(4), "11011");
        assert_eq!(super::default_spaced_pattern(5), "1101101");
    }
}
