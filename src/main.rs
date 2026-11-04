use anyhow::{bail, Context, Result};
use clap::{Args, Parser, Subcommand};
use idtaxa::classify::{classify_seq, write_assignments, ClassifyParams, OutputFormat, Strand};
use idtaxa::curate::find_conflicts;
use idtaxa::evaluate::evaluate;
use idtaxa::model::{Model, ModelMeta};
use idtaxa::primers::{self, AmpliconKind, Pcr, PcrOptions};
use idtaxa::seq::{self, create_output, SeqReader};
use idtaxa::summarize::{self, Level};
use idtaxa::taxonomy::{lineage_from_header, parse_lineage, Lineage};
use idtaxa::train::{train, TrainParams, TrainSeq};
use rand::rngs::SmallRng;
use rand::SeedableRng;
use rayon::prelude::*;
use std::collections::{BTreeMap, HashMap};
use std::io::{BufRead, Write};
use std::path::{Path, PathBuf};
use std::time::Instant;

#[derive(Parser)]
#[command(
    name = "idtaxa",
    version,
    about = "Primer-aware IDTAXA taxonomic classification for amplicon metabarcoding",
    long_about = "Primer-aware IDTAXA taxonomic classification for amplicon metabarcoding.\n\n\
Typical workflow:\n  \
1. idtaxa primers                                    # list built-in primer pairs\n  \
2. idtaxa trim  -i silva.fa --primers 515F-806R ...  # check primer coverage\n  \
3. idtaxa train -r silva.fa --primers 515F-806R -o silva_V4.idtx\n  \
4. idtaxa classify -m silva_V4.idtx -i asvs.fa -o taxonomy.tsv\n  \
5. idtaxa summarize -c taxonomy.tsv --counts table.tsv --rank genus --long ..."
)]
struct Cli {
    /// Worker threads (0 = all cores).
    #[arg(short = 't', long, global = true, default_value_t = 0)]
    threads: usize,
    #[command(subcommand)]
    cmd: Cmd,
}

#[derive(Subcommand)]
enum Cmd {
    /// Train a classifier from reference sequences (optionally trimmed to a primer region).
    Train(TrainArgs),
    /// Classify query sequences (ASVs/OTUs) with a trained model.
    Classify(ClassifyArgs),
    /// In-silico PCR: extract the amplicon region from references and report primer coverage.
    Trim(TrimArgs),
    /// List built-in primer pairs.
    Primers,
    /// Detect which catalog primers are present at the start of reads (e.g. public SRA data).
    DetectPrimers(DetectArgs),
    /// Collapse classified features into taxon tables (wide, or long format for cross-study use).
    Summarize(SummarizeArgs),
    /// Hold-out cross-validation of a reference database / primer combination.
    Evaluate(EvaluateArgs),
    /// Show model metadata as JSON.
    Info {
        /// Model file.
        model: PathBuf,
    },
}

#[derive(Args, Clone)]
struct RefArgs {
    /// Reference FASTA (lineage in header, e.g. SILVA/UNITE/DECIPHER style; may be gzipped).
    #[arg(short, long)]
    reference: PathBuf,
    /// Optional QIIME-style taxonomy TSV (id<TAB>lineage) instead of header lineages.
    #[arg(long)]
    taxonomy: Option<PathBuf>,
    /// Rank names for unprefixed lineage levels, e.g. domain,phylum,class,order,family,genus,species.
    #[arg(long, value_delimiter = ',')]
    ranks: Option<Vec<String>>,
}

#[derive(Args, Clone)]
struct PrimerArgs {
    /// Built-in primer pair (see `idtaxa primers`).
    #[arg(long, conflicts_with_all = ["forward", "reverse"])]
    primers: Option<String>,
    /// Custom forward primer (5'→3', IUPAC codes allowed).
    #[arg(long, requires = "reverse")]
    forward: Option<String>,
    /// Custom reverse primer (5'→3' as ordered, IUPAC codes allowed).
    #[arg(long, requires = "forward")]
    reverse: Option<String>,
    /// Maximum mismatches per primer.
    #[arg(long, default_value_t = 2)]
    max_mismatches: usize,
    /// Keep primer sequences in the amplicon.
    #[arg(long)]
    keep_primers: bool,
    /// Accept references where only one primer site is found.
    #[arg(long)]
    partial: bool,
    /// Minimum amplicon length.
    #[arg(long, default_value_t = 50)]
    min_length: usize,
    /// Maximum amplicon length.
    #[arg(long, default_value_t = 5000)]
    max_length: usize,
}

impl PrimerArgs {
    fn resolve(&self) -> Result<Option<Pcr>> {
        let opts = PcrOptions {
            max_mismatches: self.max_mismatches,
            keep_primers: self.keep_primers,
            partial: self.partial,
            try_revcomp: true,
        };
        if let Some(name) = &self.primers {
            let p = primers::find_pair(name)
                .with_context(|| format!("unknown primer pair '{name}' (see `idtaxa primers`)"))?;
            return Ok(Some(Pcr::from_pair(p, opts)));
        }
        match (&self.forward, &self.reverse) {
            (Some(f), Some(r)) => {
                if !primers::is_valid_primer(f) || !primers::is_valid_primer(r) {
                    bail!("primers must contain only IUPAC nucleotide codes");
                }
                Ok(Some(Pcr::new("custom", f, r, opts)))
            }
            _ => Ok(None),
        }
    }
}

#[derive(Args, Clone)]
struct TrainOpts {
    /// k-mer length (default: floor(log4(100 × 99th-percentile length))).
    #[arg(short = 'k', long)]
    k: Option<usize>,
    /// Target number of decision k-mers per taxonomic node.
    #[arg(long, default_value_t = 1000)]
    decision_kmers: usize,
    /// Keep at most N randomly chosen sequences per taxon (reduces model size).
    #[arg(long)]
    max_per_group: Option<usize>,
    /// Maximum fraction-learning iterations.
    #[arg(long, default_value_t = 10)]
    max_iterations: usize,
    /// Initial k-mer sampling fraction for tree descent.
    #[arg(long, default_value_t = 0.2)]
    max_fraction: f32,
    /// Minimum k-mer sampling fraction.
    #[arg(long, default_value_t = 0.01)]
    min_fraction: f32,
    /// Bootstrap replicates during fraction learning.
    #[arg(long, default_value_t = 20)]
    train_bootstraps: usize,
    /// Lower a node's sampling fraction only if more than this proportion of its sequences are misdirected.
    #[arg(long, default_value_t = 0.01)]
    max_error_rate: f32,
}

impl TrainOpts {
    fn params(&self, seed: u64, min_descend: f32) -> TrainParams {
        TrainParams {
            k: self.k,
            decision_kmers: self.decision_kmers,
            max_fraction: self.max_fraction,
            min_fraction: self.min_fraction,
            max_iterations: self.max_iterations,
            train_bootstraps: self.train_bootstraps,
            max_per_group: self.max_per_group,
            max_error_rate: self.max_error_rate,
            min_descend,
            seed,
            ..Default::default()
        }
    }
}

#[derive(Args, Clone)]
struct ClassifyOpts {
    /// Confidence threshold (percent) for reporting a rank.
    #[arg(long, default_value_t = 60.0)]
    threshold: f32,
    /// Bootstrap replicates.
    #[arg(long, default_value_t = 100)]
    bootstraps: usize,
    /// Minimum bootstrap support (percent) to descend the tree.
    #[arg(long, default_value_t = 98.0)]
    min_descend: f32,
    /// Query orientation.
    #[arg(long, value_enum, default_value_t = Strand::Both)]
    strand: Strand,
    /// Exponent for the number of k-mers sampled per replicate (L^x).
    #[arg(long, default_value_t = 0.47)]
    sample_exponent: f64,
    /// Number of most-similar training sequences considered in the bootstrap.
    #[arg(long, default_value_t = 500)]
    max_candidates: usize,
    /// Narrow candidates with decision-k-mer tree descent (original IDTAXA) instead of nearest-neighbour support.
    #[arg(long)]
    tree_descent: bool,
    /// Fraction of query k-mers sampled per replicate for nearest-neighbour support.
    #[arg(long, default_value_t = 0.2)]
    support_fraction: f32,
}

impl ClassifyOpts {
    fn params(&self, seed: u64) -> ClassifyParams {
        ClassifyParams {
            threshold: self.threshold / 100.0,
            bootstraps: self.bootstraps,
            min_descend: self.min_descend / 100.0,
            sample_exponent: self.sample_exponent,
            strand: self.strand,
            seed,
            max_candidates: self.max_candidates,
            tree_descent: self.tree_descent,
            support_fraction: self.support_fraction,
        }
    }
}

#[derive(Args)]
struct TrainArgs {
    #[command(flatten)]
    reference: RefArgs,
    #[command(flatten)]
    primers: PrimerArgs,
    #[command(flatten)]
    opts: TrainOpts,
    /// Output model file.
    #[arg(short, long)]
    output: PathBuf,
    /// Model name stored in metadata.
    #[arg(long)]
    name: Option<String>,
    /// Leave-one-out check of every reference; write confident label conflicts
    /// (likely mislabelled / contaminated references) to this TSV.
    #[arg(long)]
    problems: Option<PathBuf>,
    /// Remove conflicting references found by the leave-one-out check and retrain.
    #[arg(long)]
    remove_problems: bool,
    /// With --remove-problems, only remove conflicts at this rank or above
    /// (species/genus conflicts often reflect marker resolution, not errors).
    #[arg(long, default_value = "family")]
    remove_rank: String,
    /// Confidence (percent) required for a leave-one-out conflict to count.
    #[arg(long, default_value_t = 80.0)]
    problem_threshold: f32,
    /// Minimum bootstrap support (percent) to descend during training.
    #[arg(long, default_value_t = 98.0)]
    min_descend: f32,
    #[arg(long, default_value_t = 1)]
    seed: u64,
}

#[derive(Args)]
struct ClassifyArgs {
    /// Model produced by `idtaxa train`.
    #[arg(short, long)]
    model: PathBuf,
    /// Query FASTA/FASTQ (may be gzipped).
    #[arg(short, long)]
    input: PathBuf,
    /// Output file (default stdout).
    #[arg(short, long)]
    output: Option<PathBuf>,
    #[arg(long, value_enum, default_value_t = OutputFormat::Tsv)]
    format: OutputFormat,
    #[command(flatten)]
    opts: ClassifyOpts,
    #[arg(long, default_value_t = 1)]
    seed: u64,
}

#[derive(Args)]
struct TrimArgs {
    /// Reference FASTA.
    #[arg(short, long)]
    input: PathBuf,
    #[command(flatten)]
    primers: PrimerArgs,
    /// Output FASTA of amplicons (headers preserved).
    #[arg(short, long)]
    output: Option<PathBuf>,
    /// Write per-taxon primer coverage to this TSV.
    #[arg(long)]
    coverage: Option<PathBuf>,
    /// Rank at which to report coverage (needs rank-prefixed lineages or --ranks).
    #[arg(long, default_value = "phylum")]
    coverage_rank: String,
    /// Rank names for unprefixed lineage levels.
    #[arg(long, value_delimiter = ',')]
    ranks: Option<Vec<String>>,
}

#[derive(Args)]
struct DetectArgs {
    /// Read files (FASTQ/FASTA, may be gzipped), e.g. R1 and R2.
    #[arg(required = true)]
    reads: Vec<PathBuf>,
    /// Number of reads to inspect per file.
    #[arg(short, long, default_value_t = 10_000)]
    n: usize,
    /// Allowed offset of the primer from the read start (heterogeneity spacers).
    #[arg(long, default_value_t = 10)]
    max_offset: usize,
    #[arg(long, default_value_t = 2)]
    max_mismatches: usize,
}

#[derive(Args)]
struct SummarizeArgs {
    /// Classification from `idtaxa classify` (TSV or QIIME format).
    #[arg(short, long)]
    classification: PathBuf,
    /// Feature × sample count table (TSV; BIOM-TSV ok).
    #[arg(long)]
    counts: Option<PathBuf>,
    /// Count table has samples as rows (e.g. DADA2 seqtab).
    #[arg(long)]
    samples_as_rows: bool,
    /// Rank to aggregate at.
    #[arg(long, default_value = "genus", conflicts_with = "level")]
    rank: String,
    /// Aggregate at lineage depth N instead of a named rank.
    #[arg(long)]
    level: Option<usize>,
    /// Output relative abundances (wide format).
    #[arg(long)]
    relative: bool,
    /// Long/tidy output for concatenating studies.
    #[arg(long)]
    long: bool,
    /// Study label for long output.
    #[arg(long, default_value = "study")]
    study: String,
    /// Primer-set label for long output.
    #[arg(long, default_value = "NA")]
    primer_set: String,
    /// Omit the header line in long output (for appending).
    #[arg(long)]
    no_header: bool,
    #[arg(short, long)]
    output: Option<PathBuf>,
}

#[derive(Args)]
struct EvaluateArgs {
    #[command(flatten)]
    reference: RefArgs,
    #[command(flatten)]
    primers: PrimerArgs,
    #[command(flatten)]
    train: TrainOpts,
    #[command(flatten)]
    classify: ClassifyOpts,
    /// Fraction of sequences held out for testing.
    #[arg(long, default_value_t = 0.1)]
    holdout: f64,
    #[arg(short, long)]
    output: Option<PathBuf>,
    #[arg(long, default_value_t = 1)]
    seed: u64,
}

fn main() {
    if let Err(e) = run() {
        let broken_pipe = e.chain().any(|c| {
            c.downcast_ref::<std::io::Error>().map(|io| io.kind() == std::io::ErrorKind::BrokenPipe).unwrap_or(false)
        });
        if broken_pipe {
            std::process::exit(0);
        }
        eprintln!("error: {e:#}");
        std::process::exit(1);
    }
}

fn run() -> Result<()> {
    let cli = Cli::parse();
    if cli.threads > 0 {
        rayon::ThreadPoolBuilder::new().num_threads(cli.threads).build_global()?;
    }
    match cli.cmd {
        Cmd::Train(a) => cmd_train(a),
        Cmd::Classify(a) => cmd_classify(a),
        Cmd::Trim(a) => cmd_trim(a),
        Cmd::Primers => cmd_primers(),
        Cmd::DetectPrimers(a) => cmd_detect(a),
        Cmd::Summarize(a) => cmd_summarize(a),
        Cmd::Evaluate(a) => cmd_evaluate(a),
        Cmd::Info { model } => {
            let m = Model::load(&model)?;
            let mut w = create_output(None)?;
            writeln!(w, "{}", serde_json::to_string_pretty(&info_json(&m))?)?;
            w.flush()?;
            Ok(())
        }
    }
}

fn info_json(m: &Model) -> serde_json::Value {
    let meta = &m.meta;
    serde_json::json!({
        "name": meta.name,
        "tool_version": meta.tool_version,
        "created_unix": meta.created_unix,
        "reference": meta.reference,
        "primer_pair": meta.primer_pair,
        "forward_primer": meta.forward_primer,
        "reverse_primer": meta.reverse_primer,
        "k": meta.k,
        "n_input_sequences": meta.n_input_sequences,
        "n_training_sequences": meta.n_training_sequences,
        "n_groups": meta.n_groups,
        "n_taxa": meta.n_taxa,
        "n_problem_sequences": meta.n_problem_sequences,
        "ranks": meta.ranks,
        "notes": meta.notes,
    })
}

fn read_taxonomy_tsv(path: &Path) -> Result<HashMap<String, String>> {
    let r = seq::open_input(path)?;
    let mut m = HashMap::new();
    for line in r.lines() {
        let line = line?;
        if line.trim().is_empty() || line.starts_with('#') || line.starts_with("Feature ID") {
            continue;
        }
        let mut it = line.splitn(3, '\t');
        if let (Some(id), Some(lin)) = (it.next(), it.next()) {
            m.insert(id.trim().to_string(), lin.trim().to_string());
        }
    }
    Ok(m)
}

#[derive(Default)]
struct LoadStats {
    total: usize,
    no_lineage: usize,
    no_amplicon: usize,
    partial: usize,
    bad_length: usize,
}

/// Load reference sequences with lineages, optionally performing in-silico PCR.
fn load_references(r: &RefArgs, pa: &PrimerArgs, pcr: Option<&Pcr>) -> Result<(Vec<TrainSeq>, LoadStats)> {
    let tax = match &r.taxonomy {
        Some(p) => Some(read_taxonomy_tsv(p)?),
        None => None,
    };
    let ranks = r.ranks.as_deref();
    let mut st = LoadStats::default();
    let mut raw: Vec<(String, Vec<u8>, Lineage)> = Vec::new();
    let mut seen_ids: std::collections::HashSet<String> = std::collections::HashSet::new();
    for rec in SeqReader::from_path(&r.reference)? {
        let rec = rec?;
        st.total += 1;
        let lin_str = match &tax {
            Some(t) => t.get(&rec.id).cloned(),
            None => lineage_from_header(&rec.id, &rec.desc),
        };
        let lin = lin_str.map(|s| parse_lineage(&s, ranks)).unwrap_or_default();
        if lin.is_empty() {
            st.no_lineage += 1;
            continue;
        }
        // DADA2/DECIPHER-style headers are the lineage itself: use the record number as id
        let mut id = if rec.id.contains(';') { format!("record_{}", st.total) } else { rec.id };
        if !seen_ids.insert(id.clone()) {
            id = format!("{id}_record_{}", st.total);
            seen_ids.insert(id.clone());
        }
        raw.push((id, seq::clean(&rec.seq), lin));
    }
    let processed: Vec<Option<(TrainSeq, bool)>> = raw
        .into_par_iter()
        .map(|(id, s, lineage)| match pcr {
            None => Some((TrainSeq { id, seq: s, lineage }, false)),
            Some(p) => p.amplify(&s).map(|a| {
                let partial = a.kind != AmpliconKind::Full;
                (TrainSeq { id, seq: a.seq, lineage }, partial)
            }),
        })
        .collect();
    let mut out = Vec::new();
    for x in processed {
        match x {
            None => st.no_amplicon += 1,
            Some((t, partial)) => {
                if pcr.is_some() && (t.seq.len() < pa.min_length || t.seq.len() > pa.max_length) {
                    st.bad_length += 1;
                    continue;
                }
                if partial {
                    st.partial += 1;
                }
                out.push(t);
            }
        }
    }
    eprintln!(
        "read {} references: {} without lineage{}",
        st.total,
        st.no_lineage,
        if pcr.is_some() {
            format!(
                ", {} without amplicon, {} outside length limits, {} partial amplicons",
                st.no_amplicon, st.bad_length, st.partial
            )
        } else {
            String::new()
        }
    );
    if out.is_empty() {
        bail!("no usable reference sequences (check lineage format / primers)");
    }
    Ok((out, st))
}

fn cmd_train(a: TrainArgs) -> Result<()> {
    let t0 = Instant::now();
    let pcr = a.primers.resolve()?;
    let (seqs, st) = load_references(&a.reference, &a.primers, pcr.as_ref())?;
    let mut meta = ModelMeta {
        name: a
            .name
            .clone()
            .unwrap_or_else(|| a.output.file_stem().map(|s| s.to_string_lossy().to_string()).unwrap_or_default()),
        tool_version: env!("CARGO_PKG_VERSION").to_string(),
        created_unix: std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_secs())
            .unwrap_or(0),
        reference: a.reference.reference.display().to_string(),
        n_input_sequences: st.total,
        ..Default::default()
    };
    if let Some(p) = &pcr {
        meta.primer_pair = Some(p.name.clone());
        meta.forward_primer = Some(p.forward.clone());
        meta.reverse_primer = Some(p.reverse.clone());
        meta.notes.push(format!(
            "in-silico PCR: max_mismatches={}, partial={}, keep_primers={}",
            a.primers.max_mismatches, a.primers.partial, a.primers.keep_primers
        ));
    }
    let params = a.opts.params(a.seed, a.min_descend / 100.0);
    eprintln!("training on {} sequences ...", seqs.len());
    let backup = a.remove_problems.then(|| seqs.clone());
    let (mut model, mut rep) = train(seqs, &params, meta.clone())?;
    report_training(&model, &rep);
    if a.problems.is_some() || a.remove_problems {
        let cp = ClassifyParams { threshold: a.problem_threshold / 100.0, seed: a.seed, ..Default::default() };
        eprintln!("leave-one-out check of {} references ...", model.seq_kmers.len());
        let conflicts = find_conflicts(&model, &cp);
        eprintln!("{} references confidently conflict with their label", conflicts.len());
        model.meta.n_problem_sequences = conflicts.len();
        if let Some(p) = &a.problems {
            let mut w = create_output(Some(p))?;
            writeln!(w, "id\tlineage\tpredicted\tconflict_rank\tconfidence")?;
            for c in &conflicts {
                writeln!(w, "{}\t{}\t{}\t{}\t{:.1}", c.id, c.lineage, c.predicted, c.rank, c.confidence * 100.0)?;
            }
            w.flush()?;
        }
        if let Some(mut all) = backup {
            let max_depth = match model.meta.ranks.iter().position(|r| r.eq_ignore_ascii_case(&a.remove_rank)) {
                Some(i) => i + 1,
                None => {
                    bail!("--remove-rank '{}' is not one of the reference ranks {:?}", a.remove_rank, model.meta.ranks)
                }
            };
            let drop: std::collections::HashSet<&str> =
                conflicts.iter().filter(|c| c.depth <= max_depth).map(|c| c.id.as_str()).collect();
            if drop.is_empty() {
                eprintln!("no conflicts at or above {}; nothing removed", a.remove_rank);
                model.meta.notes.dedup();
                model.save(&a.output)?;
                eprintln!("wrote {} in {:.1}s", a.output.display(), t0.elapsed().as_secs_f64());
                return Ok(());
            }
            all.retain(|t| !drop.contains(t.id.as_str()));
            eprintln!(
                "retraining without {} references conflicting at {} level or above ...",
                drop.len(),
                a.remove_rank
            );
            meta.notes.push(format!(
                "{} references conflicting at {} level or above removed (leave-one-out check)",
                drop.len(),
                a.remove_rank
            ));
            (model, rep) = train(all, &params, meta)?;
            report_training(&model, &rep);
        }
    }
    model.meta.notes.dedup();
    model.save(&a.output)?;
    eprintln!("wrote {} in {:.1}s", a.output.display(), t0.elapsed().as_secs_f64());
    Ok(())
}

fn report_training(model: &Model, rep: &idtaxa::train::TrainReport) {
    eprintln!(
        "k = {}, {} duplicates removed, {} subsampled out, {} too short, {} used; {} taxa, {} groups",
        rep.k,
        rep.n_duplicates,
        rep.n_subsampled_out,
        rep.n_too_short,
        rep.n_used,
        model.meta.n_taxa,
        model.meta.n_groups
    );
}

fn cmd_classify(a: ClassifyArgs) -> Result<()> {
    let t0 = Instant::now();
    let model = Model::load(&a.model)?;
    let records = seq::read_all(&a.input)?;
    if let Some(p) = &model.meta.primer_pair {
        eprintln!("model trained for primer pair {p}; make sure queries are the same region with primers removed");
    }
    let params = a.opts.params(a.seed);
    let assigns: Vec<_> = records
        .par_iter()
        .enumerate()
        .map(|(i, r)| {
            let mut rng =
                SmallRng::seed_from_u64(params.seed.wrapping_add((i as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15)));
            classify_seq(&model, &r.seq, &params, &mut rng)
        })
        .collect();
    let ids: Vec<String> = records.iter().map(|r| r.id.clone()).collect();
    let mut w = create_output(a.output.as_deref())?;
    write_assignments(&mut w, &model, &ids, &assigns, a.format)?;
    w.flush()?;
    let n_cls = assigns.iter().filter(|x| x.n_classified > 1).count();
    eprintln!(
        "classified {} sequences ({} assigned below root) in {:.1}s",
        records.len(),
        n_cls,
        t0.elapsed().as_secs_f64()
    );
    Ok(())
}

fn cmd_trim(a: TrimArgs) -> Result<()> {
    let pcr = a.primers.resolve()?.context("specify --primers NAME or --forward/--reverse")?;
    let records = seq::read_all(&a.input)?;
    let ranks = a.ranks.as_deref();
    let results: Vec<Option<primers::Amplicon>> =
        records.par_iter().map(|r| pcr.amplify(&seq::clean(&r.seq))).collect();
    let mut w = create_output(a.output.as_deref())?;
    // coverage: taxon -> (total, full, partial)
    let mut cov: BTreeMap<String, (usize, usize, usize)> = BTreeMap::new();
    let (mut n_full, mut n_partial, mut n_len) = (0, 0, 0);
    for (r, amp) in records.iter().zip(&results) {
        let ok = amp.as_ref().filter(|x| x.seq.len() >= a.primers.min_length && x.seq.len() <= a.primers.max_length);
        if amp.is_some() && ok.is_none() {
            n_len += 1;
        }
        if let Some(x) = ok {
            if x.kind == AmpliconKind::Full {
                n_full += 1;
            } else {
                n_partial += 1;
            }
            seq::write_fasta(&mut w, &r.header(), &x.seq)?;
        }
        if a.coverage.is_some() {
            let lin = lineage_from_header(&r.id, &r.desc).map(|s| parse_lineage(&s, ranks)).unwrap_or_default();
            let key = lin
                .ranks
                .iter()
                .position(|x| x.as_deref().map(|y| y.eq_ignore_ascii_case(&a.coverage_rank)).unwrap_or(false))
                .map(|i| lin.names[..=i].join(";"))
                .unwrap_or_else(|| "unassigned".to_string());
            let e = cov.entry(key).or_insert((0, 0, 0));
            e.0 += 1;
            match ok.map(|x| x.kind) {
                Some(AmpliconKind::Full) => e.1 += 1,
                Some(_) => e.2 += 1,
                None => {}
            }
        }
    }
    w.flush()?;
    let n = records.len().max(1);
    eprintln!(
        "{}: {} references, {} full amplicons ({:.1}%), {} partial, {} outside length limits",
        pcr.name,
        records.len(),
        n_full,
        100.0 * n_full as f64 / n as f64,
        n_partial,
        n_len
    );
    if let Some(p) = &a.coverage {
        let mut cw = create_output(Some(p))?;
        writeln!(cw, "taxon\tn_reference\tn_full\tn_partial\tpct_amplified")?;
        for (t, (tot, f, pt)) in &cov {
            writeln!(cw, "{t}\t{tot}\t{f}\t{pt}\t{:.1}", 100.0 * (f + pt) as f64 / *tot as f64)?;
        }
        cw.flush()?;
    }
    Ok(())
}

fn cmd_primers() -> Result<()> {
    let mut w = create_output(None)?;
    writeln!(w, "name\tmarker\tregion\tforward\tforward_seq\treverse\treverse_seq\tnote")?;
    for p in primers::CATALOG {
        writeln!(
            w,
            "{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}",
            p.name, p.marker, p.region, p.forward_name, p.forward, p.reverse_name, p.reverse, p.note
        )?;
    }
    w.flush()?;
    Ok(())
}

fn cmd_detect(a: DetectArgs) -> Result<()> {
    // unique primer sequences with names
    let mut uniq: BTreeMap<&str, Vec<String>> = BTreeMap::new();
    for p in primers::CATALOG {
        uniq.entry(p.forward).or_default().push(format!("{} (fwd)", p.forward_name));
        uniq.entry(p.reverse).or_default().push(format!("{} (rev)", p.reverse_name));
    }
    let list: Vec<(&str, String, Vec<u8>)> = uniq
        .into_iter()
        .map(|(s, mut names)| {
            names.dedup();
            (s, names.join(", "), primers::masks(s))
        })
        .collect();
    let mut rates: HashMap<&str, f64> = HashMap::new();
    let mut w = create_output(None)?;
    writeln!(w, "file\tprimer\tsequence\treads_checked\treads_with_primer\tpct")?;
    for path in &a.reads {
        let reads: Vec<Vec<u8>> =
            SeqReader::from_path(path)?.take(a.n).map(|r| r.map(|r| r.seq)).collect::<Result<_>>()?;
        let n = reads.len().max(1);
        let mut rows: Vec<(f64, &str, &str, usize)> = list
            .par_iter()
            .map(|(s, names, m)| {
                let hits =
                    reads.iter().filter(|r| primers::primer_at_start(r, m, a.max_offset, a.max_mismatches)).count();
                (100.0 * hits as f64 / n as f64, *s, names.as_str(), hits)
            })
            .collect();
        rows.sort_by(|x, y| y.0.partial_cmp(&x.0).unwrap());
        for &(pct, s, names, hits) in &rows {
            let e = rates.entry(s).or_insert(0.0);
            *e = e.max(pct);
            if pct >= 1.0 {
                writeln!(w, "{}\t{}\t{}\t{}\t{}\t{:.1}", path.display(), names, s, reads.len(), hits, pct)?;
            }
        }
        if rows.first().map(|r| r.0 < 1.0).unwrap_or(true) {
            eprintln!(
                "{}: no catalog primer found at read starts (reads may already be primer-trimmed)",
                path.display()
            );
        }
    }
    let mut pairs: Vec<(f64, &str)> = primers::CATALOG
        .iter()
        .filter_map(|p| {
            let f = *rates.get(p.forward).unwrap_or(&0.0);
            let r = *rates.get(p.reverse).unwrap_or(&0.0);
            (f >= 20.0 && r >= 20.0).then_some((f.min(r), p.name))
        })
        .collect();
    w.flush()?;
    pairs.sort_by(|x, y| y.0.partial_cmp(&x.0).unwrap());
    for (s, name) in pairs {
        eprintln!("consistent with primer pair {name} (min hit rate {s:.1}%)");
    }
    Ok(())
}

fn cmd_summarize(a: SummarizeArgs) -> Result<()> {
    let class = summarize::read_classification(&a.classification)?;
    let counts = match &a.counts {
        Some(p) => Some(summarize::read_counts(p, a.samples_as_rows)?),
        None => None,
    };
    if let Some(c) = &counts {
        let missing = c.features.iter().filter(|f| !class.contains_key(*f)).count();
        if missing > 0 {
            eprintln!("warning: {missing} features in the count table have no classification (reported as Unassigned)");
        }
    }
    let level = match a.level {
        Some(d) => Level::Depth(d),
        None => Level::Rank(a.rank.clone()),
    };
    let table = summarize::aggregate(&class, counts.as_ref(), &level);
    let mut w = create_output(a.output.as_deref())?;
    if a.long {
        table.write_long(&mut w, &a.study, &a.primer_set, &level.label(), !a.no_header)?;
    } else {
        table.write_wide(&mut w, a.relative)?;
    }
    w.flush()?;
    eprintln!("{} taxa x {} samples", table.rows.len(), table.samples.len());
    Ok(())
}

fn cmd_evaluate(a: EvaluateArgs) -> Result<()> {
    let pcr = a.primers.resolve()?;
    let (seqs, _) = load_references(&a.reference, &a.primers, pcr.as_ref())?;
    let cp = a.classify.params(a.seed);
    let tp = a.train.params(a.seed, cp.min_descend);
    let rep = evaluate(seqs, a.holdout, &tp, &cp)?;
    eprintln!("trained on {}, tested {}", rep.n_train, rep.n_test);
    let mut w = create_output(a.output.as_deref())?;
    rep.write_tsv(&mut w)?;
    w.flush()?;
    Ok(())
}
