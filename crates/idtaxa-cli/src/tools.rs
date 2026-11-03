//! validate / calibrate / split / extract-region / clean-ref / consensus.

use std::collections::{BTreeSet, HashMap, HashSet};
use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};

use anyhow::{bail, Context, Result};
use clap::Args;
use idtaxa_core::calibration::{
    isotonic, novelty_cutoff, threshold_for_error, Calibration, NoveltyCutoff, RankCalibration,
};
use idtaxa_core::classify::SimilarityProfile;
use idtaxa_core::eval::{
    indicator, metrics, outcome, paired_bootstrap, present_paths, truth_names, EvalRecord,
};
use idtaxa_core::io::read_reference;
use idtaxa_core::{Classification, Classifier, Model};
use rand::seq::SliceRandom;
use rand::SeedableRng;
use rayon::prelude::*;

fn writer(p: &Path) -> Result<BufWriter<File>> {
    Ok(BufWriter::new(
        File::create(p).with_context(|| format!("creating {}", p.display()))?,
    ))
}

// ---------------------------------------------------------------- split

#[derive(Args)]
pub struct SplitArgs {
    #[arg(long = "ref")]
    reference: PathBuf,
    #[arg(long)]
    taxonomy: Option<PathBuf>,
    /// random | clade
    #[arg(long, default_value = "random")]
    mode: String,
    /// Fraction of sequences (random) or of taxa (clade) to hold out
    #[arg(long, default_value_t = 0.1)]
    fraction: f64,
    /// Rank depth (1 = first rank below Root) whose taxa are excluded in clade mode
    #[arg(long, default_value_t = 0)]
    rank: usize,
    #[arg(long, default_value_t = 1)]
    seed: u64,
    /// Writes <prefix>_train.fasta, <prefix>_train.tsv, <prefix>_holdout.fasta, <prefix>_holdout.tsv
    #[arg(long)]
    output_prefix: String,
}

pub fn split(a: SplitArgs, quiet: bool) -> Result<()> {
    let (recs, lin) = read_reference(&a.reference, a.taxonomy.as_deref())?;
    let names: Vec<Vec<String>> = lin.iter().map(|l| truth_names(l)).collect();
    let mut rng = rand_xoshiro::Xoshiro256PlusPlus::seed_from_u64(a.seed);
    let n = recs.len();
    let mut hold = vec![false; n];
    match a.mode.as_str() {
        "random" => {
            let mut idx: Vec<usize> = (0..n).collect();
            idx.shuffle(&mut rng);
            for &i in idx.iter().take((n as f64 * a.fraction).round() as usize) {
                hold[i] = true;
            }
        }
        "clade" => {
            if a.rank == 0 {
                bail!("--mode clade requires --rank (depth below Root, e.g. 7 for genus in a 7-rank taxonomy)");
            }
            let key = |v: &Vec<String>| -> Option<String> {
                (v.len() >= a.rank).then(|| v[..a.rank].join(";"))
            };
            let taxa: BTreeSet<String> = names.iter().filter_map(key).collect();
            // only exclude taxa whose parent keeps other members, so the
            // parent remains in the reference
            let mut taxa: Vec<String> = taxa.into_iter().collect();
            taxa.shuffle(&mut rng);
            let k = (taxa.len() as f64 * a.fraction).round() as usize;
            let excluded: HashSet<String> = taxa.into_iter().take(k).collect();
            for i in 0..n {
                if let Some(t) = key(&names[i]) {
                    hold[i] = excluded.contains(&t);
                }
            }
        }
        m => bail!("--mode must be random or clade (got {m})"),
    }
    let p = &a.output_prefix;
    for (part, want) in [("train", false), ("holdout", true)] {
        let mut fa = writer(Path::new(&format!("{p}_{part}.fasta")))?;
        let mut tx = writer(Path::new(&format!("{p}_{part}.tsv")))?;
        for i in 0..n {
            if hold[i] == want {
                writeln!(
                    fa,
                    ">{}\n{}",
                    recs[i].id,
                    String::from_utf8_lossy(&recs[i].seq)
                )?;
                writeln!(tx, "{}\t{}", recs[i].id, lin[i])?;
            }
        }
    }
    if !quiet {
        let h = hold.iter().filter(|&&x| x).count();
        eprintln!(
            "split: {} training, {} held out ({} mode)",
            n - h,
            h,
            a.mode
        );
    }
    Ok(())
}

// ---------------------------------------------------------------- validate

#[derive(Args)]
pub struct ValidateArgs {
    #[arg(long)]
    model: PathBuf,
    #[arg(long)]
    holdout: PathBuf,
    #[arg(long)]
    taxonomy: Option<PathBuf>,
    /// Output prefix: <prefix>.tsv (metrics per rank and threshold), <prefix>_summary.txt
    #[arg(long, default_value = "validate")]
    output_prefix: String,
    /// Thresholds to evaluate (comma separated)
    #[arg(long, value_delimiter = ',', default_values_t = vec![0.0, 10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0])]
    thresholds: Vec<f64>,
    /// Compare classification modes on the same held-out set, e.g. standard,balanced
    #[arg(long, value_delimiter = ',')]
    compare_mode: Option<Vec<String>>,
    /// Compare against a second model (e.g. trained with --mode spaced or balanced)
    #[arg(long)]
    compare_model: Option<PathBuf>,
    /// Calibration to apply for the comparison configuration B (per-rank thresholds)
    #[arg(long)]
    compare_calibration: Option<PathBuf>,
    /// Adaptive bootstraps for configuration B
    #[arg(long)]
    compare_adaptive: Option<u32>,
    #[arg(long, default_value_t = 60.0)]
    threshold: f64,
    #[arg(long, default_value_t = 100)]
    bootstraps: u32,
    #[arg(long, default_value = "both")]
    strand: String,
    #[arg(long, default_value_t = 42)]
    seed: u64,
    #[arg(long, default_value_t = 0.1)]
    balance_exponent: f64,
    #[arg(long, default_value_t = 1000)]
    ci_replicates: usize,
    /// Write per-sequence predictions to <prefix>_predictions.tsv
    #[arg(long)]
    write_predictions: bool,
}

pub struct Config {
    pub label: String,
    pub model: Model,
    pub params: idtaxa_core::ClassifyParams,
    pub calibration: Option<Calibration>,
    pub adaptive: Option<u32>,
}

/// Classify held-out sequences under a configuration.
pub fn classify_all(
    cfg: &Config,
    seqs: &[Vec<u8>],
    novelty: bool,
) -> Result<Vec<(Classification, Option<SimilarityProfile>)>> {
    let classifier = Classifier::new(&cfg.model, cfg.params.clone())?;
    let max_depth = cfg
        .model
        .taxonomy
        .iter()
        .map(|t| t.matches(';').count())
        .max()
        .unwrap_or(1);
    let thresholds = match &cfg.calibration {
        Some(c) => c.thresholds(max_depth, cfg.params.threshold),
        None => vec![cfg.params.threshold; max_depth],
    };
    let engine = crate::classify_cmd::Engine {
        classifier,
        calibration: None,
        thresholds,
        novelty,
        truncate_novel: false,
        adaptive: cfg.adaptive.map(|n| (n, 10.0)),
    };
    Ok(seqs
        .par_iter()
        .map_init(
            || engine.classifier.new_scratch(),
            |sc, s| {
                let norm = idtaxa_core::kmer::normalize_sequence(s);
                let c = engine.compute(&norm, sc);
                (c.cls, c.sim)
            },
        )
        .collect())
}

pub fn records(
    model: &Model,
    cls: &[(Classification, Option<SimilarityProfile>)],
    truth: &[Vec<String>],
) -> Vec<EvalRecord> {
    cls.iter()
        .zip(truth)
        .map(|((c, _), t)| EvalRecord {
            truth: t.clone(),
            pred: c.lineage[1..]
                .iter()
                .map(|&n| model.taxon(n).to_string())
                .collect(),
            conf: c.confidences[1..].to_vec(),
        })
        .collect()
}

fn rank_names(model: &Model, max_depth: usize) -> Vec<String> {
    (1..=max_depth)
        .map(|d| {
            (0..model.n_nodes() as u32)
                .find(|&i| model.level(i) as usize == d + 1)
                .and_then(|i| model.rank(i))
                .map(str::to_string)
                .unwrap_or_else(|| format!("level{d}"))
        })
        .collect()
}

pub fn validate(a: ValidateArgs, quiet: bool) -> Result<()> {
    let (recs, lin) = read_reference(&a.holdout, a.taxonomy.as_deref())?;
    let truth: Vec<Vec<String>> = lin.iter().map(|l| truth_names(l)).collect();
    let seqs: Vec<Vec<u8>> = recs.iter().map(|r| r.seq.clone()).collect();
    let mk = |mode: &str,
              model: Model,
              cal: Option<Calibration>,
              adaptive: Option<u32>|
     -> Result<Config> {
        Ok(Config {
            label: mode.to_string(),
            params: crate::classify_params_from(
                a.threshold,
                a.bootstraps,
                &a.strand,
                a.seed,
                0.98,
                0.47,
                0.0,
                mode,
                a.balance_exponent,
            )?,
            model,
            calibration: cal,
            adaptive,
        })
    };
    let mut configs = vec![];
    let modes = a
        .compare_mode
        .clone()
        .unwrap_or_else(|| vec!["standard".into()]);
    configs.push(mk(&modes[0], Model::load(&a.model)?, None, None)?);
    let cal_b = match &a.compare_calibration {
        Some(p) => Some(Calibration::load(p)?),
        None => None,
    };
    if modes.len() > 1
        || a.compare_model.is_some()
        || cal_b.is_some()
        || a.compare_adaptive.is_some()
    {
        let mode_b = modes.get(1).cloned().unwrap_or_else(|| modes[0].clone());
        let model_b = Model::load(a.compare_model.as_ref().unwrap_or(&a.model))?;
        let mut cfg = mk(&mode_b, model_b, cal_b, a.compare_adaptive)?;
        let mut label = mode_b;
        if let Some(p) = &a.compare_model {
            label = format!("{label}:{}", p.file_name().unwrap().to_string_lossy());
        }
        if cfg.calibration.is_some() {
            label.push_str("+calibrated");
        }
        if let Some(n) = cfg.adaptive {
            label.push_str(&format!("+adaptive{n}"));
        }
        cfg.label = label;
        configs.push(cfg);
    }

    let mut tsv = writer(Path::new(&format!("{}.tsv", a.output_prefix)))?;
    writeln!(
        tsv,
        "config\tdepth\trank\tthreshold\tn\tn_present\tn_absent\tTP\tMC\tOC\tUC\tTN\taccuracy\tmisclassification\toverclassification\tunderclassification\terror_rate\terror_among_assigned"
    )?;
    let mut summary = String::new();
    let mut per_config_outcomes: Vec<Vec<Vec<idtaxa_core::eval::Outcome>>> = vec![];
    for cfg in &configs {
        let t0 = std::time::Instant::now();
        let cls = classify_all(cfg, &seqs, false)?;
        if !quiet {
            eprintln!(
                "[validate] {}: classified {} held-out sequences in {:.1}s",
                cfg.label,
                seqs.len(),
                t0.elapsed().as_secs_f64()
            );
        }
        let recs_e = records(&cfg.model, &cls, &truth);
        let present = present_paths(&cfg.model);
        let max_depth = truth.iter().map(|t| t.len()).max().unwrap_or(0);
        let rnames = rank_names(&cfg.model, max_depth);
        let mut ths = a.thresholds.clone();
        if !ths.contains(&a.threshold) {
            ths.push(a.threshold);
        }
        ths.sort_by(|x, y| x.partial_cmp(y).unwrap());
        for &t in &ths {
            let tv = match &cfg.calibration {
                Some(c) if t == a.threshold => c.thresholds(max_depth, t),
                _ => vec![t; max_depth],
            };
            for d in 1..=max_depth {
                let m = metrics(&recs_e, d, &tv, &present);
                writeln!(
                    tsv,
                    "{}\t{d}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{:.4}\t{:.4}\t{:.4}\t{:.4}\t{:.4}\t{:.4}",
                    cfg.label,
                    rnames[d - 1],
                    if cfg.calibration.is_some() && t == a.threshold { "calibrated".to_string() } else { t.to_string() },
                    m.n,
                    m.n_present(),
                    m.n_absent(),
                    m.tp,
                    m.mc,
                    m.oc,
                    m.uc,
                    m.tn,
                    m.accuracy(),
                    m.misclassification_rate(),
                    m.overclassification_rate(),
                    m.underclassification_rate(),
                    m.error_rate(),
                    m.error_among_assigned()
                )?;
            }
        }
        // summary at the main threshold
        let tv = match &cfg.calibration {
            Some(c) => c.thresholds(max_depth, a.threshold),
            None => vec![a.threshold; max_depth],
        };
        summary.push_str(&format!(
            "Configuration {} (threshold {}{}), {} held-out sequences\n",
            cfg.label,
            a.threshold,
            if cfg.calibration.is_some() {
                ", calibrated per-rank thresholds"
            } else {
                ""
            },
            seqs.len()
        ));
        summary.push_str(
            "rank            present absent  accuracy  misclass  overclass underclass error\n",
        );
        let mut outs = vec![];
        for d in 1..=max_depth {
            let m = metrics(&recs_e, d, &tv, &present);
            summary.push_str(&format!(
                "{:<15} {:>7} {:>6}  {:>8.3}  {:>8.3}  {:>8.3}  {:>9.3}  {:>6.3}\n",
                rnames[d - 1],
                m.n_present(),
                m.n_absent(),
                m.accuracy(),
                m.misclassification_rate(),
                m.overclassification_rate(),
                m.underclassification_rate(),
                m.error_rate()
            ));
            outs.push(
                recs_e
                    .iter()
                    .map(|r| outcome(r, d, &tv, &present))
                    .collect(),
            );
        }
        summary.push('\n');
        per_config_outcomes.push(outs);

        if a.write_predictions {
            let mut w = writer(Path::new(&format!(
                "{}_{}_predictions.tsv",
                a.output_prefix,
                cfg.label.replace([':', '/'], "_")
            )))?;
            writeln!(w, "id\ttruth\tpredicted\tconfidences")?;
            for (i, r) in recs_e.iter().enumerate() {
                let conf: Vec<String> = r.conf.iter().map(|c| format!("{c:.1}")).collect();
                writeln!(
                    w,
                    "{}\t{}\t{}\t{}",
                    recs[i].id,
                    r.truth.join(";"),
                    r.pred.join(";"),
                    conf.join(";")
                )?;
            }
        }
    }

    if configs.len() == 2 {
        let mut cmp = writer(Path::new(&format!("{}_compare.tsv", a.output_prefix)))?;
        writeln!(
            cmp,
            "depth\tmetric\t{}\t{}\tdiff\tci_low\tci_high",
            configs[0].label, configs[1].label
        )?;
        summary.push_str(&format!(
            "Comparison {} -> {} (difference B - A, 95% paired bootstrap CI over held-out sequences)\n",
            configs[0].label, configs[1].label
        ));
        let depths = per_config_outcomes[0]
            .len()
            .min(per_config_outcomes[1].len());
        for d in 0..depths {
            for metric in [
                "correct",
                "error",
                "overclassified",
                "misclassified",
                "underclassified",
            ] {
                let xa: Vec<f64> = per_config_outcomes[0][d]
                    .iter()
                    .map(|&o| indicator(o, metric))
                    .collect();
                let xb: Vec<f64> = per_config_outcomes[1][d]
                    .iter()
                    .map(|&o| indicator(o, metric))
                    .collect();
                let ma = xa.iter().sum::<f64>() / xa.len().max(1) as f64;
                let mb = xb.iter().sum::<f64>() / xb.len().max(1) as f64;
                let (diff, lo, hi) = paired_bootstrap(&xa, &xb, a.ci_replicates, 7 + d as u64);
                writeln!(
                    cmp,
                    "{}\t{metric}\t{ma:.4}\t{mb:.4}\t{diff:.4}\t{lo:.4}\t{hi:.4}",
                    d + 1
                )?;
                if metric == "correct" || metric == "error" {
                    summary.push_str(&format!(
                        "  depth {} {:<8} A={:.4} B={:.4} diff={:+.4} [{:+.4}, {:+.4}]{}\n",
                        d + 1,
                        metric,
                        ma,
                        mb,
                        diff,
                        lo,
                        hi,
                        if lo > 0.0 || hi < 0.0 { " *" } else { "" }
                    ));
                }
            }
        }
    }
    std::fs::write(format!("{}_summary.txt", a.output_prefix), &summary)?;
    if !quiet {
        print!("{summary}");
    }
    Ok(())
}

// ---------------------------------------------------------------- calibrate

#[derive(Args)]
pub struct CalibrateArgs {
    #[arg(long)]
    model: PathBuf,
    #[arg(long)]
    holdout: PathBuf,
    #[arg(long)]
    taxonomy: Option<PathBuf>,
    /// Clade-exclusion held-out set used to fit novelty cut-offs (optional)
    #[arg(long)]
    novelty_holdout: Option<PathBuf>,
    #[arg(long)]
    novelty_taxonomy: Option<PathBuf>,
    #[arg(long, default_value_t = 0.05)]
    target_error: f64,
    #[arg(long)]
    output: PathBuf,
    /// Reliability diagram TSV (before/after calibration)
    #[arg(long)]
    reliability: Option<PathBuf>,
    #[arg(long, default_value_t = 100)]
    bootstraps: u32,
    #[arg(long, default_value = "both")]
    strand: String,
    #[arg(long, default_value_t = 42)]
    seed: u64,
}

pub fn calibrate(a: CalibrateArgs, quiet: bool) -> Result<()> {
    let model = Model::load(&a.model)?;
    let params = crate::classify_params_from(
        60.0,
        a.bootstraps,
        &a.strand,
        a.seed,
        0.98,
        0.47,
        0.0,
        "standard",
        0.0,
    )?;
    let (recs, lin) = read_reference(&a.holdout, a.taxonomy.as_deref())?;
    let truth: Vec<Vec<String>> = lin.iter().map(|l| truth_names(l)).collect();
    let seqs: Vec<Vec<u8>> = recs.iter().map(|r| r.seq.clone()).collect();
    let cfg = Config {
        label: "standard".into(),
        model,
        params,
        calibration: None,
        adaptive: None,
    };
    let cls = classify_all(&cfg, &seqs, false)?;
    let ev = records(&cfg.model, &cls, &truth);
    let max_depth = truth.iter().map(|t| t.len()).max().unwrap_or(0);
    let rnames = rank_names(&cfg.model, max_depth);
    let grid_err = [0.01, 0.02, 0.05, 0.1, 0.2];
    let mut ranks = Vec::new();
    let mut rel_rows = Vec::new();
    for d in 1..=max_depth {
        // (raw confidence, correct) for every query predicted to depth d
        let pts: Vec<(f64, bool)> = ev
            .iter()
            .filter(|r| r.conf.len() >= d)
            .map(|r| {
                let ok = r.truth.len() >= d && r.truth[..d] == r.pred[..d];
                (r.conf[d - 1], ok)
            })
            .collect();
        if pts.is_empty() {
            continue;
        }
        let (x, y) = isotonic(&pts);
        let thr = threshold_for_error(&pts, a.target_error);
        let grid = grid_err
            .iter()
            .map(|&e| (e, threshold_for_error(&pts, e)))
            .collect();
        let rc = RankCalibration {
            depth: d,
            rank: rnames[d - 1].clone(),
            n: pts.len(),
            x,
            y,
            threshold: thr,
            grid,
        };
        // reliability: 10 bins of raw confidence
        for b in 0..10 {
            let lo = b as f64 * 10.0;
            let hi = lo + 10.0;
            let inb: Vec<&(f64, bool)> = pts
                .iter()
                .filter(|p| p.0 >= lo && (p.0 < hi || (b == 9 && p.0 <= 100.0)))
                .collect();
            if inb.is_empty() {
                continue;
            }
            let n = inb.len() as f64;
            let mean_raw = inb.iter().map(|p| p.0).sum::<f64>() / n / 100.0;
            let mean_cal = inb.iter().map(|p| rc.apply(p.0)).sum::<f64>() / n;
            let acc = inb.iter().filter(|p| p.1).count() as f64 / n;
            rel_rows.push(format!(
                "{d}\t{}\t{lo}-{hi}\t{}\t{mean_raw:.4}\t{mean_cal:.4}\t{acc:.4}",
                rc.rank,
                inb.len()
            ));
        }
        ranks.push(rc);
    }

    let mut novelty = None;
    if let Some(nh) = &a.novelty_holdout {
        let (nrecs, nlin) = read_reference(nh, a.novelty_taxonomy.as_deref())?;
        let ntruth: Vec<Vec<String>> = nlin.iter().map(|l| truth_names(l)).collect();
        let nseqs: Vec<Vec<u8>> = nrecs.iter().map(|r| r.seq.clone()).collect();
        // combine random (known) and clade-exclusion (novel) sets
        let mut all_truth = truth.clone();
        all_truth.extend(ntruth);
        let mut all_seqs = seqs.clone();
        all_seqs.extend(nseqs);
        let res = classify_all(&cfg, &all_seqs, true)?;
        let present = present_paths(&cfg.model);
        let mut cuts = Vec::new();
        for d in 1..=max_depth {
            let (mut known, mut novel) = (vec![], vec![]);
            for (i, (c, sim)) in res.iter().enumerate() {
                let Some(sim) = sim else { continue };
                if c.lineage.len() <= d {
                    continue;
                }
                let t = &all_truth[i];
                let present_d = t.len() >= d && present.contains(&format!("{};", t[..d].join(";")));
                if present_d {
                    known.push(sim.per_lineage[d]);
                } else {
                    novel.push(sim.per_lineage[d]);
                }
            }
            if known.is_empty() || novel.is_empty() {
                continue;
            }
            let (cut, sens, spec) = novelty_cutoff(&known, &novel);
            cuts.push(NoveltyCutoff {
                depth: d,
                rank: rnames[d - 1].clone(),
                cutoff: cut,
                n_known: known.len(),
                n_novel: novel.len(),
                sensitivity_known: sens,
                specificity_novel: spec,
            });
        }
        novelty = Some(cuts);
    }

    let cal = Calibration {
        format: "idtaxa-calibration".into(),
        version: 1,
        model_id: cfg.model.meta.model_id.clone(),
        target_error: a.target_error,
        ranks,
        novelty,
    };
    std::fs::write(&a.output, serde_json::to_vec_pretty(&cal)?)?;
    if let Some(p) = &a.reliability {
        let mut w = writer(p)?;
        writeln!(
            w,
            "depth\trank\tbin\tn\tmean_raw_confidence\tmean_calibrated\tobserved_accuracy"
        )?;
        for r in rel_rows {
            writeln!(w, "{r}")?;
        }
    }
    if !quiet {
        eprintln!("calibration (target error {}):", a.target_error);
        for r in &cal.ranks {
            eprintln!(
                "  depth {} {:<12} n={:<6} threshold={:.1}",
                r.depth, r.rank, r.n, r.threshold
            );
        }
        if let Some(n) = &cal.novelty {
            for c in n {
                eprintln!(
                    "  novelty depth {} {:<12} cutoff={:.3} known={} novel={} sens={:.3} spec={:.3}",
                    c.depth, c.rank, c.cutoff, c.n_known, c.n_novel, c.sensitivity_known, c.specificity_novel
                );
            }
        }
    }
    Ok(())
}

// ---------------------------------------------------------------- extract-region

#[derive(Args)]
pub struct ExtractArgs {
    #[arg(long = "ref")]
    reference: PathBuf,
    #[arg(long)]
    taxonomy: Option<PathBuf>,
    /// Forward primer (5'->3', IUPAC)
    #[arg(long)]
    fwd: String,
    /// Reverse primer (5'->3' on the reverse strand, IUPAC)
    #[arg(long)]
    rev: String,
    #[arg(long, default_value_t = 2)]
    max_mismatch: usize,
    /// Number of 3'-terminal primer bases that must match exactly
    #[arg(long, default_value_t = 5)]
    three_prime_exact: usize,
    #[arg(long, default_value_t = 0)]
    min_len: usize,
    #[arg(long, default_value_t = usize::MAX)]
    max_len: usize,
    /// Keep primer sequences in the output
    #[arg(long)]
    keep_primers: bool,
    /// Rescue references lacking a primer site by k-mer anchoring to extracted regions
    #[arg(long)]
    rescue: bool,
    #[arg(long)]
    output: PathBuf,
    /// Report TSV (per-sequence status and per-rank retained taxa)
    #[arg(long)]
    report: Option<PathBuf>,
}

pub fn extract_region(a: ExtractArgs, quiet: bool) -> Result<()> {
    use idtaxa_core::region::{extract, ExtractOptions, Status};
    let (recs, lin) = match &a.taxonomy {
        Some(_) => {
            let (r, l) = read_reference(&a.reference, a.taxonomy.as_deref())?;
            (r, Some(l))
        }
        None => (idtaxa_core::io::read_all(&a.reference)?, None),
    };
    let opts = ExtractOptions {
        fwd: a.fwd.as_bytes().to_vec(),
        rev: a.rev.as_bytes().to_vec(),
        max_mismatch: a.max_mismatch,
        three_prime_exact: a.three_prime_exact,
        min_len: a.min_len,
        max_len: a.max_len,
        keep_primers: a.keep_primers,
        rescue: a.rescue,
    };
    let seqs: Vec<&[u8]> = recs.iter().map(|r| r.seq.as_slice()).collect();
    let res = extract(&seqs, &opts)?;
    let mut out = writer(&a.output)?;
    let mut counts: HashMap<&'static str, usize> = HashMap::new();
    for (i, r) in res.iter().enumerate() {
        *counts.entry(r.status.as_str()).or_default() += 1;
        if let Some(s) = &r.region {
            let hdr = match (&lin, recs[i].desc.is_empty()) {
                (Some(_), _) | (None, true) => recs[i].id.clone(),
                (None, false) => format!("{} {}", recs[i].id, recs[i].desc),
            };
            writeln!(out, ">{hdr}\n{}", String::from_utf8_lossy(s))?;
        }
    }
    out.flush()?;
    if let Some(p) = &a.report {
        let mut w = writer(p)?;
        writeln!(
            w,
            "id\tstatus\tstrand\tstart\tend\tlength\tfwd_mismatches\trev_mismatches"
        )?;
        for (i, r) in res.iter().enumerate() {
            writeln!(
                w,
                "{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}",
                recs[i].id,
                r.status.as_str(),
                if r.reverse { "-" } else { "+" },
                r.start,
                r.end,
                r.region.as_ref().map(|s| s.len()).unwrap_or(0),
                r.fwd_mm.map(|x| x.to_string()).unwrap_or_default(),
                r.rev_mm.map(|x| x.to_string()).unwrap_or_default()
            )?;
        }
        if let Some(lin) = &lin {
            // retained taxa per rank
            let mut wr = writer(&p.with_extension("ranks.tsv"))?;
            writeln!(wr, "depth\ttaxa_before\ttaxa_after\tlost")?;
            let names: Vec<Vec<String>> = lin.iter().map(|l| truth_names(l)).collect();
            let md = names.iter().map(|n| n.len()).max().unwrap_or(0);
            for d in 1..=md {
                let before: HashSet<String> = names
                    .iter()
                    .filter(|n| n.len() >= d)
                    .map(|n| n[..d].join(";"))
                    .collect();
                let after: HashSet<String> = names
                    .iter()
                    .zip(&res)
                    .filter(|(n, r)| n.len() >= d && r.region.is_some())
                    .map(|(n, _)| n[..d].join(";"))
                    .collect();
                writeln!(
                    wr,
                    "{d}\t{}\t{}\t{}",
                    before.len(),
                    after.len(),
                    before.len() - after.len()
                )?;
            }
        }
    }
    if !quiet {
        let g = |k: &str| counts.get(k).copied().unwrap_or(0);
        eprintln!(
            "extract-region: {} extracted, {} rescued, {} dropped (no primer site: {}, length filter: {})",
            g(Status::Extracted.as_str()),
            g(Status::Rescued.as_str()),
            g(Status::NoSite.as_str()) + g(Status::Length.as_str()),
            g(Status::NoSite.as_str()),
            g(Status::Length.as_str())
        );
    }
    Ok(())
}

// ---------------------------------------------------------------- clean-ref

#[derive(Args)]
pub struct CleanArgs {
    #[arg(long = "ref")]
    reference: PathBuf,
    #[arg(long)]
    taxonomy: Option<PathBuf>,
    /// Trained model whose problemSequences are reviewed
    #[arg(long)]
    model: Option<PathBuf>,
    /// Truncate lineages at placeholder/uninformative labels
    #[arg(long)]
    drop_placeholder_labels: bool,
    /// Drop problem sequences (requires --model or --retrain-rounds)
    #[arg(long)]
    drop_problem_sequences: bool,
    /// Retrain and drop problem sequences iteratively for up to N rounds
    #[arg(long, default_value_t = 0)]
    retrain_rounds: u32,
    /// Extra placeholder regex-free patterns (case-insensitive substrings)
    #[arg(long, value_delimiter = ',')]
    extra_placeholder: Vec<String>,
    /// Pad/truncate all lineages to this many ranks (0 = keep)
    #[arg(long, default_value_t = 0)]
    rank_count: usize,
    #[arg(long, default_value = "keep")]
    case: String,
    #[arg(long)]
    output_prefix: String,
    #[arg(long, default_value_t = 42)]
    seed: u64,
}

pub fn clean_ref(a: CleanArgs, quiet: bool) -> Result<()> {
    use idtaxa_core::clean::{clean_lineages, CleanOptions};
    let (recs, lin) = read_reference(&a.reference, a.taxonomy.as_deref())?;
    let mut opts = CleanOptions {
        truncate_placeholders: a.drop_placeholder_labels,
        rank_count: (a.rank_count > 0).then_some(a.rank_count),
        case: a.case.parse()?,
        ..Default::default()
    };
    opts.placeholder_patterns
        .extend(a.extra_placeholder.iter().map(|s| s.to_lowercase()));
    let cleaned = clean_lineages(&lin, &opts);
    let mut log = writer(Path::new(&format!("{}_changes.tsv", a.output_prefix)))?;
    writeln!(log, "id\taction\tbefore\tafter\tdetail")?;
    for ch in &cleaned.changes {
        writeln!(
            log,
            "{}\t{}\t{}\t{}\t{}",
            recs[ch.index].id, ch.action, ch.before, ch.after, ch.detail
        )?;
    }
    let mut keep: Vec<bool> = cleaned.lineages.iter().map(|l| l.is_some()).collect();
    let mut lineages: Vec<String> = cleaned
        .lineages
        .iter()
        .map(|l| l.clone().unwrap_or_default())
        .collect();
    for (i, k) in keep.iter().enumerate() {
        if !k {
            writeln!(
                log,
                "{}\tdropped_no_informative_rank\t{}\t\t",
                recs[i].id, lin[i]
            )?;
        }
    }
    {
        let mut c = writer(Path::new(&format!("{}_conflicts.tsv", a.output_prefix)))?;
        writeln!(c, "type\tname\tdepth\tlineages")?;
        for cf in &cleaned.conflicts {
            writeln!(
                c,
                "{}\t{}\t{}\t{}",
                cf.kind,
                cf.name,
                cf.depth,
                cf.lineages.join(" | ")
            )?;
        }
    }
    // problem sequences
    let mut review = writer(Path::new(&format!(
        "{}_problem_sequences.tsv",
        a.output_prefix
    )))?;
    writeln!(review, "round\tid\tlabelled\tpredicted")?;
    let by_id: HashMap<&str, usize> = recs
        .iter()
        .enumerate()
        .map(|(i, r)| (r.id.as_str(), i))
        .collect();
    let mut report_problems = |model: &Model,
                               round: u32,
                               keep: &mut Vec<bool>,
                               drop: bool,
                               log: &mut BufWriter<File>|
     -> Result<usize> {
        let mut n = 0;
        for p in &model.meta.problem_sequences {
            let name = &model.meta.seq_names[p.index as usize];
            writeln!(review, "{round}\t{name}\t{}\t{}", p.expected, p.predicted)?;
            if drop {
                if let Some(&i) = by_id.get(name.as_str()) {
                    if keep[i] {
                        keep[i] = false;
                        n += 1;
                        writeln!(
                            log,
                            "{name}\tdropped_problem_sequence\t{}\t\tpredicted {}",
                            p.expected, p.predicted
                        )?;
                    }
                }
            }
        }
        Ok(n)
    };
    if let Some(mp) = &a.model {
        let m = Model::load(mp)?;
        report_problems(&m, 0, &mut keep, a.drop_problem_sequences, &mut log)?;
    }
    for round in 1..=a.retrain_rounds {
        let idx: Vec<usize> = (0..recs.len()).filter(|&i| keep[i]).collect();
        let seqs: Vec<&[u8]> = idx.iter().map(|&i| recs[i].seq.as_slice()).collect();
        let l: Vec<String> = idx.iter().map(|&i| lineages[i].clone()).collect();
        let names: Vec<String> = idx.iter().map(|&i| recs[i].id.clone()).collect();
        let (m, _) = idtaxa_core::train::train(
            &seqs,
            &l,
            &names,
            &idtaxa_core::train::TrainOptions {
                seed: a.seed,
                ..Default::default()
            },
        )?;
        let dropped = report_problems(&m, round, &mut keep, true, &mut log)?;
        if !quiet {
            eprintln!(
                "clean-ref: round {round}: {} problem sequences, {dropped} newly dropped",
                m.meta.problem_sequences.len()
            );
        }
        if dropped == 0 {
            break;
        }
    }
    let mut fa = writer(Path::new(&format!("{}.fasta", a.output_prefix)))?;
    let mut tx = writer(Path::new(&format!("{}_taxonomy.tsv", a.output_prefix)))?;
    let mut n_out = 0;
    for i in 0..recs.len() {
        if keep[i] {
            n_out += 1;
            writeln!(
                fa,
                ">{}\n{}",
                recs[i].id,
                String::from_utf8_lossy(&recs[i].seq)
            )?;
            writeln!(tx, "{}\t{}", recs[i].id, std::mem::take(&mut lineages[i]))?;
        }
    }
    if !quiet {
        eprintln!(
            "clean-ref: {} of {} sequences kept, {} lineage changes, {} conflicts reported",
            n_out,
            recs.len(),
            cleaned.changes.len(),
            cleaned.conflicts.len()
        );
    }
    Ok(())
}

// ---------------------------------------------------------------- consensus

#[derive(Args)]
pub struct ConsensusArgs {
    /// idtaxa classify output (TSV, ideally with --full-confidences)
    #[arg(long)]
    idtaxa: PathBuf,
    /// gappa examine assign --per-query-results output
    #[arg(long)]
    placement: PathBuf,
    /// jplace file for pendant branch lengths (optional)
    #[arg(long)]
    jplace: Option<PathBuf>,
    #[arg(long, default_value_t = 0.9)]
    placement_min_lwr: f64,
    #[arg(long)]
    max_pendant: Option<f64>,
    /// Taxonomy file shared by both methods (id<TAB>lineage) used to check names
    #[arg(long)]
    taxonomy: Option<PathBuf>,
    /// Rule: agree (deepest agreeing rank) | idtaxa-first | placement-first
    #[arg(long, default_value = "agree")]
    rule: String,
    #[arg(long)]
    output: PathBuf,
}

pub fn consensus(a: ConsensusArgs, quiet: bool) -> Result<()> {
    use idtaxa_core::consensus::{
        consensus as run, read_gappa_per_query, read_idtaxa_tsv, read_jplace_pendant, Rule,
    };
    let id = read_idtaxa_tsv(&a.idtaxa)?;
    let pl = read_gappa_per_query(&a.placement)?;
    let pend = match &a.jplace {
        Some(p) => Some(read_jplace_pendant(p)?),
        None => None,
    };
    let known: Option<HashSet<String>> = match &a.taxonomy {
        Some(t) => {
            let m = idtaxa_core::io::read_taxonomy_tsv(t)?;
            let mut s = HashSet::new();
            for l in m.values() {
                let n = truth_names(l);
                for d in 1..=n.len() {
                    s.insert(n[..d].join(";"));
                }
            }
            Some(s)
        }
        None => None,
    };
    let rule: Rule = a.rule.parse()?;
    let res = run(
        &id,
        &pl,
        pend.as_ref(),
        a.placement_min_lwr,
        a.max_pendant,
        known.as_ref(),
        rule,
    )?;
    let mut w = writer(&a.output)?;
    writeln!(w, "id\ttaxonomy\tsupport\tidtaxa_taxonomy\tplacement_taxonomy\tplacement_lwr\tpendant_length\tflags")?;
    let mut n_dis = 0;
    for r in &res {
        if r.flags.iter().any(|f| f == "disagree") {
            n_dis += 1;
        }
        writeln!(
            w,
            "{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}",
            r.id,
            r.taxonomy.join(";"),
            r.support.join(";"),
            r.idtaxa.join(";"),
            r.placement.join(";"),
            r.placement_lwr
                .iter()
                .map(|x| format!("{x:.3}"))
                .collect::<Vec<_>>()
                .join(";"),
            r.pendant.map(|p| format!("{p:.5}")).unwrap_or_default(),
            r.flags.join(",")
        )?;
    }
    if !quiet {
        eprintln!(
            "consensus: {} queries, {} with disagreements",
            res.len(),
            n_dis
        );
    }
    Ok(())
}
