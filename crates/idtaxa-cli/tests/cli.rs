//! End-to-end tests of the `idtaxa` binary on the committed fixtures in
//! `tests/data` (a 412-sequence PR2 V4 subset trained with DECIPHER 3.9.4,
//! plus cached DECIPHER `IdTaxa` outputs for two seeds).

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::process::Command;

fn bin() -> Command {
    Command::new(env!("CARGO_BIN_EXE_idtaxa"))
}

fn data(p: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../tests/data")
        .join(p)
}

fn run(args: &[&str]) -> String {
    let out = bin().arg("-q").args(args).output().expect("run idtaxa");
    assert!(
        out.status.success(),
        "idtaxa {:?} failed: {}",
        args,
        String::from_utf8_lossy(&out.stderr)
    );
    String::from_utf8(out.stdout).unwrap()
}

fn s(p: &Path) -> &str {
    p.to_str().unwrap()
}

fn import(dir: &Path) -> PathBuf {
    let m = dir.join("mini.idtx");
    if !m.exists() {
        run(&[
            "import-decipher",
            "--input",
            s(&data("decipher_mini_export")),
            "--output",
            s(&m),
        ]);
    }
    m
}

/// (full lineage, confidences) per id from either output flavour.
fn load(p: &Path) -> HashMap<String, (Vec<String>, Vec<f64>)> {
    let text = std::fs::read_to_string(p).unwrap();
    let mut lines = text.lines().filter(|l| !l.starts_with('#'));
    let header: Vec<&str> = lines.next().unwrap().split('\t').collect();
    let (ti, ci) = match header.iter().position(|&h| h == "full_taxonomy") {
        Some(t) => (
            t,
            header
                .iter()
                .position(|&h| h == "full_confidences")
                .unwrap(),
        ),
        None => (
            header.iter().position(|&h| h == "taxonomy").unwrap(),
            header.iter().position(|&h| h == "confidences").unwrap(),
        ),
    };
    lines
        .map(|l| {
            let f: Vec<&str> = l.split('\t').collect();
            let mut tax: Vec<String> = f[ti]
                .split(';')
                .filter(|x| !x.is_empty())
                .map(String::from)
                .collect();
            let mut conf: Vec<f64> = f[ci]
                .split(';')
                .filter(|x| !x.is_empty())
                .map(|x| x.parse().unwrap())
                .collect();
            if tax.first().is_some_and(|t| t.starts_with("unclassified_")) {
                tax.clear();
                conf.clear();
            }
            (f[0].to_string(), (tax, conf))
        })
        .collect()
}

/// (assignment agreement over all ranks at threshold 60, mean |Δconf|)
fn agreement(
    a: &HashMap<String, (Vec<String>, Vec<f64>)>,
    b: &HashMap<String, (Vec<String>, Vec<f64>)>,
) -> (f64, f64) {
    let assigned = |t: &[String], c: &[f64], d: usize| -> Option<String> {
        (t.len() >= d && c[..d].iter().all(|&x| x >= 60.0)).then(|| t[..d].join(";"))
    };
    let (mut n, mut agree, mut dsum, mut dn) = (0, 0, 0.0, 0);
    for (id, (ta, ca)) in a {
        let (tb, cb) = &b[id];
        for d in 1..=ta.len().max(tb.len()) {
            n += 1;
            if assigned(ta, ca, d) == assigned(tb, cb, d) {
                agree += 1;
            }
            if d <= ca.len() && d <= cb.len() {
                dsum += (ca[d - 1] - cb[d - 1]).abs();
                dn += 1;
            }
        }
    }
    (agree as f64 / n as f64, dsum / dn as f64)
}

#[test]
fn parity_with_cached_decipher_outputs() {
    let dir = tempfile::tempdir().unwrap();
    let m = import(dir.path());
    let out = dir.path().join("rs.tsv");
    run(&[
        "classify",
        "--model",
        s(&m),
        "--input",
        s(&data("mini_queries.fasta")),
        "--output",
        s(&out),
        "--full-confidences",
    ]);
    let rs = load(&out);
    let d1 = load(&data("decipher_mini_seed1.tsv"));
    let d2 = load(&data("decipher_mini_seed2.tsv"));
    assert_eq!(rs.len(), d1.len());
    let (base_agree, base_mad) = agreement(&d1, &d2);
    let (agree, mad) = agreement(&d1, &rs);
    eprintln!("DECIPHER seed1 vs seed2: {base_agree:.4} / {base_mad:.2}; vs idtaxa-rs: {agree:.4} / {mad:.2}");
    // statistically equivalent: within DECIPHER's own run-to-run variability
    assert!(
        agree >= base_agree - 0.01,
        "agreement {agree} vs baseline {base_agree}"
    );
    assert!(agree >= 0.97);
    assert!(
        mad <= base_mad + 1.0,
        "confidence MAD {mad} vs baseline {base_mad}"
    );
}

#[test]
fn kmers_match_decipher_and_info() {
    let dir = tempfile::tempdir().unwrap();
    let m = import(dir.path());
    let out = run(&[
        "kmer-check",
        "--model",
        s(&m),
        "--ref",
        s(&data("mini_train.fasta")),
    ]);
    assert!(out.contains("mismatched\t0"), "{out}");
    let info = run(&["info", "--model", s(&m)]);
    assert!(info.contains("training_sequences\t412"));
    assert!(info.contains("K\t8"));
    assert!(info.contains("taxa\t476"));
}

#[test]
fn native_training_matches_decipher_model() {
    let dir = tempfile::tempdir().unwrap();
    let m = import(dir.path());
    let n = dir.path().join("native.idtx");
    run(&[
        "train",
        "--ref",
        s(&data("mini_train.fasta")),
        "--taxonomy",
        s(&data("mini_train_taxonomy.tsv")),
        "--k",
        "8",
        "--output",
        s(&n),
    ]);
    let cmp: serde_json::Value =
        serde_json::from_str(&run(&["compare-models", "--a", s(&m), "--b", s(&n)])).unwrap();
    assert_eq!(cmp["same_taxonomy"], true);
    assert!(cmp["mean_decision_kmer_jaccard"].as_f64().unwrap() > 0.99);
    assert!(cmp["max_idf_diff"].as_f64().unwrap() < 1e-9);
    // classification with the native model agrees with the DECIPHER model
    let a = dir.path().join("a.tsv");
    let b = dir.path().join("b.tsv");
    for (model, out) in [(&m, &a), (&n, &b)] {
        run(&[
            "classify",
            "--model",
            s(model),
            "--input",
            s(&data("mini_queries.fasta")),
            "--output",
            s(out),
            "--full-confidences",
        ]);
    }
    let (agree, _) = agreement(&load(&a), &load(&b));
    assert!(agree > 0.96, "native vs DECIPHER-trained agreement {agree}");
}

#[test]
fn deterministic_across_threads_batches_and_chunks() {
    let dir = tempfile::tempdir().unwrap();
    let m = import(dir.path());
    let q = data("mini_queries.fasta");
    let full = dir.path().join("full.tsv");
    run(&[
        "classify",
        "--model",
        s(&m),
        "--input",
        s(&q),
        "--output",
        s(&full),
        "--threads",
        "1",
    ]);
    let alt = dir.path().join("alt.tsv");
    run(&[
        "classify",
        "--model",
        s(&m),
        "--input",
        s(&q),
        "--output",
        s(&alt),
        "--threads",
        "3",
        "--batch-size",
        "7",
    ]);
    assert_eq!(std::fs::read(&full).unwrap(), std::fs::read(&alt).unwrap());
    let mut parts = vec![];
    for i in 1..=3 {
        let p = dir.path().join(format!("part{i}.tsv"));
        run(&[
            "classify",
            "--model",
            s(&m),
            "--input",
            s(&q),
            "--output",
            s(&p),
            "--chunk",
            &format!("{i}/3"),
            "--batch-size",
            "50",
        ]);
        parts.push(p);
    }
    let merged = dir.path().join("merged.tsv");
    let mut args = vec!["merge", "--output", s(&merged), "--inputs"];
    // order of inputs must not matter
    args.extend(parts.iter().rev().map(|p| s(p)));
    run(&args);
    assert_eq!(
        std::fs::read_to_string(&full).unwrap(),
        std::fs::read_to_string(&merged).unwrap()
    );
}

#[test]
fn cache_round_trip() {
    let dir = tempfile::tempdir().unwrap();
    let m = import(dir.path());
    let q = data("mini_queries.fasta");
    let cache = dir.path().join("cache.redb");
    let a = dir.path().join("a.tsv");
    let b = dir.path().join("b.tsv");
    run(&[
        "classify",
        "--model",
        s(&m),
        "--input",
        s(&q),
        "--output",
        s(&a),
        "--cache",
        s(&cache),
    ]);
    let out = bin()
        .args([
            "classify",
            "--model",
            s(&m),
            "--input",
            s(&q),
            "--output",
            s(&b),
            "--cache",
            s(&cache),
        ])
        .output()
        .unwrap();
    assert!(out.status.success());
    let err = String::from_utf8_lossy(&out.stderr);
    assert!(err.contains("572 from cache"), "{err}");
    assert_eq!(std::fs::read(&a).unwrap(), std::fs::read(&b).unwrap());
}

#[test]
fn decipher_format_and_edge_cases() {
    let dir = tempfile::tempdir().unwrap();
    let m = import(dir.path());
    let q = dir.path().join("q.fa");
    std::fs::write(&q, ">short\nACGTACGTAC\n>dup1\nNNNNNNNNNNNNNNNNNNNNNNNN\n").unwrap();
    let out = dir.path().join("o.tsv");
    run(&[
        "classify",
        "--model",
        s(&m),
        "--input",
        s(&q),
        "--output",
        s(&out),
        "--format",
        "decipher",
    ]);
    let text = std::fs::read_to_string(&out).unwrap();
    assert!(
        text.contains("short\tRoot [0%]; unclassified_Root [0%]"),
        "{text}"
    );
    // a real query produces a DECIPHER-style collapsed string
    let out2 = dir.path().join("o2.tsv");
    run(&[
        "classify",
        "--model",
        s(&m),
        "--input",
        s(&data("mini_queries.fasta")),
        "--output",
        s(&out2),
        "--format",
        "decipher",
    ]);
    let first = std::fs::read_to_string(&out2).unwrap();
    assert!(first.lines().next().unwrap().contains("\tRoot ["));
}

#[test]
fn clear_errors() {
    let dir = tempfile::tempdir().unwrap();
    let m = import(dir.path());
    // model version mismatch
    let bad = dir.path().join("bad.idtx");
    let mut bytes = std::fs::read(&m).unwrap();
    bytes[8] = 99;
    std::fs::write(&bad, &bytes).unwrap();
    let out = bin().args(["info", "--model", s(&bad)]).output().unwrap();
    assert!(!out.status.success());
    assert!(String::from_utf8_lossy(&out.stderr).contains("version mismatch"));
    // malformed FASTA
    let fa = dir.path().join("x.fa");
    std::fs::write(&fa, "this is not fasta\n").unwrap();
    let out = bin()
        .args([
            "classify",
            "--model",
            s(&m),
            "--input",
            s(&fa),
            "--output",
            s(&dir.path().join("o.tsv")),
        ])
        .output()
        .unwrap();
    assert!(!out.status.success());
    assert!(String::from_utf8_lossy(&out.stderr).contains("malformed sequence file"));
    // taxonomy mismatch
    let tx = dir.path().join("t.tsv");
    std::fs::write(&tx, "nope\tRoot;A;B\n").unwrap();
    let out = bin()
        .args([
            "train",
            "--ref",
            s(&data("mini_train.fasta")),
            "--taxonomy",
            s(&tx),
            "--output",
            s(&dir.path().join("m.idtx")),
        ])
        .output()
        .unwrap();
    assert!(!out.status.success());
    assert!(String::from_utf8_lossy(&out.stderr).contains("has no entry"));
}

#[test]
fn validate_calibrate_and_extensions_run() {
    let dir = tempfile::tempdir().unwrap();
    let d = dir.path();
    let prefix = d.join("split");
    run(&[
        "split",
        "--ref",
        s(&data("mini_train.fasta")),
        "--taxonomy",
        s(&data("mini_train_taxonomy.tsv")),
        "--fraction",
        "0.2",
        "--output-prefix",
        s(&prefix),
    ]);
    let model = d.join("m.idtx");
    let p = |x: &str| format!("{}_{x}", prefix.display());
    run(&[
        "train",
        "--ref",
        &p("train.fasta"),
        "--taxonomy",
        &p("train.tsv"),
        "--k",
        "8",
        "--output",
        s(&model),
    ]);
    let vprefix = d.join("val");
    run(&[
        "validate",
        "--model",
        s(&model),
        "--holdout",
        &p("holdout.fasta"),
        "--taxonomy",
        &p("holdout.tsv"),
        "--output-prefix",
        s(&vprefix),
        "--compare-mode",
        "standard,balanced",
        "--ci-replicates",
        "100",
    ]);
    let summary = std::fs::read_to_string(format!("{}_summary.txt", vprefix.display())).unwrap();
    assert!(summary.contains("Comparison standard -> balanced"));
    let cal = d.join("cal.json");
    run(&[
        "calibrate",
        "--model",
        s(&model),
        "--holdout",
        &p("holdout.fasta"),
        "--taxonomy",
        &p("holdout.tsv"),
        "--novelty-holdout",
        &p("holdout.fasta"),
        "--novelty-taxonomy",
        &p("holdout.tsv"),
        "--output",
        s(&cal),
        "--reliability",
        s(&d.join("rel.tsv")),
    ]);
    let out = d.join("c.tsv");
    run(&[
        "classify",
        "--model",
        s(&model),
        "--input",
        &p("holdout.fasta"),
        "--output",
        s(&out),
        "--calibration",
        s(&cal),
        "--novelty-flag",
        "--adaptive-bootstraps",
        "300",
    ]);
    let text = std::fs::read_to_string(&out).unwrap();
    let header = text.lines().next().unwrap();
    assert!(
        header.contains("calibrated_confidences")
            && header.contains("novelty_rank")
            && header.contains("bootstraps_used")
    );
    // calibration is refused for another model
    let other = d.join("o.tsv");
    let res = bin()
        .args([
            "classify",
            "--model",
            s(&import(d)),
            "--input",
            &p("holdout.fasta"),
            "--output",
            s(&other),
            "--calibration",
            s(&cal),
        ])
        .output()
        .unwrap();
    assert!(!res.status.success());
    // spaced and balanced training run
    for mode in ["spaced", "balanced"] {
        run(&[
            "train",
            "--ref",
            &p("train.fasta"),
            "--taxonomy",
            &p("train.tsv"),
            "--k",
            "8",
            "--mode",
            mode,
            "--output",
            s(&d.join(format!("{mode}.idtx"))),
        ]);
    }
    let info = run(&["info", "--model", s(&d.join("spaced.idtx"))]);
    assert!(info.contains("seed_pattern\t11011011011"));
}
