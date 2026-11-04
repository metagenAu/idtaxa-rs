//! End-to-end tests on a synthetic taxonomy with known evolutionary distances.

use idtaxa::classify::{classify_seq, ClassifyParams, Strand};
use idtaxa::evaluate::evaluate;
use idtaxa::model::{Model, ModelMeta};
use idtaxa::primers::{find_pair, Pcr, PcrOptions};
use idtaxa::seq::revcomp;
use idtaxa::taxonomy::parse_lineage;
use idtaxa::train::{train, TrainParams, TrainSeq};
use rand::rngs::SmallRng;
use rand::{Rng, SeedableRng};

const BASES: &[u8] = b"ACGT";

fn random_seq(n: usize, rng: &mut SmallRng) -> Vec<u8> {
    (0..n).map(|_| BASES[rng.gen_range(0..4)]).collect()
}

fn mutate(s: &[u8], rate: f64, rng: &mut SmallRng) -> Vec<u8> {
    s.iter()
        .map(|&b| {
            if rng.gen_bool(rate) {
                let mut n = b;
                while n == b {
                    n = BASES[rng.gen_range(0..4)];
                }
                n
            } else {
                b
            }
        })
        .collect()
}

struct World {
    refs: Vec<TrainSeq>,
    /// (species ancestor sequence, lineage string)
    species: Vec<(Vec<u8>, String)>,
    phyla: Vec<Vec<u8>>,
}

/// 3 phyla × 3 genera × 3 species × 6 sequences, 400 bp, flanked by 515F/806R sites.
fn world(seed: u64) -> World {
    let mut rng = SmallRng::seed_from_u64(seed);
    let root = random_seq(400, &mut rng);
    let mut refs = Vec::new();
    let mut species = Vec::new();
    let mut phyla = Vec::new();
    for p in 0..3 {
        let ps = mutate(&root, 0.15, &mut rng);
        phyla.push(ps.clone());
        for g in 0..3 {
            let gs = mutate(&ps, 0.06, &mut rng);
            for s in 0..3 {
                let ss = mutate(&gs, 0.02, &mut rng);
                let lin = format!("d__Bacteria;p__P{p};g__P{p}G{g};s__P{p}G{g}S{s}");
                for i in 0..6 {
                    refs.push(TrainSeq {
                        id: format!("P{p}G{g}S{s}_{i}"),
                        seq: mutate(&ss, 0.005, &mut rng),
                        lineage: parse_lineage(&lin, None),
                    });
                }
                species.push((ss, lin));
            }
        }
    }
    World { refs, species, phyla }
}

fn trained(w: &World) -> Model {
    let (m, rep) = train(w.refs.clone(), &TrainParams::default(), ModelMeta::default()).unwrap();
    assert_eq!(rep.n_used + rep.n_duplicates, w.refs.len());
    m
}

fn names(m: &Model, a: &idtaxa::classify::Assignment) -> Vec<String> {
    a.classified().iter().map(|&(n, _)| m.taxonomy.nodes[n].name.clone()).collect()
}

#[test]
fn classifies_known_species() {
    let w = world(42);
    let m = trained(&w);
    assert_eq!(m.k, 7);
    assert_eq!(m.meta.ranks, vec!["domain", "phylum", "genus", "species"]);
    let p = ClassifyParams { strand: Strand::Top, ..Default::default() };
    let mut rng = SmallRng::seed_from_u64(7);
    let (mut genus_ok, mut species_ok, mut wrong, mut n) = (0, 0, 0, 0);
    for (ss, lin) in &w.species {
        let truth = parse_lineage(lin, None).names;
        for _ in 0..2 {
            let q = mutate(ss, 0.005, &mut rng);
            let a = classify_seq(&m, &q, &p, &mut rng);
            let got = names(&m, &a);
            n += 1;
            if got.len() >= 3 && got[..3] == truth[..3] {
                genus_ok += 1;
            }
            if got == truth {
                species_ok += 1;
            }
            if got.iter().zip(&truth).any(|(a, b)| a != b) {
                wrong += 1;
            }
        }
    }
    assert!(genus_ok as f64 / n as f64 > 0.95, "genus accuracy {genus_ok}/{n}");
    assert!(species_ok as f64 / n as f64 > 0.80, "species accuracy {species_ok}/{n}");
    assert!(wrong as f64 / n as f64 <= 0.05, "misclassified {wrong}/{n}");
}

#[test]
fn reverse_complement_and_novelty() {
    let w = world(43);
    let m = trained(&w);
    let p = ClassifyParams::default(); // strand = both
    let mut rng = SmallRng::seed_from_u64(9);

    // reverse-complemented query is recognised
    let (ss, lin) = &w.species[4];
    let a = classify_seq(&m, &revcomp(&mutate(ss, 0.005, &mut rng)), &p, &mut rng);
    assert!(a.reverse);
    assert_eq!(names(&m, &a), parse_lineage(lin, None).names);

    // a novel genus within P1 is placed in P1 but not assigned a genus
    let mut over = 0;
    for _ in 0..10 {
        let novel = mutate(&mutate(&w.phyla[1], 0.06, &mut rng), 0.02, &mut rng);
        let got = names(&m, &classify_seq(&m, &novel, &p, &mut rng));
        assert_eq!(got.get(1).map(String::as_str), Some("P1"), "{got:?}");
        if got.len() > 2 {
            over += 1;
        }
    }
    assert!(over <= 2, "novel genus over-classified {over}/10 times");

    // random sequence stays unclassified
    let junk = random_seq(400, &mut rng);
    let a = classify_seq(&m, &junk, &p, &mut rng);
    assert!(a.classified().len() <= 1, "{:?}", names(&m, &a));
}

#[test]
fn model_roundtrip() {
    let w = world(44);
    let m = trained(&w);
    let dir = std::env::temp_dir().join(format!("idtaxa-test-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("m.idtx");
    m.save(&path).unwrap();
    let m2 = Model::load(&path).unwrap();
    assert_eq!(m2.k, m.k);
    assert_eq!(m2.taxonomy.len(), m.taxonomy.len());
    assert!(m2.taxonomy.child(0, "Bacteria").is_some());
    std::fs::remove_dir_all(&dir).ok();
}

#[test]
fn primer_trimmed_training() {
    let w = world(45);
    let pair = find_pair("515F-806R").unwrap();
    let fwd = b"GTGCCAGCAGCCGCGGTAA";
    let rev_rc = revcomp(b"GGACTACAAGGGTATCTAAT");
    // embed the amplicon (bases 50..300) between primer sites, inside longer flanks
    let mut rng = SmallRng::seed_from_u64(1);
    let flank5 = random_seq(30, &mut rng);
    let flank3 = random_seq(30, &mut rng);
    let full: Vec<TrainSeq> = w
        .refs
        .iter()
        .map(|t| {
            let mut s = flank5.clone();
            s.extend_from_slice(fwd);
            s.extend_from_slice(&t.seq[50..300]);
            s.extend_from_slice(&rev_rc);
            s.extend_from_slice(&flank3);
            TrainSeq { seq: s, ..t.clone() }
        })
        .collect();
    let pcr = Pcr::from_pair(pair, PcrOptions::default());
    let trimmed: Vec<TrainSeq> = full
        .iter()
        .map(|t| {
            let a = pcr.amplify(&t.seq).expect("amplicon");
            assert_eq!(a.seq.len(), 250);
            TrainSeq { seq: a.seq, ..t.clone() }
        })
        .collect();
    let (m, _) = train(trimmed, &TrainParams::default(), ModelMeta::default()).unwrap();
    let (ss, lin) = &w.species[10];
    let q = mutate(&ss[50..300], 0.004, &mut rng);
    let p = ClassifyParams { strand: Strand::Top, ..Default::default() };
    let a = classify_seq(&m, &q, &p, &mut rng);
    assert_eq!(names(&m, &a), parse_lineage(lin, None).names);
}

#[test]
fn cross_validation_report() {
    let w = world(46);
    let rep = evaluate(w.refs.clone(), 0.2, &TrainParams::default(), &ClassifyParams::default()).unwrap();
    assert_eq!(rep.levels.len(), 4);
    let genus = &rep.levels[2];
    assert_eq!(genus.rank, "genus");
    assert!(genus.correct as f64 / genus.n_known as f64 > 0.9);
}
