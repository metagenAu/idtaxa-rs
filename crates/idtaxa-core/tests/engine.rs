//! Engine-level tests on the committed DECIPHER fixture.

use std::path::Path;

use idtaxa_core::import::import_decipher;
use idtaxa_core::io::read_all;
use idtaxa_core::{Classifier, ClassifyParams, Model, Strand};

fn fixture() -> Model {
    import_decipher(
        &Path::new(env!("CARGO_MANIFEST_DIR")).join("../../tests/data/decipher_mini_export"),
    )
    .unwrap()
}

fn queries() -> Vec<Vec<u8>> {
    read_all(&Path::new(env!("CARGO_MANIFEST_DIR")).join("../../tests/data/mini_queries.fasta"))
        .unwrap()
        .into_iter()
        .map(|r| r.seq)
        .collect()
}

#[test]
fn pruned_winners_equal_full_reference_path() {
    let model = fixture();
    let qs = queries();
    for strand in [Strand::Both, Strand::Top] {
        let params = ClassifyParams {
            strand,
            ..Default::default()
        };
        let fast = Classifier::new(&model, params.clone()).unwrap();
        let mut full = Classifier::new(&model, params).unwrap();
        full.force_full = true;
        let (mut s1, mut s2) = (fast.new_scratch(), full.new_scratch());
        for q in &qs {
            assert_eq!(fast.classify(q, &mut s1), full.classify(q, &mut s2));
        }
    }
}

#[test]
fn save_load_roundtrip_preserves_results() {
    let model = fixture();
    let dir = tempfile::tempdir().unwrap();
    let p = dir.path().join("m.idtx");
    model.save(&p).unwrap();
    let loaded = Model::load(&p).unwrap();
    assert_eq!(loaded.meta.model_id, model.meta.model_id);
    assert_eq!(loaded.n_seqs(), model.n_seqs());
    assert_eq!(loaded.taxonomy, model.taxonomy);
    for i in 0..model.n_seqs() as u32 {
        assert_eq!(loaded.seq_kmers(i), model.seq_kmers(i));
    }
    let qs = queries();
    let a = Classifier::new(&model, ClassifyParams::default()).unwrap();
    let b = Classifier::new(&loaded, ClassifyParams::default()).unwrap();
    let (mut s1, mut s2) = (a.new_scratch(), b.new_scratch());
    for q in qs.iter().take(100) {
        assert_eq!(a.classify(q, &mut s1), b.classify(q, &mut s2));
    }
    // re-encoding raw parts reproduces the same model ID
    let again = Model::from_raw(loaded.to_raw()).unwrap();
    assert_eq!(again.meta.model_id, model.meta.model_id);
}

#[test]
fn confidences_are_monotone_and_bounded() {
    let model = fixture();
    let c = Classifier::new(&model, ClassifyParams::default()).unwrap();
    let mut sc = c.new_scratch();
    for q in queries() {
        let r = c.classify(&q, &mut sc);
        assert_eq!(r.lineage.len(), r.confidences.len());
        assert!(r.confidences.windows(2).all(|w| w[0] + 1e-9 >= w[1]));
        assert!(r
            .confidences
            .iter()
            .all(|&x| (0.0..=100.0 + 1e-9).contains(&x)));
        assert_eq!(r.lineage[0], 0);
    }
}
