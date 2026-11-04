//! Criterion benchmarks of the hot paths on the committed fixture:
//! `cargo bench -p idtaxa-core`.

use std::path::Path;

use criterion::{black_box, criterion_group, criterion_main, Criterion};
use idtaxa_core::import::import_decipher;
use idtaxa_core::io::read_all;
use idtaxa_core::kmer::KmerEncoder;
use idtaxa_core::{Classifier, ClassifyParams};

fn bench(c: &mut Criterion) {
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../tests/data");
    let model = import_decipher(&root.join("decipher_mini_export")).unwrap();
    let queries: Vec<Vec<u8>> = read_all(&root.join("mini_queries.fasta"))
        .unwrap()
        .into_iter()
        .map(|r| r.seq)
        .collect();

    let enc = KmerEncoder::contiguous(8).unwrap();
    c.bench_function("kmer_encode_both_450bp", |b| {
        b.iter(|| black_box(enc.encode_both(black_box(&queries[0]))))
    });

    for (name, force_full) in [
        ("classify_query_pruned", false),
        ("classify_query_full_reference_path", true),
    ] {
        let mut cl = Classifier::new(&model, ClassifyParams::default()).unwrap();
        cl.force_full = force_full;
        let mut sc = cl.new_scratch();
        let mut i = 0usize;
        c.bench_function(name, |b| {
            b.iter(|| {
                i = (i + 1) % queries.len();
                black_box(cl.classify(&queries[i], &mut sc))
            })
        });
    }
}

criterion_group! {
    name = benches;
    config = Criterion::default().sample_size(30);
    targets = bench
}
criterion_main!(benches);
