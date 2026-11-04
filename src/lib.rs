//! # idtaxa
//!
//! A Rust implementation of the IDTAXA taxonomic classification algorithm
//! (Murali, Bhargava & Wright 2018, *Microbiome* 6:140) with first-class
//! support for amplicon primer sets.
//!
//! The main building blocks are:
//!
//! * [`seq`] – FASTA/FASTQ (optionally gzipped) I/O and sequence utilities.
//! * [`kmer`] – k-mer encoding.
//! * [`taxonomy`] – lineage parsing (SILVA, GTDB, UNITE, QIIME, DECIPHER styles).
//! * [`primers`] – primer catalog, IUPAC matching and in-silico PCR.
//! * [`train`] – building a classifier ([`model::Model`]) from reference sequences.
//! * [`classify`] – assigning taxonomy with bootstrap confidences.
//! * [`curate`] – leave-one-out detection of mislabelled references.
//! * [`evaluate`] – hold-out cross-validation of a reference / primer combination.
//! * [`summarize`] – collapsing classified feature tables to taxon tables
//!   (wide or long/tidy format for cross-study integration).

pub mod classify;
pub mod curate;
pub mod evaluate;
pub mod kmer;
pub mod model;
pub mod primers;
pub mod seq;
pub mod summarize;
pub mod taxonomy;
pub mod train;
