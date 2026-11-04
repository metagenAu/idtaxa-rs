//! `idtaxa-core`: a scalable Rust port of the IDTAXA taxonomic classifier
//! (Murali, Bhargava & Wright 2018, Microbiome 6:140) as implemented by
//! `LearnTaxa()`/`IdTaxa()` in the R/Bioconductor package DECIPHER.
//!
//! Licensed GPL-3.0-or-later (a port of GPL-3 DECIPHER source).

#![allow(clippy::needless_range_loop)] // index-heavy numeric kernels read clearer with ranges

pub mod balance;
pub mod calibration;
pub mod classify;
pub mod clean;
pub mod consensus;
pub mod error;
pub mod eval;
pub mod import;
pub mod io;
pub mod kmer;
pub mod model;
pub mod output;
pub mod region;
pub mod stats;
pub mod taxonomy;
pub mod train;

pub use classify::{Classification, Classifier, ClassifyParams, Orientation, Scratch, Strand};
pub use error::{Error, Result};
pub use model::{Model, FORMAT_VERSION};
