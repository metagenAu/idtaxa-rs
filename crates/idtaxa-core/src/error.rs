//! Library error type.
use std::path::PathBuf;

/// Errors produced by `idtaxa-core`.
#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("I/O error on {path}: {source}")]
    Io {
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },
    #[error("I/O error: {0}")]
    IoPlain(#[from] std::io::Error),
    #[error("malformed sequence file {path}: {msg}")]
    Fasta { path: PathBuf, msg: String },
    #[error("taxonomy error: {0}")]
    Taxonomy(String),
    #[error("invalid parameter: {0}")]
    InvalidParameter(String),
    #[error("invalid model file: {0}")]
    InvalidModel(String),
    #[error(
        "model format version mismatch: file has version {found}, this build supports version {expected}"
    )]
    ModelVersion { found: u32, expected: u32 },
    #[error("DECIPHER export error: {0}")]
    Import(String),
    #[error("JSON error: {0}")]
    Json(#[from] serde_json::Error),
}

/// Convenience alias.
pub type Result<T, E = Error> = std::result::Result<T, E>;

impl Error {
    pub(crate) fn io(path: impl Into<PathBuf>, source: std::io::Error) -> Self {
        Error::Io {
            path: path.into(),
            source,
        }
    }
}
