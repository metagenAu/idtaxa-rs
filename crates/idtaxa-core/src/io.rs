//! Sequence and taxonomy file I/O.

use std::collections::HashMap;
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::{Path, PathBuf};

use needletail::parser::FastxReader;

use crate::error::{Error, Result};

/// One FASTA/FASTQ record.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SeqRecord {
    /// Identifier (header up to the first whitespace).
    pub id: String,
    /// Rest of the header line after the identifier (may be empty).
    pub desc: String,
    /// Sequence with gap characters ('-', '.') and whitespace removed.
    pub seq: Vec<u8>,
}

/// Streaming FASTA/FASTQ reader (plain or gzip-compressed).
pub struct SeqReader {
    inner: Box<dyn FastxReader>,
    path: PathBuf,
}

impl SeqReader {
    /// Open a FASTA/FASTQ file; `-` reads standard input.
    pub fn open(path: &Path) -> Result<Self> {
        let inner = if path.as_os_str() == "-" {
            needletail::parse_fastx_stdin().map_err(|e| Error::Fasta {
                path: path.into(),
                msg: e.to_string(),
            })?
        } else {
            needletail::parse_fastx_file(path).map_err(|e| Error::Fasta {
                path: path.into(),
                msg: e.to_string(),
            })?
        };
        Ok(Self {
            inner,
            path: path.into(),
        })
    }

    /// Read the next record.
    #[allow(clippy::should_implement_trait)]
    pub fn next(&mut self) -> Option<Result<SeqRecord>> {
        let rec = self.inner.next()?;
        Some(match rec {
            Ok(r) => {
                let header = String::from_utf8_lossy(r.id()).into_owned();
                let (id, desc) = match header.find(char::is_whitespace) {
                    Some(p) => (header[..p].to_string(), header[p..].trim().to_string()),
                    None => (header, String::new()),
                };
                let seq: Vec<u8> = r
                    .seq()
                    .iter()
                    .copied()
                    .filter(|&c| c != b'-' && c != b'.' && !c.is_ascii_whitespace())
                    .collect();
                Ok(SeqRecord { id, desc, seq })
            }
            Err(e) => Err(Error::Fasta {
                path: self.path.clone(),
                msg: e.to_string(),
            }),
        })
    }

    /// Append up to `n` records to `out`; returns false at end of input.
    pub fn read_batch(&mut self, n: usize, out: &mut Vec<SeqRecord>) -> Result<bool> {
        for _ in 0..n {
            match self.next() {
                Some(r) => out.push(r?),
                None => return Ok(false),
            }
        }
        Ok(true)
    }
}

/// Read every record of a sequence file.
pub fn read_all(path: &Path) -> Result<Vec<SeqRecord>> {
    let mut r = SeqReader::open(path)?;
    let mut v = Vec::new();
    while let Some(rec) = r.next() {
        v.push(rec?);
    }
    Ok(v)
}

/// Read a two-column taxonomy TSV (`id<TAB>lineage`). Lines starting with
/// `#` and an optional header whose second column is `taxonomy` are skipped.
pub fn read_taxonomy_tsv(path: &Path) -> Result<HashMap<String, String>> {
    let f = File::open(path).map_err(|e| Error::io(path, e))?;
    let mut map = HashMap::new();
    for (ln, line) in BufReader::new(f).lines().enumerate() {
        let line = line.map_err(|e| Error::io(path, e))?;
        let line = line.trim_end_matches('\r');
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let mut it = line.splitn(2, '\t');
        let id = it.next().unwrap().trim();
        let tax = it.next().ok_or_else(|| {
            Error::Taxonomy(format!(
                "{}:{}: expected 'id<TAB>lineage'",
                path.display(),
                ln + 1
            ))
        })?;
        if ln == 0 && tax.trim().eq_ignore_ascii_case("taxonomy") {
            continue;
        }
        if map.insert(id.to_string(), tax.trim().to_string()).is_some() {
            return Err(Error::Taxonomy(format!(
                "{}: duplicate id '{id}'",
                path.display()
            )));
        }
    }
    Ok(map)
}

/// Reference sequences with lineages, either from a taxonomy TSV or from
/// the FASTA headers (whole header after the id, or the id itself when it
/// contains ';', e.g. DADA2-formatted PR2/SILVA files).
pub fn read_reference(
    fasta: &Path,
    taxonomy: Option<&Path>,
) -> Result<(Vec<SeqRecord>, Vec<String>)> {
    let recs = read_all(fasta)?;
    let lineages = match taxonomy {
        Some(t) => {
            let map = read_taxonomy_tsv(t)?;
            let mut v = Vec::with_capacity(recs.len());
            for r in &recs {
                match map.get(&r.id) {
                    Some(l) => v.push(l.clone()),
                    None => {
                        return Err(Error::Taxonomy(format!(
                            "sequence '{}' in {} has no entry in {}",
                            r.id,
                            fasta.display(),
                            t.display()
                        )))
                    }
                }
            }
            v
        }
        None => recs
            .iter()
            .map(|r| {
                if r.id.contains(';') {
                    Ok(format!("{} {}", r.id, r.desc).trim().to_string())
                } else if r.desc.contains(';') {
                    Ok(r.desc.clone())
                } else {
                    Err(Error::Taxonomy(format!(
                        "sequence '{}' has no lineage in its header; supply --taxonomy",
                        r.id
                    )))
                }
            })
            .collect::<Result<_>>()?,
    };
    Ok((recs, lineages))
}
