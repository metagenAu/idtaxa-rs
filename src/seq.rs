//! Sequence input/output (FASTA / FASTQ, plain or gzip) and basic utilities.

use anyhow::{bail, Context, Result};
use flate2::read::MultiGzDecoder;
use flate2::write::GzEncoder;
use flate2::Compression;
use std::fs::File;
use std::io::{self, BufRead, BufReader, BufWriter, Write};
use std::path::Path;

/// A single sequence record.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Record {
    /// Identifier: header text up to the first whitespace.
    pub id: String,
    /// Remainder of the header line (may be empty).
    pub desc: String,
    /// Raw sequence bytes.
    pub seq: Vec<u8>,
}

impl Record {
    /// Full header line (without the leading `>` / `@`).
    pub fn header(&self) -> String {
        if self.desc.is_empty() {
            self.id.clone()
        } else {
            format!("{} {}", self.id, self.desc)
        }
    }
}

/// Open a file for reading, transparently decompressing gzip. `-` means stdin.
pub fn open_input(path: &Path) -> Result<Box<dyn BufRead>> {
    if path.as_os_str() == "-" {
        return Ok(Box::new(BufReader::new(io::stdin())));
    }
    let f = File::open(path).with_context(|| format!("cannot open {}", path.display()))?;
    let mut r = BufReader::with_capacity(1 << 16, f);
    let is_gz = {
        let b = r.fill_buf()?;
        b.len() >= 2 && b[0] == 0x1f && b[1] == 0x8b
    };
    if is_gz {
        Ok(Box::new(BufReader::with_capacity(1 << 16, MultiGzDecoder::new(r))))
    } else {
        Ok(Box::new(r))
    }
}

/// Create an output writer. `None` or `-` means stdout; a `.gz` extension compresses.
pub fn create_output(path: Option<&Path>) -> Result<Box<dyn Write>> {
    match path {
        None => Ok(Box::new(BufWriter::new(io::stdout()))),
        Some(p) if p.as_os_str() == "-" => Ok(Box::new(BufWriter::new(io::stdout()))),
        Some(p) => {
            let f = File::create(p).with_context(|| format!("cannot create {}", p.display()))?;
            if p.extension().map(|e| e == "gz").unwrap_or(false) {
                Ok(Box::new(GzEncoder::new(BufWriter::new(f), Compression::default())))
            } else {
                Ok(Box::new(BufWriter::new(f)))
            }
        }
    }
}

/// Streaming FASTA/FASTQ reader. The format is detected per record from the
/// header character (`>` or `@`). FASTQ records must be four lines.
pub struct SeqReader {
    inner: Box<dyn BufRead>,
    pending: Option<String>,
    buf: String,
}

impl SeqReader {
    pub fn new(inner: Box<dyn BufRead>) -> Self {
        SeqReader { inner, pending: None, buf: String::new() }
    }

    pub fn from_path(path: &Path) -> Result<Self> {
        Ok(Self::new(open_input(path)?))
    }

    fn read_line(&mut self) -> Result<Option<String>> {
        self.buf.clear();
        let n = self.inner.read_line(&mut self.buf)?;
        if n == 0 {
            Ok(None)
        } else {
            Ok(Some(self.buf.trim_end_matches(['\n', '\r']).to_string()))
        }
    }

    fn next_record(&mut self) -> Result<Option<Record>> {
        let header = match self.pending.take() {
            Some(h) => h,
            None => loop {
                match self.read_line()? {
                    None => return Ok(None),
                    Some(l) if l.trim().is_empty() => continue,
                    Some(l) => break l,
                }
            },
        };
        if let Some(h) = header.strip_prefix('>') {
            let (id, desc) = split_header(h);
            let mut seq = Vec::new();
            while let Some(l) = self.read_line()? {
                if l.starts_with('>') {
                    self.pending = Some(l);
                    break;
                }
                seq.extend(l.bytes().filter(|b| !b.is_ascii_whitespace()));
            }
            Ok(Some(Record { id, desc, seq }))
        } else if let Some(h) = header.strip_prefix('@') {
            let (id, desc) = split_header(h);
            let seq = self.read_line()?.context("truncated FASTQ record (missing sequence)")?;
            let plus = self.read_line()?.context("truncated FASTQ record (missing '+')")?;
            if !plus.starts_with('+') {
                bail!("malformed FASTQ record '{id}': expected '+' line");
            }
            self.read_line()?.context("truncated FASTQ record (missing quality)")?;
            Ok(Some(Record { id, desc, seq: seq.into_bytes() }))
        } else {
            let shown: String = header.chars().take(60).collect();
            bail!("unrecognised sequence format (expected FASTA '>' or FASTQ '@'): {shown}")
        }
    }
}

impl Iterator for SeqReader {
    type Item = Result<Record>;
    fn next(&mut self) -> Option<Self::Item> {
        self.next_record().transpose()
    }
}

/// Split a header into identifier and description.
pub fn split_header(h: &str) -> (String, String) {
    let h = h.trim();
    match h.find(char::is_whitespace) {
        Some(i) => (h[..i].to_string(), h[i..].trim().to_string()),
        None => (h.to_string(), String::new()),
    }
}

/// Read every record of a file into memory.
pub fn read_all(path: &Path) -> Result<Vec<Record>> {
    SeqReader::from_path(path)?.collect::<Result<Vec<_>>>().with_context(|| format!("reading {}", path.display()))
}

/// Upper-case, convert U→T and remove alignment gaps / whitespace.
pub fn clean(seq: &[u8]) -> Vec<u8> {
    seq.iter()
        .filter(|b| !matches!(b, b'-' | b'.' | b' ' | b'\t' | b'\r' | b'\n'))
        .map(|&b| {
            let u = b.to_ascii_uppercase();
            if u == b'U' {
                b'T'
            } else {
                u
            }
        })
        .collect()
}

/// IUPAC-aware complement of a single base.
pub fn complement(b: u8) -> u8 {
    match b.to_ascii_uppercase() {
        b'A' => b'T',
        b'C' => b'G',
        b'G' => b'C',
        b'T' | b'U' => b'A',
        b'R' => b'Y',
        b'Y' => b'R',
        b'S' => b'S',
        b'W' => b'W',
        b'K' => b'M',
        b'M' => b'K',
        b'B' => b'V',
        b'V' => b'B',
        b'D' => b'H',
        b'H' => b'D',
        _ => b'N',
    }
}

/// Reverse complement.
pub fn revcomp(seq: &[u8]) -> Vec<u8> {
    seq.iter().rev().map(|&b| complement(b)).collect()
}

/// Write a FASTA record wrapped at 80 columns.
pub fn write_fasta(w: &mut dyn Write, header: &str, seq: &[u8]) -> io::Result<()> {
    writeln!(w, ">{header}")?;
    for chunk in seq.chunks(80) {
        w.write_all(chunk)?;
        w.write_all(b"\n")?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_fasta_and_fastq() {
        let read = |d: &str| -> Vec<Record> {
            SeqReader::new(Box::new(io::Cursor::new(d.as_bytes().to_vec()))).collect::<Result<_>>().unwrap()
        };
        let recs = read(">a desc here\nACGT\nacgu\n\n>b\nNN\n");
        assert_eq!(recs.len(), 2);
        assert_eq!(recs[0].id, "a");
        assert_eq!(recs[0].desc, "desc here");
        assert_eq!(recs[0].seq, b"ACGTacgu");
        let recs = read("@r1 x\nACGT\n+\nIIII\n@r2\nGG\n+\nII\n");
        assert_eq!(recs.len(), 2);
        assert_eq!(recs[0].id, "r1");
        assert_eq!(recs[1].seq, b"GG");
    }

    #[test]
    fn cleans_and_revcomps() {
        assert_eq!(clean(b"ac-g.u"), b"ACGT");
        assert_eq!(revcomp(b"AACGTR"), b"YACGTT");
    }
}
