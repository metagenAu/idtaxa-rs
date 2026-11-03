//! In-silico PCR amplicon extraction (Phase 6.1).
//!
//! Primers are matched IUPAC-aware (a primer base matches a template base if
//! their nucleotide sets intersect) without indels, allowing up to
//! `max_mismatch` mismatches but requiring the `three_prime_exact` 3'-terminal
//! bases to match. Both template orientations are tried. References lacking
//! a primer site can optionally be rescued by anchoring the first and last
//! 20 bases of similar, successfully extracted regions.

use rayon::prelude::*;

use crate::error::{Error, Result};
use crate::kmer::{reverse_complement, KmerEncoder};

/// Nucleotide set bitmask (A=1, C=2, G=4, T=8) of an IUPAC character.
pub fn iupac_mask(c: u8) -> u8 {
    match c.to_ascii_uppercase() {
        b'A' => 1,
        b'C' => 2,
        b'G' => 4,
        b'T' | b'U' => 8,
        b'R' => 1 | 4,
        b'Y' => 2 | 8,
        b'S' => 2 | 4,
        b'W' => 1 | 8,
        b'K' => 4 | 8,
        b'M' => 1 | 2,
        b'B' => 2 | 4 | 8,
        b'D' => 1 | 4 | 8,
        b'H' => 1 | 2 | 8,
        b'V' => 1 | 2 | 4,
        b'N' => 15,
        _ => 0,
    }
}

/// Extraction options.
#[derive(Clone, Debug)]
pub struct ExtractOptions {
    pub fwd: Vec<u8>,
    pub rev: Vec<u8>,
    pub max_mismatch: usize,
    pub three_prime_exact: usize,
    pub min_len: usize,
    pub max_len: usize,
    pub keep_primers: bool,
    pub rescue: bool,
}

/// Per-sequence outcome.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Status {
    Extracted,
    Rescued,
    NoSite,
    Length,
}

impl Status {
    pub fn as_str(&self) -> &'static str {
        match self {
            Status::Extracted => "extracted",
            Status::Rescued => "rescued",
            Status::NoSite => "no_primer_site",
            Status::Length => "length_filtered",
        }
    }
}

/// Result for one reference.
#[derive(Clone, Debug)]
pub struct Extraction {
    pub status: Status,
    pub region: Option<Vec<u8>>,
    pub reverse: bool,
    /// Region coordinates in the (possibly reverse-complemented) template.
    pub start: usize,
    pub end: usize,
    pub fwd_mm: Option<usize>,
    pub rev_mm: Option<usize>,
}

/// Best match of `pat` in `seq`: (position, mismatches). `exact_end`
/// bases at the pattern end (or start if `exact_at_start`) must match.
fn find(
    seq: &[u8],
    pat: &[u8],
    max_mm: usize,
    exact: usize,
    exact_at_start: bool,
    leftmost: bool,
) -> Option<(usize, usize)> {
    if seq.len() < pat.len() || pat.is_empty() {
        return None;
    }
    let pm: Vec<u8> = pat.iter().map(|&c| iupac_mask(c)).collect();
    let exact = exact.min(pat.len());
    let mut best: Option<(usize, usize)> = None;
    for p in 0..=seq.len() - pat.len() {
        let mut mm = 0;
        let mut ok = true;
        for (j, &m) in pm.iter().enumerate() {
            if iupac_mask(seq[p + j]) & m == 0 {
                let in_exact = if exact_at_start {
                    j < exact
                } else {
                    j >= pat.len() - exact
                };
                mm += 1;
                if in_exact || mm > max_mm {
                    ok = false;
                    break;
                }
            }
        }
        if ok {
            let better = match best {
                None => true,
                Some((_, bm)) => mm < bm || (mm == bm && !leftmost),
            };
            if better {
                best = Some((p, mm));
            }
        }
    }
    best
}

fn try_orientation(
    t: &[u8],
    o: &ExtractOptions,
    rc_rev: &[u8],
) -> Option<(usize, usize, usize, usize)> {
    // forward primer: 3' end = pattern end
    let (fp, fmm) = find(t, &o.fwd, o.max_mismatch, o.three_prime_exact, false, true)?;
    let after = fp + o.fwd.len();
    // reverse primer site appears as its reverse complement; its 3' end is
    // the start of the RC pattern
    let (rp_rel, rmm) = find(
        &t[after..],
        rc_rev,
        o.max_mismatch,
        o.three_prime_exact,
        true,
        true,
    )?;
    let rp = after + rp_rel;
    let (s, e) = if o.keep_primers {
        (fp, rp + rc_rev.len())
    } else {
        (after, rp)
    };
    Some((s, e, fmm, rmm))
}

/// Extract regions from all references.
pub fn extract(seqs: &[&[u8]], o: &ExtractOptions) -> Result<Vec<Extraction>> {
    if o.fwd
        .iter()
        .chain(o.rev.iter())
        .any(|&c| iupac_mask(c) == 0)
    {
        return Err(Error::InvalidParameter(
            "primers must contain only IUPAC nucleotide codes".into(),
        ));
    }
    let rc_rev = reverse_complement(&o.rev);
    let mut res: Vec<Extraction> = seqs
        .par_iter()
        .map(|&s| {
            let fwd_try = try_orientation(s, o, &rc_rev).map(|r| (r, false, s.to_vec()));
            let found = fwd_try.or_else(|| {
                let rc = reverse_complement(s);
                try_orientation(&rc, o, &rc_rev).map(|r| (r, true, rc))
            });
            match found {
                Some(((st, en, fmm, rmm), rev, t)) => {
                    let len = en - st;
                    let status = if len < o.min_len || len > o.max_len {
                        Status::Length
                    } else {
                        Status::Extracted
                    };
                    Extraction {
                        status,
                        region: (status == Status::Extracted).then(|| t[st..en].to_vec()),
                        reverse: rev,
                        start: st,
                        end: en,
                        fwd_mm: Some(fmm),
                        rev_mm: Some(rmm),
                    }
                }
                None => Extraction {
                    status: Status::NoSite,
                    region: None,
                    reverse: false,
                    start: 0,
                    end: 0,
                    fwd_mm: None,
                    rev_mm: None,
                },
            }
        })
        .collect();

    if o.rescue {
        // seed set: up to 2000 extracted regions (deterministic stride)
        let ext: Vec<usize> = (0..res.len())
            .filter(|&i| res[i].status == Status::Extracted)
            .collect();
        if ext.is_empty() {
            return Ok(res);
        }
        let stride = (ext.len() / 2000).max(1);
        let seeds: Vec<&Vec<u8>> = ext
            .iter()
            .step_by(stride)
            .map(|&i| res[i].region.as_ref().unwrap())
            .collect();
        let enc = KmerEncoder::contiguous(8)?;
        let seed_k: Vec<Vec<u32>> = seeds.iter().map(|s| enc.encode(s).unique).collect();
        let anchor = 20usize;
        let rescued: Vec<(usize, Extraction)> = (0..res.len())
            .into_par_iter()
            .filter(|&i| res[i].status == Status::NoSite)
            .filter_map(|i| {
                for (rev, t) in [
                    (false, seqs[i].to_vec()),
                    (true, reverse_complement(seqs[i])),
                ] {
                    let tk = enc.encode(&t).unique;
                    let mut cand: Vec<(usize, usize)> = seed_k
                        .iter()
                        .enumerate()
                        .map(|(j, k)| {
                            (
                                crate::balance::jaccard(k, &tk).mul_add(1e6, 0.0) as usize,
                                j,
                            )
                        })
                        .collect();
                    cand.sort_by(|a, b| b.0.cmp(&a.0).then(a.1.cmp(&b.1)));
                    for &(_, j) in cand.iter().take(5) {
                        let s = seeds[j];
                        if s.len() < 2 * anchor {
                            continue;
                        }
                        let head = find(&t, &s[..anchor], 3, 0, false, true);
                        let tail = find(&t, &s[s.len() - anchor..], 3, 0, false, false);
                        if let (Some((h, _)), Some((tp, _))) = (head, tail) {
                            let en = tp + anchor;
                            if en > h {
                                let len = en - h;
                                if len >= o.min_len && len <= o.max_len {
                                    return Some((
                                        i,
                                        Extraction {
                                            status: Status::Rescued,
                                            region: Some(t[h..en].to_vec()),
                                            reverse: rev,
                                            start: h,
                                            end: en,
                                            fwd_mm: None,
                                            rev_mm: None,
                                        },
                                    ));
                                }
                            }
                        }
                    }
                }
                None
            })
            .collect();
        for (i, e) in rescued {
            res[i] = e;
        }
    }
    Ok(res)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn opts() -> ExtractOptions {
        ExtractOptions {
            fwd: b"ACGTRC".to_vec(),
            rev: b"GGCCAA".to_vec(), // site in template: TTGGCC
            max_mismatch: 1,
            three_prime_exact: 2,
            min_len: 0,
            max_len: usize::MAX,
            keep_primers: false,
            rescue: false,
        }
    }

    #[test]
    fn extracts_both_orientations() {
        let t = b"TTTTACGTACAAAAAGGGGGTTGGCCTTTT";
        let r = extract(&[t], &opts()).unwrap();
        assert_eq!(r[0].status, Status::Extracted);
        assert_eq!(r[0].region.as_deref(), Some(&b"AAAAAGGGGG"[..]));
        let rc = reverse_complement(t);
        let r = extract(&[&rc], &opts()).unwrap();
        assert!(r[0].reverse);
        assert_eq!(r[0].region.as_deref(), Some(&b"AAAAAGGGGG"[..]));
        let mut o = opts();
        o.keep_primers = true;
        let r = extract(&[t], &o).unwrap();
        assert_eq!(r[0].region.as_deref(), Some(&b"ACGTACAAAAAGGGGGTTGGCC"[..]));
    }

    #[test]
    fn mismatch_rules() {
        // one mismatch away from 3' end: allowed
        let t = b"TTTTTCGTACAAAAAGGGGGTTGGCCTTTT";
        assert_eq!(extract(&[t], &opts()).unwrap()[0].fwd_mm, Some(1));
        // mismatch in the 3'-terminal bases: rejected
        let t = b"TTTTACGTAGAAAAAGGGGGTTGGCCTTTT";
        assert_eq!(extract(&[t], &opts()).unwrap()[0].status, Status::NoSite);
        // length filter
        let mut o = opts();
        o.min_len = 50;
        let t = b"TTTTACGTACAAAAAGGGGGTTGGCCTTTT";
        assert_eq!(extract(&[t], &o).unwrap()[0].status, Status::Length);
    }

    #[test]
    fn rescue_by_anchors() {
        let core = b"GATTACAGATTACACCCTTTGGGAAACCCTTTGGGAAACATCATCATCATGGATCCAAGCTT";
        let mut full = b"TTTTACGTAC".to_vec();
        full.extend_from_slice(core);
        full.extend_from_slice(b"TTGGCCTTTT");
        let mut partial = b"NNNN".to_vec(); // no forward primer site
        partial.extend_from_slice(core);
        partial.extend_from_slice(b"TTTT");
        let mut o = opts();
        o.rescue = true;
        let r = extract(&[&full, &partial], &o).unwrap();
        assert_eq!(r[0].status, Status::Extracted);
        assert_eq!(r[1].status, Status::Rescued);
        assert_eq!(r[1].region.as_deref(), Some(&core[..]));
    }
}
