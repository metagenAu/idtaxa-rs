//! Primer catalog, IUPAC-aware primer matching and in-silico PCR.
//!
//! Training a classifier on only the region amplified by a primer pair
//! (rather than full-length references) substantially improves
//! classification accuracy, and primer coverage reports reveal which taxa
//! a primer set is blind to.

use crate::seq::revcomp;

/// A named primer pair.
#[derive(Clone, Debug)]
pub struct PrimerPair {
    pub name: &'static str,
    pub marker: &'static str,
    pub region: &'static str,
    pub forward_name: &'static str,
    pub forward: &'static str,
    pub reverse_name: &'static str,
    pub reverse: &'static str,
    pub note: &'static str,
}

/// Primer pairs commonly used for soil and agricultural metabarcoding.
pub const CATALOG: &[PrimerPair] = &[
    PrimerPair {
        name: "515F-806R",
        marker: "16S",
        region: "V4",
        forward_name: "515F (Parada)",
        forward: "GTGYCAGCMGCCGCGGTAA",
        reverse_name: "806R (Apprill)",
        reverse: "GGACTACNVGGGTWTCTAAT",
        note: "Earth Microbiome Project standard (modified); improved Thaumarchaeota/SAR11 coverage",
    },
    PrimerPair {
        name: "515F-806R-orig",
        marker: "16S",
        region: "V4",
        forward_name: "515F (Caporaso)",
        forward: "GTGCCAGCMGCCGCGGTAA",
        reverse_name: "806R (Caporaso)",
        reverse: "GGACTACHVGGGTWTCTAAT",
        note: "Original EMP primers; under-detect Thaumarchaeota (soil ammonia oxidisers) and SAR11",
    },
    PrimerPair {
        name: "515F-926R",
        marker: "16S",
        region: "V4-V5",
        forward_name: "515F (Parada)",
        forward: "GTGYCAGCMGCCGCGGTAA",
        reverse_name: "926R",
        reverse: "CCGYCAATTYMTTTRAGTTT",
        note: "Parada et al. 2016; also amplifies eukaryotic 18S",
    },
    PrimerPair {
        name: "341F-785R",
        marker: "16S",
        region: "V3-V4",
        forward_name: "341F",
        forward: "CCTACGGGNGGCWGCAG",
        reverse_name: "785R",
        reverse: "GACTACHVGGGTATCTAATCC",
        note: "Klindworth et al. 2013; Illumina 16S protocol",
    },
    PrimerPair {
        name: "341F-806R",
        marker: "16S",
        region: "V3-V4",
        forward_name: "341F",
        forward: "CCTACGGGNGGCWGCAG",
        reverse_name: "806R",
        reverse: "GGACTACHVGGGTWTCTAAT",
        note: "Common commercial V3-V4 set",
    },
    PrimerPair {
        name: "27F-519R",
        marker: "16S",
        region: "V1-V3",
        forward_name: "27F",
        forward: "AGAGTTTGATCMTGGCTCAG",
        reverse_name: "519R",
        reverse: "GWATTACCGCGGCKGCTG",
        note: "Australian Microbiome bacterial 16S",
    },
    PrimerPair {
        name: "799F-1193R",
        marker: "16S",
        region: "V5-V7",
        forward_name: "799F",
        forward: "AACMGGATTAGATACCCKG",
        reverse_name: "1193R",
        reverse: "ACGTCATCCCCACCTTCC",
        note: "Excludes most chloroplast; for rhizosphere/endophyte samples",
    },
    PrimerPair {
        name: "27F-1492R",
        marker: "16S",
        region: "V1-V9",
        forward_name: "27F",
        forward: "AGAGTTTGATCMTGGCTCAG",
        reverse_name: "1492R",
        reverse: "GGTTACCTTGTTACGACTT",
        note: "Near full-length (PacBio/Nanopore)",
    },
    PrimerPair {
        name: "ITS1F-ITS2",
        marker: "ITS",
        region: "ITS1",
        forward_name: "ITS1F",
        forward: "CTTGGTCATTTAGAGGAAGTAA",
        reverse_name: "ITS2",
        reverse: "GCTGCGTTCTTCATCGATGC",
        note: "EMP fungal ITS1",
    },
    PrimerPair {
        name: "ITS3-ITS4",
        marker: "ITS",
        region: "ITS2",
        forward_name: "ITS3",
        forward: "GCATCGATGAAGAACGCAGC",
        reverse_name: "ITS4",
        reverse: "TCCTCCGCTTATTGATATGC",
        note: "White et al. 1990",
    },
    PrimerPair {
        name: "fITS7-ITS4",
        marker: "ITS",
        region: "ITS2",
        forward_name: "fITS7",
        forward: "GTGARTCATCGAATCTTTG",
        reverse_name: "ITS4",
        reverse: "TCCTCCGCTTATTGATATGC",
        note: "Ihrmark et al. 2012",
    },
    PrimerPair {
        name: "gITS7-ITS4",
        marker: "ITS",
        region: "ITS2",
        forward_name: "gITS7",
        forward: "GTGARTCATCGARTCTTTG",
        reverse_name: "ITS4",
        reverse: "TCCTCCGCTTATTGATATGC",
        note: "Ihrmark et al. 2012",
    },
    PrimerPair {
        name: "ITS86F-ITS4",
        marker: "ITS",
        region: "ITS2",
        forward_name: "ITS86F",
        forward: "GTGAATCATCGAATCTTTGAA",
        reverse_name: "ITS4",
        reverse: "TCCTCCGCTTATTGATATGC",
        note: "Op De Beeck et al. 2014",
    },
    PrimerPair {
        name: "ITS1F-ITS4",
        marker: "ITS",
        region: "ITS1-5.8S-ITS2",
        forward_name: "ITS1F",
        forward: "CTTGGTCATTTAGAGGAAGTAA",
        reverse_name: "ITS4",
        reverse: "TCCTCCGCTTATTGATATGC",
        note: "Full ITS; Australian Microbiome fungal set",
    },
    PrimerPair {
        name: "AMV4.5NF-AMDGR",
        marker: "18S",
        region: "V4 (AMF)",
        forward_name: "AMV4.5NF",
        forward: "AAGCTCGTAGTTGAATTTCG",
        reverse_name: "AMDGR",
        reverse: "CCCAACTATCCCTATTAATCAT",
        note: "Arbuscular mycorrhizal fungi (Glomeromycota); better than ITS for AMF",
    },
    PrimerPair {
        name: "WANDA-AML2",
        marker: "18S",
        region: "V4 (AMF)",
        forward_name: "WANDA",
        forward: "CAGCCGCGGTAATTCCAGCT",
        reverse_name: "AML2",
        reverse: "GAACCCAAACACTTTGGTTTCC",
        note: "Arbuscular mycorrhizal fungi",
    },
    PrimerPair {
        name: "TAReuk454FWD1-TAReukREV3",
        marker: "18S",
        region: "V4",
        forward_name: "TAReuk454FWD1",
        forward: "CCAGCASCYGCGGTAATTCC",
        reverse_name: "TAReukREV3",
        reverse: "ACTTTCGTTCTTGATYRA",
        note: "Eukaryotes / protists (Stoeck et al. 2010)",
    },
];

/// Look up a catalog pair by name (case-insensitive; `/` or `-` separators).
pub fn find_pair(name: &str) -> Option<&'static PrimerPair> {
    let norm = |s: &str| s.to_ascii_lowercase().replace(['/', '_'], "-");
    let n = norm(name);
    CATALOG.iter().find(|p| norm(p.name) == n)
}

/// 4-bit IUPAC mask (A=1, C=2, G=4, T=8).
pub fn iupac_mask(b: u8) -> u8 {
    match b.to_ascii_uppercase() {
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

pub fn masks(primer: &str) -> Vec<u8> {
    primer.bytes().map(iupac_mask).collect()
}

/// Returns true if the string only contains IUPAC nucleotide codes.
pub fn is_valid_primer(p: &str) -> bool {
    !p.is_empty() && p.bytes().all(|b| iupac_mask(b) != 0)
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Hit {
    pub pos: usize,
    pub mismatches: usize,
}

/// Mask of an unambiguous template base; ambiguous template bases (N, R, ...)
/// never match, so low-quality / all-N reads cannot produce spurious hits.
#[inline]
fn template_mask(b: u8) -> u8 {
    match b {
        b'A' | b'a' => 1,
        b'C' | b'c' => 2,
        b'G' | b'g' => 4,
        b'T' | b't' | b'U' | b'u' => 8,
        _ => 0,
    }
}

/// Mismatches between a primer (as masks) and `seq` at `pos`, giving up
/// once `max_mm` is exceeded.
#[inline]
fn mismatches_at(seq: &[u8], primer: &[u8], pos: usize, max_mm: usize) -> Option<usize> {
    let mut mm = 0;
    for (i, &p) in primer.iter().enumerate() {
        if p & template_mask(seq[pos + i]) == 0 {
            mm += 1;
            if mm > max_mm {
                return None;
            }
        }
    }
    Some(mm)
}

/// Best (fewest mismatches, then leftmost) match of `primer` in `seq[start..]`.
pub fn find_primer(seq: &[u8], primer: &[u8], max_mm: usize, start: usize) -> Option<Hit> {
    if primer.is_empty() || seq.len() < primer.len() || start > seq.len() - primer.len() {
        return None;
    }
    let mut best: Option<Hit> = None;
    for pos in start..=(seq.len() - primer.len()) {
        let limit = best.map(|b| b.mismatches.saturating_sub(1)).unwrap_or(max_mm);
        if best.is_some() && best.unwrap().mismatches == 0 {
            break;
        }
        if let Some(mm) = mismatches_at(seq, primer, pos, limit) {
            if best.map(|b| mm < b.mismatches).unwrap_or(true) {
                best = Some(Hit { pos, mismatches: mm });
            }
        }
    }
    best
}

/// Whether a primer occurs at the 5' start of a read (allowing an offset for
/// heterogeneity spacers).
pub fn primer_at_start(read: &[u8], primer: &[u8], max_offset: usize, max_mm: usize) -> bool {
    let end = (max_offset + primer.len()).min(read.len());
    find_primer(&read[..end], primer, max_mm, 0).is_some()
}

/// In-silico PCR settings.
#[derive(Clone, Debug)]
pub struct PcrOptions {
    pub max_mismatches: usize,
    pub keep_primers: bool,
    /// Accept amplicons where only one primer site is present (the amplicon
    /// then extends to the end of the reference). Useful for references that
    /// are already trimmed (e.g. UNITE ITS sequences lack LSU flanks).
    pub partial: bool,
    pub try_revcomp: bool,
}

impl Default for PcrOptions {
    fn default() -> Self {
        PcrOptions { max_mismatches: 2, keep_primers: false, partial: false, try_revcomp: true }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AmpliconKind {
    Full,
    ForwardOnly,
    ReverseOnly,
}

#[derive(Clone, Debug)]
pub struct Amplicon {
    pub seq: Vec<u8>,
    pub kind: AmpliconKind,
    pub reverse_complemented: bool,
    pub mismatches: usize,
}

/// A primer pair prepared for matching.
#[derive(Clone, Debug)]
pub struct Pcr {
    pub name: String,
    pub forward: String,
    pub reverse: String,
    fwd: Vec<u8>,
    rev_rc: Vec<u8>,
    pub opts: PcrOptions,
}

impl Pcr {
    pub fn new(name: &str, forward: &str, reverse: &str, opts: PcrOptions) -> Self {
        let rc = String::from_utf8(revcomp(reverse.as_bytes())).unwrap();
        Pcr {
            name: name.to_string(),
            forward: forward.to_ascii_uppercase(),
            reverse: reverse.to_ascii_uppercase(),
            fwd: masks(forward),
            rev_rc: masks(&rc),
            opts,
        }
    }

    pub fn from_pair(p: &PrimerPair, opts: PcrOptions) -> Self {
        Self::new(p.name, p.forward, p.reverse, opts)
    }

    fn oriented(&self, seq: &[u8], rc: bool) -> Option<Amplicon> {
        let mm = self.opts.max_mismatches;
        let f = find_primer(seq, &self.fwd, mm, 0);
        let r_start = f.map(|h| h.pos + self.fwd.len()).unwrap_or(0);
        let r = find_primer(seq, &self.rev_rc, mm, r_start);
        let keep = self.opts.keep_primers;
        let (a, b, kind, m) = match (f, r) {
            (Some(f), Some(r)) => {
                let a = if keep { f.pos } else { f.pos + self.fwd.len() };
                let b = if keep { r.pos + self.rev_rc.len() } else { r.pos };
                (a, b, AmpliconKind::Full, f.mismatches + r.mismatches)
            }
            (Some(f), None) if self.opts.partial => {
                let a = if keep { f.pos } else { f.pos + self.fwd.len() };
                (a, seq.len(), AmpliconKind::ForwardOnly, f.mismatches)
            }
            (None, Some(r)) if self.opts.partial => {
                let b = if keep { r.pos + self.rev_rc.len() } else { r.pos };
                (0, b, AmpliconKind::ReverseOnly, r.mismatches)
            }
            _ => return None,
        };
        if b <= a {
            return None;
        }
        Some(Amplicon { seq: seq[a..b].to_vec(), kind, reverse_complemented: rc, mismatches: m })
    }

    /// Extract the amplicon from a reference sequence (forward orientation
    /// first, then the reverse complement). Full amplicons are preferred over
    /// partial ones.
    pub fn amplify(&self, seq: &[u8]) -> Option<Amplicon> {
        let fwd = self.oriented(seq, false);
        if matches!(fwd, Some(Amplicon { kind: AmpliconKind::Full, .. })) {
            return fwd;
        }
        if self.opts.try_revcomp {
            let rc = revcomp(seq);
            let r = self.oriented(&rc, true);
            if matches!(r, Some(Amplicon { kind: AmpliconKind::Full, .. })) {
                return r;
            }
            return fwd.or(r);
        }
        fwd
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn catalog_is_valid() {
        for p in CATALOG {
            assert!(is_valid_primer(p.forward), "{}", p.name);
            assert!(is_valid_primer(p.reverse), "{}", p.name);
        }
        assert!(find_pair("515f/806r").is_some());
    }

    #[test]
    fn degenerate_matching() {
        let p = masks("GTGYCA");
        assert_eq!(find_primer(b"AAGTGTCAAA", &p, 0, 0), Some(Hit { pos: 2, mismatches: 0 }));
        assert_eq!(find_primer(b"AAGTGACAAA", &p, 0, 0), None);
        assert_eq!(find_primer(b"AAGTGACAAA", &p, 1, 0), Some(Hit { pos: 2, mismatches: 1 }));
        assert_eq!(find_primer(b"NNNNNNNNNN", &p, 2, 0), None);
    }

    #[test]
    fn in_silico_pcr() {
        let f = "GTGCCAGCAGCCGCGGTAA";
        let r = "GGACTACAAGGGTATCTAAT";
        let insert = "ACGTACGTTTGGCCAAACGT";
        let r_rc = String::from_utf8(revcomp(r.as_bytes())).unwrap();
        let s = format!("TTTT{f}{insert}{r_rc}CCCC");
        let pcr = Pcr::from_pair(find_pair("515F-806R").unwrap(), PcrOptions::default());
        let a = pcr.amplify(s.as_bytes()).unwrap();
        assert_eq!(a.seq, insert.as_bytes());
        assert_eq!(a.kind, AmpliconKind::Full);
        // reverse-complemented reference
        let a = pcr.amplify(&revcomp(s.as_bytes())).unwrap();
        assert_eq!(a.seq, insert.as_bytes());
        assert!(a.reverse_complemented);
        // missing reverse site
        let s2 = format!("TTTT{f}{insert}");
        assert!(pcr.amplify(s2.as_bytes()).is_none());
        let opts = PcrOptions { partial: true, ..Default::default() };
        let pcr = Pcr::from_pair(find_pair("515F-806R").unwrap(), opts);
        assert_eq!(pcr.amplify(s2.as_bytes()).unwrap().kind, AmpliconKind::ForwardOnly);
    }
}
