//! K-mer encoding compatible with DECIPHER's `enumerateSequence()`.
//!
//! DECIPHER (src/EnumerateSequence.c, `enumerateSequence` with
//! `fastMovingSide = 1`) encodes bases as A=0, C=1, G=2, T=3 and computes the
//! code of the window starting at position `p` as
//! `sum_{j=0}^{K-1} base[p + j] * 4^j`, i.e. the *first* base of the window is
//! the least significant digit. Any other character (IUPAC ambiguity codes,
//! N, gaps) makes every window that contains it invalid (NA in DECIPHER).
//! `IdTaxa`/`LearnTaxa` then add 1 to obtain 1-based indices; the native
//! format of this crate stores the 0-based code, and `import-decipher`
//! subtracts 1.
//!
//! Spaced seeds generalise this: a pattern such as `1101101101101` selects
//! the "care" positions of a window of length `pattern.len()`; the code is
//! `sum_t base[p + care_t] * 4^t` over the care positions in order. A
//! contiguous K-mer is the pattern of K ones.

use crate::error::{Error, Result};

/// Invalid-base marker in [`BASE_CODE`].
pub const INVALID: u8 = 4;

/// Lookup table from ASCII byte to 2-bit base code (A=0, C=1, G=2, T/U=3),
/// case-insensitive; everything else maps to [`INVALID`].
pub const BASE_CODE: [u8; 256] = {
    let mut t = [INVALID; 256];
    t[b'A' as usize] = 0;
    t[b'a' as usize] = 0;
    t[b'C' as usize] = 1;
    t[b'c' as usize] = 1;
    t[b'G' as usize] = 2;
    t[b'g' as usize] = 2;
    t[b'T' as usize] = 3;
    t[b't' as usize] = 3;
    t[b'U' as usize] = 3;
    t[b'u' as usize] = 3;
    t
};

/// Maximum supported weight (number of informative positions) of a k-mer.
pub const MAX_K: usize = 16;

/// Normalise a nucleotide sequence for hashing/dereplication: upper case,
/// U converted to T. Other characters are kept (upper-cased).
pub fn normalize_sequence(seq: &[u8]) -> Vec<u8> {
    seq.iter()
        .map(|&c| match c.to_ascii_uppercase() {
            b'U' => b'T',
            o => o,
        })
        .collect()
}

/// Reverse complement of a nucleotide sequence (IUPAC aware, case preserved
/// in the sense that the output is upper case).
pub fn reverse_complement(seq: &[u8]) -> Vec<u8> {
    seq.iter()
        .rev()
        .map(|&c| match c.to_ascii_uppercase() {
            b'A' => b'T',
            b'C' => b'G',
            b'G' => b'C',
            b'T' | b'U' => b'A',
            b'R' => b'Y',
            b'Y' => b'R',
            b'K' => b'M',
            b'M' => b'K',
            b'B' => b'V',
            b'V' => b'B',
            b'D' => b'H',
            b'H' => b'D',
            o => o, // S, W, N and anything else are self-complementary
        })
        .collect()
}

/// K-mer encoder: contiguous K-mers or a spaced-seed pattern.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct KmerEncoder {
    k: usize,
    /// Care positions (offsets within the window), `None` for contiguous.
    care: Option<Vec<usize>>,
    span: usize,
}

/// Result of encoding one sequence.
#[derive(Clone, Debug, Default)]
pub struct EncodedKmers {
    /// Sorted, unique, 0-based k-mer codes.
    pub unique: Vec<u32>,
    /// Number of window positions with a valid (non-ambiguous) k-mer,
    /// counting duplicates. DECIPHER's `notNAs`, which drives the sample size.
    pub n_valid: usize,
}

impl KmerEncoder {
    /// Contiguous K-mers.
    pub fn contiguous(k: usize) -> Result<Self> {
        if k == 0 || k > MAX_K {
            return Err(Error::InvalidParameter(format!(
                "K must be between 1 and {MAX_K}, got {k}"
            )));
        }
        Ok(Self {
            k,
            care: None,
            span: k,
        })
    }

    /// Spaced seed from a pattern of '1' (care) and '0' (don't care)
    /// characters. The pattern must start and end with '1'.
    pub fn spaced(pattern: &str) -> Result<Self> {
        let bytes = pattern.as_bytes();
        if bytes.is_empty()
            || bytes.iter().any(|&b| b != b'0' && b != b'1')
            || bytes[0] != b'1'
            || bytes[bytes.len() - 1] != b'1'
        {
            return Err(Error::InvalidParameter(format!(
                "invalid spaced-seed pattern '{pattern}': use 0/1 characters, starting and ending with 1"
            )));
        }
        let care: Vec<usize> = bytes
            .iter()
            .enumerate()
            .filter(|(_, &b)| b == b'1')
            .map(|(i, _)| i)
            .collect();
        let k = care.len();
        if k > MAX_K {
            return Err(Error::InvalidParameter(format!(
                "spaced-seed weight must be at most {MAX_K}, got {k}"
            )));
        }
        if care.len() == bytes.len() {
            return Self::contiguous(k);
        }
        Ok(Self {
            k,
            span: bytes.len(),
            care: Some(care),
        })
    }

    /// Build from a model's K and optional pattern.
    pub fn from_model(k: usize, pattern: Option<&str>) -> Result<Self> {
        match pattern {
            Some(p) => {
                let e = Self::spaced(p)?;
                if e.k != k {
                    return Err(Error::InvalidParameter(format!(
                        "seed pattern weight {} does not match K = {k}",
                        e.k
                    )));
                }
                Ok(e)
            }
            None => Self::contiguous(k),
        }
    }

    /// Weight (number of informative positions) = K.
    pub fn k(&self) -> usize {
        self.k
    }

    /// Window length (= K for contiguous k-mers).
    pub fn span(&self) -> usize {
        self.span
    }

    /// The spaced-seed pattern, if any.
    pub fn pattern(&self) -> Option<String> {
        self.care.as_ref().map(|care| {
            let mut s = vec![b'0'; self.span];
            for &c in care {
                s[c] = b'1';
            }
            String::from_utf8(s).unwrap()
        })
    }

    /// Number of distinct k-mer codes (4^K).
    pub fn n_kmers(&self) -> u64 {
        1u64 << (2 * self.k)
    }

    /// Append the codes of every valid window of `seq` (in position order,
    /// with duplicates) to `out`; returns the number appended.
    pub fn encode_all(&self, seq: &[u8], out: &mut Vec<u32>) -> usize {
        let start = out.len();
        if seq.len() < self.span {
            return 0;
        }
        match &self.care {
            None => {
                let k = self.k;
                let shift = 2 * (k as u32 - 1);
                let mut code: u64 = 0;
                let mut run = 0usize;
                for &c in seq {
                    let b = BASE_CODE[c as usize];
                    if b == INVALID {
                        run = 0;
                        code = 0;
                        continue;
                    }
                    code = (code >> 2) | ((b as u64) << shift);
                    run += 1;
                    if run >= k {
                        out.push(code as u32);
                    }
                }
            }
            Some(care) => {
                // spaced seed: direct evaluation with an invalid-position
                // prefix count for O(1) validity checks per window
                let codes: Vec<u8> = seq.iter().map(|&c| BASE_CODE[c as usize]).collect();
                let mut bad_prefix = Vec::with_capacity(codes.len() + 1);
                bad_prefix.push(0u32);
                for &b in &codes {
                    let last = *bad_prefix.last().unwrap();
                    bad_prefix.push(last + u32::from(b == INVALID));
                }
                for p in 0..=(codes.len() - self.span) {
                    let mut ok = true;
                    let mut code: u64 = 0;
                    // only care positions need to be valid
                    if bad_prefix[p + self.span] != bad_prefix[p] {
                        for &c in care {
                            if codes[p + c] == INVALID {
                                ok = false;
                                break;
                            }
                        }
                    }
                    if ok {
                        for (t, &c) in care.iter().enumerate() {
                            code |= (codes[p + c] as u64) << (2 * t);
                        }
                        out.push(code as u32);
                    }
                }
            }
        }
        out.len() - start
    }

    /// Encode a sequence into sorted unique codes plus the valid-window count.
    pub fn encode(&self, seq: &[u8]) -> EncodedKmers {
        let mut v = Vec::with_capacity(seq.len());
        let n_valid = self.encode_all(seq, &mut v);
        v.sort_unstable();
        v.dedup();
        EncodedKmers { unique: v, n_valid }
    }

    /// Encode a sequence and its reverse complement. For contiguous k-mers
    /// both strands are produced in a single rolling pass.
    pub fn encode_both(&self, seq: &[u8]) -> (EncodedKmers, EncodedKmers) {
        match &self.care {
            Some(_) => {
                let rc = reverse_complement(seq);
                (self.encode(seq), self.encode(&rc))
            }
            None => {
                let k = self.k;
                let mut fwd = Vec::with_capacity(seq.len());
                let mut rev = Vec::with_capacity(seq.len());
                if seq.len() >= k {
                    let shift = 2 * (k as u32 - 1);
                    let mask: u64 = if k == 32 {
                        u64::MAX
                    } else {
                        (1u64 << (2 * k)) - 1
                    };
                    let mut code: u64 = 0;
                    let mut rcode: u64 = 0;
                    let mut run = 0usize;
                    for &c in seq {
                        let b = BASE_CODE[c as usize];
                        if b == INVALID {
                            run = 0;
                            code = 0;
                            rcode = 0;
                            continue;
                        }
                        code = (code >> 2) | ((b as u64) << shift);
                        rcode = ((rcode << 2) | (3 - b) as u64) & mask;
                        run += 1;
                        if run >= k {
                            fwd.push(code as u32);
                            rev.push(rcode as u32);
                        }
                    }
                }
                let n = fwd.len();
                fwd.sort_unstable();
                fwd.dedup();
                rev.sort_unstable();
                rev.dedup();
                (
                    EncodedKmers {
                        unique: fwd,
                        n_valid: n,
                    },
                    EncodedKmers {
                        unique: rev,
                        n_valid: n,
                    },
                )
            }
        }
    }
}

/// Decode a contiguous k-mer code into its string (for debugging/tests).
pub fn decode_kmer(code: u32, k: usize) -> String {
    const B: [u8; 4] = *b"ACGT";
    (0..k)
        .map(|j| B[((code >> (2 * j)) & 3) as usize] as char)
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn naive(seq: &[u8], k: usize) -> (Vec<u32>, usize) {
        let mut v = Vec::new();
        if seq.len() >= k {
            for p in 0..=seq.len() - k {
                let mut code = 0u32;
                let mut ok = true;
                for j in 0..k {
                    let b = BASE_CODE[seq[p + j] as usize];
                    if b == INVALID {
                        ok = false;
                        break;
                    }
                    code += (b as u32) * 4u32.pow(j as u32);
                }
                if ok {
                    v.push(code);
                }
            }
        }
        let n = v.len();
        v.sort();
        v.dedup();
        (v, n)
    }

    #[test]
    fn decipher_coding_order() {
        // first base least significant: "CA" -> C*1 + A*4 = 1
        let e = KmerEncoder::contiguous(2).unwrap();
        assert_eq!(e.encode(b"CA").unique, vec![1]);
        assert_eq!(e.encode(b"AC").unique, vec![4]);
        assert_eq!(e.encode(b"TT").unique, vec![15]);
        let e3 = KmerEncoder::contiguous(3).unwrap();
        // G A T = 2 + 0*4 + 3*16 = 50
        assert_eq!(e3.encode(b"GAT").unique, vec![50]);
        assert_eq!(decode_kmer(50, 3), "GAT");
    }

    #[test]
    fn matches_naive_random() {
        use rand::{Rng, SeedableRng};
        let mut rng = rand_xoshiro::Xoshiro256PlusPlus::seed_from_u64(7);
        let alphabet = b"ACGTacgtUuNRY-";
        for k in 1..=16 {
            let e = KmerEncoder::contiguous(k).unwrap();
            for _ in 0..50 {
                let len = rng.gen_range(0..80);
                let seq: Vec<u8> = (0..len)
                    .map(|_| {
                        if rng.gen_bool(0.9) {
                            alphabet[rng.gen_range(0..10)]
                        } else {
                            alphabet[rng.gen_range(0..alphabet.len())]
                        }
                    })
                    .collect();
                let (nv, nn) = naive(&seq, k);
                let got = e.encode(&seq);
                assert_eq!(
                    got.unique,
                    nv,
                    "k={k} seq={}",
                    String::from_utf8_lossy(&seq)
                );
                assert_eq!(got.n_valid, nn);
                // reverse complement in a single pass equals encoding the RC
                let (f, r) = e.encode_both(&seq);
                assert_eq!(f.unique, nv);
                let (rv, _) = naive(&reverse_complement(&seq), k);
                assert_eq!(r.unique, rv, "rc k={k}");
            }
        }
    }

    #[test]
    fn ambiguity_lowercase_and_u() {
        let e = KmerEncoder::contiguous(3).unwrap();
        assert_eq!(e.encode(b"acgu").unique, e.encode(b"ACGT").unique);
        // N breaks all windows touching it
        let r = e.encode(b"ACGNACG");
        assert_eq!(r.n_valid, 2);
        assert_eq!(r.unique, e.encode(b"ACG").unique);
        // IUPAC codes are ambiguous too
        assert_eq!(e.encode(b"ACRT").n_valid, 0);
        assert_eq!(e.encode(b"AC-GT").n_valid, 0);
    }

    #[test]
    fn shorter_than_k() {
        let e = KmerEncoder::contiguous(8).unwrap();
        let r = e.encode(b"ACGT");
        assert!(r.unique.is_empty());
        assert_eq!(r.n_valid, 0);
        let (f, rr) = e.encode_both(b"");
        assert!(f.unique.is_empty() && rr.unique.is_empty());
    }

    #[test]
    fn k16_full_range() {
        let e = KmerEncoder::contiguous(16).unwrap();
        let r = e.encode(b"TTTTTTTTTTTTTTTT");
        assert_eq!(r.unique, vec![u32::MAX]);
        assert!(KmerEncoder::contiguous(17).is_err());
        assert!(KmerEncoder::contiguous(0).is_err());
    }

    #[test]
    fn spaced_seed() {
        let e = KmerEncoder::spaced("101").unwrap();
        assert_eq!(e.k(), 2);
        assert_eq!(e.span(), 3);
        // A?C -> A + C*4 = 4 ; middle base ignored, even if ambiguous
        assert_eq!(e.encode(b"ANC").unique, vec![4]);
        assert_eq!(e.encode(b"AGC").unique, vec![4]);
        assert_eq!(e.encode(b"NAC").n_valid, 0);
        assert_eq!(e.pattern().as_deref(), Some("101"));
        // contiguous pattern collapses to a contiguous encoder
        assert_eq!(
            KmerEncoder::spaced("111").unwrap(),
            KmerEncoder::contiguous(3).unwrap()
        );
        assert!(KmerEncoder::spaced("011").is_err());
        assert!(KmerEncoder::spaced("1x1").is_err());
        let (f, r) = e.encode_both(b"ACGTTGCA");
        assert_eq!(f.unique, e.encode(b"ACGTTGCA").unique);
        assert_eq!(r.unique, e.encode(&reverse_complement(b"ACGTTGCA")).unique);
    }

    #[test]
    fn revcomp() {
        assert_eq!(reverse_complement(b"ACGTN"), b"NACGT");
        assert_eq!(reverse_complement(b"acgu"), b"ACGT");
        assert_eq!(reverse_complement(b"RYKM"), b"KMRY");
    }
}
