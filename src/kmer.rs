//! k-mer encoding. k-mers are 2-bit packed into `u32` (k ≤ 12 so that a dense
//! weight table of 4^k entries stays small). Windows containing ambiguous
//! bases are skipped.

pub const MIN_K: usize = 4;
pub const MAX_K: usize = 12;

#[inline]
pub fn base_code(b: u8) -> Option<u32> {
    match b {
        b'A' | b'a' => Some(0),
        b'C' | b'c' => Some(1),
        b'G' | b'g' => Some(2),
        b'T' | b't' | b'U' | b'u' => Some(3),
        _ => None,
    }
}

/// Sorted, de-duplicated k-mers of a sequence.
pub fn unique_kmers(seq: &[u8], k: usize) -> Vec<u32> {
    assert!((1..=16).contains(&k));
    let mask: u64 = (1u64 << (2 * k)) - 1;
    let mut out = Vec::with_capacity(seq.len());
    let mut cur: u64 = 0;
    let mut valid = 0usize;
    for &b in seq {
        match base_code(b) {
            Some(c) => {
                cur = ((cur << 2) | c as u64) & mask;
                valid += 1;
                if valid >= k {
                    out.push(cur as u32);
                }
            }
            None => {
                valid = 0;
                cur = 0;
            }
        }
    }
    out.sort_unstable();
    out.dedup();
    out
}

/// Default k-mer length following IDTAXA's `LearnTaxa`:
/// `floor(log4(100 * q99(sequence length)))`, clamped to [`MIN_K`, `MAX_K`].
/// This gives k = 8 for full-length 16S and k = 7 for V4 / ITS2 amplicons.
pub fn default_k(lengths: &mut [usize]) -> usize {
    if lengths.is_empty() {
        return 8;
    }
    lengths.sort_unstable();
    let idx = ((lengths.len() - 1) as f64 * 0.99).round() as usize;
    let q = lengths[idx].max(1) as f64;
    let k = ((100.0 * q).ln() / 4f64.ln()).floor() as usize;
    k.clamp(MIN_K, MAX_K)
}

/// Decode a packed k-mer back to bases (mainly for debugging and tests).
pub fn decode(kmer: u32, k: usize) -> String {
    (0..k)
        .rev()
        .map(|i| match (kmer >> (2 * i)) & 3 {
            0 => 'A',
            1 => 'C',
            2 => 'G',
            _ => 'T',
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn encodes_and_skips_ambiguity() {
        let k = unique_kmers(b"ACGTNACGT", 4);
        assert_eq!(k.len(), 1);
        assert_eq!(decode(k[0], 4), "ACGT");
        let k = unique_kmers(b"AAAAAA", 3);
        assert_eq!(k, vec![0]);
    }

    #[test]
    fn default_k_matches_idtaxa() {
        assert_eq!(default_k(&mut [1450; 10]), 8);
        assert_eq!(default_k(&mut [253; 10]), 7);
    }
}
