//! Taxonomy tree construction following DECIPHER's `LearnTaxa()`.
//!
//! Lineage strings are cleaned exactly as in `LearnTaxa` (R/LearnTaxa.R):
//! `gsub(" *; *", ";", x)`, the part after the first `"Root;"` is kept,
//! trailing spaces are removed and a trailing `;` is appended if missing.
//! Nodes are numbered in order of first appearance of each lineage prefix,
//! with `Root` as node 0 (DECIPHER's 1-based index minus one).

use std::collections::HashMap;

use crate::error::{Error, Result};

/// Sentinel parent value for the root node.
pub const NO_PARENT: u32 = u32::MAX;

/// Clean one lineage string; returns the lineage *below* Root with a
/// trailing `;` (e.g. `"Bacteria;Firmicutes;"`).
///
/// If the string does not contain `"Root;"`, it is treated as a lineage
/// starting below Root (`"Root;"` is prepended), which is a convenience
/// over DECIPHER (which stops with an error).
pub fn clean_lineage(raw: &str) -> Result<String> {
    // gsub(" *; *", ";", x)
    let mut s = String::with_capacity(raw.len());
    let bytes = raw.trim_end_matches(['\r', '\n']).as_bytes();
    let mut i = 0;
    while i < bytes.len() {
        if bytes[i] == b' ' {
            // look ahead: spaces followed by ';' are removed
            let mut j = i;
            while j < bytes.len() && bytes[j] == b' ' {
                j += 1;
            }
            if j < bytes.len() && bytes[j] == b';' {
                i = j;
                continue;
            }
            s.push_str(std::str::from_utf8(&bytes[i..j]).unwrap());
            i = j;
        } else if bytes[i] == b';' {
            s.push(';');
            i += 1;
            while i < bytes.len() && bytes[i] == b' ' {
                i += 1;
            }
        } else {
            // push full utf8 char
            let ch_len = utf8_len(bytes[i]);
            s.push_str(
                std::str::from_utf8(&bytes[i..i + ch_len])
                    .map_err(|_| Error::Taxonomy(format!("invalid UTF-8 in lineage '{raw}'")))?,
            );
            i += ch_len;
        }
    }
    let rest = match s.find("Root;") {
        Some(p) => {
            let after = &s[p + 5..];
            // strsplit(x, "Root;")[[1]][2]: stop at a second "Root;"
            match after.find("Root;") {
                Some(q) => after[..q].to_string(),
                None => after.to_string(),
            }
        }
        None => {
            if s == "Root" {
                String::new()
            } else {
                s.clone()
            }
        }
    };
    let mut rest = rest.trim_end_matches(' ').to_string();
    if rest.is_empty() {
        return Err(Error::Taxonomy(format!(
            "lineage '{raw}' has no ranks below Root"
        )));
    }
    if !rest.ends_with(';') {
        rest.push(';');
    }
    Ok(rest)
}

fn utf8_len(b: u8) -> usize {
    match b {
        0..=0x7f => 1,
        0xc0..=0xdf => 2,
        0xe0..=0xef => 3,
        _ => 4,
    }
}

/// A taxonomy tree in DECIPHER node order.
#[derive(Clone, Debug, Default)]
pub struct TaxonomyTree {
    /// Full path per node, e.g. `"Root;Bacteria;"`.
    pub taxonomy: Vec<String>,
    /// Last name per node.
    pub taxa: Vec<String>,
    /// Parent node per node (`NO_PARENT` for Root).
    pub parents: Vec<u32>,
    /// DECIPHER `levels`: number of names in the path (Root = 1).
    pub levels: Vec<u32>,
    /// Immediate children, in node order.
    pub children: Vec<Vec<u32>>,
    /// Node of each input sequence (DECIPHER `crossIndex` minus one).
    pub cross_index: Vec<u32>,
    /// Cleaned lineage per sequence (below Root, trailing `;`).
    pub classes: Vec<String>,
}

impl TaxonomyTree {
    /// Build the tree from per-sequence raw lineage strings.
    pub fn build<S: AsRef<str>>(lineages: &[S]) -> Result<Self> {
        let classes: Vec<String> = lineages
            .iter()
            .map(|l| clean_lineage(l.as_ref()))
            .collect::<Result<_>>()?;
        let mut index: HashMap<String, u32> = HashMap::new();
        let mut taxonomy = vec!["Root;".to_string()];
        let mut taxa = vec!["Root".to_string()];
        let mut parents = vec![NO_PARENT];
        let mut levels = vec![1u32];
        let mut cross_index = Vec::with_capacity(classes.len());
        let mut seen_class: HashMap<&str, u32> = HashMap::new();
        for c in &classes {
            if let Some(&n) = seen_class.get(c.as_str()) {
                cross_index.push(n);
                continue;
            }
            let mut parent = 0u32;
            let mut end = 0usize;
            let mut level = 1u32;
            for (pos, _) in c.match_indices(';') {
                let prefix = &c[..=pos];
                level += 1;
                let node = match index.get(prefix) {
                    Some(&n) => n,
                    None => {
                        let n = taxonomy.len() as u32;
                        index.insert(prefix.to_string(), n);
                        taxonomy.push(format!("Root;{prefix}"));
                        taxa.push(c[end..pos].to_string());
                        parents.push(parent);
                        levels.push(level);
                        n
                    }
                };
                parent = node;
                end = pos + 1;
            }
            seen_class.insert(c.as_str(), parent);
            cross_index.push(parent);
        }
        if seen_class.len() <= 1 {
            return Err(Error::Taxonomy(
                "more than one taxonomic group is required".into(),
            ));
        }
        let mut children = vec![Vec::new(); taxonomy.len()];
        for (i, &p) in parents.iter().enumerate() {
            if p != NO_PARENT {
                children[p as usize].push(i as u32);
            }
        }
        Ok(Self {
            taxonomy,
            taxa,
            parents,
            levels,
            children,
            cross_index,
            classes,
        })
    }

    /// Number of nodes.
    pub fn len(&self) -> usize {
        self.taxa.len()
    }

    /// Whether the tree has no nodes.
    pub fn is_empty(&self) -> bool {
        self.taxa.is_empty()
    }
}

/// Rebuild full path strings from names and parents.
pub fn full_paths(taxa: &[String], parents: &[u32]) -> Vec<String> {
    let mut out: Vec<String> = Vec::with_capacity(taxa.len());
    for (i, t) in taxa.iter().enumerate() {
        let p = parents[i];
        if p == NO_PARENT {
            out.push(format!("{t};"));
        } else {
            let s = format!("{}{t};", out[p as usize]);
            out.push(s);
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cleaning() {
        assert_eq!(clean_lineage("Root; A ;B").unwrap(), "A;B;");
        assert_eq!(clean_lineage("Root;A;B;").unwrap(), "A;B;");
        assert_eq!(clean_lineage("A;B").unwrap(), "A;B;");
        assert_eq!(clean_lineage("Root;A;B   ").unwrap(), "A;B;");
        assert_eq!(clean_lineage("Root;Genus sp.;").unwrap(), "Genus sp.;");
        assert!(clean_lineage("Root;").is_err());
        assert!(clean_lineage("").is_err());
    }

    #[test]
    fn tree_order_matches_decipher() {
        let t = TaxonomyTree::build(&[
            "Root;A;B;C",
            "Root;A;D;E",
            "Root;F;G;H",
            "Root;A;B;C",
            "Root;A;B",
        ])
        .unwrap();
        assert_eq!(
            t.taxonomy,
            vec![
                "Root;",
                "Root;A;",
                "Root;A;B;",
                "Root;A;B;C;",
                "Root;A;D;",
                "Root;A;D;E;",
                "Root;F;",
                "Root;F;G;",
                "Root;F;G;H;"
            ]
        );
        assert_eq!(t.cross_index, vec![3, 5, 8, 3, 2]);
        assert_eq!(t.children[0], vec![1, 6]);
        assert_eq!(t.children[1], vec![2, 4]);
        assert_eq!(t.levels[3], 4);
        assert_eq!(t.parents[0], NO_PARENT);
        assert_eq!(full_paths(&t.taxa, &t.parents), t.taxonomy);
    }

    #[test]
    fn single_group_rejected() {
        assert!(TaxonomyTree::build(&["Root;A", "Root;A;"]).is_err());
    }
}
