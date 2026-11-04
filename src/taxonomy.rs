//! Taxonomic tree and lineage parsing.
//!
//! Supported lineage styles (all `;`-separated, whitespace tolerated):
//! * DECIPHER / IDTAXA: `Root;Bacteria;Firmicutes;...`
//! * SILVA: `Bacteria;Firmicutes;...;Bacillus subtilis`
//! * GTDB / QIIME / UNITE prefixed: `d__Bacteria; p__Firmicutes; ...`
//!
//! Prefixed ranks (`d__`, `k__`, `p__`, `c__`, `o__`, `f__`, `g__`, `s__`)
//! are recognised; otherwise ranks can be supplied positionally.
//! Lineages are truncated at empty or uninformative names
//! (e.g. `g__`, `uncultured`, `unidentified`), so these never become taxa.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

pub const ROOT: usize = 0;

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct Node {
    pub name: String,
    pub rank: Option<String>,
    pub parent: Option<usize>,
    pub children: Vec<usize>,
    pub depth: usize,
}

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct Taxonomy {
    pub nodes: Vec<Node>,
    #[serde(skip)]
    lookup: HashMap<(usize, String), usize>,
}

impl Default for Taxonomy {
    fn default() -> Self {
        Self::new()
    }
}

/// A parsed lineage (root excluded).
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Lineage {
    pub names: Vec<String>,
    pub ranks: Vec<Option<String>>,
}

impl Lineage {
    pub fn is_empty(&self) -> bool {
        self.names.is_empty()
    }
    pub fn len(&self) -> usize {
        self.names.len()
    }
    pub fn truncated(&self, n: usize) -> Lineage {
        Lineage { names: self.names[..n.min(self.len())].to_vec(), ranks: self.ranks[..n.min(self.len())].to_vec() }
    }
    pub fn to_string_plain(&self) -> String {
        self.names.join(";")
    }
}

impl Taxonomy {
    pub fn new() -> Self {
        Taxonomy {
            nodes: vec![Node {
                name: "Root".to_string(),
                rank: Some("rootrank".to_string()),
                parent: None,
                children: Vec::new(),
                depth: 0,
            }],
            lookup: HashMap::new(),
        }
    }

    /// Must be called after deserialisation.
    pub fn rebuild_lookup(&mut self) {
        self.lookup.clear();
        for (i, n) in self.nodes.iter().enumerate() {
            if let Some(p) = n.parent {
                self.lookup.insert((p, n.name.clone()), i);
            }
        }
    }

    pub fn len(&self) -> usize {
        self.nodes.len()
    }

    pub fn is_empty(&self) -> bool {
        self.nodes.len() <= 1
    }

    pub fn child(&self, parent: usize, name: &str) -> Option<usize> {
        self.lookup.get(&(parent, name.to_string())).copied()
    }

    /// Insert a lineage, returning the node of its deepest taxon.
    pub fn add_lineage(&mut self, lin: &Lineage) -> usize {
        let mut cur = ROOT;
        for (i, name) in lin.names.iter().enumerate() {
            let key = (cur, name.clone());
            cur = match self.lookup.get(&key) {
                Some(&c) => {
                    if self.nodes[c].rank.is_none() {
                        self.nodes[c].rank = lin.ranks[i].clone();
                    }
                    c
                }
                None => {
                    let id = self.nodes.len();
                    let depth = self.nodes[cur].depth + 1;
                    self.nodes.push(Node {
                        name: name.clone(),
                        rank: lin.ranks[i].clone(),
                        parent: Some(cur),
                        children: Vec::new(),
                        depth,
                    });
                    self.nodes[cur].children.push(id);
                    self.lookup.insert(key, id);
                    id
                }
            };
        }
        cur
    }

    /// Find the node for a lineage, if every level exists.
    pub fn find(&self, names: &[String]) -> Option<usize> {
        let mut cur = ROOT;
        for n in names {
            cur = self.child(cur, n)?;
        }
        Some(cur)
    }

    pub fn parent(&self, node: usize) -> Option<usize> {
        self.nodes[node].parent
    }

    /// Nodes from the root to `node` (inclusive).
    pub fn path(&self, node: usize) -> Vec<usize> {
        let mut p = vec![node];
        let mut cur = node;
        while let Some(par) = self.nodes[cur].parent {
            p.push(par);
            cur = par;
        }
        p.reverse();
        p
    }

    pub fn is_ancestor_or_self(&self, anc: usize, mut node: usize) -> bool {
        let d = self.nodes[anc].depth;
        while self.nodes[node].depth > d {
            node = self.nodes[node].parent.expect("non-root node has a parent");
        }
        node == anc
    }

    /// `;`-joined names from the first rank below the root to `node`.
    pub fn lineage_string(&self, node: usize) -> String {
        self.path(node).iter().skip(1).map(|&n| self.nodes[n].name.as_str()).collect::<Vec<_>>().join(";")
    }

    /// Distinct rank names ordered by their mean depth in the tree.
    pub fn ranks(&self) -> Vec<String> {
        let mut acc: HashMap<&str, (usize, usize)> = HashMap::new();
        for n in self.nodes.iter().skip(1) {
            if let Some(r) = &n.rank {
                let e = acc.entry(r.as_str()).or_insert((0, 0));
                e.0 += n.depth;
                e.1 += 1;
            }
        }
        let mut v: Vec<(String, f64)> =
            acc.into_iter().map(|(r, (s, c))| (r.to_string(), s as f64 / c as f64)).collect();
        v.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap().then(a.0.cmp(&b.0)));
        v.into_iter().map(|x| x.0).collect()
    }
}

/// Map a QIIME/GTDB/UNITE rank prefix to a rank name.
pub fn rank_from_prefix(p: &str) -> Option<&'static str> {
    Some(match p {
        "d" => "domain",
        "k" => "kingdom",
        "p" => "phylum",
        "c" => "class",
        "o" => "order",
        "f" => "family",
        "g" => "genus",
        "s" => "species",
        _ => return None,
    })
}

/// Inverse of [`rank_from_prefix`].
pub fn prefix_for_rank(rank: &str) -> Option<&'static str> {
    Some(match rank.to_ascii_lowercase().as_str() {
        "domain" | "superkingdom" => "d",
        "kingdom" => "k",
        "phylum" => "p",
        "class" => "c",
        "order" => "o",
        "family" => "f",
        "genus" => "g",
        "species" => "s",
        _ => return None,
    })
}

fn split_prefix(tok: &str) -> Option<(&'static str, &str)> {
    let i = tok.find("__")?;
    if i == 0 || i > 2 || !tok[..i].chars().all(|c| c.is_ascii_lowercase()) {
        return None;
    }
    rank_from_prefix(&tok[..i]).map(|r| (r, &tok[i + 2..]))
}

const UNINFORMATIVE: &[&str] =
    &["unidentified", "unclassified", "unassigned", "uncultured", "unknown", "metagenome", "environmental", "na"];

fn is_uninformative(name: &str) -> bool {
    let l = name.to_ascii_lowercase();
    UNINFORMATIVE.iter().any(|u| l == *u || l.starts_with(&format!("{u} ")) || l.starts_with(&format!("{u}_")))
        || l.ends_with(" metagenome")
}

/// Parse a lineage string. `positional_ranks` gives rank names for unprefixed
/// levels (position 0 = first level below Root).
pub fn parse_lineage(s: &str, positional_ranks: Option<&[String]>) -> Lineage {
    let mut tokens: Vec<&str> = s.split(';').map(str::trim).filter(|t| !t.is_empty()).collect();
    if tokens.first().map(|t| t.eq_ignore_ascii_case("root")).unwrap_or(false) {
        tokens.remove(0);
    }
    let mut lin = Lineage::default();
    for (pos, tok) in tokens.iter().enumerate() {
        let (rank, name) = match split_prefix(tok) {
            Some((r, rest)) => (Some(r.to_string()), rest.trim()),
            None => (positional_ranks.and_then(|p| p.get(pos)).cloned(), *tok),
        };
        let name = name.trim_matches(|c: char| c == '"' || c == '\'' || c.is_whitespace());
        if name.is_empty() || is_uninformative(name) {
            break;
        }
        lin.names.push(name.to_string());
        lin.ranks.push(rank);
    }
    lin
}

/// Extract a lineage string from a FASTA header, supporting
/// `>id Lineage;...` (SILVA / DECIPHER), `>Lineage;...` and UNITE
/// `>name|acc|SH|reps|k__Fungi;p__...`.
pub fn lineage_from_header(id: &str, desc: &str) -> Option<String> {
    // Header is itself a lineage (DADA2/DECIPHER style); names may contain spaces.
    if id.contains(';') && !id.contains('|') {
        return Some(if desc.is_empty() { id.to_string() } else { format!("{id} {desc}") });
    }
    if desc.contains(';') {
        return Some(desc.to_string());
    }
    if id.contains('|') {
        if let Some(f) = id.split('|').find(|f| f.contains(';') || f.contains("__")) {
            return Some(f.to_string());
        }
    }
    if id.contains(';') {
        return Some(id.to_string());
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_prefixed_and_truncates() {
        let l = parse_lineage("d__Bacteria; p__Firmicutes; c__Bacilli; o__; f__X", None);
        assert_eq!(l.names, vec!["Bacteria", "Firmicutes", "Bacilli"]);
        assert_eq!(l.ranks[1].as_deref(), Some("phylum"));
    }

    #[test]
    fn parses_root_and_positional() {
        let r = vec!["domain".to_string(), "phylum".to_string()];
        let l = parse_lineage("Root;Bacteria;Proteobacteria;uncultured", Some(&r));
        assert_eq!(l.names, vec!["Bacteria", "Proteobacteria"]);
        assert_eq!(l.ranks[0].as_deref(), Some("domain"));
    }

    #[test]
    fn header_styles() {
        assert_eq!(lineage_from_header("AB1.1.1500", "Bacteria;Firmicutes").unwrap(), "Bacteria;Firmicutes");
        assert_eq!(
            lineage_from_header("Glomus_sp|KJ1|SH1.08FU|reps|k__Fungi;p__Glomeromycota", "").unwrap(),
            "k__Fungi;p__Glomeromycota"
        );
        assert!(lineage_from_header("seq1", "").is_none());
        assert_eq!(
            lineage_from_header("Eukaryota;Fungi;Rhizophydium", "sp. X;Rhizophydium_x;").unwrap(),
            "Eukaryota;Fungi;Rhizophydium sp. X;Rhizophydium_x;"
        );
    }

    #[test]
    fn tree_ops() {
        let mut t = Taxonomy::new();
        let a = t.add_lineage(&parse_lineage("A;B;C", None));
        let b = t.add_lineage(&parse_lineage("A;D", None));
        assert_eq!(t.lineage_string(a), "A;B;C");
        let anc = t.find(&["A".to_string()]).unwrap();
        assert!(t.is_ancestor_or_self(anc, a));
        assert!(t.is_ancestor_or_self(anc, b));
        assert!(!t.is_ancestor_or_self(a, b));
    }
}
