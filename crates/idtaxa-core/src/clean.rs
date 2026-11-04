//! Reference label cleaning (Phase 6.2).

use std::collections::{BTreeMap, BTreeSet};

use crate::error::{Error, Result};

/// Case normalisation.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum CaseMode {
    #[default]
    Keep,
    /// First letter upper case, rest unchanged.
    Capitalize,
    Lower,
}

impl std::str::FromStr for CaseMode {
    type Err = Error;
    fn from_str(s: &str) -> Result<Self> {
        match s {
            "keep" => Ok(CaseMode::Keep),
            "capitalize" => Ok(CaseMode::Capitalize),
            "lower" => Ok(CaseMode::Lower),
            _ => Err(Error::InvalidParameter(format!(
                "--case must be keep, capitalize or lower (got {s})"
            ))),
        }
    }
}

/// Default case-insensitive substrings marking uninformative labels.
pub const DEFAULT_PLACEHOLDERS: &[&str] = &[
    "uncultured",
    "unidentified",
    "unclassified",
    "unassigned",
    "unknown",
    "environmental sample",
    "metagenome",
    "incertae sedis",
    "clone",
];

/// Cleaning options.
#[derive(Clone, Debug)]
pub struct CleanOptions {
    pub truncate_placeholders: bool,
    pub placeholder_patterns: Vec<String>,
    /// Treat `Genus sp.`/`Genus_sp` species and `sp.` + strain/number
    /// labels as placeholders.
    pub sp_placeholders: bool,
    /// Also treat PR2-style filler ranks (`Name_X`, `Name_XX`) as
    /// placeholders. Off by default: in PR2 informative ranks usually
    /// follow them, and truncating there drops those ranks.
    pub x_fillers: bool,
    /// Truncate lineages to at most this many ranks.
    pub rank_count: Option<usize>,
    pub case: CaseMode,
}

impl Default for CleanOptions {
    fn default() -> Self {
        Self {
            truncate_placeholders: true,
            placeholder_patterns: DEFAULT_PLACEHOLDERS.iter().map(|s| s.to_string()).collect(),
            sp_placeholders: true,
            x_fillers: false,
            rank_count: None,
            case: CaseMode::Keep,
        }
    }
}

/// One logged change.
#[derive(Clone, Debug, PartialEq)]
pub struct Change {
    pub index: usize,
    pub action: String,
    pub before: String,
    pub after: String,
    pub detail: String,
}

/// A taxonomy conflict.
#[derive(Clone, Debug, PartialEq)]
pub struct Conflict {
    /// `homonym` (same name, same depth, different parents) or
    /// `rank_shift` (same name at different depths).
    pub kind: String,
    pub name: String,
    pub depth: usize,
    pub lineages: Vec<String>,
}

/// Cleaning output: new lineage per input (None = no informative rank).
#[derive(Clone, Debug)]
pub struct Cleaned {
    pub lineages: Vec<Option<String>>,
    pub changes: Vec<Change>,
    pub conflicts: Vec<Conflict>,
}

/// Whether a name is a placeholder under the options.
/// Split a label into lower-case word tokens; `_`, `.`, `-` and whitespace
/// separate words, so "Incertae_Sedis", ".incertae.sedis" and
/// "incertae sedis" all give `["incertae", "sedis"]`.
pub fn label_tokens(name: &str) -> Vec<String> {
    name.to_lowercase()
        .split(|c: char| c == '_' || c == '.' || c == '-' || c.is_whitespace())
        .filter(|t| !t.is_empty())
        .map(str::to_string)
        .collect()
}

/// Whether a label is a PR2-style filler rank (`Name_X`, `Name_XX`, ...).
pub fn is_x_filler(name: &str) -> bool {
    match name.rsplit_once('_') {
        Some((head, tail)) => {
            !head.is_empty() && !tail.is_empty() && tail.chars().all(|c| c == 'X' || c == 'x')
        }
        None => false,
    }
}

/// Whether a name is a placeholder under the options. Patterns match whole
/// words only (a pattern's words must appear consecutively in the label),
/// so `clone` flags "uncultured clone 12" but not "Cycloneritida".
pub fn is_placeholder(name: &str, depth_is_last: bool, o: &CleanOptions) -> bool {
    let toks = label_tokens(name);
    if toks.is_empty() || (toks.len() == 1 && (toks[0] == "na" || toks[0] == "nan")) {
        return true;
    }
    if o.x_fillers && is_x_filler(name) {
        return true;
    }
    for p in &o.placeholder_patterns {
        let pt = label_tokens(p);
        if !pt.is_empty() && toks.windows(pt.len()).any(|w| w == pt.as_slice()) {
            return true;
        }
    }
    if o.sp_placeholders && depth_is_last {
        // "Genus sp." / "Genus_sp" / "Genus sp. ABC123" / "Genus_sp._123"
        if toks.len() >= 2 && (toks[1] == "sp" || toks[1] == "spp") {
            return true;
        }
    }
    false
}

fn apply_case(s: &str, c: CaseMode) -> String {
    match c {
        CaseMode::Keep => s.to_string(),
        CaseMode::Lower => s.to_lowercase(),
        CaseMode::Capitalize => {
            let mut ch = s.chars();
            match ch.next() {
                Some(f) => f.to_uppercase().collect::<String>() + ch.as_str(),
                None => String::new(),
            }
        }
    }
}

/// Clean raw lineage strings (with or without a leading `Root`).
pub fn clean_lineages(raw: &[String], o: &CleanOptions) -> Cleaned {
    let mut out = Vec::with_capacity(raw.len());
    let mut changes = Vec::new();
    for (i, r) in raw.iter().enumerate() {
        let mut names: Vec<String> = r
            .trim()
            .trim_end_matches(';')
            .split(';')
            .map(|x| x.trim().to_string())
            .collect();
        if names.first().map(String::as_str) == Some("Root") {
            names.remove(0);
        }
        let before = names.join(";");
        let mut new: Vec<String> = names.iter().map(|n| apply_case(n, o.case)).collect();
        if o.case != CaseMode::Keep && new != names {
            changes.push(Change {
                index: i,
                action: "case".into(),
                before: before.clone(),
                after: new.join(";"),
                detail: String::new(),
            });
        }
        if let Some(rc) = o.rank_count {
            if new.len() > rc {
                let after: Vec<String> = new[..rc].to_vec();
                changes.push(Change {
                    index: i,
                    action: "truncate_rank_count".into(),
                    before: new.join(";"),
                    after: after.join(";"),
                    detail: format!("{} ranks > {rc}", new.len()),
                });
                new = after;
            }
        }
        if o.truncate_placeholders {
            let n = new.len();
            if let Some(p) = (0..n).find(|&d| is_placeholder(&new[d], d + 1 == n, o)) {
                let after: Vec<String> = new[..p].to_vec();
                changes.push(Change {
                    index: i,
                    action: "truncate_placeholder".into(),
                    before: new.join(";"),
                    after: after.join(";"),
                    detail: format!("uninformative label '{}' at rank {}", new[p], p + 1),
                });
                new = after;
            }
        }
        out.push(if new.is_empty() {
            None
        } else {
            Some(new.join(";"))
        });
    }

    // conflicts
    let mut parents: BTreeMap<(usize, String), BTreeSet<String>> = BTreeMap::new();
    let mut depths: BTreeMap<String, BTreeSet<usize>> = BTreeMap::new();
    for l in out.iter().flatten() {
        let names: Vec<&str> = l.split(';').collect();
        for d in 0..names.len() {
            parents
                .entry((d + 1, names[d].to_string()))
                .or_default()
                .insert(names[..d].join(";"));
            depths
                .entry(names[d].to_string())
                .or_default()
                .insert(d + 1);
        }
    }
    let mut conflicts = Vec::new();
    for ((d, name), ps) in &parents {
        if ps.len() > 1 {
            conflicts.push(Conflict {
                kind: "homonym".into(),
                name: name.clone(),
                depth: *d,
                lineages: ps.iter().cloned().collect(),
            });
        }
    }
    for (name, ds) in &depths {
        if ds.len() > 1 {
            conflicts.push(Conflict {
                kind: "rank_shift".into(),
                name: name.clone(),
                depth: *ds.iter().next().unwrap(),
                lineages: ds.iter().map(|d| format!("depth {d}")).collect(),
            });
        }
    }
    Cleaned {
        lineages: out,
        changes,
        conflicts,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn truncates_placeholders() {
        let raw = vec![
            "Root;Bacteria;Firmicutes;uncultured;Foo".to_string(),
            "Bacteria;Firmicutes;Bacilli;Bacillus;Bacillus_sp.".to_string(),
            "Bacteria; Firmicutes ;Bacilli;Bacillus;Bacillus subtilis".to_string(),
            "metagenome".to_string(),
        ];
        let c = clean_lineages(&raw, &CleanOptions::default());
        assert_eq!(c.lineages[0].as_deref(), Some("Bacteria;Firmicutes"));
        assert_eq!(
            c.lineages[1].as_deref(),
            Some("Bacteria;Firmicutes;Bacilli;Bacillus")
        );
        assert_eq!(
            c.lineages[2].as_deref(),
            Some("Bacteria;Firmicutes;Bacilli;Bacillus;Bacillus subtilis")
        );
        assert_eq!(c.lineages[3], None);
        assert_eq!(c.changes.len(), 3);
    }

    #[test]
    fn whole_word_matching() {
        let o = CleanOptions::default();
        assert!(!is_placeholder("Cycloneritida", false, &o));
        assert!(is_placeholder("uncultured_clone_12", false, &o));
        assert!(is_placeholder("Dinophyceae.incertae.sedis", false, &o));
        assert!(is_placeholder("Incertae_Sedis", false, &o));
        assert!(is_placeholder("Environmental_Sample", false, &o));
        assert!(!is_placeholder("Unknownia", false, &o));
        assert!(is_placeholder("Bacillus_sp.", true, &o));
        assert!(!is_placeholder("Bacillus_spizizenii", true, &o));
        assert!(!is_placeholder("Dinophyceae_X", false, &o));
        let ox = CleanOptions {
            x_fillers: true,
            ..Default::default()
        };
        assert!(is_placeholder("Dinophyceae_X", false, &ox));
        assert!(is_placeholder("Annelida_XX", false, &ox));
        assert!(!is_placeholder("Taxon_Xa", false, &ox));
    }

    #[test]
    fn conflicts_detected() {
        let raw = vec!["A;B;X".to_string(), "A;C;X".to_string(), "A;X".to_string()];
        let c = clean_lineages(&raw, &CleanOptions::default());
        assert!(c
            .conflicts
            .iter()
            .any(|x| x.kind == "homonym" && x.name == "X" && x.depth == 3));
        assert!(c
            .conflicts
            .iter()
            .any(|x| x.kind == "rank_shift" && x.name == "X"));
    }

    #[test]
    fn rank_count_and_case() {
        let o = CleanOptions {
            rank_count: Some(2),
            case: CaseMode::Capitalize,
            ..Default::default()
        };
        let c = clean_lineages(&["a;b;c".to_string()], &o);
        assert_eq!(c.lineages[0].as_deref(), Some("A;B"));
    }
}
