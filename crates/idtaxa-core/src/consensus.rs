//! Consensus of IDTAXA assignments and phylogenetic placement (Phase 6.5).
//!
//! Placement results are read from `gappa examine assign --per-query-results`
//! (`name LWR fract aLWR afract taxopath`) and optionally an EPA-ng jplace
//! file for pendant branch lengths. Placement is not reimplemented.

use std::collections::{HashMap, HashSet};
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::Path;

use crate::error::{Error, Result};

/// IDTAXA result for consensus: assigned names below Root.
#[derive(Clone, Debug, Default)]
pub struct IdtaxaCall {
    pub names: Vec<String>,
}

/// Placement support: accumulated LWR per lineage prefix.
#[derive(Clone, Debug, Default)]
pub struct PlacementCall {
    /// prefix path ("A;B") -> accumulated LWR
    pub support: HashMap<String, f64>,
}

impl PlacementCall {
    /// Deepest lineage whose every prefix has support >= `min_lwr`
    /// (greedy: best-supported child at each depth).
    pub fn lineage(&self, min_lwr: f64) -> (Vec<String>, Vec<f64>) {
        let mut names: Vec<String> = Vec::new();
        let mut lwr = Vec::new();
        loop {
            let prefix = names.join(";");
            let depth = names.len() + 1;
            let mut best: Option<(&str, f64)> = None;
            for (p, &v) in &self.support {
                let parts: Vec<&str> = p.split(';').collect();
                if parts.len() != depth {
                    continue;
                }
                let par = parts[..depth - 1].join(";");
                if par != prefix {
                    continue;
                }
                let last = parts[depth - 1];
                match best {
                    Some((bn, bv)) if v < bv || (v == bv && last > bn) => {}
                    _ => best = Some((last, v)),
                }
            }
            match best {
                Some((n, v)) if v >= min_lwr => {
                    names.push(n.to_string());
                    lwr.push(v);
                }
                _ => break,
            }
        }
        (names, lwr)
    }
}

fn open(p: &Path) -> Result<BufReader<File>> {
    Ok(BufReader::new(File::open(p).map_err(|e| Error::io(p, e))?))
}

/// Read `idtaxa classify` TSV output (`id` and `taxonomy` columns).
pub fn read_idtaxa_tsv(p: &Path) -> Result<Vec<(String, IdtaxaCall)>> {
    let mut out = Vec::new();
    let mut cols: Option<(usize, usize)> = None;
    for line in open(p)?.lines() {
        let line = line.map_err(|e| Error::io(p, e))?;
        if line.starts_with('#') || line.is_empty() {
            continue;
        }
        let f: Vec<&str> = line.split('\t').collect();
        match cols {
            None => {
                let id = f.iter().position(|&x| x == "id");
                let tx = f.iter().position(|&x| x == "taxonomy");
                match (id, tx) {
                    (Some(a), Some(b)) => cols = Some((a, b)),
                    _ => {
                        return Err(Error::InvalidParameter(format!(
                            "{}: missing id/taxonomy header",
                            p.display()
                        )))
                    }
                }
            }
            Some((a, b)) => {
                let names = f
                    .get(b)
                    .map(|t| {
                        t.split(';')
                            .filter(|x| !x.is_empty())
                            .map(str::to_string)
                            .collect()
                    })
                    .unwrap_or_default();
                out.push((f[a].to_string(), IdtaxaCall { names }));
            }
        }
    }
    Ok(out)
}

/// Read gappa per-query results; LWR of each taxopath row is accumulated
/// onto all its prefixes.
pub fn read_gappa_per_query(p: &Path) -> Result<HashMap<String, PlacementCall>> {
    let mut out: HashMap<String, PlacementCall> = HashMap::new();
    let mut cols: Option<(usize, usize, usize)> = None;
    for line in open(p)?.lines() {
        let line = line.map_err(|e| Error::io(p, e))?;
        if line.is_empty() {
            continue;
        }
        let f: Vec<&str> = line.split('\t').collect();
        match cols {
            None => {
                let name = f.iter().position(|&x| x == "name");
                let lwr = f.iter().position(|&x| x == "LWR");
                let tp = f.iter().position(|&x| x == "taxopath");
                match (name, lwr, tp) {
                    (Some(a), Some(b), Some(c)) => cols = Some((a, b, c)),
                    _ => {
                        return Err(Error::InvalidParameter(format!(
                            "{}: expected gappa per-query header with name, LWR, taxopath",
                            p.display()
                        )))
                    }
                }
            }
            Some((a, b, c)) => {
                let lwr: f64 = f.get(b).and_then(|x| x.parse().ok()).ok_or_else(|| {
                    Error::InvalidParameter(format!("{}: bad LWR value in '{line}'", p.display()))
                })?;
                let mut names: Vec<&str> = f
                    .get(c)
                    .map(|t| t.split(';').filter(|x| !x.is_empty()).collect())
                    .unwrap_or_default();
                if names.first() == Some(&"Root") {
                    names.remove(0);
                }
                let e = out.entry(f[a].to_string()).or_default();
                for d in 1..=names.len() {
                    *e.support.entry(names[..d].join(";")).or_default() += lwr;
                }
            }
        }
    }
    Ok(out)
}

/// Pendant length of the highest-LWR placement per query from a jplace file.
pub fn read_jplace_pendant(p: &Path) -> Result<HashMap<String, f64>> {
    let v: serde_json::Value = serde_json::from_reader(open(p)?)?;
    let fields: Vec<String> = v["fields"]
        .as_array()
        .ok_or_else(|| Error::InvalidParameter("jplace: missing fields".into()))?
        .iter()
        .map(|x| x.as_str().unwrap_or("").to_string())
        .collect();
    let li = fields.iter().position(|f| f == "like_weight_ratio");
    let pi = fields.iter().position(|f| f == "pendant_length");
    let (Some(li), Some(pi)) = (li, pi) else {
        return Err(Error::InvalidParameter(
            "jplace: fields must include like_weight_ratio and pendant_length".into(),
        ));
    };
    let mut out = HashMap::new();
    for pl in v["placements"].as_array().into_iter().flatten() {
        let mut best = (f64::NEG_INFINITY, f64::NAN);
        for row in pl["p"].as_array().into_iter().flatten() {
            let r = row.as_array().unwrap();
            let l = r[li].as_f64().unwrap_or(0.0);
            if l > best.0 {
                best = (l, r[pi].as_f64().unwrap_or(f64::NAN));
            }
        }
        let names: Vec<String> = match (pl.get("n"), pl.get("nm")) {
            (Some(n), _) => n
                .as_array()
                .into_iter()
                .flatten()
                .filter_map(|x| x.as_str().map(str::to_string))
                .collect(),
            (None, Some(nm)) => nm
                .as_array()
                .into_iter()
                .flatten()
                .filter_map(|x| x.as_array().and_then(|a| a[0].as_str().map(str::to_string)))
                .collect(),
            _ => vec![],
        };
        for n in names {
            out.insert(n, best.1);
        }
    }
    Ok(out)
}

/// Consensus rule.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Rule {
    /// Deepest rank where both methods (above their thresholds) agree.
    Agree,
    /// IDTAXA lineage, extended by consistent placement below it.
    IdtaxaFirst,
    /// Placement lineage, extended by consistent IDTAXA below it.
    PlacementFirst,
}

impl std::str::FromStr for Rule {
    type Err = Error;
    fn from_str(s: &str) -> Result<Self> {
        match s {
            "agree" => Ok(Rule::Agree),
            "idtaxa-first" => Ok(Rule::IdtaxaFirst),
            "placement-first" => Ok(Rule::PlacementFirst),
            _ => Err(Error::InvalidParameter(format!(
                "unknown consensus rule {s}"
            ))),
        }
    }
}

/// Consensus result for one query.
#[derive(Clone, Debug)]
pub struct ConsensusRow {
    pub id: String,
    pub taxonomy: Vec<String>,
    /// Source per consensus rank: `both`, `idtaxa`, `placement`.
    pub support: Vec<String>,
    pub idtaxa: Vec<String>,
    pub placement: Vec<String>,
    pub placement_lwr: Vec<f64>,
    pub pendant: Option<f64>,
    pub flags: Vec<String>,
}

/// Combine IDTAXA and placement calls.
pub fn consensus(
    idtaxa: &[(String, IdtaxaCall)],
    placement: &HashMap<String, PlacementCall>,
    pendant: Option<&HashMap<String, f64>>,
    min_lwr: f64,
    max_pendant: Option<f64>,
    known: Option<&HashSet<String>>,
    rule: Rule,
) -> Result<Vec<ConsensusRow>> {
    let mut unknown: Vec<String> = Vec::new();
    let mut check = |names: &[String], src: &str| {
        if let Some(k) = known {
            for d in 1..=names.len() {
                let p = names[..d].join(";");
                if !k.contains(&p) && unknown.len() < 5 {
                    unknown.push(format!("{src}: {p}"));
                }
            }
        }
    };
    let mut rows = Vec::with_capacity(idtaxa.len());
    let mut seen = HashSet::new();
    for (id, call) in idtaxa {
        seen.insert(id.as_str());
        let (pl, lwr) = placement
            .get(id)
            .map(|p| p.lineage(min_lwr))
            .unwrap_or_default();
        check(&call.names, "idtaxa");
        check(&pl, "placement");
        let mut flags = Vec::new();
        if !placement.contains_key(id) {
            flags.push("no_placement".to_string());
        }
        let common = call
            .names
            .iter()
            .zip(&pl)
            .take_while(|(a, b)| a == b)
            .count();
        let disagree = common < call.names.len().min(pl.len());
        if disagree {
            flags.push("disagree".into());
        }
        let (taxonomy, support): (Vec<String>, Vec<String>) = match rule {
            Rule::Agree => (call.names[..common].to_vec(), vec!["both".into(); common]),
            Rule::IdtaxaFirst | Rule::PlacementFirst => {
                let (first, second, a, b) = if rule == Rule::IdtaxaFirst {
                    (&call.names, &pl, "idtaxa", "placement")
                } else {
                    (&pl, &call.names, "placement", "idtaxa")
                };
                if disagree {
                    (first[..common].to_vec(), vec!["both".into(); common])
                } else {
                    let longer = if first.len() >= second.len() {
                        first
                    } else {
                        second
                    };
                    let src: Vec<String> = (0..longer.len())
                        .map(|d| {
                            if d < common {
                                "both".to_string()
                            } else if d < first.len() {
                                a.to_string()
                            } else {
                                b.to_string()
                            }
                        })
                        .collect();
                    (longer.clone(), src)
                }
            }
        };
        let pend = pendant.and_then(|p| p.get(id).copied());
        if let (Some(p), Some(m)) = (pend, max_pendant) {
            if p > m {
                flags.push("long_pendant_possible_novel_or_chimera".into());
            }
        }
        rows.push(ConsensusRow {
            id: id.clone(),
            taxonomy,
            support,
            idtaxa: call.names.clone(),
            placement: pl,
            placement_lwr: lwr,
            pendant: pend,
            flags,
        });
    }
    for (id, p) in placement {
        if !seen.contains(id.as_str()) {
            let (pl, lwr) = p.lineage(min_lwr);
            rows.push(ConsensusRow {
                id: id.clone(),
                taxonomy: vec![],
                support: vec![],
                idtaxa: vec![],
                placement: pl,
                placement_lwr: lwr,
                pendant: pendant.and_then(|x| x.get(id).copied()),
                flags: vec!["no_idtaxa".into()],
            });
        }
    }
    if !unknown.is_empty() {
        return Err(Error::Taxonomy(format!(
            "lineages not found in the shared taxonomy file (both methods must use the same taxonomy): {}",
            unknown.join(", ")
        )));
    }
    Ok(rows)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn pc(rows: &[(&str, f64)]) -> PlacementCall {
        let mut p = PlacementCall::default();
        for (t, l) in rows {
            let n: Vec<&str> = t.split(';').collect();
            for d in 1..=n.len() {
                *p.support.entry(n[..d].join(";")).or_default() += l;
            }
        }
        p
    }

    #[test]
    fn agree_rule() {
        let id = vec![(
            "q".to_string(),
            IdtaxaCall {
                names: vec!["A".into(), "B".into(), "C".into()],
            },
        )];
        let mut pl = HashMap::new();
        pl.insert("q".to_string(), pc(&[("A;B;D", 0.95), ("A;B;C", 0.05)]));
        let r = consensus(&id, &pl, None, 0.9, None, None, Rule::Agree).unwrap();
        assert_eq!(r[0].taxonomy, vec!["A", "B"]);
        assert!(r[0].flags.contains(&"disagree".to_string()));
        let r = consensus(&id, &pl, None, 0.99, None, None, Rule::IdtaxaFirst).unwrap();
        // placement only confident to A;B (LWR 1.0) -> no disagreement, idtaxa extends
        assert_eq!(r[0].taxonomy, vec!["A", "B", "C"]);
        assert_eq!(r[0].support, vec!["both", "both", "idtaxa"]);
    }

    #[test]
    fn unknown_names_error() {
        let id = vec![(
            "q".to_string(),
            IdtaxaCall {
                names: vec!["A".into()],
            },
        )];
        let pl = HashMap::new();
        let known: HashSet<String> = ["B".to_string()].into_iter().collect();
        assert!(consensus(&id, &pl, None, 0.9, None, Some(&known), Rule::Agree).is_err());
    }
}
