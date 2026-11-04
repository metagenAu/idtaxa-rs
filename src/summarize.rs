//! Collapse classified features (ASVs/OTUs) into taxon × sample tables.
//!
//! Classifying every dataset with the same reference and classifier and then
//! collapsing to a common rank (e.g. genus) is the most robust way to
//! compare studies that used different primer sets. The *long* output format
//! (`study, primer_set, sample, level, taxon, lineage, count,
//! rel_abundance`) can simply be concatenated across studies and loaded into
//! a database, statistics package or knowledge graph.

use crate::seq::open_input;
use crate::taxonomy::parse_lineage;
use anyhow::{bail, Context, Result};
use std::collections::{BTreeMap, HashMap};
use std::io::{BufRead, Write};
use std::path::Path;

/// Classified lineage of one feature.
#[derive(Clone, Debug, Default)]
pub struct FeatureTaxon {
    pub names: Vec<String>,
    pub ranks: Vec<String>,
}

/// Aggregation level.
#[derive(Clone, Debug)]
pub enum Level {
    Rank(String),
    Depth(usize),
}

impl Level {
    pub fn label(&self) -> String {
        match self {
            Level::Rank(r) => r.clone(),
            Level::Depth(d) => format!("level{d}"),
        }
    }
}

/// Read classifications produced by `idtaxa classify` (TSV or QIIME format).
pub fn read_classification(path: &Path) -> Result<HashMap<String, FeatureTaxon>> {
    let r = open_input(path)?;
    let mut lines = r.lines();
    let header = lines.next().context("empty classification file")??;
    let cols: Vec<&str> = header.split('\t').collect();
    let find = |name: &str| cols.iter().position(|c| c.eq_ignore_ascii_case(name));
    let mut out = HashMap::new();
    if let (Some(id), Some(lin)) = (find("id"), find("lineage")) {
        let rk = find("lineage_ranks");
        for line in lines {
            let line = line?;
            if line.is_empty() {
                continue;
            }
            let f: Vec<&str> = line.split('\t').collect();
            let names: Vec<String> = f
                .get(lin)
                .map(|s| s.split(';').filter(|x| !x.is_empty()).map(String::from).collect())
                .unwrap_or_default();
            let ranks: Vec<String> = match rk.and_then(|i| f.get(i)) {
                Some(s) if !s.is_empty() => s.split(';').map(String::from).collect(),
                _ => vec!["NA".to_string(); names.len()],
            };
            out.insert(f[id].to_string(), FeatureTaxon { names, ranks });
        }
    } else if let (Some(id), Some(tx)) = (find("Feature ID"), find("Taxon")) {
        for line in lines {
            let line = line?;
            if line.is_empty() || line.starts_with('#') {
                continue;
            }
            let f: Vec<&str> = line.split('\t').collect();
            let lin = parse_lineage(f.get(tx).copied().unwrap_or(""), None);
            let ranks = lin.ranks.iter().map(|r| r.clone().unwrap_or_else(|| "NA".into())).collect();
            out.insert(f[id].to_string(), FeatureTaxon { names: lin.names, ranks });
        }
    } else {
        bail!("{}: expected columns 'id' + 'lineage' (idtaxa TSV) or 'Feature ID' + 'Taxon' (QIIME)", path.display());
    }
    Ok(out)
}

/// A feature × sample count table.
#[derive(Clone, Debug, Default)]
pub struct Counts {
    pub samples: Vec<String>,
    pub features: Vec<String>,
    /// `values[feature][sample]`
    pub values: Vec<Vec<f64>>,
}

/// Read a TSV count table (features as rows by default; BIOM-TSV comment
/// lines are skipped). With `samples_as_rows`, rows are samples (e.g. a
/// DADA2 sequence table).
pub fn read_counts(path: &Path, samples_as_rows: bool) -> Result<Counts> {
    let r = open_input(path)?;
    let mut rows: Vec<Vec<String>> = Vec::new();
    let mut header: Option<Vec<String>> = None;
    for line in r.lines() {
        let line = line?;
        if line.trim().is_empty() {
            continue;
        }
        if header.is_none() {
            if line.starts_with("# ") || line.starts_with("#Constructed") {
                continue;
            }
            header = Some(line.split('\t').map(|s| s.trim().to_string()).collect());
            continue;
        }
        rows.push(line.split('\t').map(|s| s.trim().to_string()).collect());
    }
    let header = header.context("count table has no header")?;
    let cols: Vec<String> = header[1..].to_vec();
    let mut row_names = Vec::new();
    let mut mat: Vec<Vec<f64>> = Vec::new();
    for (i, r) in rows.iter().enumerate() {
        if r.len() != header.len() {
            bail!("{}: row {} has {} fields, header has {}", path.display(), i + 2, r.len(), header.len());
        }
        row_names.push(r[0].clone());
        let v: Result<Vec<f64>> =
            r[1..].iter().map(|x| x.parse::<f64>().with_context(|| format!("non-numeric count '{x}'"))).collect();
        mat.push(v?);
    }
    if samples_as_rows {
        let nf = cols.len();
        let mut values = vec![vec![0.0; row_names.len()]; nf];
        for (s, row) in mat.iter().enumerate() {
            for (f, &v) in row.iter().enumerate() {
                values[f][s] = v;
            }
        }
        Ok(Counts { samples: row_names, features: cols, values })
    } else {
        Ok(Counts { samples: cols, features: row_names, values: mat })
    }
}

/// Name and full lineage path of a feature at the requested level.
pub fn label_for(ft: Option<&FeatureTaxon>, level: &Level) -> (String, String) {
    let ft = match ft {
        Some(f) if !f.names.is_empty() => f,
        _ => return ("Unassigned".into(), "Unassigned".into()),
    };
    let idx = match level {
        Level::Rank(r) => ft.ranks.iter().position(|x| x.eq_ignore_ascii_case(r)),
        Level::Depth(d) => (*d >= 1 && ft.names.len() >= *d).then(|| d - 1),
    };
    match idx {
        Some(i) => (ft.names[i].clone(), ft.names[..=i].join(";")),
        None => {
            let last = ft.names.last().unwrap();
            (format!("unclassified_{last}"), format!("{};unclassified", ft.names.join(";")))
        }
    }
}

/// Aggregated taxon table: lineage path → (name, per-sample counts).
pub struct TaxonTable {
    pub samples: Vec<String>,
    pub rows: BTreeMap<String, (String, Vec<f64>)>,
}

pub fn aggregate(class: &HashMap<String, FeatureTaxon>, counts: Option<&Counts>, level: &Level) -> TaxonTable {
    let mut rows: BTreeMap<String, (String, Vec<f64>)> = BTreeMap::new();
    match counts {
        Some(c) => {
            for (f, vals) in c.features.iter().zip(&c.values) {
                let (name, path) = label_for(class.get(f), level);
                let e = rows.entry(path).or_insert_with(|| (name, vec![0.0; c.samples.len()]));
                for (a, b) in e.1.iter_mut().zip(vals) {
                    *a += b;
                }
            }
            TaxonTable { samples: c.samples.clone(), rows }
        }
        None => {
            for ft in class.values() {
                let (name, path) = label_for(Some(ft), level);
                rows.entry(path).or_insert_with(|| (name, vec![0.0])).1[0] += 1.0;
            }
            TaxonTable { samples: vec!["n_features".into()], rows }
        }
    }
}

impl TaxonTable {
    fn totals(&self) -> Vec<f64> {
        let mut t = vec![0.0; self.samples.len()];
        for (_, v) in self.rows.values() {
            for (a, b) in t.iter_mut().zip(v) {
                *a += b;
            }
        }
        t
    }

    pub fn write_wide(&self, w: &mut dyn Write, relative: bool) -> std::io::Result<()> {
        let totals = self.totals();
        write!(w, "lineage\ttaxon")?;
        for s in &self.samples {
            write!(w, "\t{s}")?;
        }
        writeln!(w)?;
        for (path, (name, v)) in &self.rows {
            write!(w, "{path}\t{name}")?;
            for (x, t) in v.iter().zip(&totals) {
                if relative {
                    write!(w, "\t{:.6}", if *t > 0.0 { x / t } else { 0.0 })?;
                } else {
                    write!(w, "\t{x}")?;
                }
            }
            writeln!(w)?;
        }
        Ok(())
    }

    pub fn write_long(
        &self,
        w: &mut dyn Write,
        study: &str,
        primer_set: &str,
        level: &str,
        header: bool,
    ) -> std::io::Result<()> {
        let totals = self.totals();
        if header {
            writeln!(w, "study\tprimer_set\tsample\tlevel\ttaxon\tlineage\tcount\trel_abundance")?;
        }
        for (si, s) in self.samples.iter().enumerate() {
            for (path, (name, v)) in &self.rows {
                let x = v[si];
                if x == 0.0 {
                    continue;
                }
                let rel = if totals[si] > 0.0 { x / totals[si] } else { 0.0 };
                writeln!(w, "{study}\t{primer_set}\t{s}\t{level}\t{name}\t{path}\t{x}\t{rel:.6}")?;
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn labels() {
        let ft = FeatureTaxon {
            names: vec!["Bacteria".into(), "Firmicutes".into()],
            ranks: vec!["domain".into(), "phylum".into()],
        };
        assert_eq!(label_for(Some(&ft), &Level::Rank("phylum".into())).0, "Firmicutes");
        assert_eq!(label_for(Some(&ft), &Level::Rank("genus".into())).0, "unclassified_Firmicutes");
        assert_eq!(label_for(Some(&ft), &Level::Depth(1)).1, "Bacteria");
        assert_eq!(label_for(None, &Level::Depth(1)).0, "Unassigned");
    }
}
