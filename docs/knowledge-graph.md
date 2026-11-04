# Knowledge graph: management practices → soil microbiome

This document describes how idtaxa-rs outputs feed a knowledge graph of how agricultural management
practices affect the soil microbiome. It covers the schema, the evidence model and the pipeline.

## Design principle: store effects as evidence nodes

A practice's effect on a taxon depends on context (soil, climate, crop) and on how it was measured
(primer set, reference, classifier). So effects are **evidence nodes**, never bare edges:

```
(:Practice {id:"AGRO:no_till"})-[:TESTED_IN]->(:Effect)-[:ON]->(:Taxon {name:"Glomeraceae", rank:"family"})
                                              │
              direction, log ratio, CI, p, n_samples, effect method
                                              │
       ├─[:IN_CONTEXT]──> (:Context {soil_order:"Vertosol", ph_band:"7.5-8.5", climate:"BSk", crop:"wheat"})
       ├─[:FROM]────────> (:Study {id:"PRJNA123456", doi:"10.xxxx/..."})
       ├─[:MEASURED_WITH]─> (:PrimerSet {name:"fITS7-ITS4", marker:"ITS", region:"ITS2"})
       └─[:CLASSIFIED_BY]─> (:Classifier {model:"ITS_fITS7-ITS4.idtx", reference:"UNITE 10.0", threshold:60})
```

## Node types

| label | key properties | identifiers |
|---|---|---|
| `Practice` | name, category (tillage, fertiliser, cover crop, rotation, amendment, pesticide, irrigation), dose/unit | AGRO ontology where possible |
| `Taxon` | name, rank, lineage | NCBI Taxonomy / GTDB / UNITE SH; idtaxa `lineage` column |
| `Function` | name (nitrification, N fixation, P solubilisation, AMF symbiosis, pathogen suppression) | KEGG KO / GO |
| `Outcome` | soil property or agronomic outcome (SOC, aggregate stability, N₂O, yield) | ENVO / PATO |
| `Context` | soil order/texture, pH band, climate class, crop, depth | ENVO, MIxS soil package |
| `Study` | accession, DOI, country, year, design | SRA BioProject, DOI |
| `PrimerSet` | name, marker, region, forward, reverse | `idtaxa primers` |
| `Classifier` | model file, reference + release, threshold, tool version | `idtaxa info` |
| `Effect` | direction, log ratio, CI, p/q value, n, method, evidence tier | (generated) |

Further edges:

* `(:Taxon)-[:HAS_FUNCTION {evidence}]->(:Function)`: from literature or trait databases.
* `(:PrimerSet)-[:UNDER_DETECTS {pct_amplified}]->(:Taxon)`: from `idtaxa trim --coverage`. These edges
  stop a primer blind spot being read as "no effect". For example, the original 515F/806R misses
  Thaumarchaeota, and ITS primers capture AMF poorly.
* `(:Taxon)-[:PARENT_OF]->(:Taxon)`: the taxonomy, so effects can be rolled up.

## Evidence tiers

1. **In-house trials.** Paired treatment and control plots under a known protocol.
2. **Re-processed public data.** Raw reads, primer-matched classifier, effect estimated within the study.
3. **Literature extraction.** Effects reported in papers or meta-analyses. Extraction can be assisted by
   an LLM, but every claim must be checked by a curator.

## Pipeline from idtaxa outputs

```
reads ──detect-primers──> primer set ──> choose model (one per primer set, same reference release)
ASVs ──classify──> taxonomy.tsv ──summarize --long --study S --primer-set P──> long table
long tables (all studies, concatenated) + sample metadata (MIxS/ENVO)
   └─> within-study contrasts (treatment vs control; e.g. ANCOM-BC2 / CLR log ratio per taxon)
         └─> Effect nodes (tier 1/2), with PrimerSet + Classifier provenance
               └─> random-effects meta-analysis per (Practice, Taxon, Context) → pooled Effect nodes
```

Rules that keep cross-primer comparisons honest:

* Estimate effects **within each study** and combine effect sizes. Never pool raw counts across primer sets.
* Compare at a rank the primer region resolves reliably. `idtaxa evaluate` shows this; for most short
  amplicons that means genus or family.
* An effect seen in **studies using two or more different primer sets** is strong evidence. Store
  `n_primer_sets` on pooled effects and rank results by it.
* Treat a missing taxon as "not measured", not as zero, whenever an `UNDER_DETECTS` edge exists for
  that primer set.

## Minimal Neo4j schema

```cypher
CREATE CONSTRAINT practice_id IF NOT EXISTS FOR (p:Practice) REQUIRE p.id IS UNIQUE;
CREATE CONSTRAINT taxon_lineage IF NOT EXISTS FOR (t:Taxon) REQUIRE t.lineage IS UNIQUE;
CREATE CONSTRAINT study_id IF NOT EXISTS FOR (s:Study) REQUIRE s.id IS UNIQUE;
CREATE CONSTRAINT primer_name IF NOT EXISTS FOR (p:PrimerSet) REQUIRE p.name IS UNIQUE;

// load an effects table (one row per study x practice contrast x taxon)
LOAD CSV WITH HEADERS FROM 'file:///effects.csv' AS r
MERGE (p:Practice {id: r.practice_id}) ON CREATE SET p.name = r.practice
MERGE (t:Taxon {lineage: r.lineage}) ON CREATE SET t.name = r.taxon, t.rank = r.level
MERGE (s:Study {id: r.study})
MERGE (ps:PrimerSet {name: r.primer_set})
CREATE (e:Effect {log_ratio: toFloat(r.log_ratio), ci_low: toFloat(r.ci_low), ci_high: toFloat(r.ci_high),
                  q: toFloat(r.q), n: toInteger(r.n), tier: toInteger(r.tier), method: r.method})
CREATE (p)-[:TESTED_IN]->(e)-[:ON]->(t)
CREATE (e)-[:FROM]->(s)
CREATE (e)-[:MEASURED_WITH]->(ps);

// primer blind spots from `idtaxa trim --coverage`
LOAD CSV WITH HEADERS FROM 'file:///coverage_515F-806R.csv' AS r
WITH r WHERE toFloat(r.pct_amplified) < 50
MATCH (ps:PrimerSet {name: '515F-806R'})
MERGE (t:Taxon {lineage: r.taxon})
MERGE (ps)-[u:UNDER_DETECTS]->(t) SET u.pct_amplified = toFloat(r.pct_amplified);
```

Example queries:

```cypher
// taxa that respond consistently to reduced tillage across >= 3 studies and >= 2 primer sets
MATCH (p:Practice {id:'AGRO:no_till'})-[:TESTED_IN]->(e:Effect)-[:ON]->(t:Taxon),
      (e)-[:FROM]->(s:Study), (e)-[:MEASURED_WITH]->(ps:PrimerSet)
WITH t, count(DISTINCT s) AS studies, count(DISTINCT ps) AS primer_sets,
     avg(e.log_ratio) AS mean_lr, sum(CASE WHEN e.log_ratio > 0 THEN 1 ELSE 0 END) AS n_up, count(e) AS n
WHERE studies >= 3 AND primer_sets >= 2 AND (n_up = n OR n_up = 0)
RETURN t.lineage, studies, primer_sets, mean_lr ORDER BY abs(mean_lr) DESC;

// evidence gaps: practice x soil-order combinations with no effects measured
MATCH (p:Practice), (c:Context)
WHERE NOT EXISTS { MATCH (p)-[:TESTED_IN]->(:Effect)-[:IN_CONTEXT]->(c) }
RETURN p.name, c.soil_order;
```

## Suggested first version

* Practices: tillage, N fertiliser (rate and form), cover crops.
* Markers: 16S V4 (or your in-house region) and ITS2. Add AMF SSU if mycorrhizae matter.
* Data: in-house trials plus 15–20 public studies with a treatment and a control, all re-processed.
* Rank: genus and family, plus functional groups (nitrifiers, AMF, N fixers).
* Outputs: effects per study, pooled effects per context, and a gap report.
