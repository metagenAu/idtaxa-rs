//! Persistent classification cache backed by `redb`.
//!
//! Key: xxh3-128 of (model ID, parameter fingerprint, normalised sequence);
//! value: JSON-serialised [`Computed`] result.

use std::path::Path;

use anyhow::{Context, Result};
use redb::{Database, TableDefinition};
use xxhash_rust::xxh3::Xxh3;

use crate::classify_cmd::Computed;

const TABLE: TableDefinition<u128, &[u8]> = TableDefinition::new("classifications");

pub struct Cache {
    db: Database,
    prefix: Vec<u8>,
}

impl Cache {
    pub fn open(path: &Path, model_id: &str, fingerprint: &str) -> Result<Self> {
        let db =
            Database::create(path).with_context(|| format!("opening cache {}", path.display()))?;
        let w = db.begin_write()?;
        w.open_table(TABLE)?;
        w.commit()?;
        let mut prefix = Vec::new();
        prefix.extend_from_slice(model_id.as_bytes());
        prefix.push(0);
        prefix.extend_from_slice(fingerprint.as_bytes());
        prefix.push(0);
        Ok(Self { db, prefix })
    }

    fn key(&self, seq: &[u8]) -> u128 {
        let mut h = Xxh3::new();
        h.update(&self.prefix);
        h.update(seq);
        h.digest128()
    }

    pub fn get_many<'a>(
        &self,
        seqs: impl Iterator<Item = &'a [u8]>,
    ) -> Result<Vec<Option<Computed>>> {
        let r = self.db.begin_read()?;
        let t = r.open_table(TABLE)?;
        let mut out = Vec::new();
        for s in seqs {
            let v = match t.get(self.key(s))? {
                Some(g) => serde_json::from_slice(g.value()).ok(),
                None => None,
            };
            out.push(v);
        }
        Ok(out)
    }

    pub fn put_many<'a>(
        &self,
        items: impl Iterator<Item = (&'a [u8], &'a Computed)>,
    ) -> Result<()> {
        let w = self.db.begin_write()?;
        {
            let mut t = w.open_table(TABLE)?;
            for (s, c) in items {
                let v = serde_json::to_vec(c)?;
                t.insert(self.key(s), v.as_slice())?;
            }
        }
        w.commit()?;
        Ok(())
    }
}
