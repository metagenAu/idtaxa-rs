//! Group-size balancing (Phase 6.7): cap the number of training sequences
//! per leaf group with diversity-preserving subsampling.

use rayon::prelude::*;

/// Jaccard similarity of two sorted unique k-mer lists.
pub fn jaccard(a: &[u32], b: &[u32]) -> f64 {
    let (mut i, mut j, mut inter) = (0, 0, 0usize);
    while i < a.len() && j < b.len() {
        match a[i].cmp(&b[j]) {
            std::cmp::Ordering::Less => i += 1,
            std::cmp::Ordering::Greater => j += 1,
            std::cmp::Ordering::Equal => {
                inter += 1;
                i += 1;
                j += 1;
            }
        }
    }
    let uni = a.len() + b.len() - inter;
    if uni == 0 {
        1.0
    } else {
        inter as f64 / uni as f64
    }
}

/// Select at most `cap` members of each group by greedy farthest-point
/// (max-min Jaccard distance) sampling, starting from the member with the
/// most k-mers (lowest index on ties). Exact duplicates are therefore
/// dropped first. Returns selected indices in ascending order.
pub fn cap_groups(groups: &[u32], kmers: &[Vec<u32>], cap: usize) -> Vec<usize> {
    let cap = cap.max(1);
    let mut by_group: std::collections::BTreeMap<u32, Vec<usize>> = Default::default();
    for (i, &g) in groups.iter().enumerate() {
        by_group.entry(g).or_default().push(i);
    }
    let lists: Vec<Vec<usize>> = by_group.into_values().collect();
    let mut selected: Vec<usize> = lists
        .par_iter()
        .flat_map_iter(|members| {
            if members.len() <= cap {
                return members.clone();
            }
            let first = *members
                .iter()
                .max_by(|&&a, &&b| kmers[a].len().cmp(&kmers[b].len()).then(b.cmp(&a)))
                .unwrap();
            let mut chosen = vec![first];
            // min distance of each member to the chosen set
            let mut mind: Vec<f64> = members.iter().map(|&m| 1.0 - jaccard(&kmers[m], &kmers[first])).collect();
            while chosen.len() < cap {
                let mut best = None;
                for (t, &m) in members.iter().enumerate() {
                    if chosen.contains(&m) {
                        continue;
                    }
                    match best {
                        None => best = Some(t),
                        Some(bt) if mind[t] > mind[bt] => best = Some(t),
                        _ => {}
                    }
                }
                let Some(bt) = best else { break };
                let nm = members[bt];
                chosen.push(nm);
                for (t, &m) in members.iter().enumerate() {
                    let d = 1.0 - jaccard(&kmers[m], &kmers[nm]);
                    if d < mind[t] {
                        mind[t] = d;
                    }
                }
            }
            chosen
        })
        .collect();
    selected.sort_unstable();
    selected
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn caps_and_prefers_diverse() {
        let kmers = vec![
            vec![1, 2, 3, 4],
            vec![1, 2, 3, 4], // duplicate of 0
            vec![10, 11, 12],
            vec![1, 2, 3, 5],
            vec![7],
        ];
        let groups = vec![0, 0, 0, 0, 1];
        let sel = cap_groups(&groups, &kmers, 2);
        assert_eq!(sel, vec![0, 2, 4]);
        let sel3 = cap_groups(&groups, &kmers, 3);
        assert_eq!(sel3, vec![0, 2, 3, 4]);
        assert!((jaccard(&[1, 2], &[2, 3]) - 1.0 / 3.0).abs() < 1e-12);
    }
}
