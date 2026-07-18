//! Count-Min sketch + top-k heavy-hitter tracking (hub detection without
//! per-node state — degree arrays stop scaling first; this never does).

use std::collections::BinaryHeap;

const DEPTH: usize = 4;
const WIDTH: usize = 1 << 16;

pub struct CountMinTopK {
    rows: Vec<Vec<u32>>,
    k: usize,
    /// Min-heap of (count, key) — smallest tracked count on top.
    heap: BinaryHeap<std::cmp::Reverse<(u64, u64)>>,
    tracked: std::collections::HashMap<u64, u64>,
}

impl CountMinTopK {
    pub fn new(k: usize) -> CountMinTopK {
        CountMinTopK {
            rows: vec![vec![0u32; WIDTH]; DEPTH],
            k,
            heap: BinaryHeap::new(),
            tracked: std::collections::HashMap::new(),
        }
    }

    #[inline]
    pub fn insert(&mut self, key: u64) {
        let mut est = u32::MAX;
        for (d, row) in self.rows.iter_mut().enumerate() {
            let h = mix(key, d as u64) as usize & (WIDTH - 1);
            row[h] = row[h].saturating_add(1);
            est = est.min(row[h]);
        }
        let est = est as u64;
        // Track in top-k if it qualifies.
        if let Some(c) = self.tracked.get_mut(&key) {
            *c = est;
            return;
        }
        if self.tracked.len() < self.k {
            self.tracked.insert(key, est);
            self.heap.push(std::cmp::Reverse((est, key)));
        } else if let Some(&std::cmp::Reverse((min_c, _))) = self.heap.peek() {
            if est > min_c {
                // Evict the current minimum (lazily: pop until live entry).
                while let Some(std::cmp::Reverse((c, k))) = self.heap.pop() {
                    if self.tracked.get(&k) == Some(&c) {
                        self.tracked.remove(&k);
                        break;
                    }
                }
                self.tracked.insert(key, est);
                self.heap.push(std::cmp::Reverse((est, key)));
            }
        }
    }

    /// Top heavy hitters as (key, estimated_count), descending.
    ///
    /// `tracked` is a `HashMap`, whose iteration order is randomly seeded per
    /// process, so ties must be broken on a field that gives a total order —
    /// otherwise equal-count entries come out in a different order on every run
    /// and the emitted stats file is not byte-reproducible. Ties break on the
    /// node key, ascending.
    pub fn top(&self) -> Vec<(u64, u64)> {
        let mut v: Vec<(u64, u64)> = self.tracked.iter().map(|(&k, &c)| (k, c)).collect();
        v.sort_unstable_by_key(|&(k, c)| (std::cmp::Reverse(c), k));
        v
    }
}

#[inline]
fn mix(key: u64, salt: u64) -> u64 {
    let mut z = key ^ salt.wrapping_mul(0xA24BAED4963EE407);
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
    z ^ (z >> 31)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn finds_heavy_hitters() {
        let mut cm = CountMinTopK::new(5);
        for i in 0..10_000u64 {
            cm.insert(i % 1000); // uniform background
        }
        for _ in 0..5_000 {
            cm.insert(42);
        }
        for _ in 0..3_000 {
            cm.insert(7);
        }
        let top = cm.top();
        assert_eq!(top[0].0, 42);
        assert_eq!(top[1].0, 7);
        assert!(top[0].1 >= 5_000);
    }
}
