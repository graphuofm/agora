//! Compressed sparse row adjacency over global node ids.

/// CSR indexed by *local* src index (0..n_src); targets are global node ids.
#[derive(Debug, Clone)]
pub struct Csr {
    pub offsets: Vec<u64>,
    pub targets: Vec<u64>,
}

impl Csr {
    /// Build from (src_local, dst_global) pairs via counting sort — O(E).
    pub fn from_edges(n_src: usize, edges: &[(u64, u64)]) -> Csr {
        let mut counts = vec![0u64; n_src + 1];
        for &(s, _) in edges {
            counts[s as usize + 1] += 1;
        }
        for i in 1..counts.len() {
            counts[i] += counts[i - 1];
        }
        let offsets = counts.clone();
        let mut cursor = counts;
        let mut targets = vec![0u64; edges.len()];
        for &(s, d) in edges {
            let at = cursor[s as usize];
            targets[at as usize] = d;
            cursor[s as usize] += 1;
        }
        Csr { offsets, targets }
    }

    #[inline]
    pub fn neighbors(&self, src_local: u64) -> &[u64] {
        let lo = self.offsets[src_local as usize] as usize;
        let hi = self.offsets[src_local as usize + 1] as usize;
        &self.targets[lo..hi]
    }

    pub fn n_src(&self) -> usize {
        self.offsets.len() - 1
    }

    pub fn n_edges(&self) -> usize {
        self.targets.len()
    }
}

/// One relation's materialized skeleton.
#[derive(Debug, Clone)]
pub struct RelationGraph {
    pub name: String,
    /// Global id range [start, start+n) of src / dst entity types.
    pub src_start: u64,
    pub n_src: u64,
    pub dst_start: u64,
    pub n_dst: u64,
    pub csr: Csr,
}

impl RelationGraph {
    /// Neighbors of a global src id (empty if isolated).
    #[inline]
    pub fn neighbors_of(&self, src_global: u64) -> &[u64] {
        self.csr.neighbors(src_global - self.src_start)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn counting_sort_csr() {
        let edges = vec![(2u64, 20u64), (0, 10), (2, 21), (0, 11), (1, 15)];
        let csr = Csr::from_edges(3, &edges);
        assert_eq!(csr.neighbors(0), &[10, 11]);
        assert_eq!(csr.neighbors(1), &[15]);
        assert_eq!(csr.neighbors(2), &[20, 21]);
        assert_eq!(csr.n_edges(), 5);
    }
}
