//! Skeleton builders for every `TopologyModel` (blueprint §8).
//!
//! Edge budgets derive from `mean_degree × n_src`. Growth-class models
//! (BA, forest fire) are sequential by nature; sampling-class models
//! (uniform, SBM, R-MAT, spatial, affiliation, small-world) are generated
//! per-src with independent streams and could parallelize (M6).

use anyhow::Context;
use rand::Rng;
use agora_rules::{Distribution, RelationRule, TopologyModel};
use agora_sample::{stream, DistSampler, Rng64, StreamPurpose};

use crate::csr::{Csr, RelationGraph};

/// Build one relation's skeleton.
///
/// `src_start/n_src` and `dst_start/n_dst` are the global id ranges of the
/// two entity types; `rel_idx` salts the RNG streams.
pub fn build_relation(
    rule: &RelationRule,
    src_start: u64,
    n_src: u64,
    dst_start: u64,
    n_dst: u64,
    seed: u64,
    rel_idx: u64,
) -> anyhow::Result<RelationGraph> {
    anyhow::ensure!(n_src > 0 && n_dst > 0, "relation `{}`: empty endpoint type", rule.name);
    let edges = match &rule.model {
        TopologyModel::PreferentialAttachment { m } => {
            anyhow::ensure!(
                src_start == dst_start,
                "relation `{}`: preferential attachment requires src == dst type",
                rule.name
            );
            ba(n_src, (*m).max(1) as u64, seed, rel_idx, dst_start)
        }
        TopologyModel::UniformRandom => {
            uniform_random(n_src, n_dst, dst_start, rule.mean_degree, seed, rel_idx)
        }
        TopologyModel::SmallWorld { k, beta } => {
            anyhow::ensure!(
                src_start == dst_start,
                "relation `{}`: small world requires src == dst type",
                rule.name
            );
            small_world(n_src, *k, *beta, dst_start, seed, rel_idx)
        }
        TopologyModel::ForestFire { forward_p, backward_p } => {
            anyhow::ensure!(
                src_start == dst_start,
                "relation `{}`: forest fire requires src == dst type",
                rule.name
            );
            forest_fire(n_src, *forward_p, *backward_p, dst_start, rule.mean_degree, seed, rel_idx)
        }
        TopologyModel::Sbm { communities, p_in_weight, p_out_weight } => sbm(
            n_src,
            n_dst,
            dst_start,
            *communities,
            *p_in_weight,
            *p_out_weight,
            rule.mean_degree,
            seed,
            rel_idx,
        ),
        TopologyModel::RMat { a, b, c, d } => {
            rmat(n_src, n_dst, dst_start, [*a, *b, *c, *d], rule.mean_degree, seed, rel_idx)
        }
        TopologyModel::Spatial { radius } => {
            spatial(n_src, n_dst, dst_start, *radius, rule.mean_degree, seed, rel_idx, src_start == dst_start)
        }
        TopologyModel::Affiliation { popularity } => {
            affiliation(n_src, n_dst, dst_start, popularity, rule.mean_degree, seed, rel_idx)
                .with_context(|| format!("relation `{}`", rule.name))?
        }
    };
    Ok(RelationGraph {
        name: rule.name.clone(),
        src_start,
        n_src,
        dst_start,
        n_dst,
        csr: Csr::from_edges(n_src as usize, &edges),
    })
}

/// Barabási–Albert via Batagelj–Brandes repeated-nodes: O(m·n), exact
/// preferential attachment without degree lookups. Undirected (both
/// directions stored) so every node has neighbors.
fn ba(n: u64, m: u64, seed: u64, rel_idx: u64, base: u64) -> Vec<(u64, u64)> {
    let mut rng = stream(seed, StreamPurpose::Topology, rel_idx, 0);
    let mut repeated: Vec<u64> = Vec::with_capacity((2 * m * n) as usize);
    let mut edges: Vec<(u64, u64)> = Vec::with_capacity((2 * m * n) as usize);
    for v in 0..n {
        for _ in 0..m {
            repeated.push(v);
            // Random endpoint of a random existing half-edge = degree-prop.
            let u = if repeated.len() <= 1 {
                v // self at bootstrap; harmless single self-loop at node 0
            } else {
                repeated[rng.gen_range(0..repeated.len() - 1)]
            };
            repeated.push(u);
            if u != v {
                edges.push((v, base + u));
                edges.push((u, base + v));
            }
        }
    }
    edges
}

/// Directed uniform: each src draws ~Poisson(mean_degree) uniform targets.
fn uniform_random(
    n_src: u64,
    n_dst: u64,
    dst_start: u64,
    mean_degree: f64,
    seed: u64,
    rel_idx: u64,
) -> Vec<(u64, u64)> {
    let mut edges = Vec::with_capacity((n_src as f64 * mean_degree) as usize);
    for s in 0..n_src {
        let mut rng = stream(seed, StreamPurpose::Topology, rel_idx, 1 + s);
        let d = poisson_at_least(&mut rng, mean_degree, 1);
        for _ in 0..d {
            edges.push((s, dst_start + rng.gen_range(0..n_dst)));
        }
    }
    edges
}

/// Watts–Strogatz: ring lattice (k/2 each side) with rewiring prob beta.
fn small_world(n: u64, k: u32, beta: f64, base: u64, seed: u64, rel_idx: u64) -> Vec<(u64, u64)> {
    let half = (k.max(2) / 2) as u64;
    let mut edges = Vec::with_capacity((n * half * 2) as usize);
    for v in 0..n {
        let mut rng = stream(seed, StreamPurpose::Topology, rel_idx, 1 + v);
        for j in 1..=half {
            let mut t = (v + j) % n;
            if rng.gen::<f64>() < beta {
                t = rng.gen_range(0..n);
                if t == v {
                    t = (v + 1) % n;
                }
            }
            edges.push((v, base + t));
            edges.push((t, base + v));
        }
    }
    edges
}

/// Forest fire (Leskovec et al.): each new node picks an ambassador and
/// "burns" forward/backward through its neighborhood. Burn size is capped at
/// 8×mean_degree to keep O(1) amortized per edge (risk R1).
fn forest_fire(
    n: u64,
    forward_p: f64,
    backward_p: f64,
    base: u64,
    mean_degree: f64,
    seed: u64,
    rel_idx: u64,
) -> Vec<(u64, u64)> {
    let mut rng = stream(seed, StreamPurpose::Topology, rel_idx, 0);
    let cap = (8.0 * mean_degree).max(8.0) as usize;
    // Adjacency grown incrementally (out-neighbors only, undirected emitted).
    let mut adj: Vec<Vec<u64>> = vec![Vec::new(); n as usize];
    let mut edges: Vec<(u64, u64)> = Vec::new();
    let geo = |rng: &mut Rng64, p: f64| -> usize {
        // Geometric number of links to follow: mean p/(1-p), capped later.
        if p <= 0.0 {
            0
        } else {
            let u: f64 = rng.gen::<f64>();
            (u.ln() / (1.0 - p).max(1e-12).ln()).floor() as usize
        }
    };
    for v in 1..n {
        let ambassador = rng.gen_range(0..v);
        let mut burned: Vec<u64> = vec![ambassador];
        let mut frontier = vec![ambassador];
        let mut seen = std::collections::HashSet::from([ambassador, v]);
        while let Some(w) = frontier.pop() {
            if burned.len() >= cap {
                break;
            }
            let nf = geo(&mut rng, forward_p);
            let nb = geo(&mut rng, backward_p);
            let picks = nf + nb;
            let neigh = &adj[w as usize];
            for _ in 0..picks.min(neigh.len()) {
                let u = neigh[rng.gen_range(0..neigh.len())];
                if seen.insert(u) {
                    burned.push(u);
                    frontier.push(u);
                }
            }
        }
        for &u in &burned {
            adj[v as usize].push(u);
            adj[u as usize].push(v);
            edges.push((v, base + u));
            edges.push((u, base + v));
        }
    }
    edges
}

/// Stochastic block model with contiguous equal-size blocks.
#[allow(clippy::too_many_arguments)]
fn sbm(
    n_src: u64,
    n_dst: u64,
    dst_start: u64,
    communities: u32,
    p_in: f64,
    p_out: f64,
    mean_degree: f64,
    seed: u64,
    rel_idx: u64,
) -> Vec<(u64, u64)> {
    let c = communities.max(1) as u64;
    let p_in_norm = p_in / (p_in + p_out).max(1e-12);
    let mut edges = Vec::with_capacity((n_src as f64 * mean_degree) as usize);
    let block_dst = (n_dst / c).max(1);
    for s in 0..n_src {
        let mut rng = stream(seed, StreamPurpose::Topology, rel_idx, 1 + s);
        let my_block = s * c / n_src.max(1);
        let d = poisson_at_least(&mut rng, mean_degree, 1);
        for _ in 0..d {
            let t = if rng.gen::<f64>() < p_in_norm {
                // Within own block (mapped onto dst side).
                let lo = my_block * block_dst;
                lo + rng.gen_range(0..block_dst.min(n_dst - lo.min(n_dst - 1)).max(1))
            } else {
                rng.gen_range(0..n_dst)
            };
            edges.push((s, dst_start + t.min(n_dst - 1)));
        }
    }
    edges
}

/// R-MAT: recursive quadrant descent per edge, with noise for degree spread.
fn rmat(
    n_src: u64,
    n_dst: u64,
    dst_start: u64,
    quad: [f64; 4],
    mean_degree: f64,
    seed: u64,
    rel_idx: u64,
) -> Vec<(u64, u64)> {
    let n_edges = (n_src as f64 * mean_degree) as u64;
    let bits_src = 64 - (n_src.max(2) - 1).leading_zeros();
    let bits_dst = 64 - (n_dst.max(2) - 1).leading_zeros();
    let bits = bits_src.max(bits_dst);
    let [a, b, c, _d] = quad;
    let mut edges = Vec::with_capacity(n_edges as usize);
    // Chunked streams so the loop could parallelize without changing output.
    const CHUNK: u64 = 1 << 16;
    let mut e = 0u64;
    while e < n_edges {
        let mut rng = stream(seed, StreamPurpose::Topology, rel_idx, 1 + e / CHUNK);
        let end = (e + CHUNK).min(n_edges);
        for _ in e..end {
            let (mut src, mut dst) = (0u64, 0u64);
            for _ in 0..bits {
                let r: f64 = rng.gen();
                let (sbit, dbit) = if r < a {
                    (0, 0)
                } else if r < a + b {
                    (0, 1)
                } else if r < a + b + c {
                    (1, 0)
                } else {
                    (1, 1)
                };
                src = (src << 1) | sbit;
                dst = (dst << 1) | dbit;
            }
            edges.push((src % n_src, dst_start + (dst % n_dst)));
        }
        e = end;
    }
    edges
}

/// Geometric proximity on the unit square with grid bucketing.
#[allow(clippy::too_many_arguments)]
fn spatial(
    n_src: u64,
    n_dst: u64,
    dst_start: u64,
    radius: f64,
    mean_degree: f64,
    seed: u64,
    rel_idx: u64,
    same_type: bool,
) -> Vec<(u64, u64)> {
    let r = radius.clamp(1e-4, 0.5);
    let cells = ((1.0 / r) as usize).max(1);
    // Deterministic positions for both sides.
    let pos = |id: u64, salt: u64| -> (f64, f64) {
        let mut rng = stream(seed, StreamPurpose::Topology, rel_idx ^ (salt << 32), id);
        (rng.gen::<f64>(), rng.gen::<f64>())
    };
    // Bucket dst side.
    let mut grid: Vec<Vec<u64>> = vec![Vec::new(); cells * cells];
    for t in 0..n_dst {
        let (x, y) = pos(t, 1);
        let cx = ((x * cells as f64) as usize).min(cells - 1);
        let cy = ((y * cells as f64) as usize).min(cells - 1);
        grid[cy * cells + cx].push(t);
    }
    let mut edges = Vec::with_capacity((n_src as f64 * mean_degree) as usize);
    for s in 0..n_src {
        let mut rng = stream(seed, StreamPurpose::Topology, rel_idx ^ (2 << 32), 1 + s);
        let (x, y) = pos(s, if same_type { 1 } else { 0 });
        let cx = ((x * cells as f64) as usize).min(cells - 1) as i64;
        let cy = ((y * cells as f64) as usize).min(cells - 1) as i64;
        // Candidate pool: own + 8 neighboring cells.
        let mut candidates: Vec<u64> = Vec::new();
        for dy in -1..=1i64 {
            for dx in -1..=1i64 {
                let (gx, gy) = (cx + dx, cy + dy);
                if gx >= 0 && gy >= 0 && (gx as usize) < cells && (gy as usize) < cells {
                    candidates.extend_from_slice(&grid[gy as usize * cells + gx as usize]);
                }
            }
        }
        if candidates.is_empty() {
            continue;
        }
        let d = poisson_at_least(&mut rng, mean_degree, 1) as usize;
        for _ in 0..d {
            let t = candidates[rng.gen_range(0..candidates.len())];
            if same_type && t == s {
                continue;
            }
            edges.push((s, dst_start + t));
        }
    }
    edges
}

/// Bipartite affiliation: dst popularity ~ the given distribution
/// (rank-ordered: dst id 0 is the most popular).
fn affiliation(
    n_src: u64,
    n_dst: u64,
    dst_start: u64,
    popularity: &Distribution,
    mean_degree: f64,
    seed: u64,
    rel_idx: u64,
) -> anyhow::Result<Vec<(u64, u64)>> {
    // Zipf's `n` in the rule is a hint; the actual dst count governs.
    let sampler = match popularity {
        Distribution::Zipf { exponent, .. } => {
            DistSampler::compile(&Distribution::Zipf { n: n_dst, exponent: *exponent })?
        }
        other => DistSampler::compile(other)?,
    };
    let mut edges = Vec::with_capacity((n_src as f64 * mean_degree) as usize);
    for s in 0..n_src {
        let mut rng = stream(seed, StreamPurpose::Topology, rel_idx, 1 + s);
        let d = poisson_at_least(&mut rng, mean_degree, 1);
        for _ in 0..d {
            let rank = sampler.sample(&mut rng).max(1.0) as u64;
            edges.push((s, dst_start + (rank - 1).min(n_dst - 1)));
        }
    }
    Ok(edges)
}

/// Poisson draw with a floor (no isolated nodes in the skeleton).
fn poisson_at_least(rng: &mut Rng64, mean: f64, floor: u64) -> u64 {
    use rand_distr::Distribution as _;
    let p = rand_distr::Poisson::new(mean.max(0.01)).expect("validated mean");
    (p.sample(rng) as u64).max(floor)
}

#[cfg(test)]
mod tests {
    use super::*;
    use agora_rules::Layer;

    fn rule(model: TopologyModel, mean_degree: f64) -> RelationRule {
        RelationRule {
            name: "t".into(),
            src: "a".into(),
            dst: "a".into(),
            model,
            layer: Layer::Skeleton,
            mean_degree,
        }
    }

    #[test]
    fn ba_power_law_hubs_and_determinism() {
        let r = rule(TopologyModel::PreferentialAttachment { m: 3 }, 6.0);
        let g1 = build_relation(&r, 0, 10_000, 0, 10_000, 7, 0).unwrap();
        let g2 = build_relation(&r, 0, 10_000, 0, 10_000, 7, 0).unwrap();
        assert_eq!(g1.csr.targets, g2.csr.targets, "must be bit-reproducible");
        let degrees: Vec<usize> = (0..10_000).map(|v| g1.csr.neighbors(v).len()).collect();
        let max_d = *degrees.iter().max().unwrap();
        let mean_d = degrees.iter().sum::<usize>() as f64 / 10_000.0;
        assert!(mean_d > 4.0 && mean_d < 8.0, "mean degree ≈ 2m, got {mean_d}");
        assert!(max_d > 60, "hubs must emerge under PA, max degree {max_d}");
    }

    #[test]
    fn affiliation_popularity_skew() {
        let r = rule(
            TopologyModel::Affiliation {
                popularity: Distribution::Zipf { n: 1000, exponent: 1.2 },
            },
            5.0,
        );
        let g = build_relation(&r, 0, 50_000, 100_000, 1000, 7, 1).unwrap();
        // Top dst (rank 0) must vastly out-receive the median dst.
        let mut indeg = vec![0u64; 1000];
        for &t in &g.csr.targets {
            indeg[(t - 100_000) as usize] += 1;
        }
        assert!(indeg[0] > indeg[500].max(1) * 20, "zipf head {} vs median {}", indeg[0], indeg[500]);
    }

    #[test]
    fn every_node_has_skeleton_neighbors() {
        for model in [
            TopologyModel::UniformRandom,
            TopologyModel::SmallWorld { k: 4, beta: 0.1 },
            TopologyModel::Sbm { communities: 4, p_in_weight: 0.8, p_out_weight: 0.2 },
        ] {
            let g = build_relation(&rule(model, 4.0), 0, 2000, 0, 2000, 7, 2).unwrap();
            let isolated = (0..2000).filter(|&v| g.csr.neighbors(v).is_empty()).count();
            assert_eq!(isolated, 0, "no isolated src nodes allowed");
        }
    }

    #[test]
    fn rmat_and_forest_fire_build() {
        let g = build_relation(
            &rule(TopologyModel::RMat { a: 0.57, b: 0.19, c: 0.19, d: 0.05 }, 8.0),
            0,
            4096,
            0,
            4096,
            7,
            3,
        )
        .unwrap();
        assert!(g.csr.n_edges() > 30_000);
        let g = build_relation(
            &rule(TopologyModel::ForestFire { forward_p: 0.35, backward_p: 0.2 }, 4.0),
            0,
            3000,
            0,
            3000,
            7,
            4,
        )
        .unwrap();
        assert!(g.csr.n_edges() > 5_000);
    }
}
