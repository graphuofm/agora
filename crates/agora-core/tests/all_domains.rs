//! Cross-domain generality test (risk R4): the ONE engine must generate every
//! built-in domain, with anomalies, deterministically — no per-domain code.

use agora_core::{generate, CountingSink, EventBatch, EventSink, GenParams};
use agora_rules::load_builtin_rulebase;

fn params(domain: &str, seed: u64, threads: usize) -> GenParams {
    GenParams {
        rulebase: load_builtin_rulebase(domain).unwrap(),
        nodes: 20_000,
        target_edges: 300_000,
        span_days: 30.0,
        granularity_s: 1,
        epoch_unix: 1_735_689_600,
        seed,
        threads,
        anomaly_rate: Some(0.03),
        anomaly_difficulty: Some(0.5),
        anomaly_type_mix: Vec::new(),
        anomaly_cascade: None,
        anomaly_communities: None,
        anomalies_disabled: false,
        shard_index: 0,
        shard_count: 1,
    }
}

#[test]
fn every_domain_generates_with_anomalies() {
    for domain in ["finance", "crypto", "cyber", "transport", "ecommerce", "healthcare"] {
        let mut sink = CountingSink::default();
        let summary = generate(&params(domain, 42, 8), &mut sink, None)
            .unwrap_or_else(|e| panic!("domain `{domain}` failed to generate: {e}"));
        assert!(summary.edges_written > 0, "{domain}: no edges");
        assert_eq!(summary.edges_written, sink.edges);
        assert!(summary.anomalous_edges > 0, "{domain}: no anomalies emerged");
        // Calibration holds across all domains.
        let rel = (summary.edges_written as f64 - 300_000.0).abs() / 300_000.0;
        assert!(rel < 0.15, "{domain}: budget off by {:.1}%", rel * 100.0);
        // Anomalies stay a minority of edges.
        let share = summary.anomalous_edges as f64 / summary.edges_written as f64;
        assert!(share < 0.25, "{domain}: anomalous share {share:.3} too high");
    }
}

#[test]
fn sharded_union_equals_whole_graph() {
    // The union of N shards must equal the whole-graph run exactly (disjoint,
    // complete). Compare the MULTISET of (src,dst,t,event_type,label,
    // anomaly_id) tuples — sharding partitions by src so each edge appears in
    // exactly one shard; identical tuples may occur naturally, so compare
    // sorted vectors, not sets.
    #[derive(Default)]
    struct Collect {
        edges: Vec<(u64, u64, i64, u16, u16, i64)>,
    }
    impl EventSink for Collect {
        fn write_batch(&mut self, b: &EventBatch) -> anyhow::Result<()> {
            for i in 0..b.len() {
                self.edges.push((
                    b.src[i], b.dst[i], b.t[i], b.event_type[i], b.label[i], b.anomaly_id[i],
                ));
            }
            Ok(())
        }
    }
    for domain in ["finance", "transport"] {
        let mut whole = Collect::default();
        generate(&params(domain, 5, 4), &mut whole, None).unwrap();

        let mut union: Vec<(u64, u64, i64, u16, u16, i64)> = Vec::new();
        let shard_count = 4u64;
        for idx in 0..shard_count {
            let mut p = params(domain, 5, 4);
            p.shard_index = idx;
            p.shard_count = shard_count;
            let mut shard = Collect::default();
            generate(&p, &mut shard, None).unwrap();
            // every edge in this shard must be sourced at a node it owns
            assert!(
                shard.edges.iter().all(|e| e.0 % shard_count == idx),
                "{domain}: shard {idx} emitted an edge it does not own"
            );
            union.extend(shard.edges);
        }
        assert_eq!(
            union.len(),
            whole.edges.len(),
            "{domain}: shard union edge count != whole-graph ({} vs {})",
            union.len(),
            whole.edges.len()
        );
        let mut union_sorted = union;
        let mut whole_sorted = whole.edges;
        union_sorted.sort_unstable();
        whole_sorted.sort_unstable();
        assert_eq!(
            union_sorted, whole_sorted,
            "{domain}: shard union multiset != whole-graph output"
        );
    }
}

#[test]
fn popularity_counterparty_is_skewed_not_uniform() {
    // transport's taxi_service selects zones by GlobalPopularity; the chosen
    // zone in-counts must be heavy-tailed (hubs), not uniform (§8).
    use std::collections::HashMap;
    #[derive(Default)]
    struct ZoneHits {
        trip_et: Option<u16>,
        names: Vec<String>,
        counts: HashMap<u64, u64>,
    }
    impl EventSink for ZoneHits {
        fn write_batch(&mut self, b: &EventBatch) -> anyhow::Result<()> {
            for i in 0..b.len() {
                if Some(b.event_type[i]) == self.trip_et {
                    *self.counts.entry(b.dst[i]).or_insert(0) += 1;
                }
            }
            Ok(())
        }
    }
    let mut p = params("transport", 5, 4);
    p.anomalies_disabled = true;
    p.target_edges = 1_000_000;
    // Resolve the "trip" event type id from a probe run is awkward; instead
    // capture names from the summary by a first pass.
    let mut probe = ZoneHits::default();
    let summary = generate(&p, &mut probe, None).unwrap();
    probe.names = summary.event_type_names.clone();
    probe.trip_et = probe.names.iter().position(|n| n == "trip").map(|i| i as u16);
    let mut hits = ZoneHits { trip_et: probe.trip_et, ..Default::default() };
    generate(&p, &mut hits, None).unwrap();

    let mut counts: Vec<u64> = hits.counts.values().copied().collect();
    assert!(counts.len() > 50, "need many zones hit to measure skew");
    counts.sort_unstable_by(|a, b| b.cmp(a));
    let top = counts[0];
    let median = counts[counts.len() / 2].max(1);
    assert!(
        top > median * 20,
        "popularity selection must be heavy-tailed: top {top} vs median {median}"
    );
}

#[test]
fn every_domain_is_thread_deterministic() {
    #[derive(Default)]
    struct Hash {
        h: u64,
    }
    impl EventSink for Hash {
        fn write_batch(&mut self, b: &EventBatch) -> anyhow::Result<()> {
            for i in 0..b.len() {
                // order-sensitive rolling hash of the canonical columns
                self.h = self
                    .h
                    .wrapping_mul(1_000_003)
                    .wrapping_add(b.src[i])
                    .rotate_left(7)
                    ^ (b.dst[i].wrapping_mul(2_654_435_761))
                    ^ (b.t[i] as u64)
                    ^ ((b.label[i] as u64) << 48);
            }
            Ok(())
        }
    }
    for domain in ["finance", "crypto", "cyber", "transport", "ecommerce", "healthcare"] {
        let mut a = Hash::default();
        let mut b = Hash::default();
        generate(&params(domain, 7, 1), &mut a, None).unwrap();
        generate(&params(domain, 7, 8), &mut b, None).unwrap();
        assert_eq!(a.h, b.h, "{domain}: output differs between 1 and 8 threads");
    }
}
