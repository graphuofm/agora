//! Engine integration tests: determinism, calibration, semantics.

use agora_core::{generate, EventBatch, EventSink, GenParams};
use agora_rules::load_builtin_rulebase;

/// Sink that captures everything for inspection.
#[derive(Default)]
struct CaptureSink {
    src: Vec<u64>,
    dst: Vec<u64>,
    t: Vec<i64>,
    event_type: Vec<u16>,
    label: Vec<u16>,
    amounts: Vec<f64>,
    amount_col: Option<usize>,
}

impl EventSink for CaptureSink {
    fn write_batch(&mut self, b: &EventBatch) -> anyhow::Result<()> {
        if self.amount_col.is_none() {
            self.amount_col = b.attr_names.iter().position(|a| a == "amount");
        }
        self.src.extend_from_slice(&b.src);
        self.dst.extend_from_slice(&b.dst);
        self.t.extend_from_slice(&b.t);
        self.event_type.extend_from_slice(&b.event_type);
        self.label.extend_from_slice(&b.label);
        if let Some(c) = self.amount_col {
            self.amounts.extend_from_slice(&b.attrs[c]);
        }
        Ok(())
    }
}

fn params(seed: u64, threads: usize, nodes: u64, edges: u64) -> GenParams {
    GenParams {
        rulebase: load_builtin_rulebase("finance").unwrap(),
        nodes,
        target_edges: edges,
        span_days: 14.0,
        granularity_s: 1,
        epoch_unix: 1_735_689_600,
        seed,
        threads,
        anomaly_rate: None,
        anomaly_difficulty: None,
        anomaly_type_mix: Vec::new(),
        anomaly_cascade: None,
        anomaly_communities: None,
        anomalies_disabled: true,
        shard_index: 0,
        shard_count: 1,
    }
}

#[test]
fn bit_identical_across_thread_counts() {
    let mut a = CaptureSink::default();
    let mut b = CaptureSink::default();
    generate(&params(42, 1, 5_000, 200_000), &mut a, None).unwrap();
    generate(&params(42, 8, 5_000, 200_000), &mut b, None).unwrap();
    assert_eq!(a.src, b.src, "src column must not depend on thread count");
    assert_eq!(a.dst, b.dst);
    assert_eq!(a.t, b.t);
    assert_eq!(a.event_type, b.event_type);
    assert_eq!(a.amounts, b.amounts, "attribute draws must be identical too");
}

#[test]
fn weibull_arrival_is_budget_neutral_and_deterministic() {
    use agora_rules::{ArrivalKind, RuleBase};
    // Switching every behavior to a heavy-tailed Weibull arrival must NOT
    // change the edge budget (the candidate count stays Poisson; only the
    // within-window spacing is reshaped), and must stay thread-deterministic.
    let weibull = |rb: &mut RuleBase, shape: f64| {
        for b in &mut rb.behaviors {
            b.timing.arrival = ArrivalKind::Weibull { shape };
        }
    };
    let mut poisson = CaptureSink::default();
    let p = params(3, 8, 10_000, 300_000);
    let s_poisson = generate(&p, &mut poisson, None).unwrap();

    let mut rb = load_builtin_rulebase("finance").unwrap();
    weibull(&mut rb, 0.5);
    let mut pw = params(3, 8, 10_000, 300_000);
    pw.rulebase = rb;
    let mut a = CaptureSink::default();
    let mut b = CaptureSink::default();
    let s_w = generate(&pw, &mut a, None).unwrap();
    let mut pw1 = pw.clone();
    pw1.threads = 1;
    generate(&pw1, &mut b, None).unwrap();

    // Budget neutral: within the engine's calibration tolerance of Poisson.
    let rel = (s_w.edges_written as f64 - s_poisson.edges_written as f64).abs()
        / s_poisson.edges_written as f64;
    assert!(rel < 0.05, "weibull budget drifted {:.1}% from poisson", rel * 100.0);
    // Deterministic across thread counts.
    assert_eq!(a.t, b.t, "weibull arrival must be thread-deterministic");
}

#[test]
fn hawkes_branching_is_budget_neutral_and_raises_clustering() {
    use agora_rules::RuleBase;
    // Hawkes self-excitation must (a) keep the edge budget (mean cluster size
    // 1/(1−n) is folded into the rate calibration) and (b) stay deterministic.
    let mut rb: RuleBase = load_builtin_rulebase("finance").unwrap();
    for b in &mut rb.behaviors {
        b.timing.burst_p = 0.0;
        b.timing.branching_ratio = 0.6;
        b.timing.excitation_decay_s = 600.0;
    }
    let mut p = params(5, 8, 10_000, 300_000);
    p.rulebase = rb;
    let mut a = CaptureSink::default();
    let mut b = CaptureSink::default();
    let s = generate(&p, &mut a, None).unwrap();
    let mut p1 = p.clone();
    p1.threads = 1;
    generate(&p1, &mut b, None).unwrap();

    let rel = (s.edges_written as f64 - 300_000.0).abs() / 300_000.0;
    assert!(rel < 0.08, "hawkes budget drifted {:.1}%", rel * 100.0);
    assert_eq!(a.t, b.t, "hawkes cascade must be thread-deterministic");
    // The cascade should produce clustered timestamps (many small gaps): at
    // least some events share the same second.
    let mut sorted = a.t.clone();
    sorted.sort_unstable();
    let zero_gaps = sorted.windows(2).filter(|w| w[1] - w[0] == 0).count();
    assert!(zero_gaps > 0, "self-excitation should cluster events in time");
}

#[test]
fn different_seeds_differ() {
    let mut a = CaptureSink::default();
    let mut b = CaptureSink::default();
    generate(&params(1, 4, 5_000, 100_000), &mut a, None).unwrap();
    generate(&params(2, 4, 5_000, 100_000), &mut b, None).unwrap();
    assert_ne!(a.t, b.t);
}

#[test]
fn calibration_hits_target_within_tolerance() {
    let mut s = CaptureSink::default();
    let target = 300_000u64;
    let summary = generate(&params(7, 8, 10_000, target), &mut s, None).unwrap();
    let got = summary.edges_written as f64;
    let rel_err = (got - target as f64).abs() / target as f64;
    assert!(
        rel_err < 0.05,
        "edge budget calibration off by {:.1}% ({} vs {})",
        rel_err * 100.0,
        got,
        target
    );
}

#[test]
fn output_is_time_sorted_and_in_span() {
    let mut s = CaptureSink::default();
    let p = params(9, 4, 3_000, 50_000);
    generate(&p, &mut s, None).unwrap();
    assert!(s.t.windows(2).all(|w| w[0] <= w[1]), "events must be time-sorted");
    let end = p.epoch_unix + (p.span_days * 86_400.0) as i64;
    assert!(s.t.iter().all(|&t| t >= p.epoch_unix && t <= end));
}

#[test]
fn normal_only_run_has_no_labels_and_valid_endpoints() {
    let mut s = CaptureSink::default();
    let p = params(11, 4, 3_000, 50_000);
    let summary = generate(&p, &mut s, None).unwrap();
    assert!(s.label.iter().all(|&l| l == 0));
    assert_eq!(summary.anomalous_edges, 0);
    // Endpoints must be valid node ids and transfers account->account.
    let n = p.nodes;
    assert!(s.src.iter().all(|&x| x < n));
    assert!(s.dst.iter().all(|&x| x < n));
    // Amounts present and positive wherever defined.
    assert!(s.amounts.iter().filter(|a| !a.is_nan()).all(|&a| a > 0.0));
    // Diurnal shape: business hours (9-17 UTC) busier than night (0-5).
    let hour = |t: i64| ((t - p.epoch_unix) % 86_400) / 3_600;
    let day_evts = s.t.iter().filter(|&&t| (9..17).contains(&hour(t))).count();
    let night_evts = s.t.iter().filter(|&&t| (0..5).contains(&hour(t))).count();
    assert!(
        day_evts as f64 > night_evts as f64 * 2.0,
        "diurnal modulation missing: day {day_evts} vs night {night_evts}"
    );
}
