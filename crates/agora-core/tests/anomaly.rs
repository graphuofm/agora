//! M5 tests: the five control axes, intent labeling, failure processes.

use agora_core::{generate, EventBatch, EventSink, GenParams};
use agora_rules::{load_builtin_rulebase, Distribution, FailureMode, FailureProcess, RuleBase};

#[derive(Default)]
struct Capture {
    src: Vec<u64>,
    dst: Vec<u64>,
    t: Vec<i64>,
    label: Vec<String>,
    event_type: Vec<u16>,
    amounts: Vec<f64>,
    amount_col: Option<usize>,
    intent_names: Vec<String>,
}

impl Capture {
    fn resolve(&mut self, names: &[String]) {
        self.intent_names = names.to_vec();
    }
}

impl EventSink for Capture {
    fn write_batch(&mut self, b: &EventBatch) -> anyhow::Result<()> {
        if self.amount_col.is_none() {
            self.amount_col = b.attr_names.iter().position(|a| a == "amount");
        }
        self.src.extend_from_slice(&b.src);
        self.dst.extend_from_slice(&b.dst);
        self.t.extend_from_slice(&b.t);
        self.event_type.extend_from_slice(&b.event_type);
        // labels resolved post-run via summary intent table; store raw here
        for &l in &b.label {
            self.label.push(l.to_string());
        }
        if let Some(c) = self.amount_col {
            self.amounts.extend_from_slice(&b.attrs[c]);
        }
        Ok(())
    }
}

fn params(rb: RuleBase, seed: u64, rate: Option<f64>, difficulty: Option<f64>) -> GenParams {
    GenParams {
        rulebase: rb,
        nodes: 20_000,
        target_edges: 400_000,
        span_days: 30.0,
        granularity_s: 1,
        epoch_unix: 1_735_689_600,
        seed,
        threads: 8,
        anomaly_rate: rate,
        anomaly_difficulty: difficulty,
        anomaly_type_mix: Vec::new(),
        anomaly_cascade: None,
        anomaly_communities: None,
        anomalies_disabled: false,
        shard_index: 0,
        shard_count: 1,
    }
}

fn finance() -> RuleBase {
    load_builtin_rulebase("finance").unwrap()
}

#[test]
fn labels_present_and_total_calibrated() {
    let mut s = Capture::default();
    let summary = generate(&params(finance(), 5, None, None), &mut s, None).unwrap();
    s.resolve(&summary.intent_names);
    assert!(summary.anomalous_edges > 0, "anomalies must emerge");
    // Every declared adversary intent appears.
    for (i, name) in summary.intent_names.iter().enumerate().skip(1) {
        assert!(
            summary.edges_per_intent[i] > 0,
            "intent `{name}` produced no events"
        );
    }
    // Total stays calibrated to the edge budget despite anomalies.
    let rel_err =
        (summary.edges_written as f64 - 400_000.0).abs() / 400_000.0;
    assert!(rel_err < 0.10, "budget off by {:.1}%", rel_err * 100.0);
    // Anomalies are rare (prevalence is a node fraction; edge share stays low).
    let share = summary.anomalous_edges as f64 / summary.edges_written as f64;
    assert!(share < 0.15, "anomalous edge share {share:.3} implausibly high");
}

#[test]
fn prevalence_axis_scales_anomaly_volume() {
    let mut a = Capture::default();
    let mut b = Capture::default();
    let sa = generate(&params(finance(), 7, Some(0.01), None), &mut a, None).unwrap();
    let sb = generate(&params(finance(), 7, Some(0.04), None), &mut b, None).unwrap();
    let ratio = sb.anomalous_edges as f64 / sa.anomalous_edges.max(1) as f64;
    assert!(
        (2.0..8.0).contains(&ratio),
        "4x prevalence should give ~4x anomalous edges, got {ratio:.2}x ({} vs {})",
        sa.anomalous_edges,
        sb.anomalous_edges
    );
}

#[test]
fn difficulty_axis_camouflages_structuring_amounts() {
    // At difficulty 0 every structuring deposit uses the template
    // ($8200-9900); at difficulty 1 most draw from the normal distribution.
    let near_threshold_share = |difficulty: f64| -> f64 {
        let mut s = Capture::default();
        let summary =
            generate(&params(finance(), 11, Some(0.03), Some(difficulty)), &mut s, None).unwrap();
        let structuring_idx = summary
            .intent_names
            .iter()
            .position(|n| n == "structuring")
            .unwrap()
            .to_string();
        let deposit_et = summary
            .event_type_names
            .iter()
            .position(|n| n == "cash_deposit")
            .unwrap() as u16;
        let mut total = 0u64;
        let mut near = 0u64;
        for i in 0..s.label.len() {
            if s.label[i] == structuring_idx && s.event_type[i] == deposit_et {
                total += 1;
                let a = s.amounts[i];
                if (8200.0..=9900.0).contains(&a) {
                    near += 1;
                }
            }
        }
        assert!(total > 50, "need structuring deposits to measure, got {total}");
        near as f64 / total as f64
    };
    let blatant = near_threshold_share(0.0);
    let camouflaged = near_threshold_share(1.0);
    assert!(blatant > 0.95, "difficulty 0: template share {blatant:.2} should be ~1");
    assert!(
        camouflaged < blatant - 0.3,
        "difficulty 1 must blend toward normal amounts: {camouflaged:.2} vs {blatant:.2}"
    );
}

#[test]
fn anomalies_deterministic_across_threads() {
    let run = |threads: usize| -> (Vec<u64>, Vec<String>) {
        let mut s = Capture::default();
        let mut p = params(finance(), 13, Some(0.02), None);
        p.threads = threads;
        generate(&p, &mut s, None).unwrap();
        (s.src, s.label)
    };
    let (s1, l1) = run(1);
    let (s8, l8) = run(8);
    assert_eq!(s1, s8);
    assert_eq!(l1, l8);
}

#[test]
fn failure_process_silence_and_noise() {
    // Add a sensor-fault-style failure to finance (NoiseAttr on transfers)
    // plus a Silence failure; both must label/suppress correctly.
    let mut rb = finance();
    rb.failures.push(FailureProcess {
        intent: "sensor_noise".into(),
        description: "test noise".into(),
        actor: "account".into(),
        mode: FailureMode::NoiseAttr {
            event: "transfer".into(),
            attr: "amount".into(),
            dist: Distribution::Constant { value: -999.0 },
        },
        prevalence_weight: 5.0,
        rate_per_year: 40.0,
        duration_days: Distribution::Uniform { min: 2.0, max: 5.0 },
        cascade_p: 0.0,
    });
    let mut s = Capture::default();
    let summary = generate(&params(rb, 17, Some(0.05), None), &mut s, None).unwrap();
    let noise_idx = summary.intent_names.iter().position(|n| n == "sensor_noise").unwrap();
    assert!(
        summary.edges_per_intent[noise_idx] > 0,
        "noise failure produced no labeled events"
    );
    // Every noise-labeled transfer has the corrupted value.
    let idx_str = noise_idx.to_string();
    let bad: Vec<f64> = (0..s.label.len())
        .filter(|&i| s.label[i] == idx_str)
        .map(|i| s.amounts[i])
        .collect();
    assert!(!bad.is_empty());
    assert!(bad.iter().all(|&v| v == -999.0), "noise mode must corrupt the attr");
}

#[test]
fn type_mix_axis_zeroes_and_boosts_intents() {
    // Axis 3: a type-mix that zeroes three intents and weights two must yield
    // zero edges for the zeroed intents and a boosted share for the kept ones.
    let mut p = params(finance(), 23, Some(0.03), None);
    p.anomaly_type_mix = vec![
        ("structuring".into(), 0.9),
        ("layering".into(), 0.1),
        ("fan_in_out".into(), 0.0),
        ("round_tripping".into(), 0.0),
        ("mule_network".into(), 0.0),
    ];
    let mut s = Capture::default();
    let summary = generate(&p, &mut s, None).unwrap();
    let by = |name: &str| -> u64 {
        let i = summary.intent_names.iter().position(|n| n == name).unwrap();
        summary.edges_per_intent[i]
    };
    assert_eq!(by("fan_in_out"), 0, "zeroed intent must produce no edges");
    assert_eq!(by("round_tripping"), 0);
    assert_eq!(by("mule_network"), 0);
    assert!(by("structuring") > 0 && by("layering") > 0);
    assert!(
        by("structuring") > by("layering") * 3,
        "0.9/0.1 weighting should make structuring dominate: {} vs {}",
        by("structuring"),
        by("layering")
    );
}

#[test]
fn cascade_axis_keeps_instances() {
    // Axis 5: raising the cascade multiplier spawns follow-ups, so it must not
    // reduce the number of anomaly instances vs cascade=0.
    let count = |cascade: f64| -> usize {
        let mut p = params(finance(), 29, Some(0.04), Some(0.0));
        p.anomaly_cascade = Some(cascade);
        let mut s = Capture::default();
        generate(&p, &mut s, None).unwrap().ground_truth.len()
    };
    let none = count(0.0);
    let lots = count(1.0);
    assert!(none > 0);
    assert!(lots >= none, "more cascade should not lose instances: {lots} vs {none}");
}

#[test]
fn structuring_trips_phi_constraint() {
    // The Φ rule (≥4 near-threshold deposits in 7d) must fire on smurfing
    // campaigns at difficulty 0 (no camouflage).
    let mut s = Capture::default();
    let summary = generate(&params(finance(), 19, Some(0.03), Some(0.0)), &mut s, None).unwrap();
    let deposit_et = summary
        .event_type_names
        .iter()
        .position(|n| n == "cash_deposit")
        .unwrap() as u16;
    // Count per-src near-threshold deposits within any 7-day window.
    use std::collections::HashMap;
    let mut per_src: HashMap<u64, Vec<i64>> = HashMap::new();
    for i in 0..s.label.len() {
        if s.event_type[i] == deposit_et && (7000.0..10000.0).contains(&s.amounts[i]) {
            per_src.entry(s.src[i]).or_default().push(s.t[i]);
        }
    }
    let mut tripped = 0;
    for times in per_src.values() {
        let mut ts = times.clone();
        ts.sort_unstable();
        for w in ts.windows(4) {
            if w[3] - w[0] <= 604_800 {
                tripped += 1;
                break;
            }
        }
    }
    assert!(tripped > 0, "no structuring runs tripped the CTR Φ rule");
}

// --- event_count: budget-inelastic typology volume -------------------------

/// Strip `finance` down to ONE adversary with ONE `event_count` stage, so the
/// anomalous edge count is a direct readout of the count spec.
fn count_spec_rb(count: f64, difficulty: f64) -> RuleBase {
    let mut rb = load_builtin_rulebase("finance").unwrap();
    rb.failures.clear();
    rb.adversaries.truncate(1);
    let a = &mut rb.adversaries[0];
    a.prevalence_weight = 1.0;
    a.camouflage = 0.0; // isolate VOLUME from relation/feature camouflage
    a.cascade_p = 0.0;
    a.ring_size = Distribution::Constant { value: 4.0 };
    a.stages.truncate(1);
    let s = &mut a.stages[0];
    s.duration_days = Distribution::Constant { value: 10.0 };
    s.activity_multiplier = None;
    s.rate_per_day = None;
    s.event_count = Some(Distribution::Constant { value: count });
    s.attr_overrides.clear();
    rb.control.difficulty = difficulty;
    rb.control.prevalence = 0.02;
    rb
}

fn anomalous_edges(rb: RuleBase, edges: u64) -> usize {
    let mut p = params(rb, 7, None, None);
    p.target_edges = edges;
    p.nodes = 20_000;
    let mut cap = Capture::default();
    let s = generate(&p, &mut cap, None).unwrap();
    cap.resolve(&s.intent_names);
    cap.label.iter().filter(|l| *l != "0").count()
}

/// The point of a COUNT spec: the campaign emits the typology's N events
/// whatever the edge budget. A 5x budget change must not move it.
#[test]
fn event_count_volume_is_budget_inelastic() {
    let small = anomalous_edges(count_spec_rb(6.0, 0.0), 200_000);
    let large = anomalous_edges(count_spec_rb(6.0, 0.0), 1_000_000);
    assert!(small > 0, "count spec emitted nothing at the small budget");
    let ratio = large as f64 / small as f64;
    assert!(
        (0.8..1.25).contains(&ratio),
        "event_count should be budget-inelastic, but 5x the budget moved volume \
         {small} -> {large} (ratio {ratio:.2})"
    );
}

/// Doubling the declared count doubles the emitted volume (it IS a count).
#[test]
fn event_count_scales_linearly() {
    let n1 = anomalous_edges(count_spec_rb(4.0, 0.0), 400_000);
    let n2 = anomalous_edges(count_spec_rb(8.0, 0.0), 400_000);
    let ratio = n2 as f64 / n1 as f64;
    assert!(
        (1.7..2.3).contains(&ratio),
        "doubling event_count should double volume: {n1} -> {n2} (ratio {ratio:.2})"
    );
}

/// The difficulty axis closes the volume leak ONE-SIDEDLY (P2-b fix): it may
/// only ever move an adversary's volume DOWN toward the normal rate, never up.
/// A LOUD stage (k > 1, i.e. above the ~6.9/day normal account rate over the
/// 10-day stage => count > ~69) is damped toward normal at d = 1; a QUIET stage
/// (k < 1) is left at its playbook rate, because AMPLIFYING a quiet adversary
/// would EXPOSE it — the exact crypto inversion this fix removes. The old
/// symmetric transform k^(1−d) drove BOTH sides to k = 1, i.e. it AMPLIFIED a
/// quiet count up to the normal rate; that is the behavior we deliberately drop.
#[test]
fn difficulty_one_is_one_sided_damping() {
    // LOUD side: count 300 over 10 days ~= 30/day vs normal ~6.9/day (k ~ 4.3),
    // so difficulty must damp its volume DOWN toward normal at d = 1.
    let loud0 = anomalous_edges(count_spec_rb(300.0, 0.0), 400_000);
    let loud1 = anomalous_edges(count_spec_rb(300.0, 1.0), 400_000);
    assert!(
        loud0 as f64 / loud1.max(1) as f64 > 2.0,
        "a loud (k>1) count must be damped toward normal at d=1: {loud0} -> {loud1}"
    );

    // QUIET side: count 2 over 10 days ~= 0.2/day (k ~ 0.03). Difficulty must
    // NEVER amplify it — its d = 1 volume may not exceed its d = 0 volume (a
    // small tolerance for single-run sampling noise on a handful of edges).
    let quiet0 = anomalous_edges(count_spec_rb(2.0, 0.0), 400_000);
    let quiet1 = anomalous_edges(count_spec_rb(2.0, 1.0), 400_000);
    assert!(
        quiet1 as f64 <= quiet0 as f64 * 1.35 + 5.0,
        "difficulty must never amplify a quiet (k<1) adversary: {quiet0} -> {quiet1}"
    );

    // Sanity: at d = 0 the count genuinely drives volume (30x count => >10x
    // volume), else the loud-damping check above could pass trivially.
    let base_loud = anomalous_edges(count_spec_rb(300.0, 0.0), 400_000);
    let base_quiet = anomalous_edges(count_spec_rb(10.0, 0.0), 400_000);
    assert!(
        base_loud as f64 / base_quiet.max(1) as f64 > 10.0,
        "at difficulty 0 the count must drive volume: {base_loud} vs {base_quiet}"
    );
}

/// The anomaly cap is computed from `AnomalyPlan.expected_events`, so a plan
/// that over-counts silently throttles the anomaly stream. The estimator must
/// therefore count only the stage time the simulator can actually reach.
///
/// Regression: healthcare planned 3.2x the volume it delivered, because
/// `place_window` schedules a campaign longer than the span past the span end
/// (it clamps the start, not the end) and the estimator counted those
/// unreachable tails. `kickback_ring` (3 sequential stages of 30-120 days =
/// up to 360 days) overruns a 30-day span badly.
fn planned_vs_delivered(domain: &str, span_days: f64) -> (f64, u64) {
    let rb = load_builtin_rulebase(domain).unwrap();
    let mut p = params(rb, 7, None, None);
    p.span_days = span_days;
    let world = agora_core::world::World::build(&p).unwrap();
    let plan = agora_core::anomaly::plan(&p, &world).unwrap();
    let mut cap = Capture::default();
    let s = generate(&p, &mut cap, None).unwrap();
    (plan.expected_events, s.anomalous_edges)
}

#[test]
fn campaign_estimate_matches_delivered_volume() {
    // Healthcare on a short span is the worst case: campaigns are far longer
    // than the run, so most planned stage time is unreachable.
    for span in [15.0, 30.0, 60.0] {
        let (planned, delivered) = planned_vs_delivered("healthcare", span);
        let ratio = planned / delivered.max(1) as f64;
        assert!(
            (0.75..1.25).contains(&ratio),
            "healthcare @ {span}d: planned {planned:.0} vs delivered {delivered} \
             (ratio {ratio:.2}) — the estimator must not count stage time past \
             the span end"
        );
    }
}

/// Guard the other direction: on a span longer than every campaign nothing is
/// truncated, so the clamp must be a no-op rather than a blanket discount.
#[test]
fn campaign_estimate_is_unchanged_when_nothing_overruns_the_span() {
    let (planned, delivered) = planned_vs_delivered("healthcare", 365.0);
    let ratio = planned / delivered.max(1) as f64;
    assert!(
        (0.85..1.15).contains(&ratio),
        "healthcare @ 365d: planned {planned:.0} vs delivered {delivered} (ratio {ratio:.2})"
    );
}
