//! The simulation loop (M1: normal behavior; M5 adds anomaly processes).
//!
//! Time is partitioned into 1-day windows. Within a window, generation is
//! parallel over (behavior, actor-chunk) tasks; each actor draws from its own
//! `(seed, actor, window, behavior)` stream, so output is bit-identical for
//! any thread count. Windows are merged, time-sorted, state effects applied
//! (single-threaded — all effects are commutative adds/sets applied in sorted
//! order), and the batch streams to the sink. Memory stays bounded by the
//! densest window, not the run.

use std::collections::HashSet;

use rand::Rng;
use rand_distr::Distribution as _;
use rayon::prelude::*;
use agora_sample::{stream, Rng64, StreamPurpose};

use crate::anomaly::{
    AnomalyPlan, Campaign, FailureSpan, PlannedFailureMode, PlannedScope,
};
use crate::api::{EventBatch, EventSink, GenParams, GenSummary, ProgressFn};
use crate::world::{
    AttrSampler, CompiledBehavior, CompiledCounterparty, EffectKind, World,
};

/// Whether this shard owns `node` (and thus emits its source-edges / node row).
/// `shard_count <= 1` is the normal whole-graph run.
#[inline]
fn in_shard(node: u64, params: &GenParams) -> bool {
    params.shard_count <= 1 || node % params.shard_count == params.shard_index
}

/// Keep only the rows of a node batch this shard owns (partition by id).
fn filter_node_batch(nb: &mut crate::api::NodeBatch, params: &GenParams) {
    use crate::api::NodeColumn;
    let keep: Vec<usize> = (0..nb.ids.len())
        .filter(|&i| in_shard(nb.ids[i], params))
        .collect();
    nb.ids = keep.iter().map(|&i| nb.ids[i]).collect();
    for col in nb.attrs.iter_mut() {
        match col {
            NodeColumn::Numeric(v) => *v = keep.iter().map(|&i| v[i]).collect(),
            NodeColumn::Category { codes, .. } => {
                *codes = keep.iter().map(|&i| codes[i]).collect()
            }
        }
    }
}

/// Mean intra-burst gap in seconds (follow-up events cluster within minutes).
const BURST_GAP_MEAN_S: f64 = 300.0;
/// Actors per parallel task.
const CHUNK: usize = 4096;
/// Repeat-partner memory slots per (actor, emission).
const MEM_K: usize = 2;
const MEM_EMPTY: u64 = u64::MAX;

/// Expected number of burst FOLLOW-UPS that land inside the window, exactly.
///
/// Model (mirrors the emitter): the burst starts at `t_s ~ U(0, W)`; follow-up
/// `j` is at `t_s + G_j` with `G_j ~ Gamma(j, gap)` (a sum of `j` Exp(gap)
/// gaps); the chain has geometric length with mean `m`, so it reaches step `j`
/// with probability `q^j`, `q = m/(1+m)`; and it BREAKS at the first follow-up
/// past `W`. Because `G_j` is increasing in `j`, "step `j` lands inside" already
/// implies every earlier step did, so the events are nested and
///
/// ```text
/// E[surviving] = Σ_{j≥1} q^j · P(t_s + G_j < W).
/// ```
///
/// The old code bounded the inner probability with Jensen —
/// `P(t_s + G_j < W) ≥ 1 − E[G_j]/W = 1 − j·gap/W` — and summed that to
/// `m − (gap/W)·m·(1+m)`, then used the result AS IF it were the expectation.
/// It is a strict LOWER bound (and, clamped at 0, a badly loose one once
/// `j·gap` approaches `W`), so it understates delivered normal volume and hence
/// understates the anomaly cap.
///
/// The exact inner probability is available in closed form. Conditioning on
/// `G_j` and using `t_s ~ U(0, W)`:
///
/// ```text
/// P(t_s + G_j < W) = E[(1 − G_j/W)^+]
///                  = F_j(W) − (1/W)·E[G_j · 1{G_j < W}]
///                  = F_j(W) − (j/λ)·F_{j+1}(W),      λ = W/gap
/// ```
///
/// using the standard Gamma identity `E[G·1{G<W}] = j·gap·F_{j+1}(W)`. For
/// integer `j`, `F_j(W) = P(N ≥ j)` for `N ~ Poisson(λ)` (the Erlang/Poisson
/// duality), so the whole sum is computable from Poisson tail probabilities with
/// no special functions. Terms carry the factor `q^j`, so the series converges
/// geometrically; we truncate once the remaining mass is negligible.
fn burst_survival(m: f64, gap_s: f64, window_s: f64) -> f64 {
    if m <= 0.0 || window_s <= 0.0 || gap_s <= 0.0 {
        return 0.0;
    }
    let q = m / (1.0 + m);
    let lambda = window_s / gap_s;
    // Poisson(λ) pmf/tail, built up incrementally: pmf_i = e^{-λ} λ^i / i!.
    // tail_j = P(N ≥ j) = 1 − Σ_{i<j} pmf_i.
    let mut pmf = (-lambda).exp();
    let mut cdf = pmf; // P(N ≤ 0)
    // tails[j] = P(N ≥ j); tails[1] = 1 − P(N = 0).
    let mut tail_j = 1.0 - cdf; // P(N ≥ 1)
    let mut qj = q; // q^j
    let mut sum = 0.0;
    // q^j decays geometrically; 4096 steps is far past any practical burst.
    for j in 1..4096u32 {
        // advance to P(N ≥ j+1)
        pmf *= lambda / j as f64;
        cdf += pmf;
        let tail_j1 = (1.0 - cdf).max(0.0);
        let p_inside = (tail_j - (j as f64 / lambda) * tail_j1).clamp(0.0, 1.0);
        sum += qj * p_inside;
        qj *= q;
        tail_j = tail_j1;
        // Remaining terms are bounded by Σ_{i>j} q^i = q^{j+1}/(1−q).
        if qj / (1.0 - q) < 1e-12 || (tail_j <= 0.0 && p_inside <= 0.0) {
            break;
        }
    }
    sum
}


/// Expected edges the NORMAL stream actually DELIVERS at the calibrated
/// `world.rate_scale`, before the anomaly rebalance.
///
/// `World::build`'s calibration is the *nominal* budget: it solves
/// `rate_scale` so that `Σ_b activity_sum·rate·span·burst_factor` equals the
/// edge target. Two things make the simulator deliver slightly less (or more):
///
///   * burst chains are truncated at the window boundary (`bt >= window_len_s
///     => break`), but `burst_factor = 1 + burst_p·burst_mean_len` assumes
///     every follow-up fits. Follow-up `j` survives iff `t_s + Gamma(j, gap) <
///     W`; since the chain breaks at the first overflow, the survival events
///     are nested and `E[surviving] = Σ_j q^j·P(t_s + G_j < W)` with
///     `q = m/(1+m)`. `burst_survival` evaluates that sum EXACTLY (see there).
///   * the span covers a whole number of days whose weekday mix is generally
///     unbalanced (30 days = 4 weeks + 2), so the `weekly` profile does not
///     average to 1 over the run even though it is normalized to mean 1.
///
/// The anomaly cap bounds the anomalous fraction of the DELIVERED graph, so it
/// must be computed against this rather than against the nominal target.
pub(crate) fn expected_normal_delivered(params: &GenParams, world: &World) -> f64 {
    let n_windows = params.span_days.ceil() as u64;
    world
        .behaviors
        .iter()
        .map(|b| {
            let burst_factor = if b.branching_ratio > 0.0 {
                // Hawkes children are dropped past the window edge too, but no
                // built-in rule base uses self-excitation; keep the closed form.
                1.0 / (1.0 - b.branching_ratio)
            } else {
                // Exact expectation (the window is one day of simulated time).
                let survived = burst_survival(b.burst_mean_len, BURST_GAP_MEAN_S, 86_400.0);
                1.0 + b.burst_p * survived
            };
            // Exact weekday mix over the actual windows.
            let weekday_days: f64 = (0..n_windows)
                .map(|w| {
                    let len = (params.span_days - w as f64).min(1.0);
                    let wd = (((params.epoch_unix / 86_400) + w as i64 + 3).rem_euclid(7)) as usize;
                    b.weekly[wd] * len
                })
                .sum();
            b.activity_sum * b.rate_per_day * world.rate_scale * weekday_days * burst_factor
        })
        .sum()
}

pub fn run(
    params: &GenParams,
    sink: &mut dyn EventSink,
    mut progress: Option<ProgressFn<'_>>,
) -> anyhow::Result<GenSummary> {
    params.rulebase.validate()?;
    let t0 = std::time::Instant::now();
    let mut world = World::build(params)?;

    // Plan anomaly campaigns/incidents up front (M5), then rebalance the
    // normal-behavior budget so total output still hits the edge target:
    // calibrated emergence, not injection on top (§3).
    let plan = crate::anomaly::plan(params, &world)?;
    let target = params.target_edges as f64;
    let normal_share = ((target - plan.expected_events) / target).clamp(0.3, 1.0);
    world.rate_scale *= normal_share;

    // Anomaly homophily (fidelity axis): index every marked actor by entity
    // type so a campaign can redirect a counterparty to another marked actor of
    // the SAME type. Built ONCE from the deterministic plan (identical on every
    // shard/thread), so the redirect draws stay bit-identical. `marked_set` is
    // the guard that skips counterparties already marked (structural motifs).
    let marked_by_type: Vec<Vec<u64>> = {
        let mut buckets = vec![Vec::new(); world.entities.starts.len()];
        for c in &plan.campaigns {
            for &id in &c.members {
                buckets[world.entities.type_of(id)].push(id);
            }
        }
        buckets
    };
    let marked_set: HashSet<u64> =
        plan.campaigns.iter().flat_map(|c| c.members.iter().copied()).collect();

    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(params.threads.max(1))
        .build()
        .map_err(|e| anyhow::anyhow!("cannot build thread pool: {e}"))?;

    // Fixed intent table (M1 emits only "normal"; the table already carries
    // every declared process so labels are stable across milestones).
    let mut intent_names = vec!["normal".to_string()];
    intent_names.extend(params.rulebase.adversaries.iter().map(|a| a.intent.clone()));
    intent_names.extend(params.rulebase.failures.iter().map(|f| f.intent.clone()));
    let event_type_names: Vec<String> = world.events.iter().map(|e| e.name.clone()).collect();

    // Repeat-partner memory: per behavior, interleaved [actor][emission][K].
    let mut memories: Vec<Vec<u64>> = world
        .behaviors
        .iter()
        .map(|b| vec![MEM_EMPTY; b.actors.len() * b.emissions.len() * MEM_K])
        .collect();

    let n_windows = params.span_days.ceil() as u64;
    let mut state = world.state.clone();
    let mut edges_total = 0u64;
    let mut edges_per_intent = vec![0u64; intent_names.len()];

    for w in 0..n_windows {
        let window_len_days = (params.span_days - w as f64).min(1.0);
        let weekday = (((params.epoch_unix / 86_400) + w as i64 + 3).rem_euclid(7)) as usize;

        // Generate per behavior (sequential), per actor-chunk (parallel).
        let mut window_parts: Vec<EventBatch> = Vec::new();
        for (bi, b) in world.behaviors.iter().enumerate() {
            if b.actors.is_empty() {
                continue;
            }
            let mem_stride = b.emissions.len() * MEM_K;
            let parts: Vec<EventBatch> = pool.install(|| {
                b.actors
                    .par_chunks(CHUNK)
                    .zip(memories[bi].par_chunks_mut(CHUNK * mem_stride))
                    .enumerate()
                    .map(|(chunk_i, (actors, mem))| {
                        gen_chunk(
                            params, &world, &plan, b, bi as u64, w, window_len_days, weekday,
                            actors, chunk_i * CHUNK, mem, &marked_by_type, &marked_set,
                        )
                    })
                    .collect()
            });
            window_parts.extend(parts);
        }

        // Campaign events for campaigns overlapping this window (parallel,
        // fixed order — deterministic regardless of thread count).
        let window_start_s = params.epoch_unix + (w as i64) * 86_400;
        let window_end_s = window_start_s + (window_len_days * 86_400.0) as i64;
        let campaign_parts: Vec<EventBatch> = pool.install(|| {
            plan.campaigns
                .par_iter()
                .enumerate()
                .filter(|(_, c)| {
                    c.stages
                        .iter()
                        .any(|s| s.start_s < window_end_s && s.end_s > window_start_s)
                })
                .map(|(ci, c)| {
                    gen_campaign_window(
                        params, &world, c, ci as u64, w, window_start_s, window_end_s,
                        plan.rate_scale, &marked_by_type, &marked_set,
                    )
                })
                .collect()
        });
        window_parts.extend(campaign_parts);

        // Merge + time-sort (deterministic: parts arrive in fixed order).
        let mut batch = merge_sorted(window_parts, world.attr_names.clone());

        // Granularity rounding.
        if params.granularity_s > 1 {
            let g = params.granularity_s as i64;
            for t in batch.t.iter_mut() {
                *t -= t.rem_euclid(g);
            }
        }

        // Apply state effects in time order (single-threaded, O(events)).
        apply_effects(&world, &mut state, &batch);

        edges_total += batch.len() as u64;
        for &l in &batch.label {
            edges_per_intent[l as usize] += 1;
        }
        sink.write_batch(&batch)?;
        if let Some(p) = progress.as_mut() {
            p(edges_total, (w + 1) as f64 / n_windows as f64);
        }
    }

    // Node export with final state, then close the sink. Under sharding each
    // shard writes only the node rows it owns (disjoint by id), so the union
    // of shards is the whole node set exactly once.
    let mut export_world = world;
    export_world.state = state;
    for mut nb in export_world.node_batches() {
        if params.shard_count > 1 {
            filter_node_batch(&mut nb, params);
        }
        sink.write_nodes(&nb)?;
    }
    sink.finish()?;

    let wall = t0.elapsed().as_secs_f64();
    Ok(summary(
        params, edges_total, edges_per_intent, intent_names, event_type_names,
        export_world.attr_dictionaries, plan.records, wall,
    ))
}

#[allow(clippy::too_many_arguments)]
fn summary(
    params: &GenParams,
    edges: u64,
    edges_per_intent: Vec<u64>,
    intent_names: Vec<String>,
    event_type_names: Vec<String>,
    attr_dictionaries: Vec<(String, Vec<String>)>,
    ground_truth: Vec<crate::api::AnomalyRecord>,
    wall: f64,
) -> GenSummary {
    GenSummary {
        nodes: params.nodes,
        edges_written: edges,
        anomalous_edges: edges_per_intent.iter().skip(1).sum(),
        intent_names,
        event_type_names,
        edges_per_intent,
        attr_dictionaries,
        ground_truth,
        wall_time_s: wall,
        events_per_sec: edges as f64 / wall.max(1e-9),
    }
}

/// Generate one actor-chunk of one behavior in one window.
#[allow(clippy::too_many_arguments)]
fn gen_chunk(
    params: &GenParams,
    world: &World,
    plan: &AnomalyPlan,
    b: &CompiledBehavior,
    bi: u64,
    window: u64,
    window_len_days: f64,
    weekday: usize,
    actors: &[u64],
    actor_offset: usize,
    mem: &mut [u64],
    marked_by_type: &[Vec<u64>],
    marked_set: &HashSet<u64>,
) -> EventBatch {
    let n_cols = world.attr_names.len();
    let homophily = plan.homophily;
    let mut out = EventBatch {
        attr_names: Vec::new(), // filled at merge
        attrs: vec![Vec::new(); n_cols],
        ..Default::default()
    };
    let window_start_s = params.epoch_unix + (window as i64) * 86_400;
    let window_len_s = window_len_days * 86_400.0;
    let window_end_s = window_start_s + window_len_s as i64;
    let mem_stride = b.emissions.len() * MEM_K;

    for (ai, &actor) in actors.iter().enumerate() {
        // Distributed sharding: this shard emits only edges sourced at nodes
        // it owns. Skipping is safe because each actor's stream is independent
        // of every other's — the slices union to the whole-graph output.
        if !in_shard(actor, params) {
            continue;
        }
        let mut rng = stream(
            params.seed,
            StreamPurpose::ActorWindow,
            actor,
            (window << 8) | (bi & 0xff),
        );
        let activity = b.activity[actor_offset + ai] as f64;
        let mut lam = b.rate_per_day
            * activity
            * world.rate_scale
            * b.weekly[weekday]
            * b.diurnal_max
            * window_len_days;

        // Failure incidents touching this actor in this window (M5).
        let spans: Option<&Vec<FailureSpan>> = plan.failures_by_actor.get(&actor);
        if let Some(spans) = spans {
            for sp in spans {
                if sp.start_s < window_end_s && sp.end_s > window_start_s {
                    if let PlannedFailureMode::RateShift(f) = sp.mode {
                        let ov = overlap_frac(sp, window_start_s, window_end_s);
                        lam *= 1.0 + (f - 1.0) * ov;
                    }
                }
            }
        }
        if lam <= 0.0 {
            continue;
        }
        // Candidate arrival times. The COUNT is always Poisson(lam) so the edge
        // budget is exact. Poisson arrival places them uniformly; Weibull
        // arrival gives them heavy-tailed spacing (draw N gaps ~ Weibull(1,
        // shape), cumulate, normalize to the window) — shape < 1 → bursty
        // within-window inter-event times (Barabási 2005). Budget-neutral
        // because the count is unchanged.
        let n_candidates = sample_poisson(&mut rng, lam);
        let weibull_times: Option<Vec<f64>> = b.weibull_shape.map(|k| {
            let mut cum = Vec::with_capacity(n_candidates as usize);
            let mut total = 0.0;
            for _ in 0..n_candidates {
                total += (-rng.gen::<f64>().max(1e-12).ln()).powf(1.0 / k);
                cum.push(total);
            }
            // Normalize to the window, then rotate by a uniform phase (mod the
            // window) so the heavy-tailed sequence is stationary — otherwise it
            // front-loads into the window start (midnight, low diurnal) and
            // thinning would bias the count. With the phase, diurnal acceptance
            // averages out and the edge budget matches the Poisson path.
            if total > 0.0 {
                let phase = rng.gen::<f64>() * window_len_s;
                for c in cum.iter_mut() {
                    *c = (*c / total * window_len_s + phase) % window_len_s;
                }
            }
            cum
        });
        for ci in 0..n_candidates as usize {
            let t_s = match &weibull_times {
                Some(v) => v[ci],
                None => rng.gen::<f64>() * window_len_s,
            };
            let hour = ((t_s / 3600.0) as usize).min(23) % 24;
            // Thinning: accept by diurnal shape.
            if rng.gen::<f64>() >= b.diurnal[hour] / b.diurnal_max {
                continue;
            }
            let t_abs = window_start_s + t_s as i64;
            if is_silenced(spans, t_abs) {
                continue;
            }
            let mem_base = ai * mem_stride;
            if let Some(row) = emit(world, b, actor, t_abs, &mut rng, &mut out, mem, mem_base, homophily, marked_by_type, marked_set) {
                corrupt_row(spans, &mut out, row, &mut rng);
            }

            // Hawkes self-excitation (recursive cascade): each event spawns
            // Poisson(n) children at Exp(decay) delays, each spawning its own —
            // "active stays active" clustering that lifts burstiness toward
            // real interaction data. Subcritical n<1 so it terminates; capped
            // for safety. Replaces the simple geometric burst when set.
            if b.branching_ratio > 0.0 {
                let mut stack: Vec<f64> = vec![t_s];
                let mut emitted = 0u32;
                const HAWKES_CAP: u32 = 4096;
                while let Some(parent_t) = stack.pop() {
                    let k = sample_poisson(&mut rng, b.branching_ratio);
                    for _ in 0..k {
                        let ct = parent_t + sample_exp(&mut rng, b.excitation_decay_s);
                        if ct >= window_len_s {
                            continue;
                        }
                        let t_abs = window_start_s + ct as i64;
                        if !is_silenced(spans, t_abs) {
                            if let Some(row) =
                                emit(world, b, actor, t_abs, &mut rng, &mut out, mem, mem_base, homophily, marked_by_type, marked_set)
                            {
                                corrupt_row(spans, &mut out, row, &mut rng);
                            }
                        }
                        stack.push(ct);
                        emitted += 1;
                        if emitted >= HAWKES_CAP {
                            stack.clear();
                            break;
                        }
                    }
                }
            } else if b.burst_p > 0.0 && rng.gen::<f64>() < b.burst_p {
                let extra = sample_geometric(&mut rng, b.burst_mean_len);
                let mut bt = t_s;
                for _ in 0..extra {
                    bt += sample_exp(&mut rng, BURST_GAP_MEAN_S);
                    if bt >= window_len_s {
                        break;
                    }
                    let t_abs = window_start_s + bt as i64;
                    if is_silenced(spans, t_abs) {
                        continue;
                    }
                    if let Some(row) =
                        emit(world, b, actor, t_abs, &mut rng, &mut out, mem, mem_base, homophily, marked_by_type, marked_set)
                    {
                        corrupt_row(spans, &mut out, row, &mut rng);
                    }
                }
            }
        }
    }
    out
}

#[inline]
fn overlap_frac(sp: &FailureSpan, ws: i64, we: i64) -> f64 {
    let lo = sp.start_s.max(ws);
    let hi = sp.end_s.min(we);
    ((hi - lo).max(0)) as f64 / ((we - ws).max(1)) as f64
}

#[inline]
fn is_silenced(spans: Option<&Vec<FailureSpan>>, t: i64) -> bool {
    spans.is_some_and(|spans| {
        spans.iter().any(|sp| {
            matches!(sp.mode, PlannedFailureMode::Silence) && t >= sp.start_s && t < sp.end_s
        })
    })
}

/// Apply attribute-corrupting failure modes to a just-emitted row and label
/// it with the failure's intent (label = cause, §3).
fn corrupt_row(
    spans: Option<&Vec<FailureSpan>>,
    out: &mut EventBatch,
    row: usize,
    rng: &mut Rng64,
) {
    let Some(spans) = spans else { return };
    let t = out.t[row];
    let ev = out.event_type[row] as usize;
    for sp in spans {
        if t < sp.start_s || t >= sp.end_s {
            continue;
        }
        match &sp.mode {
            PlannedFailureMode::StuckAttr(event, col, value) if *event == ev => {
                out.attrs[*col][row] = *value;
                out.label[row] = sp.intent;
                out.anomaly_id[row] = sp.id;
            }
            PlannedFailureMode::DriftAttr(event, col, per_day) if *event == ev => {
                let days = (t - sp.start_s) as f64 / 86_400.0;
                out.attrs[*col][row] += per_day * days;
                out.label[row] = sp.intent;
                out.anomaly_id[row] = sp.id;
            }
            PlannedFailureMode::NoiseAttr(event, col, sampler) if *event == ev => {
                out.attrs[*col][row] = sampler.sample(rng);
                out.label[row] = sp.intent;
                out.anomaly_id[row] = sp.id;
            }
            // RateShift: the anomaly IS the volume shift, so every event the
            // affected actor emits during the span is part of it (the slow
            // traversals of a congestion jam, the surge of an incident).
            // Label only if still normal, so attr-corrupting modes that share
            // the actor win the label.
            PlannedFailureMode::RateShift(_) if out.label[row] == 0 => {
                out.label[row] = sp.intent;
                out.anomaly_id[row] = sp.id;
            }
            // Silence drops events upstream — its signature is missing data,
            // which has no edge to label (documented in stats).
            _ => {}
        }
    }
}

/// Generate one campaign's events within one window.
#[allow(clippy::too_many_arguments)]
fn gen_campaign_window(
    params: &GenParams,
    world: &World,
    c: &Campaign,
    ci: u64,
    _window: u64,
    window_start_s: i64,
    window_end_s: i64,
    anomaly_rate_scale: f64,
    marked_by_type: &[Vec<u64>],
    marked_set: &HashSet<u64>,
) -> EventBatch {
    let n_cols = world.attr_names.len();
    let mut out = EventBatch {
        attr_names: Vec::new(),
        attrs: vec![Vec::new(); n_cols],
        ..Default::default()
    };
    let mut rng = stream(
        params.seed,
        StreamPurpose::Campaign,
        10_000 + ci,
        (window_start_s - params.epoch_unix).max(0) as u64 / 86_400,
    );
    let n_members = c.members.len();
    if n_members == 0 {
        return out;
    }
    for stage in &c.stages {
        // Shared with the planner (see `anomaly::stage_window`) so the volume
        // the cap is computed from is exactly the volume emitted here.
        let Some((t0, t1)) = crate::anomaly::stage_window(stage, window_start_s, window_end_s)
        else {
            continue;
        };
        let days = (t1 - t0) as f64 / 86_400.0;
        let stage_rate = stage.rate_per_day * anomaly_rate_scale;
        let ev = &world.events[stage.event];
        // Which members emit in this stage. Shared with the planner so the
        // calibration cap is computed against exactly this volume.
        let acting: &[u64] = crate::anomaly::acting_slice(c, stage);
        for (mi, &member) in acting.iter().enumerate() {
            let n = sample_poisson(&mut rng, stage_rate * days);
            for _ in 0..n {
                let t = t0 + (rng.gen::<f64>() * (t1 - t0) as f64) as i64;
                // The scope picks the COUNTERPARTY; which end of the edge it
                // lands on is decided by `scope.member_is_dst()` below.
                let cp = match &stage.scope {
                    PlannedScope::Ring => {
                        if n_members < 2 {
                            continue;
                        }
                        let mut d = c.members[rng.gen_range(0..n_members)];
                        if d == member {
                            d = c.members[(mi + 1) % n_members];
                        }
                        d
                    }
                    PlannedScope::Chain => c.members[(mi + 1) % n_members],
                    PlannedScope::Collector(anchor) => *anchor,
                    // Star fan-out: `member` IS members[0] (the operator; see
                    // `acting_slice`), paying a uniformly-chosen investor. No
                    // investor-to-investor edge is ever emitted, so the motif
                    // stays a star instead of collapsing to a clique.
                    PlannedScope::Hub => {
                        if n_members < 2 {
                            continue;
                        }
                        c.members[1 + rng.gen_range(0..n_members - 1)]
                    }
                    PlannedScope::Normal(cp) => match *cp {
                        CompiledCounterparty::Neighbor { relation } => {
                            let rel = &world.skeletons[relation];
                            let neigh = rel.neighbors_of(member);
                            if neigh.is_empty() {
                                rel.dst_start + rng.gen_range(0..rel.n_dst)
                            } else {
                                neigh[rng.gen_range(0..neigh.len())]
                            }
                        }
                        CompiledCounterparty::GlobalUniform { entity } => {
                            world.entities.starts[entity]
                                + rng.gen_range(0..world.entities.counts[entity])
                        }
                        CompiledCounterparty::GlobalPopularity { entity, pop } => {
                            world.entities.starts[entity]
                                + world.popularity[pop].1.sample(&mut rng) as u64
                        }
                        CompiledCounterparty::RepeatOrNeighbor { relation, .. } => {
                            let rel = &world.skeletons[relation];
                            let neigh = rel.neighbors_of(member);
                            if neigh.is_empty() {
                                rel.dst_start + rng.gen_range(0..rel.n_dst)
                            } else {
                                neigh[rng.gen_range(0..neigh.len())]
                            }
                        }
                    },
                    PlannedScope::Victims(pool) | PlannedScope::Sources(pool) => {
                        if pool.is_empty() {
                            continue;
                        }
                        pool[rng.gen_range(0..pool.len())]
                    }
                };
                // Relation camouflage (axis 2, structural): with prob c.camouflage,
                // reroute the edge to a benign counterparty, diluting the
                // fan-in / fan-out / ring / chain motif so that a
                // structure-aware detector also degrades as difficulty rises
                // (complements the attribute camouflage applied below). The
                // replacement is drawn from the COUNTERPARTY's entity type,
                // which is the event's src type for fan-in (`sources`) scopes
                // and its dst type otherwise — camouflage must never silently
                // flip the edge's direction or its endpoint types. The rng
                // draws depend only on shared campaign state, so shards stay
                // bit-identical.
                let cp_ty = stage.scope.counterparty_type(ev);
                let cp = if c.camouflage > 0.0
                    && world.entities.counts[cp_ty] > 0
                    && rng.gen::<f64>() < c.camouflage
                {
                    world.entities.starts[cp_ty] + rng.gen_range(0..world.entities.counts[cp_ty])
                } else {
                    cp
                };
                // Homophily (fidelity axis): real illicit actors preferentially
                // transact with each other. With prob c.homophily, redirect a
                // counterparty that is not already marked to a random marked
                // actor of the SAME entity type, raising anomaly homophily toward
                // real levels without disturbing the structural motifs (Ring/
                // Chain/Hub counterparties are already marked, so the not-marked
                // guard skips them). Draws depend only on shared campaign state
                // (marked_by_type and marked_set derive from the deterministic
                // plan; cp is computed shard-independently), so shards stay
                // bit-identical.
                let cp = if c.homophily > 0.0 {
                    let pool = &marked_by_type[cp_ty];
                    if !pool.is_empty() && !marked_set.contains(&cp) && rng.gen::<f64>() < c.homophily {
                        pool[rng.gen_range(0..pool.len())]
                    } else {
                        cp
                    }
                } else {
                    cp
                };
                if cp == member {
                    continue;
                }
                // Place the member on the correct end: fan-in scopes make the
                // counterparty the SOURCE and the member the DESTINATION.
                let (e_src, e_dst) = if stage.scope.member_is_dst() {
                    (cp, member)
                } else {
                    (member, cp)
                };
                // Draw all attributes FIRST (they consume rng) regardless of
                // shard ownership, so every shard replays the identical
                // campaign rng stream and keeps bit-identical edges for the
                // members it owns. Only the append below is shard-gated.
                let mut drawn: Vec<(usize, f64)> = Vec::with_capacity(ev.attrs.len());
                for ca in &ev.attrs {
                    let v = match &ca.sampler {
                        AttrSampler::Numeric(d) => d.sample(&mut rng),
                        AttrSampler::Category { table, code_map } => {
                            code_map[table.sample(&mut rng)] as f64
                        }
                        AttrSampler::Flag(p) => f64::from(rng.gen::<f64>() < *p),
                    };
                    drawn.push((ca.col, v));
                }
                // …then the adversary's override wins with prob (1 − c_eff):
                // feature camouflage blends toward the normal distribution.
                for (col, sampler) in &stage.overrides {
                    if rng.gen::<f64>() >= c.camouflage {
                        let v = sampler.sample(&mut rng);
                        if let Some(slot) = drawn.iter_mut().find(|(c, _)| c == col) {
                            slot.1 = v;
                        } else {
                            drawn.push((*col, v));
                        }
                    }
                }
                // Shard ownership is keyed on the edge's SRC, upholding the
                // graph-wide contract that shard k holds exactly the edges
                // sourced at the nodes it owns (`filter_node_batch` partitions
                // nodes the same way). For fan-in scopes the src is the outside
                // payer rather than the acting member, so this is NOT
                // `in_shard(member)` — but every shard replays the whole
                // campaign stream and only filters at the end, so `e_src` is
                // identical everywhere and each edge is still claimed exactly
                // once. Union over shards == whole graph, bit for bit.
                if !in_shard(e_src, params) {
                    continue;
                }
                out.src.push(e_src);
                out.dst.push(e_dst);
                out.t.push(t);
                out.event_type.push(stage.event as u16);
                out.label.push(c.intent);
                out.anomaly_id.push(c.id);
                for col in out.attrs.iter_mut() {
                    col.push(f64::NAN);
                }
                let row = out.src.len() - 1;
                for (col, v) in drawn {
                    out.attrs[col][row] = v;
                }
            }
        }
    }
    out
}

/// Emit one event for `actor` at `t`: choose emission, counterparty, attrs.
/// Returns the row index written (None if the event was skipped).
#[allow(clippy::too_many_arguments)]
#[inline]
fn emit(
    world: &World,
    b: &CompiledBehavior,
    actor: u64,
    t: i64,
    rng: &mut Rng64,
    out: &mut EventBatch,
    mem: &mut [u64],
    mem_base: usize,
    homophily: f64,
    marked_by_type: &[Vec<u64>],
    marked_set: &HashSet<u64>,
) -> Option<usize> {
    let em_i = b.emission_alias.sample(rng);
    let em = &b.emissions[em_i];
    let ev = &world.events[em.event];

    let dst = match &em.counterparty {
        CompiledCounterparty::Neighbor { relation } => {
            pick_neighbor(world, *relation, actor, rng)
        }
        CompiledCounterparty::RepeatOrNeighbor { relation, repeat_p } => {
            let slot = &mut mem[mem_base + em_i * MEM_K..mem_base + em_i * MEM_K + MEM_K];
            let remembered = slot.iter().filter(|&&x| x != MEM_EMPTY).count();
            if remembered > 0 && rng.gen::<f64>() < *repeat_p {
                slot[rng.gen_range(0..remembered)]
            } else {
                let d = pick_neighbor(world, *relation, actor, rng);
                // Ring update: shift in the new partner.
                slot[1] = slot[0];
                slot[0] = d;
                d
            }
        }
        CompiledCounterparty::GlobalUniform { entity } => {
            let start = world.entities.starts[*entity];
            let n = world.entities.counts[*entity];
            let mut d = start + rng.gen_range(0..n);
            if d == actor && n > 1 {
                d = start + (d - start + 1) % n;
            }
            d
        }
        CompiledCounterparty::GlobalPopularity { entity, pop } => {
            let start = world.entities.starts[*entity];
            let mut d = start + world.popularity[*pop].1.sample(rng) as u64;
            if d == actor && world.entities.counts[*entity] > 1 {
                d = start + (d - start + 1) % world.entities.counts[*entity];
            }
            d
        }
    };
    let mut dst = dst_or_skip(dst)?;

    // Anomaly homophily on the NORMAL path (fidelity axis, extends P1-a): a real
    // illicit actor's WHOLE edge set skews illicit, not only its campaign edges.
    // So when the emitting actor is marked, redirect this background edge to a
    // random marked actor of the chosen dst's entity type with prob `homophily`.
    // This is the lever the campaign-only redirect lacked: marked actors' normal
    // traffic otherwise dilutes P(dst marked | src marked) and caps homophily.
    //
    // DETERMINISM: the normal path is sharded by the emitting node, and this draw
    // consumes the actor's OWN normal rng stream, so it is identical regardless of
    // thread/shard count (an actor is always produced on its owning shard). Guarded
    // by `homophily > 0.0` and `marked_set.contains(&actor)` so that no rng is
    // drawn when the control is 0 or the actor is benign — an exact no-op for every
    // domain that leaves homophily unset, preserving their bit-identical output.
    if homophily > 0.0 && marked_set.contains(&actor) {
        let pool = &marked_by_type[world.entities.type_of(dst)];
        if !pool.is_empty() && !marked_set.contains(&dst) && rng.gen::<f64>() < homophily {
            let cand = pool[rng.gen_range(0..pool.len())];
            if cand != actor {
                dst = cand;
            }
        }
    }

    out.src.push(actor);
    out.dst.push(dst);
    out.t.push(t);
    out.event_type.push(em.event as u16);
    out.label.push(0);
    out.anomaly_id.push(-1); // normal until a failure span corrupts/labels it
    // Union columns: NaN by default, overwritten by this event's attrs.
    for col in out.attrs.iter_mut() {
        col.push(f64::NAN);
    }
    let row = out.src.len() - 1;
    for ca in &ev.attrs {
        let v = match &ca.sampler {
            AttrSampler::Numeric(d) => d.sample(rng),
            AttrSampler::Category { table, code_map } => code_map[table.sample(rng)] as f64,
            AttrSampler::Flag(p) => f64::from(rng.gen::<f64>() < *p),
        };
        out.attrs[ca.col][row] = v;
    }
    Some(row)
}

#[inline]
fn dst_or_skip(dst: u64) -> Option<u64> {
    (dst != u64::MAX).then_some(dst)
}

#[inline]
fn pick_neighbor(world: &World, relation: usize, actor: u64, rng: &mut Rng64) -> u64 {
    let rel = &world.skeletons[relation];
    let neigh = rel.neighbors_of(actor);
    if neigh.is_empty() {
        // Isolated in this relation: fall back to a uniform dst (keeps the
        // event budget; rare because skeleton builders floor degree at 1).
        rel.dst_start + rng.gen_range(0..rel.n_dst)
    } else {
        neigh[rng.gen_range(0..neigh.len())]
    }
}

/// Merge parts and sort by (t, src, dst, event_type) — a total order that
/// makes output independent of the parallel schedule.
fn merge_sorted(parts: Vec<EventBatch>, attr_names: Vec<String>) -> EventBatch {
    let total: usize = parts.iter().map(|p| p.len()).sum();
    let n_cols = attr_names.len();
    let mut merged = EventBatch {
        src: Vec::with_capacity(total),
        dst: Vec::with_capacity(total),
        t: Vec::with_capacity(total),
        event_type: Vec::with_capacity(total),
        label: Vec::with_capacity(total),
        anomaly_id: Vec::with_capacity(total),
        attrs: vec![Vec::with_capacity(total); n_cols],
        attr_names,
    };
    for p in parts {
        merged.src.extend_from_slice(&p.src);
        merged.dst.extend_from_slice(&p.dst);
        merged.t.extend_from_slice(&p.t);
        merged.event_type.extend_from_slice(&p.event_type);
        merged.label.extend_from_slice(&p.label);
        merged.anomaly_id.extend_from_slice(&p.anomaly_id);
        for (c, col) in p.attrs.into_iter().enumerate() {
            merged.attrs[c].extend_from_slice(&col);
        }
    }
    // Parallel argsort permutation, then apply to every column in parallel.
    let mut idx: Vec<u32> = (0..total as u32).collect();
    idx.par_sort_unstable_by_key(|&i| {
        let i = i as usize;
        (merged.t[i], merged.src[i], merged.dst[i], merged.event_type[i])
    });
    apply_perm(&mut merged, &idx);
    merged
}

fn apply_perm(b: &mut EventBatch, idx: &[u32]) {
    fn perm<T: Copy + Send + Sync>(v: &[T], idx: &[u32]) -> Vec<T> {
        idx.par_iter().map(|&i| v[i as usize]).collect()
    }
    // Permute the fixed columns and every attr column concurrently.
    let (src, dst, t, et, label) = {
        let (a, b2) = rayon::join(
            || rayon::join(|| perm(&b.src, idx), || perm(&b.dst, idx)),
            || {
                rayon::join(
                    || perm(&b.t, idx),
                    || rayon::join(|| perm(&b.event_type, idx), || perm(&b.label, idx)),
                )
            },
        );
        let ((src, dst), (t, (et, label))) = (a, b2);
        (src, dst, t, et, label)
    };
    b.src = src;
    b.dst = dst;
    b.t = t;
    b.event_type = et;
    b.label = label;
    b.anomaly_id = perm(&b.anomaly_id, idx);
    let attrs = std::mem::take(&mut b.attrs);
    b.attrs = attrs.into_par_iter().map(|col| perm(&col, idx)).collect();
}

/// Apply state effects of every event, in time order.
fn apply_effects(world: &World, state: &mut [Vec<Vec<f64>>], batch: &EventBatch) {
    for i in 0..batch.len() {
        let ev = &world.events[batch.event_type[i] as usize];
        for eff in &ev.effects {
            let node = if eff.on_src { batch.src[i] } else { batch.dst[i] };
            let local = (node - world.entities.starts[eff.entity]) as usize;
            let slot = &mut state[eff.entity][eff.var][local];
            match eff.kind {
                EffectKind::Add => {
                    let v = batch.attrs[eff.from_col][i];
                    if v.is_finite() {
                        *slot += v;
                    }
                }
                EffectKind::Sub => {
                    let v = batch.attrs[eff.from_col][i];
                    if v.is_finite() {
                        *slot -= v;
                    }
                }
                EffectKind::Increment => *slot += 1.0,
                EffectKind::Set => *slot = eff.value,
            }
        }
    }
}

// --- small deterministic samplers -------------------------------------------

#[inline]
fn sample_poisson(rng: &mut Rng64, lam: f64) -> u64 {
    if lam < 1e-9 {
        return 0;
    }
    rand_distr::Poisson::new(lam).map(|p| p.sample(rng) as u64).unwrap_or(0)
}

/// Geometric with the given mean (number of burst follow-ups).
#[inline]
fn sample_geometric(rng: &mut Rng64, mean: f64) -> u64 {
    if mean <= 0.0 {
        return 0;
    }
    let p = 1.0 / (1.0 + mean);
    let u: f64 = rng.gen();
    (u.ln() / (1.0 - p).ln()).floor() as u64
}

#[inline]
fn sample_exp(rng: &mut Rng64, mean: f64) -> f64 {
    let u: f64 = rng.gen::<f64>().max(1e-12);
    -mean * u.ln()
}

#[cfg(test)]
mod tests {
    use super::*;
    use agora_sample::{stream, StreamPurpose};

    /// Replicates the burst emitter EXACTLY (same geometric, same Exp gaps,
    /// same `>= W` break) and counts surviving follow-ups, so the closed form
    /// in `burst_survival` is checked against the thing it claims to describe
    /// rather than against a restatement of its own algebra.
    fn mc_survival(m: f64, gap: f64, w: f64, iters: u32) -> f64 {
        let mut rng = stream(0xB0_1E, StreamPurpose::Campaign, 7, 0);
        let mut total = 0u64;
        for _ in 0..iters {
            let t_s = rng.gen::<f64>() * w;
            let extra = sample_geometric(&mut rng, m);
            let mut bt = t_s;
            for _ in 0..extra {
                bt += sample_exp(&mut rng, gap);
                if bt >= w {
                    break;
                }
                total += 1;
            }
        }
        total as f64 / iters as f64
    }

    #[test]
    fn burst_survival_matches_monte_carlo() {
        // (mean chain length, gap seconds, window seconds)
        for &(m, gap, w) in &[
            (2.0, 300.0, 86_400.0),   // the shipped regime: gaps << window
            (5.0, 300.0, 86_400.0),
            (0.5, 300.0, 86_400.0),
            (3.0, 20_000.0, 86_400.0), // gaps comparable to the window
            (8.0, 40_000.0, 86_400.0), // truncation dominates
        ] {
            let exact = burst_survival(m, gap, w);
            let mc = mc_survival(m, gap, w, 400_000);
            assert!(
                (exact - mc).abs() < 0.02 * mc.max(0.05),
                "m={m} gap={gap} w={w}: closed form {exact:.4} vs monte carlo {mc:.4}"
            );
        }
    }

    /// The old Jensen form was a strict LOWER bound; the fix must never sit
    /// below it, and must be strictly above it wherever truncation bites.
    #[test]
    fn burst_survival_dominates_the_old_jensen_bound() {
        for &(m, gap, w) in &[(2.0f64, 300.0f64, 86_400.0f64), (3.0, 20_000.0, 86_400.0)] {
            let jensen = (m - (gap / w) * m * (1.0 + m)).max(0.0);
            let exact = burst_survival(m, gap, w);
            assert!(exact >= jensen - 1e-9, "m={m} gap={gap}: {exact} < bound {jensen}");
        }
        // Where gaps are a large fraction of the window the bound is not merely
        // loose, it collapses to the 0 clamp while real bursts still land
        // (Monte Carlo puts the truth near 0.89 follow-ups per burst here).
        let (m, gap, w) = (8.0f64, 40_000.0f64, 86_400.0f64);
        assert_eq!((m - (gap / w) * m * (1.0 + m)).max(0.0), 0.0);
        assert!(burst_survival(m, gap, w) > 0.5);
    }

    /// Degenerate inputs must not produce NaN/Inf on the cap path.
    #[test]
    fn burst_survival_handles_degenerate_inputs() {
        for &(m, gap, w) in &[(0.0, 300.0, 86_400.0), (2.0, 0.0, 86_400.0), (2.0, 300.0, 0.0)] {
            let v = burst_survival(m, gap, w);
            assert!(v.is_finite() && v >= 0.0, "m={m} gap={gap} w={w} -> {v}");
        }
    }
}
