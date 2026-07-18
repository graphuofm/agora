//! Anomaly processes (M5, the P0 core): calibrated emergence.
//!
//! Anomalies are not injected templates — they are *generative processes*
//! (AdversaryProcess campaigns, FailureProcess incidents) whose events emerge
//! from the same simulation loop as normal behavior and are LABELED BY INTENT
//! (the cause), not by post-hoc rule matching (§3, §5).
//!
//! The five control axes (§3):
//!   1 PREVALENCE   fraction of actor nodes recruited into anomalous
//!                  processes; sets the campaign/incident budget.
//!   2 DIFFICULTY   global d ∈ [0,1] blends the adversary toward a normal
//!                  actor along three independent leaks:
//!                    * FEATURE  — c_eff = clamp(c·2d); with prob c_eff an
//!                      event's attributes come from the NORMAL event
//!                      distribution instead of the playbook's override.
//!                    * RELATION — with prob c_eff the counterparty is
//!                      rerouted to a benign node, diluting the motif.
//!                    * VOLUME   — a stage's activity ratio k (× the actor
//!                      type's normal per-actor rate) is shrunk in LOG space,
//!                      k_eff = k^(1−d), so d = 1 gives k_eff = 1 exactly: the
//!                      adversary is no busier (nor quieter) than a normal
//!                      actor and the volume channel carries no signal.
//!                  At d = 1 all three leaks close, which is what makes
//!                  difficulty a true detectability axis rather than a
//!                  partial damping.
//!   3 TYPE-MIX     weights over intents (config override or per-process
//!                  prevalence_weight).
//!   4 PLACEMENT    which communities (contiguous id blocks of the actor
//!                  range) and/or time window campaigns concentrate in.
//!   5 CASCADE      multiplier on per-process cascade_p: finished campaigns
//!                  recruit follow-up campaigns; failures propagate to
//!                  skeleton neighbors (congestion-style).
//!
//! Everything is planned up front from `(seed, Campaign/Failure, …)` streams,
//! so anomaly placement is bit-reproducible and thread-count independent.

use std::collections::HashMap;

use anyhow::Context;
use rand::Rng;
use agora_rules::{ControlParams, FailureMode, Placement, StageScope, TypeWeight};
use agora_sample::{stream, DistSampler, Rng64, StreamPurpose};

use crate::api::{AnomalyRecord, GenParams};
use crate::world::{CompiledCounterparty, World};

/// Hard caps keeping pathological configs bounded (risk R1).
const MAX_CASCADE_FRACTION: f64 = 0.5;
const MAX_RING: usize = 1 << 16;

/// Anomalies must stay a MINORITY of edges (they are rare by definition,
/// §3 / DOMAINS.md cross-domain note). If the planned processes would exceed
/// this fraction of the edge budget, campaign rates are scaled down uniformly
/// — legitimate calibration (set process parameters to hit targets), and the
/// scale is reported so the cap is never silent.
const MAX_ANOMALY_FRACTION: f64 = 0.30;

pub struct AnomalyPlan {
    pub campaigns: Vec<Campaign>,
    /// Failure incidents indexed by affected actor.
    pub failures_by_actor: HashMap<u64, Vec<FailureSpan>>,
    /// One omniscient record per anomaly instance (the multi-attribute ground
    /// truth written to `ground_truth.json`).
    pub records: Vec<AnomalyRecord>,
    /// Expected anomalous event count AFTER rate scaling.
    pub expected_events: f64,
    /// Uniform multiplier applied to campaign stage rates to honor the cap
    /// (1.0 = uncapped). Reported in the run summary.
    pub rate_scale: f64,
    /// Global anomaly-homophily control in [0,1], copied from `ControlParams`.
    /// The normal-emission path reads it directly to bias a marked actor's
    /// background edges toward other marked actors (see `sim::emit`). 0.0 is an
    /// exact no-op (no rng drawn), so domains that leave it unset are untouched.
    pub homophily: f64,
}

pub struct Campaign {
    /// Stable ground-truth instance id (matches the edge `anomaly_id` column).
    pub id: i64,
    /// Intent label index (into the run's intent table; 0 = normal).
    pub intent: u16,
    pub members: Vec<u64>,
    pub stages: Vec<PlannedStage>,
    /// Effective camouflage after the difficulty axis.
    pub camouflage: f64,
    /// Placement community block (-1 if not clustered).
    pub community: i64,
    pub cascade: bool,
    /// Fidelity axis — probability that a non-marked counterparty is redirected
    /// to a random marked actor of the same entity type (anomaly homophily).
    /// Global: copied uniformly from `ControlParams`, never scaled by difficulty.
    pub homophily: f64,
}

pub struct PlannedStage {
    pub start_s: i64,
    pub end_s: i64,
    /// Events per member per day (already camouflage-damped).
    pub rate_per_day: f64,
    pub event: usize,
    pub scope: PlannedScope,
    /// (union attr column, override sampler).
    pub overrides: Vec<(usize, DistSampler)>,
}

/// Where a stage's counterparty comes from, and — crucially — which END of the
/// emitted edge the campaign member occupies.
///
/// Every variant but `Sources` puts the member at the edge's SRC (the member
/// acts outward). `Sources` is the fan-IN mirror: the member is the DST and the
/// drawn counterparty is the SRC. See `member_is_dst`.
pub enum PlannedScope {
    /// Random other ring member (same-type events only).
    Ring,
    /// members[i] -> members[(i+1) % n] (chains and conserved cycles;
    /// same-type events only).
    Chain,
    /// All acting members -> one anchor node of the event's DST type
    /// (fan-in). For same-type events the anchor is members[0]; for
    /// heterogeneous events it is a dedicated dst-type node.
    Collector(u64),
    /// Star fan-OUT: `members[0]` (the operator) is the sole actor and pays the
    /// other members. Same-type events only. Dual of `Collector`.
    Hub,
    /// The actor's normal counterparty policy for this event type.
    Normal(CompiledCounterparty),
    /// Pre-sampled victim pool of the event's DST type, outside the ring.
    /// member -> victim.
    Victims(Vec<u64>),
    /// Pre-sampled pool of payers of the event's SRC type, outside the ring.
    /// payer -> member (fan-in). The ONLY scope where the member is the DST.
    Sources(Vec<u64>),
}

impl PlannedScope {
    /// True iff the campaign member is the DESTINATION of the emitted edge
    /// (i.e. the counterparty is the source). Only `Sources` fans in.
    #[inline]
    pub(crate) fn member_is_dst(&self) -> bool {
        matches!(self, PlannedScope::Sources(_))
    }

    /// Entity type of the COUNTERPARTY end of the edge for `ev`. Relation
    /// camouflage reroutes that end, so it must draw from this population.
    #[inline]
    pub(crate) fn counterparty_type(&self, ev: &crate::world::CompiledEvent) -> usize {
        if self.member_is_dst() { ev.src_type } else { ev.dst_type }
    }
}

#[derive(Clone)]
pub struct FailureSpan {
    /// Stable ground-truth instance id (matches the edge `anomaly_id` column).
    pub id: i64,
    pub start_s: i64,
    pub end_s: i64,
    pub intent: u16,
    pub mode: PlannedFailureMode,
}

#[derive(Clone)]
pub enum PlannedFailureMode {
    Silence,
    RateShift(f64),
    /// (event idx, union col, stuck value)
    StuckAttr(usize, usize, f64),
    /// (event idx, union col, drift per day)
    DriftAttr(usize, usize, f64),
    /// (event idx, union col, replacement sampler)
    NoiseAttr(usize, usize, DistSampler),
}

pub fn plan(params: &GenParams, world: &World) -> anyhow::Result<AnomalyPlan> {
    if params.anomalies_disabled {
        return Ok(AnomalyPlan {
            campaigns: Vec::new(),
            failures_by_actor: HashMap::new(),
            records: Vec::new(),
            expected_events: 0.0,
            rate_scale: 1.0,
            homophily: 0.0,
        });
    }
    let rb = &params.rulebase;
    // Apply the CLI/runtime control-axis overrides (axes 3/4/5) onto the
    // rule-base defaults, then use this effective control everywhere.
    let control = effective_control(&rb.control, params);
    let control = &control;
    let prevalence = params.anomaly_rate.unwrap_or(control.prevalence).clamp(0.0, 0.5);
    let difficulty = params.anomaly_difficulty.unwrap_or(control.difficulty).clamp(0.0, 1.0);
    let span_s = (params.span_days * 86_400.0) as i64;

    // Type-mix weights, normalized over all processes (adversaries then
    // failures, matching the intent table order).
    let n_proc = rb.adversaries.len() + rb.failures.len();
    if n_proc == 0 || prevalence == 0.0 {
        return Ok(AnomalyPlan {
            campaigns: Vec::new(),
            failures_by_actor: HashMap::new(),
            records: Vec::new(),
            expected_events: 0.0,
            rate_scale: 1.0,
            homophily: 0.0,
        });
    }
    let mut weights: Vec<f64> = rb
        .adversaries
        .iter()
        .map(|a| (a.intent.clone(), a.prevalence_weight))
        .chain(rb.failures.iter().map(|f| (f.intent.clone(), f.prevalence_weight)))
        .map(|(intent, w)| {
            control
                .type_mix
                .iter()
                .find(|tw| tw.intent == intent)
                .map(|tw| tw.weight)
                .unwrap_or(w)
        })
        .collect();
    let wsum: f64 = weights.iter().sum();
    anyhow::ensure!(wsum > 0.0, "all anomaly type weights are zero");
    for w in weights.iter_mut() {
        *w /= wsum;
    }

    if std::env::var("AGORA_DEBUG_RATES").is_ok() {
        eprintln!("[rates] domain={} rate_scale={:.6}", rb.meta.id, world.rate_scale);
        for (ti, name) in world.entities.names.iter().enumerate() {
            // scale-1 rate = the events/day/actor the YAML itself declares,
            // before the edge budget rescales the world. This is the
            // denominator for deriving `activity_multiplier` from a legacy
            // absolute `rate_per_day`.
            eprintln!(
                "[rates]   {name}: n={} delivered={:.6}/day scale1={:.6}/day",
                world.entities.counts[ti],
                world.normal_rate_per_actor(ti),
                world.normal_rate_per_actor(ti) / world.rate_scale
            );
        }
    }

    let mut campaigns = Vec::new();
    let mut failures_by_actor: HashMap<u64, Vec<FailureSpan>> = HashMap::new();
    let mut records: Vec<AnomalyRecord> = Vec::new();
    let mut expected_events = 0.0f64;
    let mut next_id: i64 = 0; // stable, deterministic instance ids in plan order

    // --- adversary campaigns -------------------------------------------------
    for (pi, proc) in rb.adversaries.iter().enumerate() {
        let intent = 1 + pi as u16;
        let actor_ty = world.entities.index_of(&proc.actor);
        let n_actors = world.entities.counts[actor_ty];
        let actor_start = world.entities.starts[actor_ty];
        let ring_sampler = DistSampler::compile(&proc.ring_size)
            .with_context(|| format!("adversary `{}` ring_size", proc.intent))?;
        let mean_ring = ring_sampler.mean().max(1.0);
        let budget_nodes = prevalence * n_actors as f64 * weights[pi];
        let n_campaigns = (budget_nodes / mean_ring).round().max(if weights[pi] > 0.0 { 1.0 } else { 0.0 }) as u64;
        // Heavy-tail (super-hub) actors the campaign must NOT recruit: an
        // exchange hot wallet or MEV bot cannot be commandeered into a fraud
        // ring, and its normal traffic would drown the campaign's edges. Empty
        // unless the actor mix is genuinely heavy-tailed (see `heavy_tail_actors`).
        let reject: Vec<u64> = world.heavy_tail_actors(actor_ty);
        if std::env::var("AGORA_DEBUG_RATES").is_ok() && !reject.is_empty() {
            eprintln!(
                "[rates]   adversary `{}`: excluding {} heavy-tail actors of `{}` from recruitment",
                proc.intent, reject.len(), proc.actor
            );
        }
        let c_eff = (proc.camouflage * 2.0 * difficulty).clamp(0.0, 1.0);
        let cascade_p = (proc.cascade_p * control.cascade).clamp(0.0, 1.0);
        let max_extra = (n_campaigns as f64 * MAX_CASCADE_FRACTION).ceil() as u64;

        let mut extra_spawned = 0u64; // cascade children count
        for ci in 0..n_campaigns {
            let mut rng = stream(params.seed, StreamPurpose::Campaign, pi as u64, ci);
            let mut camp = plan_campaign(
                params, world, control, pi, proc, intent, c_eff, difficulty, actor_start, n_actors, span_s,
                &ring_sampler, &reject, &mut rng,
            )?;
            expected_events += campaign_expected_events(&camp, params.epoch_unix, span_s);
            // Cascade (axis 5): the finished campaign seeds a follow-up
            // starting near its end, same neighborhood.
            if extra_spawned < max_extra && rng.gen::<f64>() < cascade_p {
                let mut crng =
                    stream(params.seed, StreamPurpose::Campaign, 1000 + pi as u64, ci);
                let mut child = plan_campaign(
                    params, world, control, pi, proc, intent, c_eff, difficulty, actor_start, n_actors, span_s,
                    &ring_sampler, &reject, &mut crng,
                )?;
                expected_events += campaign_expected_events(&child, params.epoch_unix, span_s);
                child.id = next_id;
                child.cascade = true;
                child.community = community_of(&control.placement, n_actors, actor_start, &child.members);
                records.push(campaign_record(&child, &proc.intent));
                next_id += 1;
                campaigns.push(child);
                extra_spawned += 1;
            }
            camp.id = next_id;
            camp.community = community_of(&control.placement, n_actors, actor_start, &camp.members);
            records.push(campaign_record(&camp, &proc.intent));
            next_id += 1;
            campaigns.push(camp);
        }
    }

    // --- failure incidents ---------------------------------------------------
    for (fi, proc) in rb.failures.iter().enumerate() {
        let intent = 1 + rb.adversaries.len() as u16 + fi as u16;
        let actor_ty = world.entities.index_of(&proc.actor);
        let n_actors = world.entities.counts[actor_ty];
        let actor_start = world.entities.starts[actor_ty];
        let widx = rb.adversaries.len() + fi;
        let n_affected =
            (prevalence * n_actors as f64 * weights[widx]).round().max(1.0) as u64;
        let dur_sampler = DistSampler::compile(&proc.duration_days)
            .with_context(|| format!("failure `{}` duration", proc.intent))?;
        let mode = compile_failure_mode(&proc.mode, world)
            .with_context(|| format!("failure `{}`", proc.intent))?;
        let incidents_per_entity =
            (proc.rate_per_year * params.span_days / 365.0).max(1e-9);
        let cascade_p = (proc.cascade_p * control.cascade).clamp(0.0, 1.0);
        // Skeleton relation used for failure propagation (first with matching
        // src type), if any.
        let neighbor_rel = rb
            .relations
            .iter()
            .position(|r| r.src == proc.actor && r.dst == proc.actor);

        for ai in 0..n_affected {
            let mut rng = stream(params.seed, StreamPurpose::Failure, fi as u64, ai);
            let entity = actor_start + place_node(&control.placement, n_actors, &mut rng);
            let n_inc = sample_count(&mut rng, incidents_per_entity);
            for _ in 0..n_inc {
                let (t0, t1) = place_window(
                    &control.placement,
                    params.epoch_unix,
                    span_s,
                    (dur_sampler.sample(&mut rng).max(0.01) * 86_400.0) as i64,
                    &mut rng,
                );
                let id = next_id;
                next_id += 1;
                let span = FailureSpan { id, start_s: t0, end_s: t1, intent, mode: mode.clone() };
                failures_by_actor.entry(entity).or_default().push(span.clone());
                records.push(failure_record(&span, &proc.intent, entity, false));
                // Cascade: propagate to a skeleton neighbor with delay.
                if rng.gen::<f64>() < cascade_p {
                    if let Some(ri) = neighbor_rel {
                        let neigh = world.skeletons[ri].neighbors_of(entity);
                        if !neigh.is_empty() {
                            let nb = neigh[rng.gen_range(0..neigh.len())];
                            let delay = (t1 - t0) / 4;
                            let child_id = next_id;
                            next_id += 1;
                            let child = FailureSpan {
                                id: child_id,
                                start_s: t0 + delay,
                                end_s: t1 + delay,
                                ..span
                            };
                            records.push(failure_record(&child, &proc.intent, nb, true));
                            failures_by_actor.entry(nb).or_default().push(child);
                        }
                    }
                }
            }
        }
    }

    // Calibration cap: keep anomalous events a minority of the DELIVERED edges.
    // Failures (RateShift/corruption) ride the normal stream and are bounded
    // by it, so only campaign volume is scaled here.
    //
    // The cap must bound the anomalous fraction of the graph the run actually
    // emits, so it is solved against the delivered total, not the nominal
    // target: `sim::run` scales the normal stream by `(target − A)/target`, so
    // planning campaign volume A the expected delivered total is
    //     D(A) = A + (1 − A/target)·normal_full.
    // Setting A/D(A) = MAX_ANOMALY_FRACTION and solving for A gives `cap`.
    // (With normal_full = target this reduces to the old MAX·target.)
    let target = params.target_edges as f64;
    let normal_full = crate::sim::expected_normal_delivered(params, world);
    let cap = MAX_ANOMALY_FRACTION * normal_full
        / (1.0 - MAX_ANOMALY_FRACTION + MAX_ANOMALY_FRACTION * normal_full / target);

    // A safety rail, not the mechanism: with budget-elastic stage rates the
    // planned volume is set by the domain's activity multipliers and normally
    // sits well under this, so the rail does not bind. It only catches
    // pathological configs (risk R1).
    let rate_scale = if expected_events > cap && expected_events > 0.0 {
        cap / expected_events
    } else {
        1.0
    };

    if std::env::var("AGORA_DEBUG_CAP").is_ok() {
        eprintln!(
            "[cap-dbg] prevalence={prevalence} expected_events={expected_events:.1} cap={cap:.1} rate_scale={rate_scale:.4} campaigns={} bind={}",
            campaigns.len(),
            rate_scale < 1.0
        );
    }

    // The cap scaled campaign rates; reflect that in the recorded camouflage?
    // No — camouflage is a difficulty attribute, independent of the volume cap.
    Ok(AnomalyPlan {
        campaigns,
        failures_by_actor,
        records,
        expected_events: expected_events * rate_scale,
        rate_scale,
        homophily: control.homophily,
    })
}

#[allow(clippy::too_many_arguments)]
fn plan_campaign(
    params: &GenParams,
    world: &World,
    control: &agora_rules::ControlParams,
    pi: usize,
    proc: &agora_rules::AdversaryProcess,
    intent: u16,
    c_eff: f64,
    difficulty: f64,
    actor_start: u64,
    n_actors: u64,
    span_s: i64,
    ring_sampler: &DistSampler,
    reject: &[u64],
    rng: &mut Rng64,
) -> anyhow::Result<Campaign> {
    let ring = (ring_sampler.sample(rng).round().max(2.0) as usize).min(MAX_RING.min(n_actors as usize));
    // Placement (axis 4): recruit within one community block, rejecting
    // heavy-tail (super-hub) actors so the ring is ordinary/dormant accounts.
    // Bounded retries keep this deterministic and terminating even if a whole
    // community block were heavy-tail; the recruitable population is the vast
    // majority (only the thin tail is rejected), so retries are rare.
    const RECRUIT_RETRIES: u32 = 32;
    let mut members = Vec::with_capacity(ring);
    for _ in 0..ring {
        let mut cand = actor_start + place_node(&control.placement, n_actors, rng);
        for _ in 0..RECRUIT_RETRIES {
            if reject.binary_search(&cand).is_err() {
                break;
            }
            cand = actor_start + place_node(&control.placement, n_actors, rng);
        }
        members.push(cand);
    }
    members.dedup();

    // Total campaign duration to place its start.
    let mut total_dur_s = 0i64;
    let mut stage_durs = Vec::with_capacity(proc.stages.len());
    for s in &proc.stages {
        let d = (DistSampler::compile(&s.duration_days)?.sample(rng).max(0.05) * 86_400.0) as i64;
        stage_durs.push(d);
        total_dur_s += d;
    }
    let (mut cursor, _) =
        place_window(&control.placement, params.epoch_unix, span_s, total_dur_s, rng);

    let mut stages = Vec::with_capacity(proc.stages.len());
    for (si, s) in proc.stages.iter().enumerate() {
        let ev = world
            .events
            .iter()
            .position(|e| e.name == s.event)
            .ok_or_else(|| anyhow::anyhow!("adversary `{}`: unknown event `{}`", proc.intent, s.event))?;
        // Stage rate, normalised to a single currency: k, the adversary's
        // activity RATIO to a normal actor of its type. All three spec forms
        // reduce to k, so the difficulty rule and the calibration cap below see
        // one uniform quantity.
        //
        //   * `activity_multiplier` IS k. Budget-elastic: it rides
        //     `world.rate_scale` exactly like normal behavior, so the
        //     adversary/normal ratio is invariant to `--edges`.
        //   * `event_count` is a total per member for the stage, so the implied
        //     rate is count/duration and k = count/(normal_rate·duration).
        //     Budget-INELASTIC by construction — that is the point: a typology
        //     states "N deposits", not "N/day relative to whatever density the
        //     budget happens to imply".
        //   * `rate_per_day` is the legacy absolute form; k = rate/normal_rate.
        //
        // Recovering k for the inelastic forms is what lets difficulty still
        // close their volume leak (see below) instead of leaving them loud.
        let normal_rate = world.normal_rate_per_actor(world.entities.index_of(&proc.actor));
        let dur_days = (stage_durs[si] as f64 / 86_400.0).max(1e-9);
        let k = match (&s.activity_multiplier, &s.event_count, &s.rate_per_day) {
            (Some(m), None, None) => DistSampler::compile(m)?.sample(rng).max(0.0),
            (None, Some(c), None) => {
                let count = DistSampler::compile(c)?.sample(rng).max(0.0);
                // rate = count/duration  =>  k = rate/normal_rate
                let denom = normal_rate * dur_days;
                if denom > 0.0 { count / denom } else { 0.0 }
            }
            (None, None, Some(r)) => {
                let abs = DistSampler::compile(r)?.sample(rng).max(0.0);
                if normal_rate > 0.0 { abs / normal_rate } else { 0.0 }
            }
            (None, None, None) => anyhow::bail!(
                "adversary `{}` stage `{}`: no rate specified (set exactly one of \
                 activity_multiplier, event_count, rate_per_day)",
                proc.intent,
                s.name
            ),
            _ => anyhow::bail!(
                "adversary `{}` stage `{}`: set exactly ONE of activity_multiplier, \
                 event_count, rate_per_day",
                proc.intent,
                s.name
            ),
        };
        // Difficulty (axis 2) drives the VOLUME leak to zero — but ONE-SIDED:
        // difficulty may only ever move an adversary's volume DOWN toward the
        // normal rate, never up.
        //
        // k is the adversary's activity RATIO to a normal actor. For a LOUDER-
        // than-normal adversary (k > 1) the volume signal a rate-based detector
        // reads is ln k > 0, and we shrink it linearly in difficulty d:
        //     ln k_eff = (1 − d)·ln k     <=>     k_eff = k^(1−d)
        // At d = 0 the playbook's own k stands; at d = 1, k_eff = 1 exactly —
        // the adversary is as active as a normal actor and NO volume signal
        // remains.
        //
        // For a QUIETER-than-normal adversary (k < 1) we DO NOT touch k: the
        // old symmetric transform k^(1−d) drove such k UP toward 1 as d rose,
        // which INCREASES a quiet adversary's volume and thereby EXPOSES it —
        // the exact opposite of what the difficulty knob is for. Concretely,
        // every crypto stage has k ~ 0.001-0.09 (sub-1 vs the loud normal EOA
        // rate), so the symmetric form made rising difficulty explode anomalous
        // edge volume (~55x) and unmask the still-uncamouflaged attribute and
        // structure signals — the P2-b inversion (docs/ENGINE_ISSUES.md). We
        // therefore leave k < 1 stages at their playbook rate and let ATTRIBUTE
        // camouflage (axis 1) do the hiding for them. This does not weaken the
        // knob where it already worked: finance stages with k > 1 still damp to
        // 1, and finance's sub-1 stages were never the ones difficulty needed to
        // quiet (their loudness is not a rate excess).
        //
        // One-sided: damps toward k = 1 from above, fixed point at k = 1,
        // monotone (non-increasing volume) in d, never amplifies.
        let k_eff = if k > 1.0 { k.powf(1.0 - difficulty) } else { k };
        let rate = k_eff * normal_rate;
        let mut overrides = Vec::new();
        for ov in &s.attr_overrides {
            let col = world
                .attr_names
                .iter()
                .position(|a| a == &ov.attr)
                .ok_or_else(|| {
                    anyhow::anyhow!(
                        "adversary `{}` stage `{}`: unknown attr `{}`",
                        proc.intent,
                        s.name,
                        ov.attr
                    )
                })?;
            overrides.push((col, DistSampler::compile(&ov.dist)?));
        }
        let dst_ty = world.events[ev].dst_type;
        let same_type = dst_ty == world.entities.index_of(&proc.actor);
        let scope = match &s.scope {
            StageScope::Ring => PlannedScope::Ring,
            StageScope::Chain => PlannedScope::Chain,
            StageScope::Collector => {
                // Anchor the fan-in on a real dst-type node: a ring member if
                // the event is same-type, else a dedicated dst-type node
                // placed in the same community.
                let anchor = if same_type && !members.is_empty() {
                    members[0]
                } else {
                    world.entities.starts[dst_ty]
                        + place_node(&control.placement, world.entities.counts[dst_ty], rng)
                };
                PlannedScope::Collector(anchor)
            }
            StageScope::Hub => PlannedScope::Hub,
            StageScope::Normal => PlannedScope::Normal(normal_counterparty(world, ev)),
            StageScope::Victims { count } => {
                let n = DistSampler::compile(count)?.sample(rng).round().max(1.0) as usize;
                let dst_ty = world.events[ev].dst_type;
                let (vs, vn) =
                    (world.entities.starts[dst_ty], world.entities.counts[dst_ty]);
                let pool = (0..n.min(MAX_RING))
                    .map(|_| vs + rng.gen_range(0..vn))
                    .collect();
                PlannedScope::Victims(pool)
            }
            // Fan-in mirror of Victims: the pool is drawn from the event's SRC
            // type, because these nodes are the SOURCE of the emitted edge and
            // the member is the destination.
            StageScope::Sources { count } => {
                let n = DistSampler::compile(count)?.sample(rng).round().max(1.0) as usize;
                let src_ty = world.events[ev].src_type;
                let (ps, pn) =
                    (world.entities.starts[src_ty], world.entities.counts[src_ty]);
                let pool = (0..n.min(MAX_RING))
                    .map(|_| ps + rng.gen_range(0..pn))
                    .collect();
                PlannedScope::Sources(pool)
            }
        };
        let end = cursor + stage_durs[si];
        stages.push(PlannedStage {
            start_s: cursor,
            end_s: end,
            rate_per_day: rate,
            event: ev,
            scope,
            overrides,
        });
        cursor = end;
    }
    let _ = pi;
    // id/community/cascade are stamped by the caller (it owns the counter).
    Ok(Campaign {
        id: -1,
        intent,
        members,
        stages,
        camouflage: c_eff,
        community: -1,
        cascade: false,
        // Global axis: copied uniformly to every campaign (never scaled by
        // difficulty), so anomalous actors cluster at real illicit levels.
        homophily: control.homophily,
    })
}

/// The members that ACT in a stage — i.e. the ones the simulator loops over,
/// each drawing its own Poisson event count. `sim::gen_campaign_window` uses
/// THIS function for its loop, so the calibration cap (which sizes planned
/// volume via `acting_members`) can never drift out of lockstep with the volume
/// the simulator actually produces.
///
/// Two scopes act on a subset:
///   * `Collector` whose anchor IS `members[0]` (same-type event, so the fan-in
///     target is itself a ring member): that member RECEIVES, so the other
///     `n-1` act. A heterogeneous Collector anchors on a dedicated dst-type
///     node and every member acts.
///   * `Hub`: the operator `members[0]` is the SOLE actor — a Ponzi operator
///     makes every payout itself; the investors do not pay anyone.
pub(crate) fn acting_slice<'a>(c: &'a Campaign, s: &PlannedStage) -> &'a [u64] {
    let n = c.members.len();
    match s.scope {
        PlannedScope::Collector(anchor) if n > 1 && anchor == c.members[0] => &c.members[1..],
        PlannedScope::Hub if n > 1 => &c.members[..1],
        _ => &c.members,
    }
}

/// Number of members that ACT (emit) in a stage; see `acting_slice`.
pub(crate) fn acting_members(c: &Campaign, s: &PlannedStage) -> usize {
    acting_slice(c, s).len()
}

/// The slice of a stage that falls inside `[lo, hi)` — i.e. the only part the
/// simulator can ever emit into. `None` when the stage misses the interval.
///
/// This is the time-domain twin of [`acting_slice`]: `sim::gen_campaign_window`
/// clips each stage to the window it is generating, and the planner clips to
/// the whole run span. Both go through THIS function so planned volume cannot
/// drift out of lockstep with delivered volume.
///
/// It matters because [`place_window`] deliberately schedules a campaign past
/// the span end when the campaign is longer than the span itself (it clamps the
/// start, not the end). Counting those unreachable tails is what made the
/// healthcare estimate over-count by ~3x.
pub(crate) fn stage_window(s: &PlannedStage, lo: i64, hi: i64) -> Option<(i64, i64)> {
    let t0 = s.start_s.max(lo);
    let t1 = s.end_s.min(hi);
    (t0 < t1).then_some((t0, t1))
}

/// Expected events from a campaign over a run of `span_s` seconds from
/// `epoch_unix` — counting only the stage time the simulator can reach.
fn campaign_expected_events(c: &Campaign, epoch_unix: i64, span_s: i64) -> f64 {
    c.stages
        .iter()
        .filter_map(|s| {
            let (t0, t1) = stage_window(s, epoch_unix, epoch_unix + span_s)?;
            let days = (t1 - t0) as f64 / 86_400.0;
            Some(s.rate_per_day * days * acting_members(c, s) as f64)
        })
        .sum()
}

/// Community block index of a node set under a clustered placement (-1 if the
/// placement isn't community-clustered).
fn community_of(placement: &Placement, n_actors: u64, actor_start: u64, members: &[u64]) -> i64 {
    let n_comm = match placement {
        Placement::Clustered { n_communities }
        | Placement::ClusteredBursty { n_communities, .. } => *n_communities as u64,
        _ => return -1,
    };
    let Some(&first) = members.first() else { return -1 };
    let c = n_comm.clamp(1, n_actors.max(1));
    let block = (n_actors / c).max(1);
    (((first - actor_start) / block).min(c - 1)) as i64
}

/// Capped sample of member ids (full ground truth is the simulator's; the
/// JSON record stays bounded for very large rings).
fn member_sample(members: &[u64], cap: usize) -> Vec<u64> {
    members.iter().take(cap).copied().collect()
}

fn campaign_record(c: &Campaign, intent_name: &str) -> AnomalyRecord {
    let start = c.stages.iter().map(|s| s.start_s).min().unwrap_or(0);
    let end = c.stages.iter().map(|s| s.end_s).max().unwrap_or(0);
    AnomalyRecord {
        id: c.id,
        intent: intent_name.to_string(),
        kind: "adversary".into(),
        camouflage: c.camouflage,
        members: member_sample(&c.members, 256),
        n_members: c.members.len() as u64,
        community: c.community,
        start_t: start,
        end_t: end,
        cascade: c.cascade,
    }
}

fn failure_record(s: &FailureSpan, intent_name: &str, entity: u64, cascade: bool) -> AnomalyRecord {
    AnomalyRecord {
        id: s.id,
        intent: intent_name.to_string(),
        kind: "failure".into(),
        camouflage: f64::NAN, // failures are not adversarially camouflaged
        members: vec![entity],
        n_members: 1,
        community: -1,
        start_t: s.start_s,
        end_t: s.end_s,
        cascade,
    }
}

/// The actor's normal counterparty policy for this event (degrading
/// repeat-partner memory to plain neighbor picks — campaign actors have no
/// behavior-local memory slots).
fn normal_counterparty(world: &World, ev: usize) -> CompiledCounterparty {
    for b in &world.behaviors {
        for em in &b.emissions {
            if em.event == ev {
                return match &em.counterparty {
                    CompiledCounterparty::RepeatOrNeighbor { relation, .. }
                    | CompiledCounterparty::Neighbor { relation } => {
                        CompiledCounterparty::Neighbor { relation: *relation }
                    }
                    CompiledCounterparty::GlobalUniform { entity } => {
                        CompiledCounterparty::GlobalUniform { entity: *entity }
                    }
                    CompiledCounterparty::GlobalPopularity { entity, pop } => {
                        CompiledCounterparty::GlobalPopularity { entity: *entity, pop: *pop }
                    }
                };
            }
        }
    }
    CompiledCounterparty::GlobalUniform { entity: world.events[ev].dst_type }
}

fn compile_failure_mode(mode: &FailureMode, world: &World) -> anyhow::Result<PlannedFailureMode> {
    let col_of = |attr: &str| -> anyhow::Result<usize> {
        world
            .attr_names
            .iter()
            .position(|a| a == attr)
            .ok_or_else(|| anyhow::anyhow!("unknown attr `{attr}` in failure mode"))
    };
    let ev_of = |event: &str| -> anyhow::Result<usize> {
        world
            .events
            .iter()
            .position(|e| e.name == event)
            .ok_or_else(|| anyhow::anyhow!("unknown event `{event}` in failure mode"))
    };
    Ok(match mode {
        FailureMode::Silence => PlannedFailureMode::Silence,
        FailureMode::RateShift { factor } => PlannedFailureMode::RateShift(*factor),
        FailureMode::StuckAttr { event, attr, value } => {
            PlannedFailureMode::StuckAttr(ev_of(event)?, col_of(attr)?, *value)
        }
        FailureMode::DriftAttr { event, attr, drift_per_day } => {
            PlannedFailureMode::DriftAttr(ev_of(event)?, col_of(attr)?, *drift_per_day)
        }
        FailureMode::NoiseAttr { event, attr, dist } => {
            PlannedFailureMode::NoiseAttr(ev_of(event)?, col_of(attr)?, DistSampler::compile(dist)?)
        }
    })
}

/// Pick a node offset honoring the placement axis (clustered = contiguous
/// community blocks of the actor id range).
fn place_node(placement: &Placement, n_actors: u64, rng: &mut Rng64) -> u64 {
    match placement {
        Placement::Clustered { n_communities }
        | Placement::ClusteredBursty { n_communities, .. } => {
            let c = (*n_communities as u64).clamp(1, n_actors.max(1));
            let block = n_actors / c;
            let chosen = rng.gen_range(0..c);
            chosen * block + rng.gen_range(0..block.max(1))
        }
        _ => rng.gen_range(0..n_actors.max(1)),
    }
}

/// Pick a start so [start, start+dur] honors the time-window placement.
fn place_window(
    placement: &Placement,
    epoch: i64,
    span_s: i64,
    dur_s: i64,
    rng: &mut Rng64,
) -> (i64, i64) {
    let (lo_f, hi_f) = match placement {
        Placement::TimeWindow { start_frac, end_frac }
        | Placement::ClusteredBursty { start_frac, end_frac, .. } => (*start_frac, *end_frac),
        _ => (0.0, 1.0),
    };
    let lo = epoch + (span_s as f64 * lo_f.clamp(0.0, 1.0)) as i64;
    let hi = epoch + (span_s as f64 * hi_f.clamp(0.0, 1.0)) as i64;
    let latest_start = (hi - dur_s).max(lo);
    let start = if latest_start > lo { lo + rng.gen_range(0..(latest_start - lo) as u64) as i64 } else { lo };
    (start, start + dur_s)
}

fn sample_count(rng: &mut Rng64, mean: f64) -> u64 {
    use rand_distr::Distribution as _;
    if mean <= 0.0 {
        return 0;
    }
    if mean < 1.0 {
        // Bernoulli for sub-unit means (at least the chance of one incident).
        return u64::from(rng.gen::<f64>() < mean);
    }
    rand_distr::Poisson::new(mean).map(|p| p.sample(rng) as u64).unwrap_or(0)
}

/// Apply the CLI/runtime control-axis overrides onto the rule-base defaults.
/// Axis 3 (type-mix), 4 (placement community count) and 5 (cascade) can each
/// be set per-run without editing the rule base; unset axes keep their domain
/// defaults. Prevalence (1) and difficulty (2) are applied at their use sites.
fn effective_control(base: &ControlParams, params: &GenParams) -> ControlParams {
    let mut c = base.clone();

    // Axis 5: cascade multiplier.
    if let Some(v) = params.anomaly_cascade {
        c.cascade = v.max(0.0);
    }

    // Axis 3: type-mix overrides (replace/add per intent).
    for (intent, weight) in &params.anomaly_type_mix {
        match c.type_mix.iter_mut().find(|tw| &tw.intent == intent) {
            Some(tw) => tw.weight = *weight,
            None => c.type_mix.push(TypeWeight { intent: intent.clone(), weight: *weight }),
        }
    }

    // Axis 4: override the clustered-community count, preserving the placement
    // family (uniform stays uniform; a time window keeps its window).
    if let Some(n) = params.anomaly_communities {
        let n = n.max(1);
        c.placement = match c.placement {
            Placement::Uniform | Placement::Clustered { .. } => {
                Placement::Clustered { n_communities: n }
            }
            Placement::TimeWindow { start_frac, end_frac }
            | Placement::ClusteredBursty { start_frac, end_frac, .. } => {
                Placement::ClusteredBursty { n_communities: n, start_frac, end_frac }
            }
        };
    }
    c
}
