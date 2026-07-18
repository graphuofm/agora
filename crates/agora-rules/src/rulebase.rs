//! The rule base: one domain = one instantiation of the seven primitives
//! (blueprint §5). This is the *compiled, static* config the engine consumes —
//! authored by hand for the 6 built-ins, or synthesized by the offline RAG
//! (§9) for a natural-language domain. No LLM ever runs after this is built.
//!
//! Schema-encoding decision (§19): strongly-typed serde structs serialized as
//! YAML (human-editable, diffable) with JSON interchange for the RAG's
//! grammar-constrained decoding. Every variant the engine supports appears
//! here as a typed enum case — never a free-form expression in the hot path.

use std::collections::HashSet;

use serde::{Deserialize, Serialize};
use thiserror::Error;

// ---------------------------------------------------------------------------
// Top level
// ---------------------------------------------------------------------------

/// A complete domain rule base: the seven primitives plus provenance.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RuleBase {
    pub meta: RuleBaseMeta,
    /// Primitive 1 — typed node classes with state + attributes.
    pub entity_types: Vec<EntityType>,
    /// Primitive 2 — who CAN interact: pluggable evolution models (§8).
    pub relations: Vec<RelationRule>,
    /// Primitive 3 — interactions that emit temporal edges.
    pub event_types: Vec<EventType>,
    /// Primitive 4 — normal agent dynamics.
    pub behaviors: Vec<BehaviorProcess>,
    /// Primitive 5 — legality predicates Φ (a detector's-eye view; labels
    /// anchor to intent, not to these).
    #[serde(default)]
    pub constraints: Vec<Constraint>,
    /// Primitive 6a — intentional anomaly sources (label = intent).
    #[serde(default)]
    pub adversaries: Vec<AdversaryProcess>,
    /// Primitive 6b — unintentional/emergent anomaly sources.
    #[serde(default)]
    pub failures: Vec<FailureProcess>,
    /// Primitive 7 — the five calibrated control axes (§3), domain defaults.
    pub control: ControlParams,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RuleBaseMeta {
    /// Stable identifier, e.g. "finance".
    pub id: String,
    pub name: String,
    pub description: String,
    /// Rule-base schema version (bump on breaking schema change).
    pub schema_version: u32,
    /// Where these rules come from: standards, papers, datasets (§9 grounding).
    #[serde(default)]
    pub provenance: Vec<Provenance>,
}

/// Traceability of a rule to an authoritative source (CORPUS.md).
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Provenance {
    pub source: String,
    #[serde(default)]
    pub url: Option<String>,
    #[serde(default)]
    pub version: Option<String>,
    /// CORPUS.md license tier: "A" | "B" | "C".
    #[serde(default)]
    pub license_tier: Option<String>,
}

// ---------------------------------------------------------------------------
// Primitive 1: EntityType
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct EntityType {
    pub name: String,
    /// Relative share of the total node budget (normalized across types).
    pub population_weight: f64,
    /// Immutable-at-birth attributes (may be graded/hierarchical).
    #[serde(default)]
    pub attributes: Vec<AttributeDef>,
    /// Mutable simulation state variables.
    #[serde(default)]
    pub state: Vec<StateVar>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AttributeDef {
    pub name: String,
    pub kind: AttributeKind,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AttributeKind {
    /// Unordered categories with sampling weights.
    Categorical { values: Vec<String>, weights: Vec<f64> },
    /// Ordered tiers (graded attribute, e.g. risk_tier or kyc_level).
    Ordinal { tiers: Vec<String>, weights: Vec<f64> },
    /// Numeric value drawn from a distribution.
    Numeric { dist: Distribution },
    /// Taxonomy-valued: a path in a hierarchy, e.g. a MITRE tactic→technique.
    Taxonomy { paths: Vec<String>, weights: Vec<f64> },
    /// Boolean flag with probability of `true`.
    Flag { p: f64 },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct StateVar {
    pub name: String,
    pub init: Distribution,
}

/// The closed set of distributions the samplers implement (agora-sample).
/// Closed and typed on purpose: the RAG can only emit these, so a drafted rule
/// base can never smuggle an unimplementable distribution into the engine.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Distribution {
    Constant { value: f64 },
    Uniform { min: f64, max: f64 },
    Normal { mean: f64, std: f64 },
    /// Heavy-tailed workhorse (amounts, durations, volumes).
    LogNormal { mu: f64, sigma: f64 },
    Exponential { rate: f64 },
    /// Power-law tail (Pareto type I).
    Pareto { scale: f64, shape: f64 },
    /// Zipf over ranks 1..=n.
    Zipf { n: u64, exponent: f64 },
    Poisson { lambda: f64 },
}

// ---------------------------------------------------------------------------
// Primitive 2: Relation (the skeleton; §8 pluggable evolution models)
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RelationRule {
    pub name: String,
    /// Source / target entity-type names.
    pub src: String,
    pub dst: String,
    pub model: TopologyModel,
    /// Anatomical layer this relation belongs to (§5 layered view).
    #[serde(default)]
    pub layer: Layer,
    /// Average skeleton degree per source node (drives edge budget split).
    pub mean_degree: f64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Layer {
    /// Stable structural backbone.
    #[default]
    Skeleton,
    /// High-traffic flow/interaction edges.
    Vessels,
    /// Functional repeated-interaction edges built over time.
    Muscle,
    /// Peripheral/surface edges.
    Skin,
}

/// Pluggable structure/evolution models (§8). All O(1)-per-edge generable.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TopologyModel {
    /// Barabási–Albert preferential attachment; `m` edges per arriving node.
    PreferentialAttachment { m: u32 },
    /// Erdős–Rényi G(n, p)-style uniform random matching.
    UniformRandom,
    /// Watts–Strogatz small world: ring degree `k`, rewire prob `beta`.
    SmallWorld { k: u32, beta: f64 },
    /// Forest-fire growth (natural-growth family).
    ForestFire { forward_p: f64, backward_p: f64 },
    /// Stochastic block model: `communities` blocks, in/out densities.
    Sbm { communities: u32, p_in_weight: f64, p_out_weight: f64 },
    /// R-MAT/Kronecker recursive quadrant probabilities (a+b+c+d = 1).
    RMat { a: f64, b: f64, c: f64, d: f64 },
    /// Geometric proximity in a unit square; connect within `radius`-ish
    /// neighborhoods (grid-bucketed for O(1) candidate lookup).
    Spatial { radius: f64 },
    /// Bipartite affiliation: dst-side popularity follows `popularity`.
    Affiliation { popularity: Distribution },
}

// ---------------------------------------------------------------------------
// Primitive 3: EventType
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct EventType {
    pub name: String,
    pub src: String,
    pub dst: String,
    /// Edge attribute schema; each attribute sampled per event (may be
    /// overridden by behavior/adversary context).
    #[serde(default)]
    pub attributes: Vec<AttributeDef>,
    /// Deterministic effects applied to entity state when the event fires.
    #[serde(default)]
    pub effects: Vec<StateEffect>,
}

/// State updates the engine applies atomically with the event.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum StateEffect {
    /// dst_state[var] += amount_attr (e.g. balance += amount on transfer).
    AddToDst { var: String, from_attr: String },
    /// src_state[var] += amount_attr (e.g. balance += amount on deposit).
    AddToSrc { var: String, from_attr: String },
    /// src_state[var] -= amount_attr.
    SubFromSrc { var: String, from_attr: String },
    /// dst_state[var] -= amount_attr.
    SubFromDst { var: String, from_attr: String },
    /// src_state[var] += 1 (counters: txn_count, review_count…).
    IncrementSrc { var: String },
    IncrementDst { var: String },
    /// Set a state var to a constant on both endpoints.
    SetSrc { var: String, value: f64 },
    SetDst { var: String, value: f64 },
}

// ---------------------------------------------------------------------------
// Primitive 4: BehaviorProcess (normal dynamics)
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct BehaviorProcess {
    pub name: String,
    /// Which entity type acts.
    pub actor: String,
    /// Restrict the behavior to actors matching all filters (e.g. only
    /// `type == business` accounts). Empty = all actors of the type.
    #[serde(default)]
    pub actor_filter: Vec<AttrFilter>,
    /// WHEN: per-actor event arrival process.
    pub timing: TimingModel,
    /// WHAT + WHOM: mix of event emissions (weights normalized), each with
    /// its own counterparty policy (the per-event hot operation, §8).
    pub events: Vec<EventEmission>,
    /// Heterogeneity: per-actor activity multiplier (species variation, §5).
    #[serde(default = "default_activity")]
    pub activity: Distribution,
}

fn default_activity() -> Distribution {
    Distribution::LogNormal { mu: 0.0, sigma: 0.5 }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AttrFilter {
    pub attr: String,
    pub value: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct EventEmission {
    pub event: String,
    pub weight: f64,
    pub counterparty: CounterpartyModel,
}

/// Arrival process: base Poisson rate modulated by seasonality + bursts.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TimingModel {
    /// Mean events per actor per day (before modulation).
    pub rate_per_day: f64,
    /// 24-hourly multipliers (diurnal pattern); empty = flat.
    #[serde(default)]
    pub diurnal: Vec<f64>,
    /// 7 weekday multipliers, Monday first; empty = flat.
    #[serde(default)]
    pub weekly: Vec<f64>,
    /// Burstiness: probability that an event triggers a follow-up burst,
    /// geometric burst length (heavy-tailed inter-arrivals à la real traces).
    #[serde(default)]
    pub burst_p: f64,
    #[serde(default)]
    pub burst_mean_len: f64,
    /// Hawkes self-excitation branching ratio n ∈ [0,1) (Hawkes & Oakes 1974
    /// cluster representation). 0 = use the simple geometric burst above; n > 0
    /// replaces it with a RECURSIVE cascade: each event spawns Poisson(n)
    /// children, each of which spawns its own — endogenous "active stays active"
    /// clustering that raises burstiness toward real interaction data (measured
    /// gap: real B≈0.7–0.9 vs Poisson+burst ≈0.4–0.6). Subcritical (n<1) so the
    /// cascade terminates; mean cluster size = 1/(1−n).
    #[serde(default)]
    pub branching_ratio: f64,
    /// Mean delay (seconds) of a Hawkes child after its parent (the excitation
    /// kernel timescale, 1/β). Defaults to 300 s.
    #[serde(default)]
    pub excitation_decay_s: f64,
    /// The per-actor arrival process. Default = inhomogeneous Poisson (thinned
    /// by diurnal/weekly). `Weibull { shape }` makes inter-event times
    /// heavy-tailed/bursty (Barabási 2005; Unicomb et al. 2021): shape = 1 is
    /// exponential (Poisson-like), shape < 1 is bursty (burstiness B rises),
    /// shape > 1 is more regular. The mean rate is preserved either way, so the
    /// edge budget is unaffected.
    #[serde(default)]
    pub arrival: ArrivalKind,
}

#[derive(Debug, Clone, Copy, PartialEq, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ArrivalKind {
    /// Inhomogeneous Poisson via thinning (exponential inter-event times).
    #[default]
    Poisson,
    /// Weibull renewal process; `shape` < 1 → bursty, = 1 → exponential.
    Weibull { shape: f64 },
}

/// Counterparty selection — must be O(1) amortized per event (§8, risk R1).
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CounterpartyModel {
    /// Pick uniformly among the actor's skeleton neighbors in `relation`.
    SkeletonNeighbor { relation: String },
    /// Repeat-partner bias: with prob `repeat_p` reuse a recent counterparty
    /// (muscle-building), else sample a skeleton neighbor.
    RepeatOrNeighbor { relation: String, repeat_p: f64 },
    /// Global popularity-proportional pick over an entity type (alias table).
    GlobalPopularity { entity: String },
    /// Uniform over an entity type.
    GlobalUniform { entity: String },
}

// ---------------------------------------------------------------------------
// Primitive 5: Constraint Φ
// ---------------------------------------------------------------------------

/// Legality predicates. Violations are *candidate* anomalies from a
/// detector's point of view; ground-truth labels still anchor to intent.
/// Encoded as a closed set of checkable forms (no expression language in the
/// hot path; complex checks run post-hoc in `agora validate`).
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Constraint {
    pub name: String,
    pub check: ConstraintCheck,
    #[serde(default)]
    pub description: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ConstraintCheck {
    /// Event attribute must lie in [min, max].
    AttrRange { event: String, attr: String, min: f64, max: f64 },
    /// A state var must stay non-negative (e.g. balance).
    StateNonNegative { entity: String, var: String },
    /// At most `k` events of `event` per src within a sliding `window_s`.
    RateLimit { event: String, k: u32, window_s: u64 },
    /// Count of events with attr in [`floor`, `threshold`) per src within
    /// `window_s` must be < `k` (structuring detection, CTR-style: only
    /// NEAR-threshold amounts are suspicious, hence the floor).
    SubThresholdCount {
        event: String,
        attr: String,
        threshold: f64,
        #[serde(default)]
        floor: f64,
        k: u32,
        window_s: u64,
    },
    /// Free-text check evaluated only by `agora validate` (documented, not
    /// engine-enforced).
    Documented { rule: String },
}

// ---------------------------------------------------------------------------
// Primitive 6: AdversaryProcess / FailureProcess
// ---------------------------------------------------------------------------

/// Intentional anomaly source. Events it emits are LABELED BY INTENT — the
/// generative cause — not by a post-hoc rule check (§3, §5).
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AdversaryProcess {
    /// Intent label written on every event this process causes
    /// (e.g. "structuring", "wash_trading", "lateral_movement").
    pub intent: String,
    pub description: String,
    /// Which entity type the adversary recruits/controls.
    pub actor: String,
    /// Strategy: a staged policy automaton (cross-domain parameterization).
    pub stages: Vec<Stage>,
    /// 0 = blatant template, 1 = maximal mimicry of normal behavior.
    /// THE difficulty lever (§3 axis 2): blends adversary parameters toward
    /// the actor's normal BehaviorProcess.
    pub camouflage: f64,
    /// Relative share of the anomaly budget this typology receives
    /// (normalized across processes; scaled by the global anomaly rate).
    pub prevalence_weight: f64,
    /// How many controlled actors participate per campaign instance.
    pub ring_size: Distribution,
    /// Cascade: probability a finished campaign recruits a neighbor campaign.
    #[serde(default)]
    pub cascade_p: f64,
}

/// One stage of an adversary campaign: emit events for `duration` days,
/// choosing counterparties by `scope`.
///
/// The stage's event rate is given EITHER as `activity_multiplier` (preferred)
/// or as `rate_per_day` (legacy). They differ in how they react to the run's
/// edge budget:
///
///   * `rate_per_day` is an ABSOLUTE events/day per acting member. Normal
///     behavior is not absolute — `world.rate_scale` rescales every normal
///     rate so the run hits `--edges`. So an absolute adversary rate makes the
///     adversary/normal activity ratio a function of the edge budget: the same
///     campaign emits the same COUNT at `--edges 200000` and `--edges
///     2000000`, i.e. 14.2% vs 1.4% of the graph. The anomaly rate then is not
///     a property of the domain, it is an artifact of the budget.
///   * `activity_multiplier` is a MULTIPLE of what a normal actor of this
///     adversary's `actor` type emits per day. It rides `rate_scale` exactly
///     like normal behavior, so the adversary/normal activity ratio — the
///     thing that actually decides detectability — is an explicit, invariant
///     property of the rule base. `k = 1` means "as active as a normal actor"
///     (no volume signal at all).
///   * `event_count` is a COUNT of events per acting member for the WHOLE
///     stage; `duration_days` then only shapes WHEN they land, not how many.
///     This is the form typologies are actually written in — "a smurf ring
///     makes N deposits below the reporting threshold", "a scanner probes N
///     hosts". It matters because the source standards legislate counts,
///     thresholds and code-sets but deliberately NOT cadence: 31 CFR
///     1010.100(xx) defines structuring as "one or more transactions in
///     currency, in any amount ... on one or more days" — rate-free on
///     purpose. A count is therefore the only form much of the corpus can
///     ground at all.
///
///     TRADE-OFF (state it, do not hide it): a count is budget-INELASTIC like
///     `rate_per_day`, so as `--edges` shrinks the adversary's share of the
///     graph grows and the VOLUME leak reopens. What makes it acceptable here
///     — and what plain `rate_per_day` never had — is that it is converted to
///     an `activity_multiplier` internally (`k = count / (normal_rate ·
///     duration)`), so the difficulty axis still closes that leak: at d = 1,
///     `k_eff = 1` and the adversary emits normal-actor volume regardless of
///     how the stage was declared. Counts give a faithful typology at d = 0;
///     difficulty still buys indistinguishability at d = 1.
///
/// Exactly one of the three must be set.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Stage {
    pub name: String,
    pub duration_days: Distribution,
    /// Events/day per acting member, as a multiple of the actor type's normal
    /// per-actor rate. Budget-elastic.
    #[serde(default)]
    pub activity_multiplier: Option<Distribution>,
    /// Total events per acting member over the whole stage. Budget-INELASTIC,
    /// but still difficulty-damped (see the type docs). Preferred when the
    /// typology states a count rather than a cadence — which is the usual case.
    #[serde(default)]
    pub event_count: Option<Distribution>,
    /// Absolute events/day per acting member. Budget-INELASTIC; legacy.
    #[serde(default)]
    pub rate_per_day: Option<Distribution>,
    /// Event type emitted in this stage.
    pub event: String,
    /// Counterparty scope within the stage.
    pub scope: StageScope,
    /// Attribute overrides, e.g. amount: sub-threshold log-normal.
    #[serde(default)]
    pub attr_overrides: Vec<AttrOverride>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum StageScope {
    /// Within the recruited ring (wash trading, round-tripping).
    Ring,
    /// Ring members in a chain/cycle order (layering hops).
    Chain,
    /// One designated collector node (fan-in: phishing, smurf aggregation).
    /// Members are the SOURCE; the collector is the DESTINATION.
    Collector,
    /// Star fan-OUT from the ring's operator: `members[0]` is the only actor
    /// and pays/serves the other members (Ponzi payouts, franchise payroll,
    /// C2 tasking). The dual of `Collector`: operator -> members, and — unlike
    /// `Ring` — members never transact with each other, so the motif is a STAR
    /// rather than a clique. Same-type events only.
    Hub,
    /// Normal counterparties (integration/mimicry stages).
    Normal,
    /// Random victims outside the ring (scans, spam, fake reviews).
    /// Members are the SOURCE; each victim is the DESTINATION.
    Victims { count: Distribution },
    /// Fan-IN from outside the ring: a pre-sampled pool of counterparties of
    /// the event's SRC type pays INTO the acting member. The mirror of
    /// `Victims`: here each member is the DESTINATION and the drawn
    /// counterparty is the SOURCE. This is what "receive"/"drain"/"deposit"
    /// stages mean — money flows toward the adversary (mule receipts,
    /// phishing drains, collector inflow).
    Sources { count: Distribution },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AttrOverride {
    pub attr: String,
    pub dist: Distribution,
}

/// Unintentional/emergent anomaly source (sensor fault, congestion, outage).
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FailureProcess {
    /// Label, e.g. "sensor_fault", "congestion_cascade".
    pub intent: String,
    pub description: String,
    /// Entity type that fails.
    pub actor: String,
    pub mode: FailureMode,
    pub prevalence_weight: f64,
    /// Mean failures per affected entity per simulated year.
    pub rate_per_year: f64,
    pub duration_days: Distribution,
    /// Contagion: probability the failure propagates to a skeleton neighbor.
    #[serde(default)]
    pub cascade_p: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum FailureMode {
    /// Entity stops emitting events (dropout/outage).
    Silence,
    /// An event attribute is corrupted: stuck at a constant value.
    StuckAttr { event: String, attr: String, value: f64 },
    /// An event attribute drifts by `drift_per_day` (sensor drift).
    DriftAttr { event: String, attr: String, drift_per_day: f64 },
    /// Activity multiplies by `factor` (congestion = slowdown, surge = spike).
    RateShift { factor: f64 },
    /// Attribute distribution replaced (impossible values).
    NoiseAttr { event: String, attr: String, dist: Distribution },
}

// ---------------------------------------------------------------------------
// Primitive 7: ControlParams (the five axes, §3)
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ControlParams {
    /// Axis 1 — fraction of nodes participating in any anomalous process.
    pub prevalence: f64,
    /// Axis 2 — global difficulty in [0,1]; scales every process's camouflage.
    pub difficulty: f64,
    /// Axis 3 — type mix: intent → weight (overrides per-process
    /// prevalence_weight when present).
    #[serde(default)]
    pub type_mix: Vec<TypeWeight>,
    /// Axis 4 — placement policy.
    pub placement: Placement,
    /// Axis 5 — global cascade multiplier applied to per-process cascade_p.
    pub cascade: f64,
    /// Fidelity axis — anomaly homophily in [0,1]. With this probability a
    /// campaign counterparty that is not already marked is redirected to a
    /// random marked actor of the same entity type, so anomalous actors
    /// preferentially transact with each other (matching real illicit graphs).
    /// Default 0.0 preserves the pre-fix behavior. Global: copied uniformly to
    /// every campaign, never scaled by difficulty.
    #[serde(default)]
    pub homophily: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TypeWeight {
    pub intent: String,
    pub weight: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Placement {
    /// Recruit actors uniformly at random.
    Uniform,
    /// Concentrate in `n_communities` skeleton communities.
    Clustered { n_communities: u32 },
    /// Concentrate campaigns in a time window [start_frac, end_frac] of the
    /// simulated span (bursty placement).
    TimeWindow { start_frac: f64, end_frac: f64 },
    /// Clustered in space AND time.
    ClusteredBursty { n_communities: u32, start_frac: f64, end_frac: f64 },
}

// ---------------------------------------------------------------------------
// Validation
// ---------------------------------------------------------------------------

#[derive(Debug, Error)]
pub enum RuleBaseError {
    #[error("rule base `{0}`: {1}")]
    Invalid(String, String),
}

/// Validate a distribution's parameters are finite and in the sampler's valid
/// range, so the engine can compile it without panicking (untrusted config,
/// §6). Returns the error message body (the caller wraps it with the rule-base
/// id). Mirrors the checks in `agora-sample::DistSampler::compile`.
fn check_dist(d: &Distribution, ctx: &str) -> Result<(), String> {
    let bad = |m: &str| Err(format!("{ctx}: {m}"));
    let fin = |x: f64| x.is_finite();
    match *d {
        Distribution::Constant { value } if !fin(value) => bad("constant value must be finite"),
        Distribution::Uniform { min, max } if !(fin(min) && fin(max) && max >= min) => {
            bad("uniform needs finite min ≤ max")
        }
        Distribution::Normal { mean, std } if !(fin(mean) && fin(std) && std >= 0.0) => {
            bad("normal needs finite mean and std ≥ 0")
        }
        Distribution::LogNormal { mu, sigma } if !(fin(mu) && fin(sigma) && sigma >= 0.0) => {
            bad("log_normal needs finite mu and sigma ≥ 0")
        }
        Distribution::Exponential { rate } if !(fin(rate) && rate > 0.0) => {
            bad("exponential rate must be finite > 0")
        }
        Distribution::Pareto { scale, shape } if !(fin(scale) && fin(shape) && scale > 0.0 && shape > 0.0) => {
            bad("pareto needs finite scale > 0 and shape > 0")
        }
        Distribution::Zipf { n, exponent } if n == 0 || !(fin(exponent) && exponent > 0.0) => {
            bad("zipf needs n ≥ 1 and finite exponent > 0")
        }
        Distribution::Poisson { lambda } if !(fin(lambda) && lambda > 0.0) => {
            bad("poisson lambda must be finite > 0")
        }
        _ => Ok(()),
    }
}

impl RuleBase {
    pub fn from_yaml(s: &str) -> anyhow::Result<RuleBase> {
        let rb: RuleBase = serde_yaml::from_str(s)?;
        rb.validate()?;
        Ok(rb)
    }

    pub fn to_yaml(&self) -> anyhow::Result<String> {
        Ok(serde_yaml::to_string(self)?)
    }

    /// Structural consistency lint: every name referenced anywhere must be
    /// defined; weights/probabilities in range. This is the validation gate
    /// the RAG's drafts must pass (§9 step 5) and the loader always runs.
    pub fn validate(&self) -> Result<(), RuleBaseError> {
        let err = |m: String| Err(RuleBaseError::Invalid(self.meta.id.clone(), m));

        if self.entity_types.is_empty() {
            return err("at least one entity type is required".into());
        }
        let entities: HashSet<&str> =
            self.entity_types.iter().map(|e| e.name.as_str()).collect();
        if entities.len() != self.entity_types.len() {
            return err("duplicate entity type names".into());
        }
        let _events: HashSet<&str> = self.event_types.iter().map(|e| e.name.as_str()).collect();
        let _relations: HashSet<&str> = self.relations.iter().map(|r| r.name.as_str()).collect();

        for e in &self.entity_types {
            if e.population_weight <= 0.0 {
                return err(format!("entity `{}`: population_weight must be > 0", e.name));
            }
        }
        for r in &self.relations {
            for side in [&r.src, &r.dst] {
                if !entities.contains(side.as_str()) {
                    return err(format!("relation `{}`: unknown entity `{side}`", r.name));
                }
            }
            if r.mean_degree <= 0.0 {
                return err(format!("relation `{}`: mean_degree must be > 0", r.name));
            }
            if let TopologyModel::RMat { a, b, c, d } = r.model {
                if (a + b + c + d - 1.0).abs() > 1e-6 {
                    return err(format!("relation `{}`: R-MAT a+b+c+d must equal 1", r.name));
                }
            }
        }
        for ev in &self.event_types {
            for side in [&ev.src, &ev.dst] {
                if !entities.contains(side.as_str()) {
                    return err(format!("event `{}`: unknown entity `{side}`", ev.name));
                }
            }
        }
        for b in &self.behaviors {
            if !entities.contains(b.actor.as_str()) {
                return err(format!("behavior `{}`: unknown actor `{}`", b.name, b.actor));
            }
            if b.events.is_empty() {
                return err(format!("behavior `{}`: needs at least one event", b.name));
            }
            for we in &b.events {
                let Some(ev) = self.event_types.iter().find(|e| e.name == we.event) else {
                    return err(format!("behavior `{}`: unknown event `{}`", b.name, we.event));
                };
                // The acting entity emits the event, so the event's src type
                // must be the behavior's actor type.
                if ev.src != b.actor {
                    return err(format!(
                        "behavior `{}`: actor `{}` cannot emit event `{}` (its src is `{}`)",
                        b.name, b.actor, we.event, ev.src
                    ));
                }
                match &we.counterparty {
                    CounterpartyModel::SkeletonNeighbor { relation }
                    | CounterpartyModel::RepeatOrNeighbor { relation, .. } => {
                        let Some(rel) = self.relations.iter().find(|r| &r.name == relation) else {
                            return err(format!(
                                "behavior `{}` event `{}`: unknown relation `{relation}`",
                                b.name, we.event
                            ));
                        };
                        // Neighbor lookup is keyed by the actor's id, so the
                        // relation's src type must be the actor type.
                        if rel.src != b.actor {
                            return err(format!(
                                "behavior `{}` event `{}`: relation `{relation}` has src `{}`, \
                                 but the actor is `{}` (a behavior can only follow neighbors of a \
                                 relation rooted at its own type)",
                                b.name, we.event, rel.src, b.actor
                            ));
                        }
                    }
                    CounterpartyModel::GlobalPopularity { entity }
                    | CounterpartyModel::GlobalUniform { entity } => {
                        if !entities.contains(entity.as_str()) {
                            return err(format!(
                                "behavior `{}` event `{}`: unknown entity `{entity}`",
                                b.name, we.event
                            ));
                        }
                    }
                }
            }
            if let ArrivalKind::Weibull { shape } = b.timing.arrival {
                if !(shape.is_finite() && shape > 0.0) {
                    return err(format!("behavior `{}`: weibull shape must be finite > 0", b.name));
                }
            }
            if !(b.timing.branching_ratio.is_finite() && (0.0..1.0).contains(&b.timing.branching_ratio)) {
                return err(format!("behavior `{}`: branching_ratio must be in [0,1)", b.name));
            }
            if !b.timing.diurnal.is_empty() && b.timing.diurnal.len() != 24 {
                return err(format!("behavior `{}`: diurnal must have 24 entries", b.name));
            }
            if !b.timing.weekly.is_empty() && b.timing.weekly.len() != 7 {
                return err(format!("behavior `{}`: weekly must have 7 entries", b.name));
            }
        }
        for a in &self.adversaries {
            if !entities.contains(a.actor.as_str()) {
                return err(format!("adversary `{}`: unknown actor `{}`", a.intent, a.actor));
            }
            if !(0.0..=1.0).contains(&a.camouflage) {
                return err(format!("adversary `{}`: camouflage must be in [0,1]", a.intent));
            }
            if a.stages.is_empty() {
                return err(format!("adversary `{}`: needs at least one stage", a.intent));
            }
            for s in &a.stages {
                let Some(ev) = self.event_types.iter().find(|e| e.name == s.event) else {
                    return err(format!(
                        "adversary `{}` stage `{}`: unknown event `{}`",
                        a.intent, s.name, s.event
                    ));
                };
                // Which END of the edge is the campaign member? Every scope but
                // `sources` puts the member at the SRC; `sources` is the
                // fan-in mirror and puts it at the DST.
                if matches!(s.scope, StageScope::Sources { .. }) {
                    // Fan-in: outsiders (event's src type) pay INTO the member,
                    // so the member — and hence the actor type — is the DST.
                    if ev.dst != a.actor {
                        return err(format!(
                            "adversary `{}` stage `{}`: sources scope makes actor `{}` the \
                             DESTINATION, but event `{}` goes {} -> {} (its dst is `{}`)",
                            a.intent, s.name, a.actor, s.event, ev.src, ev.dst, ev.dst
                        ));
                    }
                } else if ev.src != a.actor {
                    // Campaign members (of the actor type) emit the event, so
                    // the event's src type must be the actor type.
                    return err(format!(
                        "adversary `{}` stage `{}`: actor `{}` cannot emit event `{}` (its src is `{}`)",
                        a.intent, s.name, a.actor, s.event, ev.src
                    ));
                }
                // ring/chain/hub are homogeneous-ring patterns: the event's src
                // and dst entity types must match, else counterparty selection
                // would target the wrong population (collector/victims/sources
                // handle heterogeneous events).
                if matches!(s.scope, StageScope::Ring | StageScope::Chain | StageScope::Hub)
                    && ev.src != ev.dst
                {
                    return err(format!(
                        "adversary `{}` stage `{}`: ring/chain/hub scope needs a same-type event, \
                         but `{}` goes {} -> {} (use collector, victims or sources instead)",
                        a.intent, s.name, s.event, ev.src, ev.dst
                    ));
                }
            }
        }
        for f in &self.failures {
            if !entities.contains(f.actor.as_str()) {
                return err(format!("failure `{}`: unknown actor `{}`", f.intent, f.actor));
            }
        }
        let c = &self.control;
        if !(0.0..=1.0).contains(&c.prevalence) {
            return err("control.prevalence must be in [0,1]".into());
        }
        if !(0.0..=1.0).contains(&c.difficulty) {
            return err("control.difficulty must be in [0,1]".into());
        }
        let known_intents: HashSet<&str> = self
            .adversaries
            .iter()
            .map(|a| a.intent.as_str())
            .chain(self.failures.iter().map(|f| f.intent.as_str()))
            .collect();
        for tw in &c.type_mix {
            if !known_intents.contains(tw.intent.as_str()) {
                return err(format!("control.type_mix: unknown intent `{}`", tw.intent));
            }
            if !(tw.weight.is_finite() && tw.weight >= 0.0) {
                return err(format!("control.type_mix `{}`: weight must be finite ≥ 0", tw.intent));
            }
        }
        if let Placement::Clustered { n_communities }
        | Placement::ClusteredBursty { n_communities, .. } = c.placement
        {
            if n_communities == 0 {
                return err("control.placement: n_communities must be ≥ 1".into());
            }
        }

        // Hardening for untrusted configs (§6): every weight vector and every
        // distribution must be well-formed, so the engine's samplers and alias
        // tables can never panic at world-build time. Fail at LOAD with an
        // actionable message instead.
        for et in &self.entity_types {
            for a in &et.attributes {
                self.check_attr_kind(&a.kind, &format!("entity `{}` attr `{}`", et.name, a.name))?;
            }
            for sv in &et.state {
                check_dist(&sv.init, &format!("entity `{}` state `{}`", et.name, sv.name))
                    .map_err(|m| RuleBaseError::Invalid(self.meta.id.clone(), m))?;
            }
        }
        for ev in &self.event_types {
            for a in &ev.attributes {
                self.check_attr_kind(&a.kind, &format!("event `{}` attr `{}`", ev.name, a.name))?;
            }
        }
        for b in &self.behaviors {
            check_dist(&b.activity, &format!("behavior `{}` activity", b.name))
                .map_err(|m| RuleBaseError::Invalid(self.meta.id.clone(), m))?;
            let mut wsum = 0.0;
            for we in &b.events {
                if !(we.weight.is_finite() && we.weight >= 0.0) {
                    return err(format!("behavior `{}` event `{}`: weight must be finite ≥ 0", b.name, we.event));
                }
                wsum += we.weight;
            }
            if wsum <= 0.0 {
                return err(format!("behavior `{}`: event weights sum to 0", b.name));
            }
        }
        for r in &self.relations {
            if let TopologyModel::Affiliation { popularity } = &r.model {
                check_dist(popularity, &format!("relation `{}` popularity", r.name))
                    .map_err(|m| RuleBaseError::Invalid(self.meta.id.clone(), m))?;
            }
        }
        for a in &self.adversaries {
            check_dist(&a.ring_size, &format!("adversary `{}` ring_size", a.intent))
                .map_err(|m| RuleBaseError::Invalid(self.meta.id.clone(), m))?;
            for s in &a.stages {
                check_dist(&s.duration_days, &format!("adversary `{}` stage `{}` duration_days", a.intent, s.name))
                    .map_err(|m| RuleBaseError::Invalid(self.meta.id.clone(), m))?;
                // Exactly one of the three rate forms (see `Stage` docs).
                let forms: [(&str, &Option<Distribution>); 3] = [
                    ("activity_multiplier", &s.activity_multiplier),
                    ("event_count", &s.event_count),
                    ("rate_per_day", &s.rate_per_day),
                ];
                let set: Vec<&str> =
                    forms.iter().filter(|(_, d)| d.is_some()).map(|(n, _)| *n).collect();
                match set.len() {
                    1 => {}
                    0 => {
                        return err(format!(
                            "adversary `{}` stage `{}`: needs one of `event_count` (total events \
                             per member for the stage — the form most typologies state), \
                             `activity_multiplier` (events/day as a multiple of the actor type's \
                             normal rate), or `rate_per_day` (absolute, legacy)",
                            a.intent, s.name
                        ))
                    }
                    _ => {
                        return err(format!(
                            "adversary `{}` stage `{}`: set exactly ONE rate form, found {}",
                            a.intent,
                            s.name,
                            set.join(" + ")
                        ))
                    }
                }
                for (name, d) in forms.iter().filter(|(_, d)| d.is_some()) {
                    check_dist(
                        d.as_ref().unwrap(),
                        &format!("adversary `{}` stage `{}` {}", a.intent, s.name, name),
                    )
                    .map_err(|m| RuleBaseError::Invalid(self.meta.id.clone(), m))?
                }
                for ov in &s.attr_overrides {
                    check_dist(&ov.dist, &format!("adversary `{}` stage `{}` override `{}`", a.intent, s.name, ov.attr))
                        .map_err(|m| RuleBaseError::Invalid(self.meta.id.clone(), m))?;
                }
            }
        }
        for f in &self.failures {
            check_dist(&f.duration_days, &format!("failure `{}` duration_days", f.intent))
                .map_err(|m| RuleBaseError::Invalid(self.meta.id.clone(), m))?;
            if !(f.rate_per_year.is_finite() && f.rate_per_year >= 0.0) {
                return err(format!("failure `{}`: rate_per_year must be finite ≥ 0", f.intent));
            }
        }
        Ok(())
    }

    /// Validate one attribute kind: weight vectors must be same-length as
    /// values, finite, non-negative, and sum to a positive total (so the
    /// engine's alias tables never panic); numeric kinds get a distribution
    /// check; flags get a probability check.
    fn check_attr_kind(&self, kind: &AttributeKind, ctx: &str) -> Result<(), RuleBaseError> {
        let err = |m: String| Err(RuleBaseError::Invalid(self.meta.id.clone(), m));
        match kind {
            AttributeKind::Categorical { values, weights }
            | AttributeKind::Ordinal { tiers: values, weights }
            | AttributeKind::Taxonomy { paths: values, weights } => {
                if values.is_empty() {
                    return err(format!("{ctx}: needs at least one value"));
                }
                if values.len() != weights.len() {
                    return err(format!(
                        "{ctx}: {} values but {} weights",
                        values.len(),
                        weights.len()
                    ));
                }
                let mut sum = 0.0;
                for w in weights {
                    if !(w.is_finite() && *w >= 0.0) {
                        return err(format!("{ctx}: weight must be finite ≥ 0"));
                    }
                    sum += *w;
                }
                if sum <= 0.0 {
                    return err(format!("{ctx}: weights sum to 0"));
                }
            }
            AttributeKind::Numeric { dist } => {
                check_dist(dist, ctx).map_err(|m| RuleBaseError::Invalid(self.meta.id.clone(), m))?;
            }
            AttributeKind::Flag { p } => {
                if !(p.is_finite() && (0.0..=1.0).contains(p)) {
                    return err(format!("{ctx}: flag probability must be in [0,1]"));
                }
            }
        }
        Ok(())
    }
}
