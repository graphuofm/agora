//! World construction: RuleBase + scale -> typed populations with sampled
//! attributes/state, materialized skeletons, and compiled behaviors.
//!
//! Everything random here derives from `(seed, NodeInit/Topology/ActorStatic,
//! …)` streams, so world construction is bit-reproducible and (where columnar)
//! parallelizable without changing output.

use std::collections::HashMap;

use anyhow::Context;
use rand::Rng;
use agora_rules::{
    AttributeKind, BehaviorProcess, CounterpartyModel, EventType, RuleBase, StateEffect,
};
use agora_sample::{stream, AliasTable, DistSampler, StreamPurpose};
use agora_topology::{build_relation, RelationGraph};

use crate::api::{GenParams, NodeBatch, NodeColumn};

/// Global id layout: entity type i owns ids [starts[i], starts[i]+counts[i]).
pub struct EntityIndex {
    pub names: Vec<String>,
    pub starts: Vec<u64>,
    pub counts: Vec<u64>,
}

impl EntityIndex {
    pub fn type_of(&self, id: u64) -> usize {
        // Few entity types: linear scan beats binary search in practice.
        for i in (0..self.starts.len()).rev() {
            if id >= self.starts[i] {
                return i;
            }
        }
        0
    }

    pub fn index_of(&self, name: &str) -> usize {
        self.names.iter().position(|n| n == name).expect("validated name")
    }
}

/// One sampled per-node attribute column.
pub enum AttrColumn {
    Numeric(Vec<f64>),
    Category { codes: Vec<u16>, names: Vec<String> },
}

/// Compiled per-event-attribute sampler writing into a union column.
pub struct CompiledAttr {
    /// Index into the union attr columns.
    pub col: usize,
    pub sampler: AttrSampler,
}

pub enum AttrSampler {
    Numeric(DistSampler),
    /// Categorical/ordinal/taxonomy: alias over local value indices, then
    /// remapped into the column's union dictionary (different event types may
    /// define different value sets for the same attr name).
    Category { table: AliasTable, code_map: Vec<u16> },
    Flag(f64),
}

/// Union categorical dictionaries per event-attribute name, in first-seen
/// order. The single source of truth shared by the engine (emission codes)
/// and the writers (decoding) — both must agree bit-for-bit.
pub fn union_attr_dictionaries(rb: &RuleBase) -> Vec<(String, Vec<String>)> {
    let mut dicts: Vec<(String, Vec<String>)> = Vec::new();
    for et in &rb.event_types {
        for a in &et.attributes {
            let values = match &a.kind {
                AttributeKind::Categorical { values, .. } => values,
                AttributeKind::Ordinal { tiers, .. } => tiers,
                AttributeKind::Taxonomy { paths, .. } => paths,
                _ => continue,
            };
            let entry = match dicts.iter_mut().find(|(n, _)| n == &a.name) {
                Some((_, e)) => e,
                None => {
                    dicts.push((a.name.clone(), Vec::new()));
                    &mut dicts.last_mut().expect("just pushed").1
                }
            };
            for v in values {
                if !entry.contains(v) {
                    entry.push(v.clone());
                }
            }
        }
    }
    dicts
}

/// Compiled state effect with resolved indices.
pub struct CompiledEffect {
    pub kind: EffectKind,
    /// (entity_type_idx, state_var_idx) of the touched variable.
    pub entity: usize,
    pub var: usize,
    /// Union attr column the amount comes from (usize::MAX for counters).
    pub from_col: usize,
    /// Constant for Set effects.
    pub value: f64,
    /// true = applies to src endpoint, false = dst.
    pub on_src: bool,
}

#[derive(Clone, Copy, PartialEq)]
pub enum EffectKind {
    Add,
    Sub,
    Increment,
    Set,
}

/// Compiled event type: union-attr samplers + effects.
pub struct CompiledEvent {
    pub name: String,
    pub src_type: usize,
    pub dst_type: usize,
    pub attrs: Vec<CompiledAttr>,
    pub effects: Vec<CompiledEffect>,
}

pub enum CompiledCounterparty {
    /// Uniform among skeleton neighbors of `relation`.
    Neighbor { relation: usize },
    /// With prob `repeat_p` reuse one of the K remembered partners.
    RepeatOrNeighbor { relation: usize, repeat_p: f64 },
    /// Uniform global pick over an entity type.
    GlobalUniform { entity: usize },
    /// Popularity-proportional global pick over an entity type — O(1) via a
    /// precomputed Vose alias table weighted by the node's total skeleton
    /// in-degree (§8: hubs receive disproportionately). `pop` indexes
    /// `World::popularity`.
    GlobalPopularity { entity: usize, pop: usize },
}

pub struct CompiledEmission {
    pub event: usize,
    pub counterparty: CompiledCounterparty,
}

pub struct CompiledBehavior {
    pub name: String,
    /// Entity-type index this behavior's actors are drawn from.
    pub actor_type: usize,
    /// Global actor ids participating in this behavior (after filters).
    pub actors: Vec<u64>,
    /// Per-actor activity multiplier, parallel to `actors`.
    pub activity: Vec<f32>,
    pub rate_per_day: f64,
    /// 24 diurnal multipliers normalized to mean 1 (flat = all 1).
    pub diurnal: [f64; 24],
    pub diurnal_max: f64,
    /// 7 weekday multipliers normalized to mean 1.
    pub weekly: [f64; 7],
    pub burst_p: f64,
    pub burst_mean_len: f64,
    /// Hawkes self-excitation branching ratio (0 = simple geometric burst).
    pub branching_ratio: f64,
    /// Hawkes child-delay mean in seconds.
    pub excitation_decay_s: f64,
    /// Weibull arrival shape (None = inhomogeneous Poisson). The candidate
    /// COUNT stays Poisson(lam) (budget-exact); shape only reshapes the
    /// within-window inter-event spacing. shape < 1 → bursty.
    pub weibull_shape: Option<f64>,
    pub emissions: Vec<CompiledEmission>,
    pub emission_alias: AliasTable,
    /// Σ activity (for closed-form calibration).
    pub activity_sum: f64,
}

pub struct World {
    pub entities: EntityIndex,
    /// Per entity type: attribute columns (parallel to rule base attrs).
    pub node_attrs: Vec<Vec<AttrColumn>>,
    /// Per entity type: attribute names (parallel to `node_attrs`).
    pub attr_rule_names: Vec<Vec<String>>,
    /// Per entity type: state variable columns.
    pub state: Vec<Vec<Vec<f64>>>,
    pub state_names: Vec<Vec<String>>,
    pub skeletons: Vec<RelationGraph>,
    pub events: Vec<CompiledEvent>,
    pub behaviors: Vec<CompiledBehavior>,
    /// Union event-attribute schema.
    pub attr_names: Vec<String>,
    /// Categorical dictionaries for union columns that are categorical.
    pub attr_dictionaries: Vec<(String, Vec<String>)>,
    /// Popularity alias tables for `GlobalPopularity` counterparty selection,
    /// one per (entity type) that any behavior selects by popularity. Weighted
    /// by total skeleton in-degree (§8 hubs). Indexed by `CompiledCounterparty
    /// ::GlobalPopularity::pop`; `.0` is the entity-type index.
    pub popularity: Vec<(usize, AliasTable)>,
    /// Events/day rate multiplier that calibrates expected totals to target.
    pub rate_scale: f64,
}

impl World {
    pub fn build(params: &GenParams) -> anyhow::Result<World> {
        let rb = &params.rulebase;
        let entities = layout_entities(rb, params.nodes);
        let node_attrs = sample_node_attrs(rb, &entities, params.seed)?;
        let (state, state_names) = init_state(rb, &entities, params.seed)?;
        let skeletons = build_skeletons(rb, &entities, params.seed)?;
        let (events, attr_names, attr_dictionaries) = compile_events(rb, &entities)?;
        // Popularity tables (in-degree-weighted) for any entity type a behavior
        // selects by GlobalPopularity. Built before behaviors so they can be
        // assigned `pop` indices.
        let (popularity, pop_index) = build_popularity(rb, &entities, &skeletons);
        let behaviors = compile_behaviors(
            rb, &entities, &node_attrs, &skeletons, &events, &pop_index, params.seed,
        )?;

        // Closed-form calibration (§3): expected events at scale 1.0, then
        // scale so the expectation hits the target edge budget.
        let span = params.span_days;
        let expected: f64 = behaviors
            .iter()
            .map(|b| {
                // Expected events per immigrant: Hawkes cluster mean 1/(1−n) if
                // self-exciting, else the geometric burst expectation.
                let burst_factor = if b.branching_ratio > 0.0 {
                    1.0 / (1.0 - b.branching_ratio)
                } else {
                    1.0 + b.burst_p * b.burst_mean_len
                };
                b.activity_sum * b.rate_per_day * span * burst_factor
            })
            .sum();
        anyhow::ensure!(
            expected > 0.0,
            "rule base `{}` produces no events (no behaviors or zero rates)",
            rb.meta.id
        );
        let rate_scale = params.target_edges as f64 / expected;

        let attr_rule_names = rb
            .entity_types
            .iter()
            .map(|e| e.attributes.iter().map(|a| a.name.clone()).collect())
            .collect();

        Ok(World {
            entities,
            node_attrs,
            attr_rule_names,
            state,
            state_names,
            skeletons,
            events,
            behaviors,
            attr_names,
            attr_dictionaries,
            popularity,
            rate_scale,
        })
    }

    /// Mean events/day a NORMAL actor of entity type `ti` emits, at the
    /// calibrated `rate_scale` — i.e. the actual per-node activity level of
    /// this run, after the edge budget has been applied.
    ///
    /// This is the yardstick adversary stage rates are expressed against
    /// (`Stage::activity_multiplier`). Because it carries `rate_scale`, an
    /// adversary rate defined as `k ×` this is budget-ELASTIC exactly like
    /// normal behavior, so `k` — the adversary/normal activity ratio, which is
    /// what makes a campaign detectable by volume alone — stays invariant to
    /// `--edges` instead of being an artifact of it.
    ///
    /// Averaged over ALL nodes of the type (not just the ones a filtered
    /// behavior covers), since that is the population an adversary recruits
    /// from and the population a detector compares against. Includes the
    /// burst/self-excitation factor and the mean activity multiplier, so it is
    /// a delivered-events/day figure, not a nominal one.
    pub fn normal_rate_per_actor(&self, ti: usize) -> f64 {
        let n = self.entities.counts[ti] as f64;
        if n <= 0.0 {
            return 0.0;
        }
        self.behaviors
            .iter()
            .filter(|b| b.actor_type == ti)
            .map(|b| {
                let burst_factor = if b.branching_ratio > 0.0 {
                    1.0 / (1.0 - b.branching_ratio)
                } else {
                    1.0 + b.burst_p * b.burst_mean_len
                };
                b.activity_sum * b.rate_per_day * burst_factor * self.rate_scale / n
            })
            .sum()
    }

    /// Global actor ids of the HEAVY-TAIL (super-hub) behavior classes for
    /// entity type `ti`: classes whose per-actor normal rate is more than an
    /// order of magnitude above the median rate of the *active* population.
    ///
    /// The premise of [`normal_rate_per_actor`] — "the population an adversary
    /// recruits from" — is false when the actor mix is heavy-tailed: for crypto
    /// the eoa type is ~88% retail (<1 tx/day) plus a thin tail of exchange hot
    /// wallets (~1300 tx/day) and MEV bots (~7700 tx/day). A fraud ring is
    /// assembled from ordinary/dormant accounts, NOT from those super-hubs — an
    /// operator cannot commandeer a real exchange's hot wallet or an arbitrage
    /// bot into a sybil cycle. When such wallets ARE recruited (uniform draw
    /// over the id range), their high-volume *normal* traffic to random
    /// counterparties swamps the campaign edges, collapsing anomaly homophily to
    /// chance and inverting the in/out-degree signature.
    ///
    /// The reference is the median rate of the *active* actors (rate > 0), NOT
    /// the whole-type mean: a large dormant sub-population (e.g. healthcare
    /// providers of specialties with no billing behavior) would drag a mean
    /// below the ordinary active classes and wrongly flag them. The median of
    /// the active bulk is the robust "typical active actor", and a super-hub is
    /// one an order of magnitude (10×) above it — the textbook heavy-tail
    /// separation. This is derived solely from the domain's declared activity
    /// rates, never from any evaluation target; and the excluded SET is
    /// insensitive to the exact factor across a wide range (crypto's tail is
    /// >1000× the bulk, so any factor in [2, 1000] excludes exactly the same
    /// two classes; domains whose classes sit within an order of magnitude of
    /// each other exclude nothing). The median-containing class always has rate
    /// ≤ median < 10·median, so the bulk is never excluded. Returned SORTED for
    /// binary-search membership in the recruiter.
    pub fn heavy_tail_actors(&self, ti: usize) -> Vec<u64> {
        /// A super-hub is > this many times the typical active actor's rate.
        /// One order of magnitude — the exclusion set is insensitive to the
        /// exact value (see doc comment), so it is not a tuned parameter.
        const HEAVY_TAIL_FACTOR: f64 = 10.0;
        let burst_of = |b: &CompiledBehavior| {
            if b.branching_ratio > 0.0 {
                1.0 / (1.0 - b.branching_ratio)
            } else {
                1.0 + b.burst_p * b.burst_mean_len
            }
        };
        // Per-actor scale-1 rate of each behavior class of this type (rate_scale
        // cancels in the ratio, so it is omitted). Classes with no actors are
        // skipped; actors covered by no behavior are dormant (rate 0) and are
        // never a super-hub.
        let mut classes: Vec<(f64, &[u64])> = Vec::new();
        for b in self.behaviors.iter().filter(|b| b.actor_type == ti) {
            let n_c = b.actors.len();
            if n_c == 0 {
                continue;
            }
            let per_actor = b.activity_sum * b.rate_per_day * burst_of(b) / n_c as f64;
            classes.push((per_actor, b.actors.as_slice()));
        }
        // Median rate over ACTIVE actors (rate > 0), weighted by class size.
        let mut active: Vec<(f64, u64)> =
            classes.iter().filter(|(r, _)| *r > 0.0).map(|(r, a)| (*r, a.len() as u64)).collect();
        if active.is_empty() {
            return Vec::new();
        }
        active.sort_by(|x, y| x.0.partial_cmp(&y.0).unwrap_or(std::cmp::Ordering::Equal));
        let total: u64 = active.iter().map(|(_, c)| c).sum();
        let mut cum = 0u64;
        let mut median = active.last().map(|(r, _)| *r).unwrap_or(0.0);
        for (r, c) in &active {
            cum += c;
            if cum * 2 >= total {
                median = *r;
                break;
            }
        }
        let bound = median * HEAVY_TAIL_FACTOR;
        let mut out = Vec::new();
        for (r, a) in &classes {
            if *r > bound {
                out.extend_from_slice(a);
            }
        }
        out.sort_unstable();
        out.dedup();
        out
    }

    /// Node tables (attributes + final state) for export.
    pub fn node_batches(&self) -> Vec<NodeBatch> {
        let mut out = Vec::new();
        for ti in 0..self.entities.names.len() {
            let n = self.entities.counts[ti] as usize;
            let start = self.entities.starts[ti];
            let mut attr_names = Vec::new();
            let mut attrs = Vec::new();
            for (ai, col) in self.node_attrs[ti].iter().enumerate() {
                attr_names.push(attr_name_of(self, ti, ai));
                attrs.push(match col {
                    AttrColumn::Numeric(v) => NodeColumn::Numeric(v.clone()),
                    AttrColumn::Category { codes, names } => {
                        NodeColumn::Category { codes: codes.clone(), names: names.clone() }
                    }
                });
            }
            for (si, sv) in self.state[ti].iter().enumerate() {
                attr_names.push(self.state_names[ti][si].clone());
                attrs.push(NodeColumn::Numeric(sv.clone()));
            }
            out.push(NodeBatch {
                entity_type: self.entities.names[ti].clone(),
                ids: (start..start + n as u64).collect(),
                attr_names,
                attrs,
            });
        }
        out
    }
}

fn attr_name_of(w: &World, ti: usize, ai: usize) -> String {
    w.attr_rule_names[ti][ai].clone()
}

// --- construction helpers ---------------------------------------------------

fn layout_entities(rb: &RuleBase, total_nodes: u64) -> EntityIndex {
    let weight_sum: f64 = rb.entity_types.iter().map(|e| e.population_weight).sum();
    let mut names = Vec::new();
    let mut starts = Vec::new();
    let mut counts = Vec::new();
    let mut cursor = 0u64;
    for (i, e) in rb.entity_types.iter().enumerate() {
        let mut c = ((e.population_weight / weight_sum) * total_nodes as f64).round() as u64;
        c = c.max(1);
        // Last type absorbs rounding drift.
        if i == rb.entity_types.len() - 1 {
            let assigned: u64 = counts.iter().sum::<u64>() + c;
            if assigned != total_nodes && total_nodes > counts.iter().sum::<u64>() {
                c = total_nodes - counts.iter().sum::<u64>();
            }
        }
        names.push(e.name.clone());
        starts.push(cursor);
        counts.push(c);
        cursor += c;
    }
    EntityIndex { names, starts, counts }
}

fn sample_node_attrs(
    rb: &RuleBase,
    entities: &EntityIndex,
    seed: u64,
) -> anyhow::Result<Vec<Vec<AttrColumn>>> {
    let mut all = Vec::new();
    for (ti, et) in rb.entity_types.iter().enumerate() {
        let n = entities.counts[ti] as usize;
        let start = entities.starts[ti];
        let mut cols = Vec::new();
        for (ai, attr) in et.attributes.iter().enumerate() {
            let col = match &attr.kind {
                AttributeKind::Numeric { dist } => {
                    let s = DistSampler::compile(dist)
                        .with_context(|| format!("entity `{}` attr `{}`", et.name, attr.name))?;
                    let mut v = vec![0.0f64; n];
                    for (i, slot) in v.iter_mut().enumerate() {
                        let mut rng =
                            stream(seed, StreamPurpose::NodeInit, start + i as u64, ai as u64);
                        *slot = s.sample(&mut rng);
                    }
                    AttrColumn::Numeric(v)
                }
                AttributeKind::Categorical { values, weights }
                | AttributeKind::Ordinal { tiers: values, weights }
                | AttributeKind::Taxonomy { paths: values, weights } => {
                    anyhow::ensure!(
                        values.len() == weights.len() && !values.is_empty(),
                        "entity `{}` attr `{}`: values/weights mismatch",
                        et.name,
                        attr.name
                    );
                    let table = AliasTable::new(weights);
                    let mut codes = vec![0u16; n];
                    for (i, slot) in codes.iter_mut().enumerate() {
                        let mut rng =
                            stream(seed, StreamPurpose::NodeInit, start + i as u64, ai as u64);
                        *slot = table.sample(&mut rng) as u16;
                    }
                    AttrColumn::Category { codes, names: values.clone() }
                }
                AttributeKind::Flag { p } => {
                    let mut codes = vec![0u16; n];
                    for (i, slot) in codes.iter_mut().enumerate() {
                        let mut rng =
                            stream(seed, StreamPurpose::NodeInit, start + i as u64, ai as u64);
                        *slot = u16::from(rng.gen::<f64>() < *p);
                    }
                    AttrColumn::Category { codes, names: vec!["false".into(), "true".into()] }
                }
            };
            cols.push(col);
        }
        all.push(cols);
    }
    Ok(all)
}

#[allow(clippy::type_complexity)]
fn init_state(
    rb: &RuleBase,
    entities: &EntityIndex,
    seed: u64,
) -> anyhow::Result<(Vec<Vec<Vec<f64>>>, Vec<Vec<String>>)> {
    let mut state = Vec::new();
    let mut names = Vec::new();
    for (ti, et) in rb.entity_types.iter().enumerate() {
        let n = entities.counts[ti] as usize;
        let start = entities.starts[ti];
        let mut vars = Vec::new();
        let mut vnames = Vec::new();
        for (si, sv) in et.state.iter().enumerate() {
            let s = DistSampler::compile(&sv.init)
                .with_context(|| format!("entity `{}` state `{}`", et.name, sv.name))?;
            let mut v = vec![0.0f64; n];
            // Salt state vars after attributes to keep streams distinct.
            let salt = 1000 + si as u64;
            for (i, slot) in v.iter_mut().enumerate() {
                let mut rng = stream(seed, StreamPurpose::NodeInit, start + i as u64, salt);
                *slot = s.sample(&mut rng);
            }
            vars.push(v);
            vnames.push(sv.name.clone());
        }
        state.push(vars);
        names.push(vnames);
    }
    Ok((state, names))
}

fn build_skeletons(
    rb: &RuleBase,
    entities: &EntityIndex,
    seed: u64,
) -> anyhow::Result<Vec<RelationGraph>> {
    rb.relations
        .iter()
        .enumerate()
        .map(|(ri, rule)| {
            let si = entities.index_of(&rule.src);
            let di = entities.index_of(&rule.dst);
            build_relation(
                rule,
                entities.starts[si],
                entities.counts[si],
                entities.starts[di],
                entities.counts[di],
                seed,
                ri as u64,
            )
        })
        .collect()
}

/// Build popularity alias tables for every entity type any behavior selects by
/// `GlobalPopularity`. Each node's weight is `1 + total skeleton in-degree`
/// (the +1 keeps zero-in-degree nodes selectable), so hubs are picked
/// proportionally more often (§8). Returns the tables plus an entity-type →
/// table-index map. O(total skeleton edges); built once.
fn build_popularity(
    rb: &RuleBase,
    entities: &EntityIndex,
    skeletons: &[RelationGraph],
) -> (Vec<(usize, AliasTable)>, HashMap<usize, usize>) {
    // Which entity types are popularity targets?
    let mut targets: Vec<usize> = rb
        .behaviors
        .iter()
        .flat_map(|b| b.events.iter())
        .filter_map(|we| match &we.counterparty {
            CounterpartyModel::GlobalPopularity { entity } => Some(entities.index_of(entity)),
            _ => None,
        })
        .collect();
    targets.sort_unstable();
    targets.dedup();

    let mut tables = Vec::new();
    let mut index = HashMap::new();
    for ety in targets {
        let start = entities.starts[ety];
        let n = entities.counts[ety] as usize;
        let mut indeg = vec![1.0f64; n]; // +1 smoothing
        for rel in skeletons {
            // Only relations whose destination is this entity type contribute.
            if rel.dst_start == start {
                for &t in &rel.csr.targets {
                    indeg[(t - start) as usize] += 1.0;
                }
            }
        }
        index.insert(ety, tables.len());
        tables.push((ety, AliasTable::new(&indeg)));
    }
    (tables, index)
}

#[allow(clippy::type_complexity)]
fn compile_events(
    rb: &RuleBase,
    entities: &EntityIndex,
) -> anyhow::Result<(Vec<CompiledEvent>, Vec<String>, Vec<(String, Vec<String>)>)> {
    // Union attr schema across event types, in first-seen order.
    let mut attr_names: Vec<String> = Vec::new();
    let dictionaries = union_attr_dictionaries(rb);
    let mut col_of = |name: &str| -> usize {
        match attr_names.iter().position(|a| a == name) {
            Some(i) => i,
            None => {
                attr_names.push(name.to_string());
                attr_names.len() - 1
            }
        }
    };

    // Pass 1: union schema + attr samplers.
    let mut events = Vec::new();
    for et in &rb.event_types {
        let src_type = entities.index_of(&et.src);
        let dst_type = entities.index_of(&et.dst);
        let mut attrs = Vec::new();
        for a in &et.attributes {
            let col = col_of(&a.name);
            let sampler = match &a.kind {
                AttributeKind::Numeric { dist } => AttrSampler::Numeric(
                    DistSampler::compile(dist)
                        .with_context(|| format!("event `{}` attr `{}`", et.name, a.name))?,
                ),
                AttributeKind::Categorical { values, weights }
                | AttributeKind::Ordinal { tiers: values, weights }
                | AttributeKind::Taxonomy { paths: values, weights } => {
                    let union = &dictionaries
                        .iter()
                        .find(|(n, _)| n == &a.name)
                        .expect("built from same rule base")
                        .1;
                    let code_map = values
                        .iter()
                        .map(|v| {
                            union.iter().position(|u| u == v).expect("union superset") as u16
                        })
                        .collect();
                    AttrSampler::Category { table: AliasTable::new(weights), code_map }
                }
                AttributeKind::Flag { p } => AttrSampler::Flag(*p),
            };
            attrs.push(CompiledAttr { col, sampler });
        }
        events.push(CompiledEvent {
            name: et.name.clone(),
            src_type,
            dst_type,
            attrs,
            effects: Vec::new(),
        });
    }
    let _ = col_of; // release the borrow of attr_names before pass 2
    // Pass 2: effects (need the complete union schema).
    for (ei, et) in rb.event_types.iter().enumerate() {
        events[ei].effects = compile_effects(rb, et, entities, &attr_names)?;
    }
    Ok((events, attr_names, dictionaries))
}

fn compile_effects(
    rb: &RuleBase,
    et: &EventType,
    entities: &EntityIndex,
    attr_names: &[String],
) -> anyhow::Result<Vec<CompiledEffect>> {
    let src_type = entities.index_of(&et.src);
    let dst_type = entities.index_of(&et.dst);
    let state_idx = |ty: usize, var: &str| -> anyhow::Result<usize> {
        rb.entity_types[ty]
            .state
            .iter()
            .position(|s| s.name == var)
            .ok_or_else(|| {
                anyhow::anyhow!(
                    "event `{}`: state var `{var}` not defined on entity `{}`",
                    et.name,
                    rb.entity_types[ty].name
                )
            })
    };
    let col_idx = |attr: &str| -> anyhow::Result<usize> {
        attr_names
            .iter()
            .position(|a| a == attr)
            .ok_or_else(|| anyhow::anyhow!("event `{}`: unknown attr `{attr}` in effect", et.name))
    };
    let mut out = Vec::new();
    for eff in &et.effects {
        let c = match eff {
            StateEffect::AddToDst { var, from_attr } => CompiledEffect {
                kind: EffectKind::Add,
                entity: dst_type,
                var: state_idx(dst_type, var)?,
                from_col: col_idx(from_attr)?,
                value: 0.0,
                on_src: false,
            },
            StateEffect::AddToSrc { var, from_attr } => CompiledEffect {
                kind: EffectKind::Add,
                entity: src_type,
                var: state_idx(src_type, var)?,
                from_col: col_idx(from_attr)?,
                value: 0.0,
                on_src: true,
            },
            StateEffect::SubFromSrc { var, from_attr } => CompiledEffect {
                kind: EffectKind::Sub,
                entity: src_type,
                var: state_idx(src_type, var)?,
                from_col: col_idx(from_attr)?,
                value: 0.0,
                on_src: true,
            },
            StateEffect::SubFromDst { var, from_attr } => CompiledEffect {
                kind: EffectKind::Sub,
                entity: dst_type,
                var: state_idx(dst_type, var)?,
                from_col: col_idx(from_attr)?,
                value: 0.0,
                on_src: false,
            },
            StateEffect::IncrementSrc { var } => CompiledEffect {
                kind: EffectKind::Increment,
                entity: src_type,
                var: state_idx(src_type, var)?,
                from_col: usize::MAX,
                value: 1.0,
                on_src: true,
            },
            StateEffect::IncrementDst { var } => CompiledEffect {
                kind: EffectKind::Increment,
                entity: dst_type,
                var: state_idx(dst_type, var)?,
                from_col: usize::MAX,
                value: 1.0,
                on_src: false,
            },
            StateEffect::SetSrc { var, value } => CompiledEffect {
                kind: EffectKind::Set,
                entity: src_type,
                var: state_idx(src_type, var)?,
                from_col: usize::MAX,
                value: *value,
                on_src: true,
            },
            StateEffect::SetDst { var, value } => CompiledEffect {
                kind: EffectKind::Set,
                entity: dst_type,
                var: state_idx(dst_type, var)?,
                from_col: usize::MAX,
                value: *value,
                on_src: false,
            },
        };
        out.push(c);
    }
    Ok(out)
}

#[allow(clippy::too_many_arguments)]
fn compile_behaviors(
    rb: &RuleBase,
    entities: &EntityIndex,
    node_attrs: &[Vec<AttrColumn>],
    skeletons: &[RelationGraph],
    events: &[CompiledEvent],
    pop_index: &HashMap<usize, usize>,
    seed: u64,
) -> anyhow::Result<Vec<CompiledBehavior>> {
    let rel_idx: HashMap<&str, usize> = rb
        .relations
        .iter()
        .enumerate()
        .map(|(i, r)| (r.name.as_str(), i))
        .collect();
    let event_idx: HashMap<&str, usize> =
        events.iter().enumerate().map(|(i, e)| (e.name.as_str(), i)).collect();

    let mut out = Vec::new();
    for (bi, b) in rb.behaviors.iter().enumerate() {
        let ti = entities.index_of(&b.actor);
        let actors = filter_actors(b, ti, entities, &rb.entity_types[ti], node_attrs)?;
        // Activity multipliers (heterogeneity), deterministic per actor.
        let act_sampler = DistSampler::compile(&b.activity)
            .with_context(|| format!("behavior `{}` activity", b.name))?;
        let mut activity = Vec::with_capacity(actors.len());
        let mut activity_sum = 0.0f64;
        for &a in &actors {
            let mut rng = stream(seed, StreamPurpose::ActorStatic, a, bi as u64);
            let v = act_sampler.sample(&mut rng).max(0.0);
            activity.push(v as f32);
            activity_sum += v;
        }

        let mut diurnal = [1.0f64; 24];
        if !b.timing.diurnal.is_empty() {
            let mean: f64 = b.timing.diurnal.iter().sum::<f64>() / 24.0;
            for (i, v) in b.timing.diurnal.iter().enumerate() {
                diurnal[i] = v / mean;
            }
        }
        let mut weekly = [1.0f64; 7];
        if !b.timing.weekly.is_empty() {
            let mean: f64 = b.timing.weekly.iter().sum::<f64>() / 7.0;
            for (i, v) in b.timing.weekly.iter().enumerate() {
                weekly[i] = v / mean;
            }
        }
        let diurnal_max = diurnal.iter().cloned().fold(0.0, f64::max);

        let mut emissions = Vec::new();
        let mut weights = Vec::new();
        for we in &b.events {
            let ei = event_idx[we.event.as_str()];
            let counterparty = match &we.counterparty {
                CounterpartyModel::SkeletonNeighbor { relation } => {
                    CompiledCounterparty::Neighbor { relation: rel_idx[relation.as_str()] }
                }
                CounterpartyModel::RepeatOrNeighbor { relation, repeat_p } => {
                    CompiledCounterparty::RepeatOrNeighbor {
                        relation: rel_idx[relation.as_str()],
                        repeat_p: *repeat_p,
                    }
                }
                CounterpartyModel::GlobalPopularity { entity } => {
                    let e = entities.index_of(entity);
                    CompiledCounterparty::GlobalPopularity {
                        entity: e,
                        pop: pop_index[&e],
                    }
                }
                CounterpartyModel::GlobalUniform { entity } => {
                    CompiledCounterparty::GlobalUniform { entity: entities.index_of(entity) }
                }
            };
            emissions.push(CompiledEmission { event: ei, counterparty });
            weights.push(we.weight);
        }

        out.push(CompiledBehavior {
            name: b.name.clone(),
            actor_type: ti,
            actors,
            activity,
            rate_per_day: b.timing.rate_per_day,
            diurnal,
            diurnal_max,
            weekly,
            burst_p: b.timing.burst_p,
            burst_mean_len: b.timing.burst_mean_len,
            branching_ratio: b.timing.branching_ratio,
            excitation_decay_s: if b.timing.excitation_decay_s > 0.0 {
                b.timing.excitation_decay_s
            } else {
                300.0
            },
            weibull_shape: match b.timing.arrival {
                agora_rules::ArrivalKind::Weibull { shape } => Some(shape),
                agora_rules::ArrivalKind::Poisson => None,
            },
            emissions,
            emission_alias: AliasTable::new(&weights),
            activity_sum,
        });
    }
    // Skeleton sanity: a behavior whose relations have no edges can't emit.
    let _ = skeletons;
    Ok(out)
}

fn filter_actors(
    b: &BehaviorProcess,
    ti: usize,
    entities: &EntityIndex,
    et_rule: &agora_rules::EntityType,
    node_attrs: &[Vec<AttrColumn>],
) -> anyhow::Result<Vec<u64>> {
    let start = entities.starts[ti];
    let n = entities.counts[ti];
    if b.actor_filter.is_empty() {
        return Ok((start..start + n).collect());
    }
    // Resolve each filter to (attr_idx, code).
    let mut checks: Vec<(usize, u16)> = Vec::new();
    for f in &b.actor_filter {
        let ai = et_rule
            .attributes
            .iter()
            .position(|a| a.name == f.attr)
            .ok_or_else(|| {
                anyhow::anyhow!("behavior `{}`: filter attr `{}` not on `{}`", b.name, f.attr, b.actor)
            })?;
        let code = match &node_attrs[ti][ai] {
            AttrColumn::Category { names, .. } => {
                names.iter().position(|v| v == &f.value).ok_or_else(|| {
                    anyhow::anyhow!(
                        "behavior `{}`: filter value `{}` not a value of `{}` (known: {})",
                        b.name,
                        f.value,
                        f.attr,
                        names.join(", ")
                    )
                })? as u16
            }
            AttrColumn::Numeric(_) =>

                anyhow::bail!("behavior `{}`: cannot filter on numeric attr `{}`", b.name, f.attr),
        };
        checks.push((ai, code));
    }
    let mut actors = Vec::new();
    for i in 0..n as usize {
        let ok = checks.iter().all(|&(ai, code)| match &node_attrs[ti][ai] {
            AttrColumn::Category { codes, .. } => codes[i] == code,
            AttrColumn::Numeric(_) => false,
        });
        if ok {
            actors.push(start + i as u64);
        }
    }
    Ok(actors)
}
