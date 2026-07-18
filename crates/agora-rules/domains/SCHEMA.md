# AGORA rule-base YAML schema (authoring reference)

One domain = one YAML file instantiating the **seven primitives** (mustread.txt
§5). Loaded + validated by `agora-rules`. YAML enums use serde **external
tagging**: write `!variant_name { fields }`. Validation runs on load (`agora
domains --show <id>` after wiring into `domains.rs`).

## Top level
```yaml
meta: { id, name, description, schema_version: 1, provenance: [{source, url?, version?, license_tier?}] }
entity_types: [EntityType, ...]
relations: [RelationRule, ...]
event_types: [EventType, ...]
behaviors: [BehaviorProcess, ...]
constraints: [Constraint, ...]      # optional
adversaries: [AdversaryProcess, ...]  # optional
failures: [FailureProcess, ...]       # optional
control: ControlParams
```

## Distribution (used everywhere a value/dist is needed)
`!constant {value}` · `!uniform {min,max}` · `!normal {mean,std}` ·
`!log_normal {mu,sigma}` · `!exponential {rate}` · `!pareto {scale,shape}` ·
`!zipf {n,exponent}` · `!poisson {lambda}`
(log_normal mean = exp(mu+sigma²/2); pick mu = ln(desired_median).)

## AttributeKind (for entity attributes AND event attributes)
- `!categorical {values: [...], weights: [...]}`  (len must match)
- `!ordinal {tiers: [...], weights: [...]}`
- `!taxonomy {paths: [...], weights: [...]}`  (e.g. MITRE tactic→technique strings)
- `!numeric {dist: <Distribution>}`
- `!flag {p}`  (boolean true-prob; decodes to values "false"/"true")

## EntityType
```yaml
- name: account
  population_weight: 0.9          # share of node budget (normalized across types)
  attributes: [{name, kind: <AttributeKind>}, ...]   # immutable at birth
  state: [{name, init: <Distribution>}, ...]         # mutable sim variables
```

## RelationRule  (the skeleton: who CAN interact; O(1)/edge models)
```yaml
- name: net
  src: account
  dst: account
  model: <TopologyModel>
  layer: skeleton|vessels|muscle|skin   # default skeleton
  mean_degree: 6.0
```
TopologyModel variants (note constraints):
- `!preferential_attachment {m}`  — **requires src == dst type** (hubs emerge)
- `!uniform_random`
- `!small_world {k, beta}`  — **src == dst**
- `!forest_fire {forward_p, backward_p}`  — **src == dst**
- `!sbm {communities, p_in_weight, p_out_weight}`
- `!rmat {a,b,c,d}`  — **a+b+c+d == 1**
- `!spatial {radius}`  (unit square, grid-bucketed; radius in (0,0.5])
- `!affiliation {popularity: <Distribution>}`  — bipartite; dst rank-popularity
  (use `!zipf` for a heavy head; `n` is overridden by the actual dst count)

## EventType  (emits a temporal edge; applies state effects)
```yaml
- name: transfer
  src: account
  dst: account
  attributes: [{name, kind: <AttributeKind>}, ...]
  effects: [<StateEffect>, ...]
```
StateEffect variants (var must be a state var on the named endpoint's type;
from_attr must be one of this event's numeric attributes):
- `!add_to_dst {var, from_attr}` · `!add_to_src {var, from_attr}`
- `!sub_from_src {var, from_attr}` · `!sub_from_dst {var, from_attr}`
- `!increment_src {var}` · `!increment_dst {var}`
- `!set_src {var, value}` · `!set_dst {var, value}`

## BehaviorProcess  (normal dynamics: when × whom × what)
```yaml
- name: retail_activity
  actor: account
  actor_filter: [{attr: type, value: retail}]   # optional; categorical attrs only
  timing:
    rate_per_day: 1.8
    diurnal: [24 floats]   # optional, must be exactly 24 (normalized internally)
    weekly:  [7 floats]    # optional, Monday first, exactly 7
    burst_p: 0.15          # optional: prob an event triggers a follow-up burst
    burst_mean_len: 2.0
  events:                  # WHAT + WHOM, weights normalized
    - event: purchase
      weight: 0.7
      counterparty: <CounterpartyModel>
  activity: <Distribution>  # optional per-actor heterogeneity multiplier (default log_normal 0,0.5)
```
CounterpartyModel (O(1) per event):
- `!skeleton_neighbor {relation}`  — uniform among skeleton neighbors
- `!repeat_or_neighbor {relation, repeat_p}`  — reuse a recent partner (muscle)
- `!global_popularity {entity}` / `!global_uniform {entity}`

## Constraint  (Φ: legality predicate; violation = candidate, NOT the label)
```yaml
- name: balance_non_negative
  check: <ConstraintCheck>
  description: ...
```
ConstraintCheck:
- `!attr_range {event, attr, min, max}`
- `!state_non_negative {entity, var}`
- `!rate_limit {event, k, window_s}`  (≤ k events/src in window)
- `!sub_threshold_count {event, attr, threshold, floor, k, window_s}`  (≥k events
  with attr in [floor,threshold) per src in window ⇒ candidate)
- `!documented {rule: "free text"}`  (checked only by humans / future passes)

## AdversaryProcess  (intent label = the event's ground-truth cause)
```yaml
- intent: structuring          # the label written on every event it causes
  description: ...
  actor: account
  stages:                      # staged policy automaton
    - name: smurf_deposits
      duration_days: <Distribution>
      rate_per_day: <Distribution>
      event: cash_deposit
      scope: <StageScope>
      attr_overrides: [{attr, dist: <Distribution>}, ...]   # optional
  camouflage: 0.5              # 0=blatant, 1=mimic normal (THE difficulty lever)
  prevalence_weight: 0.3       # share of anomaly budget (normalized w/ others)
  ring_size: <Distribution>    # controlled actors per campaign
  cascade_p: 0.1               # optional: prob of spawning a follow-up campaign
```
StageScope:
- `!ring` (random other ring member) · `!chain` (member i → i+1; cycles) ·
  `!collector` (all → members[0]; fan-in) · `!normal` (the actor's normal
  counterparty for this event; mimicry stage) ·
  `!victims { count: <Distribution> }` (fresh nodes outside the ring; fan-out)

## FailureProcess  (non-adversarial/emergent twin; same labeling discipline)
```yaml
- intent: sensor_fault
  description: ...
  actor: sensor
  mode: <FailureMode>
  prevalence_weight: 2.0
  rate_per_year: 12.0          # incidents per affected entity per simulated year
  duration_days: <Distribution>
  cascade_p: 0.2               # optional: propagate to a skeleton neighbor
```
FailureMode:
- `!silence` (suppress the actor's events for the span)
- `!stuck_attr {event, attr, value}` · `!drift_attr {event, attr, drift_per_day}`
- `!rate_shift {factor}` (multiply activity; >1 surge, <1 slowdown)
- `!noise_attr {event, attr, dist: <Distribution>}` (replace attr w/ a dist)

## ControlParams  (domain defaults for the five axes; CLI flags override)
```yaml
control:
  prevalence: 0.02     # axis 1: node fraction in anomalous processes (keep ≈0.01–0.05; anomalies are RARE)
  difficulty: 0.5      # axis 2 ∈ [0,1]
  type_mix: [{intent, weight}, ...]   # axis 3; empty = per-process weights
  placement: <Placement>              # axis 4
  cascade: 1.0         # axis 5: multiplier on per-process cascade_p
```
Placement: `!uniform` · `!clustered {n_communities}` ·
`!time_window {start_frac, end_frac}` · `!clustered_bursty {n_communities, start_frac, end_frac}`

## Authoring rules
- Every name referenced (entity/event/relation/attr/state var) must be defined.
- Anomalies are RARE: control.prevalence ≈ 0.01–0.05.
- Ground truth = INTENT (adversary/failure), not Φ. Φ is the detector's-eye view.
- Counterparty selection must stay O(1) (it's the per-event hot op).
- For a heavy-tailed quantity use log_normal/pareto/zipf, not normal.
- Provenance: cite the CORPUS.md standards each rule is grounded in.
- See `finance.yaml` for a complete worked example.
