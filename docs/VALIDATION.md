# M7 validation — fidelity vs real data (first results)

This is the evidence behind the central claim (Risk R3 / demo P1): realism
**measured, not asserted**. The `agora_eval` harness computes the same
distributional statistics on a real temporal graph and a AGORA-generated one and
reports per-metric distances + an overall fidelity score in [0,1].

> Status: first round, single-node, no tuning yet. These numbers are the honest
> starting point that tells us **which realism gaps actually matter** — they are
> not yet the paper's final fidelity claims.

## Datasets

| dataset | kind | edges | nodes | time | why |
|---|---|---|---|---|---|
| SNAP CollegeMsg | temporal interaction (UC Irvine DMs) | 59,835 | 1,899 | 194 d, unix s | agent-interaction structure — the **right kind** of comparison for AGORA |
| SNAP email-Eu | temporal interaction (institute email) | 332,334 | 986 | 804 d | same |
| Elliptic Bitcoin | transaction **DAG** (tx-level nodes) | 234,355 | 203,769 | 49 coarse steps | finance/crypto-relevant, but a *different graph kind* (caveat below) |

AGORA graphs were scale-matched (same edge/node/time budget) and generated with
anomalies off, so the comparison is on the **normal** structure.

## Results (fidelity 1.0 = identical distributions)

### AGORA finance vs CollegeMsg — fidelity **0.788**
| metric | real | AGORA | read |
|---|---|---|---|
| mean degree | 63.0 | 62.9 | ✅ matched (scale control) |
| power-law α | 4.10 | 2.03 | AGORA tail too shallow |
| degree KS (total) | — | 0.33 | moderate shape gap |
| inter-event KS | — | 0.29 | moderate |
| **burstiness B** | **0.676** | **0.417** | ⚠️ AGORA less bursty than real |
| reciprocity | 0.636 | 0.126 | ⚠️ (partly domain: payments ≠ DMs) |
| clustering | 0.138 | 0.083 | AGORA less clustered |
| repeat-edge | 0.661 | 0.827 | close-ish |

### AGORA finance vs email-Eu — fidelity **0.799**
| metric | real | AGORA | read |
|---|---|---|---|
| mean degree | 674 | 671 | ✅ matched |
| power-law α | 2.45 | 2.19 | ✅ close |
| **burstiness B** | **0.783** | **0.493** | ⚠️ AGORA less bursty |
| reciprocity | 0.711 | 0.313 | ⚠️ |
| clustering | 0.450 | 0.128 | ⚠️ real email is densely triadic |
| repeat-edge | 0.925 | 0.966 | ✅ close |

### AGORA crypto vs Elliptic — fidelity **0.601** (interpret with the caveat)
Elliptic's nodes are **transactions** (used once) over **49 coarse time steps**;
AGORA's nodes are **accounts** (reused) with fine timestamps. So: inter-event
KS=0.999 (Elliptic time is degenerate), repeat-edge 0.00 vs 0.79 (tx-DAG has no
recurrence; accounts do), mean degree 2.3 vs 11 (tx-DAG is sparse). These are
**graph-kind differences, not generator error** — Elliptic is the wrong target
for an account-level generator. Recorded for completeness; the account-level
real benchmark (IBM AMLworld) is Kaggle-gated and not yet acquired.

## What the data tells us (evidence-guided, not guessing)

1. **Burstiness is the clear, domain-independent gap.** Real human/interaction
   temporal data sits at B≈0.68–0.78; AGORA is at 0.42–0.49. The inhomogeneous
   Poisson + geometric bursts + (budget-neutral) Weibull spacing do not reach
   real burstiness. → **This is the data-justified case for Hawkes
   self-excitation** (endogenous cascades; research item #3) and/or stronger
   burst parameters. Next step measures whether parameter tuning closes it
   before building Hawkes.
2. **Reciprocity & clustering are low** vs interaction nets. Partly honest
   domain mismatch (finance payments are not reciprocal/triadic like messaging),
   but for the social/communication-shaped domains a triadic-closure / reciprocal
   counterparty option would help. → evidence for a structural-realism knob.
3. **Degree level, power-law α, and recurrence are already in the right
   ballpark** — AGORA reproduces the gross degree and recurrence structure;
   the gaps are in the *higher-order temporal and triadic* structure.
4. **Need a matched real benchmark per domain.** The cleanest test is AGORA
   finance vs a real account-level transaction graph (AMLworld); without it,
   cross-comparing to messaging conflates domain difference with generator
   fidelity. Acquiring AMLworld/account-level data is a priority.

## Domain-matched results (the clean comparisons)

After acquiring **domain-matched** real data, the comparison is same-domain
(not finance-vs-messaging), which corrects and sharpens the picture.

### AGORA ecommerce vs tgbl-review (real Amazon reviews) — same domain
All-edges fidelity **0.785**; **burstiness B real 0.387 vs AGORA 0.381 (near
perfect)** and **reciprocity 0 vs 0** — i.e. AGORA matches the right domain's
temporal/structural signature. Like-for-like (AGORA `WROTE_REVIEW` edges only,
via `--event-type`): fidelity 0.66, and the real gaps are **total-degree
(KS 0.83)** and **repeat-edge (0.03 real vs 0.39 AGORA)** — AGORA users review
too many items and re-review the same products. Root cause measured to be the
**review-per-user degree calibration** (rate × span too high), NOT the
counterparty: two candidate one-line fixes (lower repeat_p; switch to
global-popularity products) BOTH left repeat-edge at 0.39 — the harness caught
that they were wrong before anything shipped. → review-rate recalibration is
the data-justified ecommerce fix (M7.4).

### AGORA cyber vs CICIDS2017 benign flows — same domain
Fidelity **0.660** (row-order time, as CICIDS timestamps are truncated). Real
gaps: **out-degree KS 0.93** (real networks have a few super-hub servers; AGORA
spreads connections too evenly), **reciprocity 0.01 real vs 0.19 AGORA** (flows
are one-directional client→server; AGORA host traffic is too bidirectional), and
heavier real degree tail (α 2.05 vs 3.15). → cyber needs hub-dominated host
activity + less reciprocal flow.

## Cross-cutting lessons (this is what M7 is for)

1. **Burstiness is DOMAIN-SPECIFIC, not a universal deficit.** Reviews B≈0.39
   (AGORA matches exactly); messaging/email/flows B≈0.68–0.91 (AGORA at 0.40–0.49
   under-matches). So the fix is per-domain burst calibration (and, for the
   bursty interaction domains, Hawkes self-excitation), NOT a global change.
   The earlier finance-vs-messaging "burstiness gap" was a domain artifact.
2. **Compare like-for-like.** A multi-relation AGORA output vs a single-relation
   real dataset inflates repeat-edge; the harness's `--event-type` filter is
   required for a fair comparison. (Added after this bit us.)
3. **Measure before fixing — twice this turn the obvious fix was wrong.** The
   ecommerce repeat-edge looked like a counterparty bug; the data proved it was
   a degree-calibration issue. Without the harness we would have shipped two
   incorrect changes. This is the central argument for building the instrument
   first.
4. **AGORA already reproduces the gross structure** (degree level, power-law α,
   recurrence, and — in the matched domain — burstiness and reciprocity). The
   remaining, now-pinpointed gaps are higher-order: per-domain burstiness,
   degree-tail shape, hub concentration (cyber), review-per-user (ecommerce).

## Data-justified M7.4 backlog (ranked by evidence strength)

| fix | evidence | domain |
|---|---|---|
| review-per-user degree recalibration | total-degree KS 0.83, repeat-edge 0.39→0.03 | ecommerce |
| hub-dominated host out-degree | out-degree KS 0.93 | cyber |
| less reciprocal flows | reciprocity 0.19→0.01 | cyber |
| higher burstiness for interaction domains (Hawkes / burst params) | B 0.4 vs 0.7–0.9 | cyber, social |
| (account-level finance comparison) | need AMLworld (gated) | finance |

## M7.4 — evidence-guided fixes (measure → fix → re-measure)

Worked the ranked backlog with the discipline that every change must be
confirmed by re-measuring. The honest, repeated lesson: **measuring first
prevented several wrong "fixes."**

- **ecommerce review degree (KS 0.83).** Two "obvious" one-line fixes — lower
  `repeat_p`, then switch to global-popularity products — BOTH left repeat-edge
  at 0.39 (the harness caught them as non-fixes). Adding a matched k≥5 node
  filter (TGB pre-filters low-activity nodes; we didn't) dropped degree KS from
  **0.83 → 0.40**: most of the gap was a *benchmark-filtering artifact*, not a
  generator flaw. The residual (repeat-edge 0.4 vs 0.03) is a real but harder
  popularity-vs-repeat tension; bumping catalog fanout barely moved it (Zipf
  head dominates) and slightly hurt fidelity, so it was reverted. → needs a
  review-selection redesign, not a parameter tweak.
- **cyber out-degree (KS 0.93).** Real CICIDS-monday has only ~90 source hosts
  (a tiny specific capture, super-concentrated); AGORA spreads across 8,781. The
  gap is largely a network-size/topology artifact of a single capture — not a
  pure generator defect.
- **burstiness (the cleanest, multi-dataset signal).** Cranking the existing
  burst params raised B 0.44 → **0.59** (toward real interaction ≈0.7) but
  plateaued below it — so the data justified adding **Hawkes self-excitation**
  (recursive endogenous cascades; Hawkes & Oakes 1974 cluster representation),
  now in the engine (`TimingModel.branching_ratio`, budget-neutral via the
  1/(1−n) factor, thread-deterministic). Hawkes raises B 0.44 → 0.52 at n=0.7 —
  a real, correct capability, but it lifts B *less* than the burst crank because
  BOTH are within-window mechanisms. **The global per-source B is dominated by
  the cross-window activity envelope (overnight/weekend gaps), so fully reaching
  real B≈0.7 needs a self-similar / heavy-tailed cross-window activity envelope
  (research item #5) — within-window bursts/Hawkes/Weibull are necessary but not
  sufficient.** This is now precisely scoped.

### Meta-finding (worth stating in the paper)
Naive fidelity comparison is confounded by **benchmark-specific preprocessing**
(TGB k-core filtering, CICIDS single-capture topology, multi- vs single-relation
outputs, coarse timestamps). Honest fidelity evaluation must control for these —
the harness's `--event-type` and `filter_min_degree` exist for exactly that.
Once controlled, AGORA's gaps are smaller and more specific than the raw scores
suggest, and they point to two concrete generator items: a self-similar activity
envelope (burstiness) and a review-selection redesign (e-commerce repeat).

## Reproduce

```bash
# all edges
python3 -m agora_eval compare --synth <agora_out_dir> --real <real_path> [--t-col N] [--t-scale S]
# like-for-like single relation
python3 -m agora_eval compare --synth <agora_out_dir> --event-type WROTE_REVIEW --real <reviews.csv> --src-col 1 --dst-col 2 --t-col 0 --header
```
