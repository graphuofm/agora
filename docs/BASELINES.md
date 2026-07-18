# BASELINES.md — AGORA vs classical graph generators

The full-paper comparison the demo lacked: AGORA next to the generators it is
compared to (and accused of *being*). Harness: `python/agora_eval/baselines.py`
(`python3 -m agora_eval baselines --real <path> --synth <agora_dir>`). Each
classical generator is matched to the real graph's node/edge count; since none
produce timestamps, they get the naive practitioner treatment — **uniform-random
timestamps over the real span** — which is precisely what exposes the gap.

## Result: vs CollegeMsg (real temporal DM network; 1,899 nodes, 59,835 edges)

AGORA = finance domain, scale-matched, anomalies off (normal structure), seed 42.

| generator | edges | clustering | reciprocity | burstiness B | α | **fidelity** | time | attrs | labels |
|---|--:|--:|--:|--:|--:|--:|:--:|:--:|:--:|
| **real (reference)** | 59,835 | 0.138 | 0.636 | 0.676 | 4.10 | 1.000 | ✅ | ✅ | ✅ |
| **AGORA (ours)** | 59,309 | 0.086 | 0.124 | 0.416 | 2.43 | **0.793** | ✅ | ✅ | ✅ |
| Erdős–Rényi | 59,835 | 0.033 | 0.015 | −0.00 | 15.8 | 0.558 | ~ | ❌ | ❌ |
| Barabási–Albert | 59,744 | 0.084 | 0.000 | 0.300 | 2.92 | 0.642 | ~ | ❌ | ❌ |
| R-MAT | 59,835 | 0.184 | 0.101 | 0.311 | 2.52 | 0.705 | ~ | ❌ | ❌ |
| configuration model | 59,835 | 0.388 | 0.190 | −0.00 | 4.10 | 0.816 | ~ | ❌ | ❌ |
| Watts–Strogatz | 60,768 | 0.543 | 0.000 | 0.691 | 25.5 | 0.557 | ~ | ❌ | ❌ |

(~ = "temporal" only via the fake random timestamps; the generators have no real
dynamics.)

## What this shows (and the honest nuances)

1. **"Isn't this just BA?" — answered with numbers.** Barabási–Albert scores
   **0.642**; AGORA scores **0.793**. BA also has reciprocity 0.000 (real 0.636)
   and no attributes or labels. AGORA beats BA, R-MAT (0.705), ER and WS (~0.56)
   on the single fidelity number, and is the *only* generator that carries
   attributes and anomaly labels at all.

2. **The configuration model is the trap that proves the point.** It attains the
   highest structural fidelity (**0.816**) — but *only because it is handed the
   real degree sequence* (its α = 4.10 exactly matches real). It then fails
   everything a benchmark needs: clustering 0.388 (real 0.138, 2.8× too high),
   reciprocity 0.190 (real 0.636), burstiness −0.00 (real 0.676 — random
   timestamps have no burstiness), and **zero attributes/labels**. This is the
   live demonstration of `docs/EVAL.md`'s thesis: **a single fidelity number is
   confounded by the degree distribution**; the capability matrix and the
   per-layer breakdown are what matter.

3. **Honest caveats.**
   - AGORA's reciprocity (0.124) and burstiness (0.416) undershoot this *messaging*
     target — a domain mismatch (finance payments are not reciprocal/bursty like
     DMs; see `docs/VALIDATION.md`). The domain-matched e-commerce comparison
     (AGORA-ecommerce vs tgbl-review, fidelity 0.785, burstiness 0.381 vs real
     0.387) is the fairer test.
   - The discriminative-score (C2ST) is ≈1.0 for *every* generator here
     (config-model 0.918): a classifier trivially separates any generator from
     real messaging using timing features — a harsh, domain-sensitive metric we
     report rather than hide.

## Capability matrix (the definitive differentiator)

Only real data and AGORA are complete. Classical generators produce bare topology;
LDBC/S3G2 adds semantics but no anomaly labels; AMLSim injects labels but is
single-domain and small; deep temporal generators (TIGGER/TagGen) don't scale and
have no controllable difficulty. `[Experiments & Analysis]` reviewers care about
exactly this table.

| capability | BA/R-MAT | LDBC | AMLSim | CTGAN | TGB | TIGGER | **AGORA** |
|---|:--:|:--:|:--:|:--:|:--:|:--:|:--:|
| power-law / community | ✅ | ✅ | ~ | ❌ | — | ~ | ✅ |
| temporal edges | ❌ | ✅ | ✅ | ❌ | ✅ | ✅ | ✅ |
| rich attributes | ❌ | ~ | ✅ | ✅ | — | ❌ | ✅ |
| anomaly ground truth | ❌ | ❌ | injected | ❌ | ~ | ❌ | ✅ emergent |
| controllable difficulty | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ✅ |
| multi-domain (zero-code) | ❌ | ❌ | ❌ | ~ | ✅ | ❌ | ✅ |
| scale (≥10⁸ edges) | ✅ | ✅ | ❌ | ❌ | — | ❌ | ✅ |
| fidelity measured | ~ | ~ | ❌ | ✅ | — | ✅ | ✅ |

*Next: run the domain-matched baseline table (tgbl-review at full scale on
the cluster) and the heavy external baselines (AMLSim, CTGAN, TIGGER) as documented
jobs; add both to the paper.*
