# EVAL.md — AGORA's Evaluation Methodology (how we out-rigor LDBC)

The serious question: **how do we prove AGORA's graphs are good — more rigorously
than LDBC / TrillionG / GraphRNN?** This is the design of the evaluation chapter.
All claims are citation-grounded (refs in `references.bib`); `VERIFY` = confirm a
citation detail before the paper.

## The gap we exploit

LDBC SNB evaluates via **(a) structural fidelity** (a Facebook-derived power-law
degree distribution + attribute-correlated edges → realistic clustering; Erling
et al. SIGMOD 2015, DATAGEN) and **(b) choke-point query benchmarking** (28 query
choke points in 6 categories, seeded from TPC-H; Boncz, Neumann & Erling, TPCTC
2013). That is a **query-engine** benchmark with a structurally-realistic graph.
It has **no temporal-graph fidelity, no downstream-utility check, no privacy
analysis, and — because it has no anomaly labels — no notion of anomaly
difficulty.** AGORA is evaluated on **six layers**; LDBC touches only Layer 1.

---

## Layer 1 — Structural fidelity (do it, but rigorously)

- **Graph-MMD, done right.** The GraphRNN protocol (You et al., ICML 2018)
  compares generated vs real *sets* of graphs by MMD over the **degree,
  clustering, and 4-node orbit** distributions with a Gaussian-EMD kernel. But
  O'Bray et al. (ICLR 2022, spotlight) show graph-MMD is **hyperparameter-fragile
  and the EMD kernel is often not PSD** — "any model can rank first" under bad σ.
  We therefore: use **PSD kernels**, **fix σ from the real data** (median
  heuristic), and — the key rigor flex — run the **perturbation test**: MMD must
  increase **monotonically** as we add random edge/node perturbations to the real
  graph. We report the perturbation-monotonicity curve, which most generator
  papers do not.
- **Graphlet/orbit distance.** 4-node orbit counts via **ORCA** (Hočevar &
  Demšar, Bioinformatics 2014) and the **Graphlet Correlation Distance (GCD-11)**
  (Yaveroğlu et al., Sci. Rep. 2014) — noise-tolerant, size-comparable.
- **Spectral.** Laplacian-eigenvalue MMD and **NetLSD** heat-trace signatures
  (Tsitsulin et al., KDD 2018) — permutation/size-invariant.
- **Single robust scalar.** A **GNN-based FID-analog** (Thompson et al., ICLR
  2022): embed graphs with an **untrained random GNN**, score with Fréchet
  distance — one number for model selection, feature-aware.
- **Portrait divergence** (Bagrow & Bollt, 2019) as an all-scales JS-divergence
  pseudometric.

## Layer 2 — Statistical rigor (beat "a line on a log-log plot")

- **Distribution distance:** prefer **1-D Wasserstein / energy distance** over KS.
  Energy distance = distance-based **MMD** (Sejdinovic et al., Ann. Stat. 2013);
  it has units, obeys the triangle inequality, and is far more stable than KS.
  Use **Anderson–Darling** (tail-weighted) for heavy-tailed **degree** tails where
  KS is blind.
- **Power laws the CSN way** (Clauset, Shalizi & Newman, SIAM Rev. 2009):
  MLE α̂ = 1 + n/Σln(xᵢ/x_min), x_min by KS-minimization, **semiparametric
  bootstrap goodness-of-fit p-value** (p>0.1 ⇒ plausible), and a **Vuong
  likelihood-ratio test** (power-law vs **log-normal**/exponential/cutoff). Report
  whether real AND synthetic are *both* plausibly power-law and *which* model
  wins — not just a fitted α. (`powerlaw` pkg, Alstott et al. 2014.)
- **Report effect sizes, not p-values.** KS/AD **reject on trivial differences at
  large n** (critical D ∝ √((n₁+n₂)/n₁n₂) → 0). Lead with D, W₁, energy distance
  (+ CIs); apply **Benjamini–Hochberg FDR** across the per-domain × per-metric grid.

## Layer 3 — Temporal fidelity (the axis LDBC has none of)

- **Temporal motifs** (Paranjape, Benson & Leskovec, WSDM 2017): the **36-motif
  (2-/3-node, 3-edge, δ-window) fingerprint**, exact in ~O(m) for star motifs.
  This is a domain fingerprint of blocking vs non-blocking dynamics — a direct,
  temporal-specific fidelity test no static generator can pass by construction.
- **Burstiness B** (Goh & Barabási, EPL 2008) with the **finite-size correction
  Bₙ** (Kim & Jo, PRE 2016 — `VERIFY` denominator) so unequal-length sequences
  compare fairly; **memory M** (lag-1 IET correlation) as the orthogonal axis.
- **Temporal correlation coefficient C** (Nicosia et al. 2013) — neighborhood
  overlap between consecutive snapshots; **temporal reachability / efficiency**
  (Holme & Saramäki, Phys. Rep. 2012) over time-respecting paths.
- Peer generators (TagGen KDD 2020; **TIGGER, AAAI 2022**) converge on exactly
  this set (degree + IET + temporal-motif profile) — we match and exceed it.

## Layer 4 — Extrinsic / downstream utility (the gold standard; LDBC skips it)

Does a model **trained on AGORA transfer to real data**? Two now-standard probes:
- **Discriminative score (C2ST).** Train a classifier to tell real vs synthetic
  edges/sequences; report **|accuracy − 0.5|** (0 = indistinguishable = best).
  The Classifier Two-Sample Test (Lopez-Paz & Oquab, ICLR 2017); temporal form in
  **TimeGAN** (Yoon et al., NeurIPS 2019).
- **TSTR — Train on Synthetic, Test on Real** (Esteban, Hyland & Rätsch, 2017):
  train a **temporal link predictor** (TGN/GraphMixer) and/or an **anomaly
  detector** entirely on AGORA, evaluate on a held-out real graph; a small AP/MRR
  (or AUC) gap ⇒ AGORA captured the real predictive structure. TRTS catches
  off-manifold/mode-collapse. ML-efficacy protocol as in CTGAN (Xu et al. 2019).

## Layer 5 — Privacy / disclosure (safe to release; real data is private)

Since we release datasets publicly, we show they don't memorize a private source:
- **DCR** (distance to closest real record) + **NNDR** (nearest/2nd-nearest ratio);
- the **distance-ratio test** (DCR-to-training vs DCR-to-holdout — memorization if
  systematically closer to training);
- **membership-inference AUC** (DOMIAS, van Breugel et al., **AISTATS 2023** — not
  NeurIPS; density-ratio overfitting detector). AUC≈0.5 ⇒ private. (Note the "DCR
  Delusion" 2025 caveat: DCR alone is weak; pair with a principled MIA.)

## Layer 6 — Anomaly-benchmark quality (AGORA-unique; the headline)

This is where AGORA has no competition. LDBC/AMLSim/TGB cannot do any of it.

- **Choke points for DETECTION, not queries.** We reframe Boncz/LDBC choke-point
  design: AGORA's five control axes are **choke points for anomaly detectors** —
  each stresses a specific detector weakness (camouflage → feature/relation
  robustness; cascade → temporal-propagation modeling; placement → community
  awareness).
- **The difficulty→detectability curve (the money figure).** Emmott et al. (ODD
  2013; meta-analysis 2015) construct AD benchmarks by varying **point difficulty,
  relative frequency, clusteredness, feature relevance** — which map **exactly**
  onto AGORA's axes (difficulty/camouflage, prevalence, placement, type-mix).
  We run a **panel of SOTA detectors** (from GADBench, Tang et al. NeurIPS 2023;
  temporal detectors TADDY/StrGNN/SLADE) across our difficulty sweep and show
  **detector AUC/AP degrades monotonically as the camouflage knob rises** — proof
  the difficulty axis is *real, meaningful, and controllable*. A benchmark that
  spans easy→hard and separates weak from strong detectors is exactly what the
  field asks for (Emmott 2015; ADBench, Han et al. NeurIPS 2022).
- **Exact label quality.** Current AD benchmarks are criticized for **mislabeled
  ground truth and trivial anomalies** (Wu & Keogh 2021; "Rethink Benchmarking in
  AD" 2025). AGORA's labels are **generative ground truth** (label = the process
  that emitted the edge) — exact by construction, the direct answer to that
  critique.

---

## Positioning (why this is more rigorous than LDBC)

| layer | LDBC SNB | GraphRNN | TGB | AMLSim | **AGORA** |
|---|:--:|:--:|:--:|:--:|:--:|
| structural fidelity | ✅ | ✅ (MMD) | — (real) | ~ | ✅ + perturbation test |
| statistical rigor (CSN/energy/AD) | ~ | ~ | — | — | ✅ |
| **temporal-graph fidelity** | ❌ | ❌ | ~ | ~ | ✅ (motifs+B+C+reach) |
| **extrinsic utility (TSTR/C2ST)** | ❌ | ~ | ✅ (link pred) | ❌ | ✅ |
| **privacy/disclosure** | ❌ | ❌ | ❌ | ❌ | ✅ |
| **anomaly difficulty curve** | ❌ | ❌ | ❌ | ❌ | ✅ (the headline) |

## Implementation status (in `python/agora_eval`)

- **Have:** degree KS, IET KS, CSN α (KS-min x_min), burstiness B, memory M,
  clustering, reciprocity, repeat-edge, temporal activity, an honest fidelity
  score (see `docs/VALIDATION.md`).
- **Building (this milestone, ranked):** (1) temporal motifs (Layer 3); (2)
  energy/Wasserstein + Anderson–Darling (Layer 2); (3) discriminative score C2ST
  (Layer 4); (4) the **difficulty→detector-AUC** harness (Layer 6, headline);
  (5) proper graph-MMD + perturbation test & GCD (Layer 1); (6) DCR/NNDR (Layer 5).
- **Discipline (from M7):** measure like-for-like (`--event-type`), control for
  benchmark preprocessing (k-core filtering), report effect sizes + the honest
  scorecard. Naive single-number fidelity is confounded; we report the layer
  breakdown.

*The evaluation is itself a contribution: a multi-layer, temporal-aware,
utility-and-difficulty-grounded protocol for judging a temporal-graph anomaly
benchmark — the missing rigor the AD-benchmark critiques (Wu & Keogh; ADBench;
"Rethink Benchmarking") call for.*
