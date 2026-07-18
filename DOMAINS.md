# DOMAINS.md — AGORA Rule Dossiers for the 6 Industries

These are the per-domain knowledge dossiers that (a) feed the **self-built RAG
corpus** (mustread.txt §9) and (b) specify how each domain instantiates the
seven primitives (mustread.txt §5). Each follows the same structure: **Entities ·
Events · Normal behavior · Anomaly playbooks · Constraints · Datasets**.

> Citations are grounded but **`VERIFY`-flagged where uncertain** — confirm
> against DBLP/ACM DL before the paper. Distribution fits (log-normal, power-law,
> Benford) are *population-level regularities, not guarantees* — validate per
> dataset before treating them as discriminative.
>
> **The authoritative STANDARDS to download for each domain's RAG corpus** (FATF,
> MITRE ATT&CK, CMS NCCI, HCM, FTC rules, …) — with formats, URLs, and license
> tiers — are in [`CORPUS.md`](CORPUS.md). That corpus is what grounds the
> zero-code rule synthesis described in `mustread.txt` §9.

Anomaly source type per domain: **ADVERSARIAL** (intentional `AdversaryProcess`)
vs **NATURAL** (`FailureProcess`). Domain 4 is the deliberate non-adversarial
stress test.

---

## 1. Finance / Anti-Money-Laundering  *(adversarial)*

**Entities + state.** `account`(balance, status{active,dormant,frozen,closed},
risk_tier, kyc_level, age_days, jurisdiction, type{retail,business,shell});
`customer`(occupation/industry, PEP_flag, sanctions_flag); `institution`(country,
correspondent_flag); `merchant`(MCC, cash_intensive). State evolves: balance per
txn, risk re-scoring, dormancy→activation, cumulative in/out.

**Events + attributes.** `transfer(src→dst)`(amount, currency, channel{wire,ACH,
card,cash,crypto_onramp}, ts, type{credit/debit}, is_cross_border, memo);
`cash_deposit`/`withdrawal`(amount, branch/ATM, ts); `account_open/close`;
`ownership`(customer→account beneficial-owner). Directed, timestamped, weighted,
multigraph.

**Normal + distributions.** Amounts heavy-tailed **log-normal** (legit aggregates
conform to **Benford**; deviation is a forensic signal); power-law tails for
large transfers. Temporal: diurnal + business-hours, weekly + payroll/month-end
spikes; **bursty** heavy-tailed inter-arrivals. **Power-law degree** (hubs =
merchants/payroll). Illicit nodes RARE (~2% Elliptic, ~1.3% DGraph-Fin) — preserve
extreme imbalance.

**Anomaly playbooks (FATF typologies).** *Placement* (sub-threshold cash bursts);
*Structuring/smurfing* (many sub-$10k-CTR deposits via mules; camouflage = jitter
amounts/channels/timing); *Layering* (rapid pass-through chains across
jurisdictions, near-zero net balance; camouflage = legit intermediaries + dwell
time); *Integration* (shell→business with plausible memos); *Fan-in/fan-out*
(scatter-gather degree spikes); *Mule/shell* (in≈out, short dwell, reactivated
dormant); *Round-tripping* (directed cycle A→…→A, conserved value); *Rapid
movement* (low dwell, high pass-through). **Camouflage strength = the difficulty
lever.**

**Constraints (violation ⇒ candidate).** outflow ≤ inflow+balance; dwell ≥ τ for
non-mules; sub-threshold count in window ≤ k; no short-horizon conserved cycle;
cross-border-to-high-risk needs kyc ≥ x; out/in≈1 + short dwell ⇒ mule.

**Datasets.** **Elliptic** (Weber et al., KDD AI-in-Finance ws, 2019); **IBM
AMLworld** (Altman et al., NeurIPS D&B, 2023); **DGraph-Fin** (Huang et al.,
NeurIPS D&B, 2022); **AMLSim** (IBM; software/tech report — `VERIFY` venue; the
conceptual ancestor we must beat).

---

## 2. Crypto / Blockchain (Ethereum / DeFi)  *(adversarial)*

**Entities + state.** `EOA`(balance, nonce, age, label{exchange,user,mixer,scam});
`contract`(bytecode_hash, type{ERC-20/721,DEX_pool,lending,Ponzi,proxy}, TVL,
is_verified, creator); `token`(supply, holders, liquidity); `liquidity_pool`
(reserves, price, LP_supply).

**Events + attributes.** `tx(from→to)`(value, gas_price, gas_used, ts, block,
input_data, is_contract_call, status); `token_transfer`(token, amount);
`swap`(pool, token_in/out, amount_in/out, price_impact); `add/remove_liquidity`,
`contract_creation`, `approve`. Total order via (block_number, tx_index).

**Normal + distributions.** Value heavy-tailed **log-normal/power-law**; gas
spikes with congestion; many dust calls. Strong power-law degree (exchange hot
wallets = super-hubs). 24/7 with market-cycle correlation; MEV bots = sub-second
bursts. High address churn / low reuse.

**Anomaly playbooks.** *Wash trading* (self-trading tight cycles, volume ≫ unique
counterparties; camouflage via sybil wallets); *Phishing/scam* (victims→1
collector fan-in, drain after `approve`; camouflage = relay to fresh wallets,
fake airdrop); *Mixing/tumbling* (fixed-denomination in/out breaks linkability;
camouflage = delays, relayers); *Pump-and-dump* (synchronized buys→spike→insider
sells); *Rug pull* (creator `remove_liquidity` near-total / hidden mint /
honeypot); *Ponzi* (later deposits pay earlier; collapse on inflow drop).

**Constraints.** token conservation (excess mint ⇒ flag); LP invariant x·y=k
(abrupt drain ⇒ rug); buy/sell symmetry (sell reverts ⇒ honeypot); no frequent
near-closed conserved cycles among related addrs (wash); unmatched mixer
denominations (obfuscation); holder-concentration bound.

**Datasets.** **XBlock Ethereum phishing** (Chen et al., IEEE TKDE/TNSE ~2020–21 —
`VERIFY` exact paper); **Elliptic++** (Elmougy & Liu, 2023 — `VERIFY` venue);
**Ethereum Ponzi** (Chen et al., WWW 2018); **DeFi attacks SoK** (Zhou et al.,
IEEE S&P 2023); **DeFiHackLabs**; traces via Etherscan / BigQuery Ethereum.

---

## 3. Cybersecurity — Network IDS + APT  *(adversarial)*

**Entities + state.** `host`(OS, open_ports, compromise-stage, creds, priv_tier);
`user/account`(role, tokens, priv_level, sessions, host-affinity); `process`(PID,
parent, integrity, alive); `service/port`(listening, proto); `flow`(transient);
`adversary`(current ATT&CK tactic, footholds, objective).

**Events + attributes.** flow `host→host`(ts, src/dst IP+port, proto, duration,
fwd/bwd bytes+packets, flags, mean IAT); auth `user→host`(ts, src/dst host, auth
type{Kerberos/NTLM}, logon type, success/fail, LogOn/LogOff); process `proc→proc`
(ts, parent→child exec, image, cmdline); file/registry `proc→object`(ts, action);
DNS/C2(ts, query, beacon interval).

**Normal + distributions.** Diurnal/weekly seasonality. Packet sizes **bimodal**
(~40–100 B control vs ~1500 B MTU). Flow duration/bytes **heavy-tailed/log-normal**
(elephant vs mice). **Self-similar** bursty arrivals (Leland 1994). Stable small
user→host affinity; mostly successful auth; power-law degree.

**Anomaly playbooks (MITRE ATT&CK / kill-chain).** *Recon scan* (T1046: 1→many,
tiny flows, RSTs; camo = slow/distributed); *Brute force* (T1110: many fail→1
success; camo = password spraying); *DDoS* (T1498: massive fan-in, low entropy;
camo = mimic flash-crowd); *Lateral movement* (T1021/T1570: novel auth edges
between low-affinity hosts; camo = valid creds, business hours); *Priv-esc* (T1068:
integrity jump; camo = LOLBins); *C2 beacon* (T1071: periodic low-volume egress;
camo = jitter, HTTPS/DGA); *Exfiltration* (T1041/T1048: large outbound to rare dst,
off-hours; camo = low-and-slow chunking, backup windows). APT chains these over
weeks (low base-rate stealth — OpTC's purpose).

**Constraints.** bytes/pkts ≥ 0; duration ≥ 0; flags consistent w/ handshake; auth
needs prior valid session; child needs live parent; no priv jump without escalation
event; egress dst in known-good set; flow conserves counts.

**Datasets.** **CICIDS2017** (Sharafaldin et al., ICISSP 2018); **DARPA OpTC**
(~17.4B events, red-team APT; Anjum et al. ~2020 — `VERIFY` count/attribution);
**LANL** (Kent, LANL 2015, 58-day auth/flow/process/DNS, labeled red-team);
**UNSW-NB15** (Moustafa & Slay, MilCIS 2015); **CTU-13** (Garcia 2014, botnet).

---

## 4. Transportation / Mobility  *(mostly NATURAL — the non-adversarial stress test)*

**Entities + state.** `vehicle/agent`(position, speed, accel, lane, route,
driver-aggressiveness, OD); `road_segment`(occupancy/density, capacity, free-flow
speed, signal phase, incident_flag); `intersection`(signal phase, queues);
`sensor`(flow, occupancy, speed, health/fault); `trip`(pickup/dropoff, fare,
passengers).

**Events + attributes.** traversal `node→node`(ts, vehicle, travel_time, mean
speed, segment); sensor reading(ts, flow veh/h, occupancy %, speed); trip
`pickup_zone→dropoff_zone`(ts_pickup/dropoff, distance, duration, fare,
passengers); micro car-following leader→follower(gap, headway).

**Normal + distributions.** **Krauss** safe-speed car-following (SUMO default).
**Bimodal diurnal commute** peaks; weekday/weekend seasonality. **Fundamental
diagram** (speed–flow–density; Greenshields); congestion = high-density/low-speed.
Trip lengths/durations **log-normal**; gravity-model spatial OD; headways ~
exponential at low density; taxi demand spatially clustered + diurnal.

**Anomaly playbooks.** *Congestion cascade* (NATURAL/emergent: density crosses
critical → backward-moving shockwave ~15–20 km/h; phantom jams without incident);
*Accident/incident* (NATURAL: abrupt local capacity drop, queue spillback);
*Sensor fault* (NATURAL/non-traffic: stuck-at, dropout, drift, impossible values —
**not corroborated by neighbors**, the key discriminator); *Special event/weather*
(NATURAL demand surge / capacity loss); *Intentional* (thin adversarial tail:
coordinated rerouting, fake-trip injection, GPS spoofing; camo = mimic organic
surge).

**Constraints.** speed ∈ [0,v_max]; flow ≤ capacity; density ≤ jam; travel_time ≥
length/v_max; vehicle conservation at nodes; occupancy ∈ [0,1]; no teleport
(Δposition ≤ v_max·Δt); sensor in physical range + consistent with neighbors.

**Datasets.** **SUMO** (Lopez et al., IEEE ITSC 2018); **Krauss** model (Krauß,
DLR thesis ~1998 — `VERIFY` year); **METR-LA / PEMS-BAY** (Li et al. DCRNN, ICLR
2018); **PeMS/Caltrans** (Chen et al., TRB 2001); **NYC/Chicago taxi** open data;
fundamental diagram (Greenshields 1935).

---

## 5. E-commerce / Review & Rating Fraud  *(adversarial)*

**Entities + state.** `user/reviewer`(account_age, review_count, avg_rating,
burstiness, verified_purchase_ratio, IP/device fp, paid/organic); `product`
(cumulative_rating, review_velocity, price, category, rank); `seller`(rating,
return_rate, age); `review`(rating 1–5, text, helpful_votes, verified);
`device/IP`(shared-fp for sockpuppet linkage).

**Events + attributes.** `WROTE_REVIEW`(user→product: ts, star, text,
verified_purchase, helpful_votes); `PURCHASED`(ts, price, returned?); `RATED`;
`RETURNED/REFUNDED`(ts, reason, amount); `VOTED_HELPFUL`(user→review);
`SHARES_DEVICE/IP`(user↔user). Heterogeneous/multi-relation (YelpChi: R-U-R,
R-S-R, R-T-R).

**Normal + distributions.** Reviews per user/product **power-law** (most users ≤1;
McAuley & Leskovec, WWW 2013); ratings **J-shaped**, 5★-skewed; bursty inter-
arrivals with long idle gaps; small rating-deviation for honest users.

**Anomaly playbooks.** *Fake reviews* (Ott et al., ACL 2011: extreme rating,
exaggerated sentiment, low concrete detail); *Collusion rings / spammer groups*
(Mukherjee et al., WWW 2012: group co-review of a target set in a tight window;
camo = spread members across many products, interleave benign); *Sockpuppets*
(shared device/IP, stylometric similarity; camo = account aging, IP rotation);
*Singleton/burst manipulation* (Xie et al., KDD 2012); *Bot accounts*; *Return/
refund fraud* (wardrobing, empty-box). **Camouflage (CARE-GNN, Dou et al., CIKM
2020): feature + relation camouflage** (connect to benign entities to dilute the
neighborhood).

**Constraints.** review ⇒ usually a verified PURCHASED edge; |rating−product_mean|
bounded; per-user inter-arrival ≥ τ; group Jaccard(targets) > θ in window ⇒
suspicious; ¬(positive_review ∧ later return); users-per-device ≤ k; account_age ≥
min before high-impact reviews.

**Datasets.** **YelpChi** (Rayana & Akoglu SpEagle, KDD 2015); **YelpNYC/YelpZip**;
**Amazon** review/fraud graph (McAuley & Leskovec, WWW/RecSys 2013; popularized by
Dou et al. 2020 / DGL FraudDataset); **REV2** (Kumar et al., WSDM 2018 — closest
TEMPORAL rating-fraud option). *Gap: no single canonical large temporal e-commerce-
fraud benchmark — flag in paper.*

---

## 6. Healthcare / Insurance Claims Fraud  *(adversarial)*

**Entities + state.** `provider`(NPI, specialty, billing_volume, avg_claim_$,
patient_count, services/patient, charge-to-payment, peer-percentile); `patient`
(age, chronic conditions, claim_count, distinct_providers, distinct_pharmacies);
`claim`(total_charge, allowed, status); `procedure`(HCPCS/CPT, ICD-10, DRG);
`pharmacy`/`drug`(NDC, opioid_flag); `facility`.

**Events + attributes.** `SUBMITTED_CLAIM`(provider→patient: ts, CPT/HCPCS set,
ICD-10 dx, charge, allowed, place_of_service, units); `RENDERED_SERVICE`;
`PRESCRIBED`(provider→patient→drug: ts, NDC, days_supply, qty); `DISPENSED`
(pharmacy→patient); `REFERRED`(provider→provider); `PAID`(payer→provider).
Heterogeneous tripartite **provider–patient–pharmacy/code**, timestamped, weighted
by $ and units.

**Normal + distributions.** Billing volumes/$ heavy-tailed **log-normal/power-
law**; specialty determines a code "fingerprint"; plausible care pathways
(dx→procedure coherence); bounded services/beneficiary and distinct-providers/
patient per specialty; short patient–provider distance (local care). Anchors: CMS
Provider Utilization & Payment PUFs; CMS Part D opioid benchmarks.

**Anomaly playbooks (NHCAA/OIG/GAO).** *Upcoding* (bill higher-intensity code;
right-shifted code-level vs peers; camo = stay just under audit thresholds + mix
legit codes); *Phantom billing* (services never rendered; impossible daily volumes;
camo = bill real/stolen-identity patients); *Unbundling* (component codes
violating NCCI edits); *Collusion rings/kickbacks* (Anti-Kickback/Stark: dense
provider–patient–pharmacy cores, reciprocal referral loops; camo = small per-
patient amounts, recruited patients); *Doctor/pharmacy shopping* (one beneficiary,
many prescribers for controlled substances; overlapping days_supply); *Identity/
phantom-patient fraud* (patient claimed in disjoint geographies simultaneously).
Broad camo = stay within peer-percentile bands, embed in high-legit-volume
practices, distribute across many beneficiaries.

**Constraints (NCCI/medical-necessity).** unbundled component pair forbidden when
bundle billed; mutually-exclusive codes not co-billed; gender/age–procedure
compatibility; dx–procedure coherence; daily service time ≤ 24h/provider; units ≤
MUE; patient not in two places at one ts; no post-death claims; controlled-
substance days_supply overlap from distinct prescribers ≤ threshold; provider
code-mix within peer band.

**Datasets.** **CMS Medicare PUFs** (Physician/Supplier Part B, Part D prescriber,
DMEPOS — cms.gov); **LEIE** (OIG exclusions = fraud labels; joined by Bauder &
Khoshgoftaar, J. Big Data 2018 — `VERIFY`); **Synthea** (Walonoski et al., JAMIA
2018, synthetic EHR/claims — prior art for synthetic claim graphs); **CMS DE-
SynPUF**. *Gap: no standard heterogeneous-temporal healthcare-fraud GRAPH
benchmark exists — precisely the gap AGORA fills; flag in paper.*

---

### Cross-domain notes for design
- **Base rates are low everywhere** (≈1–2% anomalous) — the generator must
  preserve extreme class imbalance, and the **difficulty/camouflage knob** is the
  scientific control variable across all six.
- **Two anomaly sources**: domains 1,2,3,5,6 are adversary-driven; domain 4 is
  failure/emergent-driven — both must flow through the same schema (§5), which is
  the strongest test of domain-agnosticism (Risk R4).
- Several domains **lack a standard temporal-graph labeled benchmark** — that
  absence is the market for AGORA, and should be stated explicitly in the paper.
