# CORPUS.md — RAG Knowledge-Base Acquisition Manifest

**The RAG corpus is REAL.** AGORA's zero-code domain migration is powered by a
curated knowledge base of **actual, downloaded authoritative standards,
regulations, and taxonomy documents** — not a hand-waved "corpus". *Building this
corpus is a first-class engineering task and part of the contribution* (see
`mustread.txt` §9). This file is the executable acquisition manifest: what to
fetch, in what format, and under what license.

## How we ship it (the licensing model — drives what the public artifact can contain)

A SIGMOD artifact must be publicly released, so the corpus is assembled by a
**fetch script** (`agora rules corpus --fetch`) that pulls from official sources at
build time, rather than bundling copyrighted PDFs. Three tiers:

- **TIER A — OPEN** (public domain / open license): *bundle freely* in the released
  artifact and reproducibility package. Mostly US-federal + MITRE + EU + CC0.
- **TIER B — FREE, COPYRIGHT-RESERVED**: *fetch locally for the RAG; DO NOT
  redistribute*. The artifact ships the URL + a fetcher, not the file.
- **TIER C — REFERENCE-ONLY / UNCLEAR**: *do not bundle*; link/cite only; use for
  human reference and rule cross-checking.

Prefer machine-readable sources (STIX/JSON/XML/CSV/code-sets) as the backbone;
PDF-only sources need a parse step. `VERIFY` = confirm URL/edition at fetch time.

---

## Domain 1 — Finance / AML
- **FATF — The 40 Recommendations** (FATF; 2012, upd. through Oct 2025). Obliged-entity types, CDD/KYC, beneficial ownership, PEPs, wire/travel-rule fields (R.16), thresholds (EUR/USD 15,000 occasional; 1,000 wire). *PDF*. **TIER B**. fatf-gafi.org `VERIFY` live PDF name.
- **FATF typologies / red-flag reports** — *Professional Money Laundering* (2018), *TBML Risk Indicators* (2021), *Virtual Assets Red-Flag Indicators* (2020). The anomaly playbooks. *PDF*. **TIER B**.
- **FinCEN / Bank Secrecy Act — 31 CFR Chapter X** (FinCEN/Treasury). **CTR >$10k** (1010.311), **SAR $5k** banks (1020.320), **structuring** def (1010.100). FFIEC BSA/AML Manual App. F/G red flags + structuring. *Machine-readable XML/JSON via eCFR API*. **TIER A** (US federal, public domain). ecfr.gov/current/title-31/.../chapter-X
- **Wolfsberg Group** — Correspondent Banking Principles (2022), Payment Transparency Standards (2023), Monitoring Statement Pt II (2025). *PDF*. **TIER B**. wolfsberg-group.org `VERIFY` deep links.
- **Egmont Group** — *FIUs in Action: 100 Cases* + PML Facilitators Bulletin (2019). Sanitized real ML cases + 6 recurring indicators. *PDF*. **TIER B/C** (sanitized cases, awareness-use). `VERIFY` canonical hosting.
- **EU AML** — 4/5AMLD (2015/849, 2018/843), 2024 package: AMLR 2024/1624 + AMLD6 2024/1640 + AMLA 2024/1620. *Machine-readable XML (Formex) + ELI + CELLAR API*. **TIER A** (reuse w/ attribution). eur-lex.europa.eu (CELEX IDs in DOMAINS.md research).
- **Basel — BCBS d505** (sound ML/TF risk mgmt; BIS). *PDF*. **TIER B**. Note: *Basel AML Index* = Basel Institute on Governance (≠ BIS), partly paywalled.

## Domain 2 — Crypto / Blockchain
- **FATF VA/VASP Guidance** (2021) + **Targeted Update** (June 2025). VASP types, travel rule (R.15/R.16), VA de-minimis USD/EUR 1,000. *PDF*. **TIER B**.
- **OFAC SDN + Digital Currency Address list** (Treasury/OFAC). Sanctioned crypto addresses (XBT/ETH/USDT/… tags) → blocklist + entity attribution. *Machine-readable XML/CSV*; crypto-only extractor repo (TXT/JSON). **TIER A** (US gov, public domain). treasury SDN + github.com/0xB10C/ofac-sanctioned-digital-currency-addresses
- **Chainalysis Crypto Crime Report** (2026 ed.). Scam/ransomware/DeFi-exploit/sanctions typologies + thresholds. *Gated PDF; free ungated blog chapters*. **TIER B/C** (proprietary; gated form).
- **SoK: DeFi Attacks** (Zhou et al., IEEE S&P 2023). DeFi attack taxonomy (flash loan, oracle, reentrancy, governance). *arXiv PDF + incident dataset repo*. **TIER A** (arXiv; dataset repo license `VERIFY`). arxiv.org/abs/2208.13035
- **DeFiHackLabs** (~700 reproduced incident PoCs) + **Rekt leaderboard**. Real attack mechanics + USD-loss ranking. *Solidity/Git + web*. **TIER C** (DeFiHackLabs = UNLICENSED/all-rights-reserved; Rekt proprietary) — reference only.
- **Ethereum EIPs** — ERC-20/721/1155. Canonical event/method signatures (`Transfer`, `Approval`, …) for event schemas. *Markdown/HTML*. **TIER A (CC0)**. eips.ethereum.org
- **Etherscan labels** — entity attribution (exchange/mixer/scam/bridge). *Community JSON/CSV mirror*. **TIER C** (proprietary, anti-scrape, mirror stale + data rights unclear).

## Domain 3 — Cybersecurity (IDS + APT)
- **MITRE ATT&CK** (v19.x; MITRE). Tactics→techniques→sub-techniques, groups, software, mitigations, detections, campaigns. *Machine-readable STIX 2.1 JSON* (`mitre-attack/attack-stix-data`, `pip install mitreattack-python`). **TIER A** (Terms of Use, attribution). `VERIFY` current version.
- **Lockheed Martin Cyber Kill Chain** (Hutchins et al., 2011). 7-phase intrusion model + courses-of-action. *PDF*. **TIER B** (© Lockheed; cite, don't redistribute).
- **NIST SP 800-61** (Rev 3 final 2025; + Rev 2 for the classic 4-phase IR lifecycle). *PDF*. **TIER A** (public domain). csrc.nist.gov
- **NIST SP 800-94** (IDPS, 2007 — Rev 1 withdrawn). 3 detection methodologies (signature / anomaly / stateful-protocol) + 4 IDPS classes. *PDF*. **TIER A**.
- **MITRE CAR / CAPEC / CWE**. CAR = detection analytics + pseudocode (YAML, **Apache-2.0**); CAPEC = 559 attack patterns (XML); CWE = weakness taxonomy (XML) + REST API. *Machine-readable*. **TIER A**.
- **CICIDS2017 (+ CSE-CIC-IDS2018)** (CIC/UNB). Labeled flows, 80+ CICFlowMeter features; attack families (brute force, DoS/DDoS, web, infiltration, botnet, portscan, heartbleed). *CSV + PCAP* (IDS2018 on AWS Open Data). **TIER A** (research-use + cite). unb.ca/cic/datasets
- **CVE / NVD** — CVE List (JSON 5.x, `CVEProject/cvelistV5`) + NVD API 2.0 (CVSS/CWE/CPE). *Machine-readable JSON*. **TIER A**.

## Domain 4 — Transportation / Mobility
- **Highway Capacity Manual, 7th ed.** (TRB, 2022). LOS A–F thresholds, capacities (~2,400 pc/h/ln), sat-flow (~1,900), delay/queue/density formulas — the "normal vs congested" parameters. *eBook*. **TIER B/PAYWALLED (~$250)** — use parameter VALUES as facts; do NOT ingest full copyrighted text.
- **FHWA Traffic Flow Theory Monograph + Traffic Analysis Toolbox** (FHWA). Car-following, fundamental diagram, shockwaves; calibration/validation criteria (GEH thresholds). *Free PDF*. **TIER A** (US gov). ops.fhwa.dot.gov/trafficanalysistools `VERIFY` monograph hosting.
- **SUMO docs + Krauss model spec** (DLR/Eclipse SUMO). Safe-velocity eqn + params (`accel`,`decel`,`emergencyDecel`,`sigma`,`tau`,`minGap`); IDM/Wiedemann alternatives. *Web docs + code*. **TIER A** (CC/EPL-2.0). sumo.dlr.de/docs/Car-Following-Models.html. Cite Krauß (1998) thesis separately.
- **NGSIM trajectories — data dictionary** (FHWA/US DOT). 10 Hz real trajectories (I-80, US-101, Lankershim, Peachtree); columns for gap/headway/accel/lane-change → calibration + anomalous-maneuver ground truth. *CSV + dict*. **TIER A** (public domain). catalog.data.gov NGSIM. Note known accel-noise; pick a reconstruction `VERIFY`.
- **PeMS / METR-LA / PEMS-BAY docs** (Caltrans / DCRNN repo). Loop-detector 5-min flow/occupancy/speed; sensor graphs + **adjacency matrices** (207 / 325 sensors). *PeMS account-gated; METR-LA/PEMS-BAY via github.com/liyaguang/DCRNN HDF5/CSV*. **TIER A-research** (`VERIFY` redistribution).
- **MUTCD 11th ed.** (FHWA, 2023, Rev 1 2025). Ch. 4C signal warrants (9 warrants, volume thresholds), timing/clearance standards. *Free PDF*. **TIER A**. mutcd.fhwa.dot.gov/pdfs/11th_Edition

## Domain 5 — E-commerce / Review & Rating Fraud
- **FTC 16 CFR Part 465 — Consumer Reviews & Testimonials Rule** (eff. Oct 2024). THE review-fraud taxonomy: fake/AI reviews, buying/selling reviews, undisclosed insider reviews, fake independent sites, review suppression, fake social indicators; penalties. *Machine-readable via eCFR API + FRN PDF*. **TIER A** (US federal, public domain). ecfr.gov title-16 part-465
- **FTC 16 CFR Part 255 — Endorsement Guides** (rev. 2023) + "What People Are Asking". Material-connection disclosure logic; AI/virtual-influencer endorsers. *eCFR + PDF*. **TIER A**.
- **Amazon Community Guidelines / Yelp Content Guidelines** (platforms). Operational prohibited-behavior definitions (compensated reviews, manipulation, review rings, coordinated voting; Yelp recommendation-software + Consumer Alerts). *HTML*. **TIER C** (proprietary, anti-scrape, reference-only). `VERIFY` current pages.
- **Academic review-fraud taxonomies** — Rayana & Akoglu (KDD 2015, YelpChi/Zip/NYC), Mukherjee et al. (ICWSM 2013), Jindal & Liu (WSDM 2008), Ott et al. (ACL 2011). Feature/behavior taxonomies + labeled datasets. *Papers free; datasets by-request*. **TIER A-paper / C-dataset** (`VERIFY` dataset rights).

## Domain 6 — Healthcare / Insurance Fraud
- **CMS NCCI — PTP edits, Add-On edits, MUE** (CMS, quarterly). PTP code-pairs (Column1/2 + modifier indicator) = unbundling rules; MUE max-units/day = implausible-quantity rules; + NCCI Policy Manual (rationale). *Downloadable CSV/Excel*. **TIER A** (public domain; some DME MUE values confidential). cms.gov/.../national-correct-coding-initiative-ncci-edits
- **CMS Medicare Manuals — Pub. 100-04 (Claims Processing) + 100-08 (Program Integrity)** (CMS). Compliant-claim rules; fraud-vs-abuse definitions; medical-review processes. *PDF chapters*. **TIER A**. Pin transmittal version.
- **OIG Work Plan + LEIE** (HHS-OIG). Active fraud-focus areas; excluded-provider roster = ground-truth bad-actor node attribute. *LEIE CSV + Work Plan HTML*. **TIER A**. oig.hhs.gov/exclusions
- **NHCAA — "The Challenge of Health Care Fraud"** (trade assoc.). Provider fraud-scheme taxonomy (phantom billing, upcoding, unbundling, unnecessary services, kickbacks, falsified dx). *HTML summary free; detailed schemes member-gated*. **TIER B/C**.
- **Code sets — ICD-10-CM/PCS (CDC/CMS), HCPCS Level II (CMS), CPT (AMA)**. Diagnosis/procedure/service vocabularies for realistic claims + dx-procedure consistency checks. *ICD/HCPCS free machine-readable*; **CPT long descriptors AMA-licensed/PAYWALLED — codes only, not text**. **TIER A (ICD/HCPCS) / B (CPT)**.
- **Stark Law (42 USC 1395nn) + Anti-Kickback (42 USC 1320a-7b)** + 42 CFR safe harbors. Self-referral + kickback = graph/edge anomaly patterns. *eCFR XML/JSON + US Code*. **TIER A**.
- **CMS Public Utilization Data Dictionaries** (Physician & Other Practitioners; Part D; DMEPOS). Real NPI×HCPCS billing distributions = empirical "normal" baselines for overutilization outliers. *CSV bulk + API + dict*. **TIER A** (privacy-suppressed <11; ~2yr lag). data.cms.gov

---

## Cross-domain ingestion summary
- **OPEN backbone (bundle freely):** all US-federal (FinCEN/eCFR, OFAC, NIST, MITRE ATT&CK/CAPEC/CWE/CAR, CMS NCCI/manuals/LEIE/utilization, FTC, MUTCD, NGSIM, FHWA, ICD-10/HCPCS, US Code), EUR-Lex (attribution), Ethereum EIPs (CC0), CICIDS (cite), CVE/NVD.
- **Free but copyright-reserved (fetch locally, don't redistribute):** FATF, Wolfsberg, Egmont, BIS, Lockheed Kill Chain, Chainalysis, NIST is fine (gov) — and **paywalled/licensed**: HCM 7 (TRB), CPT descriptors (AMA).
- **Reference-only (don't bundle):** DeFiHackLabs, Rekt, Etherscan labels, Amazon/Yelp policy text, NHCAA gated schemes, by-request academic datasets.
- **Best machine-readable (RAG backbone):** MITRE STIX/XML, OFAC XML/CSV, eCFR + EUR-Lex XML/JSON, CVE JSON 5.x, NVD API, CICIDS CSV, CMS NCCI/utilization CSV, Ethereum EIP markdown. Everything PDF-only (FATF/Wolfsberg/BIS/Lockheed/HCM) needs a parse step.

*This manifest is consumed by the corpus fetcher (build milestone, `mustread.txt`
§18 M3). Each fetched document is chunked, embedded, and indexed (faiss) per
`mustread.txt` §9; provenance (source URL + version + license tier) is recorded so
every extracted rule is traceable to an authoritative source — the grounding that
makes zero-code domain migration trustworthy and reproducible.*
