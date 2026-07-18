"""The CORPUS.md acquisition manifest as structured Python data.

Every source below mirrors an entry in /CORPUS.md. URLs were verified live on
2026-06-11 (HTTP status + content type + size) unless the notes say otherwise.

License tiers (see CORPUS.md):
  TIER A — open / public domain: may be redistributed with the artifact.
  TIER B — free but copyright-reserved: fetch locally, do NOT redistribute.
  TIER C — reference-only / unclear rights: never fetched (fetch=False).

URL corrections vs CORPUS.md (verified 2026-06-11):
  * Ethereum EIPs: ERC-20/721/1155 moved from ethereum/EIPs to the
    ethereum/ERCs repo (the EIPs copies are 130-byte "moved" stubs).
  * EUR-Lex: eur-lex.europa.eu HTML endpoints answer 202 (bot challenge) to
    non-browser clients; the Publications Office CELLAR content-negotiation
    endpoint (publications.europa.eu/resource/celex/<CELEX> with
    Accept: application/xhtml+xml) serves the full XHTML text.
  * NIST nvlpubs rejects HEAD with 404 but serves GET fine (no change needed,
    just noted for verification tooling).
  * MUTCD 11th ed.: per-chapter PDF part4.pdf (Part 4 = signals, the chapter
    DOMAINS.md actually uses) — full-edition file is mutcd11thedition.pdf.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

MAX_DOWNLOAD_BYTES: int = 200 * 1024 * 1024  # hard cap per single download
USER_AGENT: str = "agora-corpus-fetcher/0.1"

DOMAINS: List[str] = [
    "finance",
    "crypto",
    "cyber",
    "transport",
    "ecommerce",
    "healthcare",
]


@dataclass(frozen=True)
class Source:
    """One acquisition target from CORPUS.md."""

    id: str
    domain: str
    name: str
    urls: List[str]
    format: str  # pdf | html | xml | json | csv | stix | markdown | zip
    license_tier: str  # "A" | "B" | "C"
    redistributable: bool  # True only for TIER A
    notes: str = ""
    fetch: bool = True  # False for TIER C (reference-only) and unverifiable
    headers: Dict[str, str] = field(default_factory=dict)  # extra HTTP headers

    def __post_init__(self) -> None:
        if self.license_tier not in ("A", "B", "C"):
            raise ValueError(f"{self.id}: bad tier {self.license_tier!r}")
        if self.redistributable and self.license_tier != "A":
            raise ValueError(f"{self.id}: only TIER A is redistributable")
        if self.license_tier == "C" and self.fetch:
            raise ValueError(f"{self.id}: TIER C sources are reference-only")


SOURCES: List[Source] = [
    # ------------------------------------------------------------------
    # Domain 1 — Finance / AML
    # ------------------------------------------------------------------
    Source(
        id="fincen_31cfr_chx",
        domain="finance",
        name="FinCEN / BSA — 31 CFR Chapter X (eCFR full XML)",
        urls=[
            "https://www.ecfr.gov/api/versioner/v1/full/2026-06-01/title-31.xml?chapter=X"
        ],
        format="xml",
        license_tier="A",
        redistributable=True,
        notes="CTR >$10k (1010.311), SAR $5k (1020.320), structuring "
        "(1010.100). US federal, public domain. eCFR Versioner API, "
        "point-in-time snapshot 2026-06-01.",
    ),
    Source(
        id="eu_amlr_2024_1624",
        domain="finance",
        name="EU AML Regulation (AMLR) 2024/1624 — CELLAR XHTML",
        urls=["http://publications.europa.eu/resource/celex/32024R1624"],
        format="html",
        license_tier="A",
        redistributable=True,
        notes="EUR-Lex reuse with attribution. Fetched via Publications "
        "Office CELLAR content negotiation (eur-lex.europa.eu blocks "
        "non-browser clients with HTTP 202).",
        headers={"Accept": "application/xhtml+xml", "Accept-Language": "eng"},
    ),
    Source(
        id="eu_amld6_2024_1640",
        domain="finance",
        name="EU AMLD6 Directive 2024/1640 — CELLAR XHTML",
        urls=["http://publications.europa.eu/resource/celex/32024L1640"],
        format="html",
        license_tier="A",
        redistributable=True,
        notes="Companion directive to AMLR in the 2024 EU AML package.",
        headers={"Accept": "application/xhtml+xml", "Accept-Language": "eng"},
    ),
    Source(
        id="bcbs_d505",
        domain="finance",
        name="Basel BCBS d505 — Sound management of ML/FT risks (BIS)",
        urls=["https://www.bis.org/bcbs/publ/d505.pdf"],
        format="pdf",
        license_tier="B",
        redistributable=False,
        notes="(c) BIS — fetch locally only, do not redistribute.",
    ),
    Source(
        id="fatf_40_recommendations",
        domain="finance",
        name="FATF — The 40 Recommendations (2012, updated)",
        urls=[
            "https://www.fatf-gafi.org/content/dam/fatf-gafi/recommendations/FATF%20Recommendations%202012.pdf.coredownload.inline.pdf"
        ],
        format="pdf",
        license_tier="B",
        redistributable=False,
        notes="fatf-gafi.org serves HTTP 403 to non-browser clients "
        "(bot protection, verified 2026-06-11); expected to fail in "
        "automated runs — download manually in a browser if needed.",
    ),
    Source(
        id="wolfsberg_standards",
        domain="finance",
        name="Wolfsberg Group standards (Correspondent Banking, Payment Transparency)",
        urls=["https://wolfsberg-group.org/"],
        format="pdf",
        license_tier="B",
        redistributable=False,
        fetch=False,
        notes="Deep PDF links are GUID-based and served by a JS-driven asset "
        "host (db.wolfsberg-group.org); not stable for automated fetch. "
        "Tier B: download manually, keep local only.",
    ),
    Source(
        id="egmont_100_cases",
        domain="finance",
        name="Egmont Group — FIUs in Action: 100 Cases",
        urls=["https://egmontgroup.org/"],
        format="pdf",
        license_tier="C",
        redistributable=False,
        fetch=False,
        notes="Tier B/C in CORPUS.md; canonical hosting unverified — "
        "reference-only here.",
    ),
    # ------------------------------------------------------------------
    # Domain 2 — Crypto / Blockchain
    # ------------------------------------------------------------------
    Source(
        id="ofac_sdn",
        domain="crypto",
        name="OFAC Specially Designated Nationals (SDN) list — CSV",
        urls=["https://www.treasury.gov/ofac/downloads/sdn.csv"],
        format="csv",
        license_tier="A",
        redistributable=True,
        notes="US gov, public domain. Includes digital-currency address "
        "entries (XBT/ETH/... tags).",
    ),
    Source(
        id="ofac_crypto_addresses",
        domain="crypto",
        name="OFAC sanctioned digital-currency addresses (0xB10C extractor)",
        urls=[
            "https://raw.githubusercontent.com/0xB10C/ofac-sanctioned-digital-currency-addresses/lists/sanctioned_addresses_ETH.json",
            "https://raw.githubusercontent.com/0xB10C/ofac-sanctioned-digital-currency-addresses/lists/sanctioned_addresses_XBT.json",
        ],
        format="json",
        license_tier="A",
        redistributable=True,
        notes="Machine extraction of US-gov public-domain SDN data; repo "
        "github.com/0xB10C/ofac-sanctioned-digital-currency-addresses "
        "('lists' branch).",
    ),
    Source(
        id="ethereum_ercs",
        domain="crypto",
        name="Ethereum ERC-20 / ERC-721 / ERC-1155 token standards",
        urls=[
            "https://raw.githubusercontent.com/ethereum/ERCs/master/ERCS/erc-20.md",
            "https://raw.githubusercontent.com/ethereum/ERCs/master/ERCS/erc-721.md",
            "https://raw.githubusercontent.com/ethereum/ERCs/master/ERCS/erc-1155.md",
        ],
        format="markdown",
        license_tier="A",
        redistributable=True,
        notes="CC0. CORRECTED vs CORPUS.md: canonical files moved from "
        "ethereum/EIPs to the ethereum/ERCs repo (EIPs copies are stubs).",
    ),
    Source(
        id="sok_defi_attacks",
        domain="crypto",
        name="SoK: Decentralized Finance Attacks (Zhou et al., IEEE S&P 2023)",
        urls=["https://arxiv.org/pdf/2208.13035"],
        format="pdf",
        license_tier="A",
        redistributable=True,
        notes="arXiv open access; DeFi attack taxonomy (flash loan, oracle, "
        "reentrancy, governance). Incident-dataset repo license unverified.",
    ),
    Source(
        id="fatf_vasp_guidance",
        domain="crypto",
        name="FATF — Updated Guidance for a Risk-Based Approach to VAs and VASPs (2021)",
        urls=[
            "https://www.fatf-gafi.org/content/dam/fatf-gafi/guidance/Updated-Guidance-VA-VASP.pdf.coredownload.inline.pdf"
        ],
        format="pdf",
        license_tier="B",
        redistributable=False,
        notes="fatf-gafi.org bot protection (403 to scripts); expected to "
        "fail in automated runs.",
    ),
    Source(
        id="defihacklabs",
        domain="crypto",
        name="DeFiHackLabs reproduced incident PoCs + Rekt leaderboard",
        urls=[
            "https://github.com/SunWeb3Sec/DeFiHackLabs",
            "https://rekt.news/leaderboard/",
        ],
        format="html",
        license_tier="C",
        redistributable=False,
        fetch=False,
        notes="DeFiHackLabs is UNLICENSED (all rights reserved); Rekt "
        "proprietary. Reference-only.",
    ),
    Source(
        id="etherscan_labels",
        domain="crypto",
        name="Etherscan entity labels (exchange/mixer/scam/bridge)",
        urls=["https://etherscan.io/labelcloud"],
        format="json",
        license_tier="C",
        redistributable=False,
        fetch=False,
        notes="Proprietary, anti-scrape ToS; community mirrors stale with "
        "unclear data rights. Reference-only.",
    ),
    Source(
        id="chainalysis_crime_report",
        domain="crypto",
        name="Chainalysis Crypto Crime Report (2026 ed.)",
        urls=["https://www.chainalysis.com/blog/"],
        format="pdf",
        license_tier="C",
        redistributable=False,
        fetch=False,
        notes="Proprietary, form-gated PDF. Reference-only (free blog "
        "chapters may be read manually).",
    ),
    # ------------------------------------------------------------------
    # Domain 3 — Cybersecurity (IDS + APT)
    # ------------------------------------------------------------------
    Source(
        id="mitre_attack_enterprise",
        domain="cyber",
        name="MITRE ATT&CK Enterprise — STIX 2.1 JSON",
        urls=[
            "https://raw.githubusercontent.com/mitre-attack/attack-stix-data/master/enterprise-attack/enterprise-attack.json"
        ],
        format="stix",
        license_tier="A",
        redistributable=True,
        notes="MITRE Terms of Use (attribution required). ~53 MB; master = "
        "latest release of the enterprise matrix.",
    ),
    Source(
        id="nist_sp800_61r3",
        domain="cyber",
        name="NIST SP 800-61 Rev 3 — Incident Response Recommendations (2025)",
        urls=[
            "https://nvlpubs.nist.gov/nistpubs/SpecialPublications/NIST.SP.800-61r3.pdf"
        ],
        format="pdf",
        license_tier="A",
        redistributable=True,
        notes="US gov public domain. nvlpubs answers 404 to HEAD but serves "
        "GET normally (verified 2026-06-11).",
    ),
    Source(
        id="nist_sp800_61r2",
        domain="cyber",
        name="NIST SP 800-61 Rev 2 — Computer Security Incident Handling Guide",
        urls=[
            "https://nvlpubs.nist.gov/nistpubs/SpecialPublications/NIST.SP.800-61r2.pdf"
        ],
        format="pdf",
        license_tier="A",
        redistributable=True,
        notes="Classic 4-phase IR lifecycle (superseded by r3 but cited).",
    ),
    Source(
        id="nist_sp800_94",
        domain="cyber",
        name="NIST SP 800-94 — Guide to IDPS (2007)",
        urls=[
            "https://nvlpubs.nist.gov/nistpubs/Legacy/SP/nistspecialpublication800-94.pdf"
        ],
        format="pdf",
        license_tier="A",
        redistributable=True,
        notes="3 detection methodologies + 4 IDPS classes.",
    ),
    Source(
        id="mitre_capec",
        domain="cyber",
        name="MITRE CAPEC — attack pattern catalog (XML)",
        urls=["https://capec.mitre.org/data/xml/capec_latest.xml"],
        format="xml",
        license_tier="A",
        redistributable=True,
        notes="559 attack patterns; MITRE terms, attribution.",
    ),
    Source(
        id="nvd_cve_sample",
        domain="cyber",
        name="NVD CVE API 2.0 — bounded sample (2,000 most recently published 2025 CVEs)",
        urls=[
            "https://services.nvd.nist.gov/rest/json/cves/2.0?pubStartDate=2025-01-01T00:00:00.000&pubEndDate=2025-04-30T23:59:59.999&resultsPerPage=2000"
        ],
        format="json",
        license_tier="A",
        redistributable=True,
        notes="Bounded subset per CORPUS.md guidance (full CVE list is "
        "multi-GB). No API key (public rate limits).",
    ),
    Source(
        id="lockheed_kill_chain",
        domain="cyber",
        name="Lockheed Martin — Intelligence-Driven Defense / Cyber Kill Chain (Hutchins et al. 2011)",
        urls=[
            "https://www.lockheedmartin.com/content/dam/lockheed-martin/rms/documents/cyber/LM-White-Paper-Intel-Driven-Defense.pdf"
        ],
        format="pdf",
        license_tier="B",
        redistributable=False,
        notes="(c) Lockheed Martin — cite, do not redistribute.",
    ),
    Source(
        id="cicids2017",
        domain="cyber",
        name="CICIDS2017 / CSE-CIC-IDS2018 labeled IDS flows (CIC/UNB)",
        urls=["https://www.unb.ca/cic/datasets/ids-2017.html"],
        format="csv",
        license_tier="C",
        redistributable=False,
        fetch=False,
        notes="Research-use license requires registration/citation; multi-GB "
        "PCAP/CSV — reference-only for the text corpus.",
    ),
    # ------------------------------------------------------------------
    # Domain 4 — Transportation / Mobility
    # ------------------------------------------------------------------
    Source(
        id="mutcd_11th_part4",
        domain="transport",
        name="MUTCD 11th ed. (FHWA 2023) — Part 4: Highway Traffic Signals",
        urls=["https://mutcd.fhwa.dot.gov/pdfs/11th_Edition/part4.pdf"],
        format="pdf",
        license_tier="A",
        redistributable=True,
        notes="Ch. 4C signal warrants + timing/clearance standards. Per-part "
        "PDF (full edition = mutcd11thedition.pdf, ~24 MB).",
    ),
    Source(
        id="fhwa_tat_vol3",
        domain="transport",
        name="FHWA Traffic Analysis Toolbox Vol. III — microsimulation guidelines",
        urls=[
            "https://ops.fhwa.dot.gov/trafficanalysistools/tat_vol3/vol3_guidelines.pdf"
        ],
        format="pdf",
        license_tier="A",
        redistributable=True,
        notes="Calibration/validation criteria incl. GEH thresholds. US gov.",
    ),
    Source(
        id="sumo_car_following",
        domain="transport",
        name="SUMO docs — car-following models (Krauss/IDM/Wiedemann) + vehicle params",
        urls=[
            "https://sumo.dlr.de/docs/Car-Following-Models.html",
            "https://sumo.dlr.de/docs/Definition_of_Vehicles%2C_Vehicle_Types%2C_and_Routes.html",
        ],
        format="html",
        license_tier="A",
        redistributable=True,
        notes="Eclipse SUMO docs (CC/EPL-2.0). accel/decel/sigma/tau/minGap "
        "parameter semantics. Cite Krauss (1998) thesis separately.",
    ),
    Source(
        id="ngsim_metadata",
        domain="transport",
        name="NGSIM vehicle trajectories — dataset metadata + column dictionary (US DOT)",
        urls=["https://data.transportation.gov/api/views/8ect-6jqj.json"],
        format="json",
        license_tier="A",
        redistributable=True,
        notes="Socrata views API: dataset description, citation, and full "
        "column dictionary for the I-80/US-101/Lankershim/Peachtree "
        "trajectories. Public domain. Known accel noise — pick a "
        "reconstruction for calibration use.",
    ),
    Source(
        id="hcm7",
        domain="transport",
        name="Highway Capacity Manual, 7th ed. (TRB 2022)",
        urls=["https://www.trb.org/publications/hcm7thedition.aspx"],
        format="pdf",
        license_tier="C",
        redistributable=False,
        fetch=False,
        notes="Paywalled (~$250) (c) TRB. Use parameter VALUES as facts "
        "(LOS thresholds, ~2,400 pc/h/ln, sat-flow ~1,900); never ingest "
        "the copyrighted text.",
    ),
    Source(
        id="metr_la_pems_bay",
        domain="transport",
        name="METR-LA / PEMS-BAY sensor graphs (DCRNN repo)",
        urls=["https://github.com/liyaguang/DCRNN"],
        format="csv",
        license_tier="C",
        redistributable=False,
        fetch=False,
        notes="Redistribution rights unverified (PeMS data is "
        "account-gated upstream). Reference-only.",
    ),
    # ------------------------------------------------------------------
    # Domain 5 — E-commerce / Review & Rating Fraud
    # ------------------------------------------------------------------
    Source(
        id="ftc_16cfr_465",
        domain="ecommerce",
        name="FTC 16 CFR Part 465 — Consumer Reviews and Testimonials Rule (eCFR XML)",
        urls=[
            "https://www.ecfr.gov/api/versioner/v1/full/2026-06-01/title-16.xml?part=465"
        ],
        format="xml",
        license_tier="A",
        redistributable=True,
        notes="THE review-fraud taxonomy: fake/AI reviews, buying reviews, "
        "insider reviews, suppression, fake social indicators. Public domain.",
    ),
    Source(
        id="ftc_16cfr_255",
        domain="ecommerce",
        name="FTC 16 CFR Part 255 — Endorsement Guides (eCFR XML)",
        urls=[
            "https://www.ecfr.gov/api/versioner/v1/full/2026-06-01/title-16.xml?part=255"
        ],
        format="xml",
        license_tier="A",
        redistributable=True,
        notes="Material-connection disclosure logic (rev. 2023).",
    ),
    Source(
        id="ftc_endorsement_faq",
        domain="ecommerce",
        name="FTC — Endorsement Guides: What People Are Asking (HTML)",
        urls=[
            "https://www.ftc.gov/business-guidance/resources/ftcs-endorsement-guides-what-people-are-asking"
        ],
        format="html",
        license_tier="A",
        redistributable=True,
        notes="Operational FAQ incl. AI/virtual-influencer endorsers.",
    ),
    Source(
        id="ott_2011_deceptive_spam",
        domain="ecommerce",
        name="Ott et al. (ACL 2011) — Finding Deceptive Opinion Spam",
        urls=["https://aclanthology.org/P11-1032.pdf"],
        format="pdf",
        license_tier="A",
        redistributable=True,
        notes="ACL Anthology open access (CC BY). Deception-feature taxonomy. "
        "Companion datasets (YelpChi etc.) are by-request — Tier C.",
    ),
    Source(
        id="platform_review_policies",
        domain="ecommerce",
        name="Amazon Community Guidelines / Yelp Content Guidelines",
        urls=[
            "https://www.amazon.com/gp/help/customer/display.html?nodeId=GLHXEX85MENUE4XF",
            "https://www.yelp.com/guidelines",
        ],
        format="html",
        license_tier="C",
        redistributable=False,
        fetch=False,
        notes="Proprietary platform policy text, anti-scrape ToS. "
        "Reference-only.",
    ),
    # ------------------------------------------------------------------
    # Domain 6 — Healthcare / Insurance Fraud
    # ------------------------------------------------------------------
    Source(
        id="cms_ncci_mue_practitioner",
        domain="healthcare",
        name="CMS NCCI MUE — practitioner services max-units table (2026 Q3)",
        urls=[
            "https://www.cms.gov/files/zip/medicare-ncci-2026-q3-practitioner-services-mue-table.zip"
        ],
        format="zip",
        license_tier="A",
        redistributable=True,
        notes="Implausible-quantity rules (max units/day per HCPCS). Zip "
        "contains CSV + XLSX; quarterly URL — re-VERIFY each quarter. Some "
        "DME MUE values confidential (not in this file).",
    ),
    Source(
        id="cms_ncci_ptp_practitioner",
        domain="healthcare",
        name="CMS NCCI PTP — practitioner procedure-to-procedure edits (2026 Q3, file 1 of 4)",
        urls=[
            "https://www.cms.gov/files/zip/medicare-ncci-2026q3-practitioner-ptp-edits-ccipra-v322r0-f1.zip"
        ],
        format="zip",
        license_tier="A",
        redistributable=True,
        notes="Unbundling rules: Column1/Column2 code pairs + modifier "
        "indicator. ~20 MB zip; files 2-4 omitted to bound corpus size.",
    ),
    Source(
        id="oig_leie",
        domain="healthcare",
        name="HHS-OIG LEIE — excluded individuals/entities (CSV)",
        urls=["https://oig.hhs.gov/exclusions/downloadables/UPDATED.csv"],
        format="csv",
        license_tier="A",
        redistributable=True,
        notes="Ground-truth bad-actor roster (~15 MB). Public domain.",
    ),
    Source(
        id="icd10cm_codes",
        domain="healthcare",
        name="ICD-10-CM 2026 code descriptions (CMS, tabular order)",
        urls=[
            "https://www.cms.gov/files/zip/2026-code-descriptions-tabular-order.zip"
        ],
        format="zip",
        license_tier="A",
        redistributable=True,
        notes="Diagnosis vocabulary (icd10cm_codes_2026.txt inside zip).",
    ),
    Source(
        id="hcpcs_level2",
        domain="healthcare",
        name="HCPCS Level II — April 2026 alpha-numeric file (CMS)",
        urls=[
            "https://www.cms.gov/files/zip/april-2026-alpha-numeric-hcpcs-file.zip"
        ],
        format="zip",
        license_tier="A",
        redistributable=True,
        notes="Service/supply vocabulary (HCPC2026_APR_ANWEB.txt inside zip).",
    ),
    Source(
        id="ecfr_42cfr_1001",
        domain="healthcare",
        name="42 CFR Part 1001 — OIG exclusions + Anti-Kickback safe harbors (eCFR XML)",
        urls=[
            "https://www.ecfr.gov/api/versioner/v1/full/2026-06-01/title-42.xml?part=1001"
        ],
        format="xml",
        license_tier="A",
        redistributable=True,
        notes="Safe harbors at 1001.952; pairs with Stark 42 USC 1395nn and "
        "AKS 42 USC 1320a-7b.",
    ),
    Source(
        id="cms_pub100_04_ch1",
        domain="healthcare",
        name="CMS Pub. 100-04 Claims Processing Manual — Chapter 1 (general billing)",
        urls=[
            "https://www.cms.gov/regulations-and-guidance/guidance/manuals/downloads/clm104c01.pdf"
        ],
        format="pdf",
        license_tier="A",
        redistributable=True,
        notes="Compliant-claim rules. Pin transmittal version at fetch time.",
    ),
    Source(
        id="cpt_descriptors",
        domain="healthcare",
        name="CPT long descriptors (AMA)",
        urls=["https://www.ama-assn.org/practice-management/cpt"],
        format="csv",
        license_tier="C",
        redistributable=False,
        fetch=False,
        notes="AMA-licensed/paywalled. Codes may be referenced as facts; "
        "descriptor TEXT must never be ingested or bundled. (Tier B in "
        "CORPUS.md but unfetchable without a license -> reference-only.)",
    ),
    Source(
        id="nhcaa_fraud_taxonomy",
        domain="healthcare",
        name="NHCAA — The Challenge of Health Care Fraud",
        urls=[
            "https://www.nhcaa.org/tools-insights/about-health-care-fraud/the-challenge-of-health-care-fraud/"
        ],
        format="html",
        license_tier="C",
        redistributable=False,
        fetch=False,
        notes="Trade-association content; detailed schemes member-gated. "
        "Reference-only.",
    ),
]


def by_domain(domain: str) -> List[Source]:
    """All sources for one domain."""
    return [s for s in SOURCES if s.domain == domain]


def get(source_id: str) -> Optional[Source]:
    """Look up a single source by id."""
    for s in SOURCES:
        if s.id == source_id:
            return s
    return None


def fetchable(
    domains: Optional[List[str]] = None, tiers: Optional[List[str]] = None
) -> List[Source]:
    """Sources eligible for download (fetch=True, tier A/B), filtered."""
    out: List[Source] = []
    for s in SOURCES:
        if not s.fetch or s.license_tier == "C":
            continue
        if domains and s.domain not in domains:
            continue
        if tiers and s.license_tier not in tiers:
            continue
        out.append(s)
    return out
