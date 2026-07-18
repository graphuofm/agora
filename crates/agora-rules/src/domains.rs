//! Registry of the six built-in domains (blueprint §14) and loading of their
//! embedded rule bases. Rule bases are compiled into the binary with
//! `include_str!` so a downloaded AGORA needs no extra files.

use crate::rulebase::RuleBase;

#[derive(Debug, Clone)]
pub struct DomainInfo {
    pub id: &'static str,
    pub name: &'static str,
    pub anomaly_source: &'static str,
    pub summary: &'static str,
    /// Embedded rule-base YAML; `None` until the milestone that ships it (M4).
    pub rulebase_yaml: Option<&'static str>,
}

pub fn builtin_domains() -> Vec<DomainInfo> {
    vec![
        DomainInfo {
            id: "finance",
            name: "Finance / AML",
            anomaly_source: "adversarial",
            summary: "Payment network; FATF typologies: structuring, layering, fan-in/out, round-tripping, mules.",
            rulebase_yaml: Some(include_str!("../domains/finance.yaml")),
        },
        DomainInfo {
            id: "crypto",
            name: "Crypto / Blockchain (Ethereum/DeFi)",
            anomaly_source: "adversarial",
            summary: "On-chain txns/DeFi; wash trading, phishing, mixing, pump-and-dump, rug pull, Ponzi.",
            rulebase_yaml: Some(include_str!("../domains/crypto.yaml")),
        },
        DomainInfo {
            id: "cyber",
            name: "Cybersecurity (IDS + APT)",
            anomaly_source: "adversarial",
            summary: "Flows/auth/process events; MITRE ATT&CK kill-chains: scan, brute force, lateral movement, C2, exfil.",
            rulebase_yaml: Some(include_str!("../domains/cyber.yaml")),
        },
        DomainInfo {
            id: "transport",
            name: "Transportation / Mobility",
            anomaly_source: "natural (failure-driven)",
            summary: "Traffic/trips; congestion cascades, incidents, sensor faults + a thin adversarial tail.",
            rulebase_yaml: Some(include_str!("../domains/transport.yaml")),
        },
        DomainInfo {
            id: "ecommerce",
            name: "E-commerce / Review Fraud",
            anomaly_source: "adversarial",
            summary: "User–product–review; fake reviews, collusion rings, sockpuppets, return fraud.",
            rulebase_yaml: Some(include_str!("../domains/ecommerce.yaml")),
        },
        DomainInfo {
            id: "healthcare",
            name: "Healthcare / Insurance Fraud",
            anomaly_source: "adversarial",
            summary: "Provider–patient–claim; upcoding, phantom billing, unbundling, kickback rings, doctor shopping.",
            rulebase_yaml: Some(include_str!("../domains/healthcare.yaml")),
        },
    ]
}

/// Load a built-in domain's rule base by id, or a custom one from a path.
pub fn load_builtin_rulebase(domain: &str) -> anyhow::Result<RuleBase> {
    if let Some(info) = builtin_domains().into_iter().find(|d| d.id == domain) {
        match info.rulebase_yaml {
            Some(yaml) => return RuleBase::from_yaml(yaml),
            None => anyhow::bail!(
                "domain `{domain}` is registered but its rule base is not shipped yet \
                 (lands at milestone M4); available now: {}",
                shipped_ids().join(", ")
            ),
        }
    }
    // Not a built-in id: treat as a path to a custom compiled rule base.
    let p = std::path::Path::new(domain);
    if p.exists() {
        let text = std::fs::read_to_string(p)?;
        return RuleBase::from_yaml(&text);
    }
    anyhow::bail!(
        "unknown domain `{domain}`: expected one of [{}] or a path to a rule-base YAML",
        builtin_domains().iter().map(|d| d.id).collect::<Vec<_>>().join(", ")
    )
}

fn shipped_ids() -> Vec<&'static str> {
    builtin_domains()
        .iter()
        .filter(|d| d.rulebase_yaml.is_some())
        .map(|d| d.id)
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn finance_rulebase_parses_and_validates() {
        let rb = load_builtin_rulebase("finance").unwrap();
        assert_eq!(rb.meta.id, "finance");
        assert_eq!(rb.entity_types.len(), 3);
        assert!(rb.adversaries.len() >= 4);
        assert!((rb.control.prevalence - 0.02).abs() < 1e-9);
    }

    /// Untrusted-config hardening (§6): malformed rule bases must be REJECTED
    /// at load with an actionable error — never reach the engine and panic.
    #[test]
    fn malformed_rulebases_error_not_panic() {
        // Each mutation takes the valid finance base and corrupts one field
        // that, before the loader hardening, would have panicked at world
        // build (alias-table construction, sampler compile, etc.).
        let cases: Vec<(&str, Box<dyn Fn(&mut RuleBase)>)> = vec![
            (
                "all-zero categorical weights",
                Box::new(|rb: &mut RuleBase| {
                    if let crate::rulebase::AttributeKind::Categorical { weights, .. } =
                        &mut rb.entity_types[0].attributes[0].kind
                    {
                        weights.iter_mut().for_each(|w| *w = 0.0);
                    }
                }),
            ),
            (
                "NaN categorical weight",
                Box::new(|rb: &mut RuleBase| {
                    if let crate::rulebase::AttributeKind::Categorical { weights, .. } =
                        &mut rb.entity_types[0].attributes[0].kind
                    {
                        weights[0] = f64::NAN;
                    }
                }),
            ),
            (
                "negative log_normal sigma in state",
                Box::new(|rb: &mut RuleBase| {
                    rb.entity_types[0].state[0].init =
                        crate::rulebase::Distribution::LogNormal { mu: 0.0, sigma: -1.0 };
                }),
            ),
            (
                "behavior event weights sum to 0",
                Box::new(|rb: &mut RuleBase| {
                    rb.behaviors[0].events.iter_mut().for_each(|e| e.weight = 0.0);
                }),
            ),
            (
                "infinite uniform bound in adversary stage",
                Box::new(|rb: &mut RuleBase| {
                    // Built-in domains express stage rates as `activity_multiplier`;
                    // the bound check must reject a non-finite dist there too.
                    rb.adversaries[0].stages[0].activity_multiplier =
                        Some(crate::rulebase::Distribution::Uniform { min: 1.0, max: f64::INFINITY });
                    rb.adversaries[0].stages[0].rate_per_day = None;
                }),
            ),
            (
                "zero communities in placement",
                Box::new(|rb: &mut RuleBase| {
                    rb.control.placement = crate::rulebase::Placement::Clustered { n_communities: 0 };
                }),
            ),
        ];
        for (name, mutate) in cases {
            let mut rb = load_builtin_rulebase("finance").unwrap();
            mutate(&mut rb);
            let res = rb.validate();
            assert!(res.is_err(), "malformed case `{name}` should be rejected, but validated");
        }
    }

    #[test]
    fn unknown_domain_error_lists_options() {
        let err = load_builtin_rulebase("nope").unwrap_err().to_string();
        assert!(err.contains("finance"), "got: {err}");
    }

    #[test]
    fn all_six_domains_load_and_validate() {
        let ids = ["finance", "crypto", "cyber", "transport", "ecommerce", "healthcare"];
        for id in ids {
            let rb = load_builtin_rulebase(id)
                .unwrap_or_else(|e| panic!("domain `{id}` failed to load: {e}"));
            assert_eq!(rb.meta.id, id);
            assert!(!rb.entity_types.is_empty());
            assert!(!rb.event_types.is_empty());
            assert!(!rb.behaviors.is_empty());
            // Every domain must declare at least one anomaly process.
            assert!(
                !rb.adversaries.is_empty() || !rb.failures.is_empty(),
                "domain `{id}` has no anomaly processes"
            );
            // Anomalies are rare everywhere (the cross-domain invariant).
            assert!(
                rb.control.prevalence <= 0.05,
                "domain `{id}` prevalence {} too high",
                rb.control.prevalence
            );
        }
        assert_eq!(builtin_domains().len(), 6);
        assert!(builtin_domains().iter().all(|d| d.rulebase_yaml.is_some()));
    }

    #[test]
    fn transport_is_failure_dominated() {
        // The deliberate non-adversarial stress test (mustread.txt §14).
        let rb = load_builtin_rulebase("transport").unwrap();
        assert!(
            rb.failures.len() > rb.adversaries.len(),
            "transport must be failure-driven: {} failures vs {} adversaries",
            rb.failures.len(),
            rb.adversaries.len()
        );
    }
}
