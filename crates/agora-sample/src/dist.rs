//! Samplers for the closed `Distribution` set in the rule-base schema.
//!
//! Pre-resolved at setup ([`DistSampler::compile`]) so the per-event path is
//! a single match + draw with no allocation (risk R1: no hot-path creep).
//! The analytic mean is computed at compile time (rand_distr does not expose
//! its parameters) and drives closed-form rate calibration (§3).

use rand::Rng;
use rand_distr::{Distribution as RandDist, Exp, LogNormal, Normal, Poisson, Zipf};
use agora_rules::Distribution;

use crate::rng::Rng64;

/// A compiled, ready-to-draw distribution.
#[derive(Debug, Clone)]
pub struct DistSampler {
    kind: Kind,
    mean: f64,
}

#[derive(Debug, Clone)]
enum Kind {
    Constant(f64),
    Uniform { min: f64, span: f64 },
    Normal(Normal<f64>),
    LogNormal(LogNormal<f64>),
    Exp(Exp<f64>),
    Pareto { scale: f64, inv_shape: f64 },
    Zipf(Zipf<f64>),
    Poisson(Poisson<f64>),
}

impl DistSampler {
    /// Validate + pre-compute. Errors are actionable (§13) and surface at
    /// rule-base load time, never mid-simulation.
    pub fn compile(d: &Distribution) -> anyhow::Result<DistSampler> {
        let (kind, mean) = match *d {
            Distribution::Constant { value } => (Kind::Constant(value), value),
            Distribution::Uniform { min, max } => {
                anyhow::ensure!(max >= min, "uniform: max ({max}) < min ({min})");
                (Kind::Uniform { min, span: max - min }, (min + max) / 2.0)
            }
            Distribution::Normal { mean, std } => (
                Kind::Normal(Normal::new(mean, std).map_err(|e| anyhow::anyhow!("normal: {e}"))?),
                mean,
            ),
            Distribution::LogNormal { mu, sigma } => (
                Kind::LogNormal(
                    LogNormal::new(mu, sigma).map_err(|e| anyhow::anyhow!("log_normal: {e}"))?,
                ),
                (mu + sigma * sigma / 2.0).exp(),
            ),
            Distribution::Exponential { rate } => (
                Kind::Exp(Exp::new(rate).map_err(|e| anyhow::anyhow!("exponential: {e}"))?),
                1.0 / rate,
            ),
            Distribution::Pareto { scale, shape } => {
                anyhow::ensure!(scale > 0.0 && shape > 0.0, "pareto: scale and shape must be > 0");
                let mean = if shape > 1.0 { shape * scale / (shape - 1.0) } else { f64::INFINITY };
                (Kind::Pareto { scale, inv_shape: 1.0 / shape }, mean)
            }
            Distribution::Zipf { n, exponent } => (
                Kind::Zipf(
                    Zipf::new(n, exponent).map_err(|e| anyhow::anyhow!("zipf: {e}"))?,
                ),
                zipf_mean(n, exponent),
            ),
            Distribution::Poisson { lambda } => (
                Kind::Poisson(Poisson::new(lambda).map_err(|e| anyhow::anyhow!("poisson: {e}"))?),
                lambda,
            ),
        };
        Ok(DistSampler { kind, mean })
    }

    /// Analytic mean (∞ for shape ≤ 1 Pareto).
    pub fn mean(&self) -> f64 {
        self.mean
    }

    #[inline]
    pub fn sample(&self, rng: &mut Rng64) -> f64 {
        match &self.kind {
            Kind::Constant(v) => *v,
            Kind::Uniform { min, span } => min + rng.gen::<f64>() * span,
            Kind::Normal(d) => d.sample(rng),
            Kind::LogNormal(d) => d.sample(rng),
            Kind::Exp(d) => d.sample(rng),
            Kind::Pareto { scale, inv_shape } => {
                // Inverse CDF: scale / U^(1/shape).
                scale / rng.gen::<f64>().max(1e-15).powf(*inv_shape)
            }
            Kind::Zipf(d) => d.sample(rng),
            Kind::Poisson(d) => d.sample(rng),
        }
    }
}

/// E[Zipf(n,s)] = H(n, s-1) / H(n, s); exact for small n, integral
/// approximation for large n (calibration-grade accuracy is enough).
fn zipf_mean(n: u64, s: f64) -> f64 {
    if n <= 10_000 {
        let (mut num, mut den) = (0.0, 0.0);
        for k in 1..=n {
            let kf = k as f64;
            num += kf.powf(1.0 - s);
            den += kf.powf(-s);
        }
        num / den
    } else {
        let h = |p: f64| -> f64 {
            // ∫1..n x^-p dx + 0.5·(1 + n^-p) (trapezoid endpoint correction)
            let nf = n as f64;
            let integral = if (p - 1.0).abs() < 1e-9 {
                nf.ln()
            } else {
                (nf.powf(1.0 - p) - 1.0) / (1.0 - p)
            };
            integral + 0.5 * (1.0 + nf.powf(-p))
        };
        h(s - 1.0) / h(s)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rng::{stream, StreamPurpose};

    fn mean_of(d: &Distribution, n: usize) -> f64 {
        let s = DistSampler::compile(d).unwrap();
        let mut rng = stream(11, StreamPurpose::NodeInit, 0, 0);
        (0..n).map(|_| s.sample(&mut rng)).sum::<f64>() / n as f64
    }

    #[test]
    fn empirical_means_match_analytic() {
        for d in [
            Distribution::Uniform { min: 2.0, max: 4.0 },
            Distribution::Exponential { rate: 2.0 },
            Distribution::LogNormal { mu: 0.0, sigma: 0.5 },
            Distribution::Poisson { lambda: 3.5 },
            Distribution::Pareto { scale: 1.0, shape: 3.0 },
        ] {
            let s = DistSampler::compile(&d).unwrap();
            let emp = mean_of(&d, 300_000);
            assert!(
                (emp - s.mean()).abs() / s.mean() < 0.03,
                "{d:?}: analytic {} vs empirical {emp}",
                s.mean()
            );
        }
    }

    #[test]
    fn bad_params_fail_at_compile_not_runtime() {
        assert!(DistSampler::compile(&Distribution::Uniform { min: 5.0, max: 1.0 }).is_err());
        assert!(DistSampler::compile(&Distribution::Pareto { scale: -1.0, shape: 2.0 }).is_err());
    }
}
