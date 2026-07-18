//! Welford running moments + deterministic reservoir sample for quantiles.

use serde::Serialize;

#[derive(Default, Clone)]
pub struct Moments {
    n: u64,
    mean: f64,
    m2: f64,
    min: f64,
    max: f64,
    /// Fixed-size reservoir for quantile estimation.
    reservoir: Vec<f64>,
    /// Deterministic counter-hash replacement (no RNG dependency).
    seen: u64,
}

const RESERVOIR: usize = 16_384;

impl Moments {
    #[inline]
    pub fn add(&mut self, x: f64) {
        if self.n == 0 {
            self.min = x;
            self.max = x;
        } else {
            self.min = self.min.min(x);
            self.max = self.max.max(x);
        }
        self.n += 1;
        let d = x - self.mean;
        self.mean += d / self.n as f64;
        self.m2 += d * (x - self.mean);

        // Reservoir: fill, then replace at a hash-derived slot with
        // probability RESERVOIR/seen (Vitter's algorithm R, derandomized).
        if self.reservoir.len() < RESERVOIR {
            self.reservoir.push(x);
        } else {
            self.seen += 1;
            let h = splitmix(self.seen ^ x.to_bits());
            let j = (h % (RESERVOIR as u64 + self.seen)) as usize;
            if j < RESERVOIR {
                self.reservoir[j] = x;
            }
        }
    }

    pub fn summary(&self) -> MomentsSummary {
        let mut q = self.reservoir.clone();
        q.sort_unstable_by(|a, b| a.partial_cmp(b).expect("no NaN reaches reservoir"));
        let quantile = |p: f64| -> f64 {
            if q.is_empty() {
                f64::NAN
            } else {
                q[((q.len() - 1) as f64 * p) as usize]
            }
        };
        MomentsSummary {
            count: self.n,
            mean: self.mean,
            std: if self.n > 1 { (self.m2 / (self.n - 1) as f64).sqrt() } else { 0.0 },
            min: self.min,
            max: self.max,
            p50: quantile(0.50),
            p90: quantile(0.90),
            p99: quantile(0.99),
        }
    }
}

#[derive(Debug, Clone, Serialize)]
pub struct MomentsSummary {
    pub count: u64,
    pub mean: f64,
    pub std: f64,
    pub min: f64,
    pub max: f64,
    pub p50: f64,
    pub p90: f64,
    pub p99: f64,
}

#[inline]
fn splitmix(mut z: u64) -> u64 {
    z = z.wrapping_add(0x9E3779B97F4A7C15);
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
    z ^ (z >> 31)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn moments_and_quantiles_on_uniform() {
        let mut m = Moments::default();
        for i in 0..100_000 {
            m.add(i as f64 / 100_000.0);
        }
        let s = m.summary();
        assert!((s.mean - 0.5).abs() < 0.01);
        assert!((s.p50 - 0.5).abs() < 0.03);
        assert!((s.p99 - 0.99).abs() < 0.03);
        assert_eq!(s.count, 100_000);
    }
}
