//! HyperLogLog distinct counter (p = 14: 16 KiB, ~0.8% standard error).

const P: u32 = 14;
const M: usize = 1 << P;

pub struct Hll {
    regs: Vec<u8>,
}

impl Default for Hll {
    fn default() -> Self {
        Hll { regs: vec![0; M] }
    }
}

impl Hll {
    #[inline]
    pub fn insert(&mut self, value: u64) {
        let h = hash64(value);
        let idx = (h >> (64 - P)) as usize;
        let rest = h << P;
        let rank = (rest.leading_zeros() + 1).min(64 - P + 1) as u8;
        if rank > self.regs[idx] {
            self.regs[idx] = rank;
        }
    }

    /// Insert a pair (e.g. (src,dst) for distinct-edge counting).
    #[inline]
    pub fn insert_pair(&mut self, a: u64, b: u64) {
        self.insert(hash64(a).wrapping_mul(0x9E3779B97F4A7C15) ^ b);
    }

    pub fn estimate(&self) -> f64 {
        let m = M as f64;
        let alpha = 0.7213 / (1.0 + 1.079 / m);
        let mut sum = 0.0f64;
        let mut zeros = 0usize;
        for &r in &self.regs {
            sum += 1.0 / (1u64 << r) as f64;
            if r == 0 {
                zeros += 1;
            }
        }
        let raw = alpha * m * m / sum;
        if raw <= 2.5 * m && zeros > 0 {
            // Linear counting for the small range.
            m * (m / zeros as f64).ln()
        } else {
            raw
        }
    }
}

#[inline]
fn hash64(mut z: u64) -> u64 {
    z = z.wrapping_add(0x9E3779B97F4A7C15);
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
    z ^ (z >> 31)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn estimates_within_two_percent() {
        for &n in &[1_000u64, 100_000, 2_000_000] {
            let mut h = Hll::default();
            for i in 0..n {
                h.insert(i);
            }
            let est = h.estimate();
            let err = (est - n as f64).abs() / n as f64;
            assert!(err < 0.02, "n={n}: est {est:.0}, err {:.2}%", err * 100.0);
        }
    }
}
