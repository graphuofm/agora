//! The streaming stats collector: an `EventSink` tee'd with the file writer.
//!
//! Everything here is O(1) or O(attrs) per event plus O(nodes) memory for
//! exact degrees (skipped above [`EXACT_DEGREE_NODE_CAP`] — the sketches keep
//! working at any scale). Label introspection is free and exact because
//! label = generative cause (blueprint §11b).

use std::collections::HashMap;

use agora_core::{EventBatch, EventSink};
use serde::Serialize;

use crate::cm::CountMinTopK;
use crate::hll::Hll;
use crate::moments::{Moments, MomentsSummary};

/// Above this node count the exact degree arrays (8 B/node) are skipped.
const EXACT_DEGREE_NODE_CAP: u64 = 400_000_000;

pub struct StatsCollector {
    n_nodes: u64,
    event_type_names: Vec<String>,
    intent_names: Vec<String>,
    attr_dicts: Vec<(String, Vec<String>)>,
    // resolved on first batch
    attr_names: Vec<String>,
    numeric_moments: Vec<Option<Moments>>,
    cat_hist: Vec<Option<(Vec<String>, Vec<u64>)>>,

    total: u64,
    per_event_type: Vec<u64>,
    per_intent: Vec<u64>,
    t_min: i64,
    t_max: i64,
    /// Events per simulated day (day = (t - t_min_day) / 86400).
    daily: HashMap<i64, u64>,
    out_deg: Vec<u32>,
    in_deg: Vec<u32>,
    hll_src: Hll,
    hll_dst: Hll,
    hll_pairs: Hll,
    top_dst: CountMinTopK,
    inter_event_gap: Moments,
    last_t: Option<i64>,
    /// Per-source-node temporal structure (burstiness B, memory M, inter-event
    /// CDF), computed only when exact per-node arrays are affordable.
    temporal: Temporal,
}

/// Number of log2-spaced bins for the inter-event-time CDF (gap in seconds,
/// bin i covers [2^i, 2^{i+1})), up to ~2^31 s ≈ 68 years.
const CDF_BINS: usize = 32;

/// Per-source-node inter-event-time structure (blueprint §11b realism, made
/// MEASURABLE not asserted). All O(1)/event:
///   - burstiness B = (σ−μ)/(σ+μ) over pooled per-source gaps (Goh & Barabási
///     2008): −1 periodic, 0 Poisson/exponential, →1 bursty.
///   - memory M = lag-1 Pearson correlation of consecutive gaps per source.
///   - inter-event-time CDF as a log2 histogram (KS-comparable to real data).
#[derive(Default)]
struct Temporal {
    /// Per global node id: last event time (i64::MIN = unseen). Empty when not
    /// affordable (huge node counts) — then B/M/CDF are reported as absent.
    last_t: Vec<i64>,
    /// Per node: previous inter-event gap (NaN = none yet) for the memory M.
    prev_gap: Vec<f64>,
    // Welford over all per-source gaps → mean/std → B.
    n: u64,
    mean: f64,
    m2: f64,
    // Streaming sums for the lag-1 (prev_gap, gap) Pearson correlation → M.
    m_n: u64,
    sum_a: f64,
    sum_b: f64,
    sum_aa: f64,
    sum_bb: f64,
    sum_ab: f64,
    cdf: [u64; CDF_BINS],
}

impl Temporal {
    fn new(n_nodes: u64, exact: bool) -> Temporal {
        let cap = if exact { n_nodes as usize } else { 0 };
        Temporal {
            last_t: vec![i64::MIN; cap],
            prev_gap: vec![f64::NAN; cap],
            cdf: [0; CDF_BINS],
            ..Default::default()
        }
    }

    /// Record one event sourced at `src` at time `t` (events for a given src
    /// arrive in non-decreasing t, since each window is time-sorted and
    /// windows are emitted in order).
    #[inline]
    fn observe(&mut self, src: u64, t: i64) {
        if self.last_t.is_empty() {
            return; // per-source temporal stats not affordable at this scale
        }
        let i = src as usize;
        let lt = self.last_t[i];
        self.last_t[i] = t;
        if lt == i64::MIN {
            return; // first event for this source: no gap yet
        }
        let gap = (t - lt).max(0) as f64;
        // Welford for B.
        self.n += 1;
        let d = gap - self.mean;
        self.mean += d / self.n as f64;
        self.m2 += d * (gap - self.mean);
        // CDF log2 bin.
        let bin = if gap < 1.0 {
            0
        } else {
            (63 - (gap as u64).leading_zeros() as usize).min(CDF_BINS - 1)
        };
        self.cdf[bin] += 1;
        // Memory M: correlate consecutive gaps per source.
        let pg = self.prev_gap[i];
        if pg.is_finite() {
            self.m_n += 1;
            self.sum_a += pg;
            self.sum_b += gap;
            self.sum_aa += pg * pg;
            self.sum_bb += gap * gap;
            self.sum_ab += pg * gap;
        }
        self.prev_gap[i] = gap;
    }

    fn burstiness(&self) -> Option<f64> {
        if self.n < 2 {
            return None;
        }
        let std = (self.m2 / (self.n - 1) as f64).sqrt();
        let denom = std + self.mean;
        if denom <= 0.0 {
            return None;
        }
        Some((std - self.mean) / denom)
    }

    fn memory(&self) -> Option<f64> {
        let n = self.m_n as f64;
        if self.m_n < 2 {
            return None;
        }
        let cov = n * self.sum_ab - self.sum_a * self.sum_b;
        let va = n * self.sum_aa - self.sum_a * self.sum_a;
        let vb = n * self.sum_bb - self.sum_b * self.sum_b;
        let den = (va * vb).sqrt();
        if den <= 0.0 {
            return None;
        }
        Some(cov / den)
    }
}

impl StatsCollector {
    pub fn new(
        n_nodes: u64,
        event_type_names: Vec<String>,
        intent_names: Vec<String>,
        attr_dicts: Vec<(String, Vec<String>)>,
    ) -> StatsCollector {
        let exact = n_nodes <= EXACT_DEGREE_NODE_CAP;
        StatsCollector {
            n_nodes,
            per_event_type: vec![0; event_type_names.len()],
            per_intent: vec![0; intent_names.len()],
            event_type_names,
            intent_names,
            attr_dicts,
            attr_names: Vec::new(),
            numeric_moments: Vec::new(),
            cat_hist: Vec::new(),
            total: 0,
            t_min: i64::MAX,
            t_max: i64::MIN,
            daily: HashMap::new(),
            out_deg: if exact { vec![0; n_nodes as usize] } else { Vec::new() },
            in_deg: if exact { vec![0; n_nodes as usize] } else { Vec::new() },
            hll_src: Hll::default(),
            hll_dst: Hll::default(),
            hll_pairs: Hll::default(),
            top_dst: CountMinTopK::new(20),
            inter_event_gap: Moments::default(),
            last_t: None,
            temporal: Temporal::new(n_nodes, exact),
        }
    }

    pub fn report(&self) -> StatsReport {
        let degree = |v: &Vec<u32>| -> Option<DegreeSummary> {
            if v.is_empty() {
                return None;
            }
            let active = v.iter().filter(|&&d| d > 0).count() as u64;
            let max = *v.iter().max().unwrap_or(&0) as u64;
            let sum: u64 = v.iter().map(|&d| d as u64).sum();
            // Log2-binned histogram: bin i = degree in [2^i, 2^(i+1)).
            let mut bins = vec![0u64; 33];
            for &d in v.iter().filter(|&&d| d > 0) {
                bins[(63 - (d as u64).leading_zeros() as usize).min(32)] += 1;
            }
            while bins.last() == Some(&0) {
                bins.pop();
            }
            Some(DegreeSummary {
                active_nodes: active,
                mean: sum as f64 / active.max(1) as f64,
                max,
                log2_histogram: bins,
            })
        };

        let mut days: Vec<(i64, u64)> = self.daily.iter().map(|(&d, &c)| (d, c)).collect();
        days.sort_unstable();

        StatsReport {
            total_edges: self.total,
            nodes: self.n_nodes,
            distinct_src: self.hll_src.estimate() as u64,
            distinct_dst: self.hll_dst.estimate() as u64,
            distinct_pairs: self.hll_pairs.estimate() as u64,
            t_min: self.t_min,
            t_max: self.t_max,
            events_per_event_type: self
                .event_type_names
                .iter()
                .cloned()
                .zip(self.per_event_type.iter().copied())
                .collect(),
            label_introspection: self
                .intent_names
                .iter()
                .cloned()
                .zip(self.per_intent.iter().copied())
                .collect(),
            out_degree: degree(&self.out_deg),
            in_degree: degree(&self.in_deg),
            top_in_nodes: self.top_dst.top(),
            daily_events: days.iter().map(|&(_, c)| c).collect(),
            inter_event_gap_s: self.inter_event_gap.summary(),
            burstiness_b: self.temporal.burstiness(),
            memory_m: self.temporal.memory(),
            inter_event_cdf_log2: if self.temporal.last_t.is_empty() {
                None
            } else {
                // Trim trailing empty bins for readability.
                let mut bins = self.temporal.cdf.to_vec();
                while bins.last() == Some(&0) {
                    bins.pop();
                }
                Some(bins)
            },
            // Edge recurrence (TGB's headline temporal statistic): fraction of
            // edges that are NOT the first occurrence of their (src,dst) pair,
            // estimated from the distinct-pair HLL.
            repeat_edge_ratio: if self.total > 0 {
                let distinct = self.hll_pairs.estimate().min(self.total as f64);
                (1.0 - distinct / self.total as f64).clamp(0.0, 1.0)
            } else {
                0.0
            },
            numeric_attrs: self
                .attr_names
                .iter()
                .zip(&self.numeric_moments)
                .filter_map(|(n, m)| m.as_ref().map(|m| (n.clone(), m.summary())))
                .collect(),
            categorical_attrs: self
                .attr_names
                .iter()
                .zip(&self.cat_hist)
                .filter_map(|(n, h)| {
                    h.as_ref().map(|(vals, counts)| {
                        (n.clone(), vals.iter().cloned().zip(counts.iter().copied()).collect())
                    })
                })
                .collect(),
        }
    }
}

impl EventSink for StatsCollector {
    fn write_batch(&mut self, b: &EventBatch) -> anyhow::Result<()> {
        if self.attr_names.is_empty() && !b.attr_names.is_empty() {
            self.attr_names = b.attr_names.clone();
            for name in &self.attr_names {
                match self.attr_dicts.iter().find(|(n, _)| n == name) {
                    Some((_, vals)) => {
                        self.cat_hist.push(Some((vals.clone(), vec![0; vals.len()])));
                        self.numeric_moments.push(None);
                    }
                    None => {
                        self.cat_hist.push(None);
                        self.numeric_moments.push(Some(Moments::default()));
                    }
                }
            }
        }
        for i in 0..b.len() {
            let (s, d, t) = (b.src[i], b.dst[i], b.t[i]);
            self.total += 1;
            self.per_event_type[b.event_type[i] as usize] += 1;
            self.per_intent[b.label[i] as usize] += 1;
            self.t_min = self.t_min.min(t);
            self.t_max = self.t_max.max(t);
            *self.daily.entry(t.div_euclid(86_400)).or_insert(0) += 1;
            if !self.out_deg.is_empty() {
                self.out_deg[s as usize] += 1;
                self.in_deg[d as usize] += 1;
            }
            self.hll_src.insert(s);
            self.hll_dst.insert(d);
            self.hll_pairs.insert_pair(s, d);
            self.top_dst.insert(d);
            self.temporal.observe(s, t);
            if let Some(lt) = self.last_t {
                let gap = (t - lt) as f64;
                if gap >= 0.0 {
                    self.inter_event_gap.add(gap);
                }
            }
            self.last_t = Some(t);
            for (c, col) in b.attrs.iter().enumerate() {
                let v = col[i];
                if v.is_nan() {
                    continue;
                }
                if let Some(m) = &mut self.numeric_moments[c] {
                    m.add(v);
                } else if let Some((vals, counts)) = &mut self.cat_hist[c] {
                    let code = (v as usize).min(vals.len() - 1);
                    counts[code] += 1;
                }
            }
        }
        Ok(())
    }
}

#[derive(Debug, Serialize)]
pub struct DegreeSummary {
    pub active_nodes: u64,
    pub mean: f64,
    pub max: u64,
    /// bin i = node count with degree in [2^i, 2^(i+1)).
    pub log2_histogram: Vec<u64>,
}

#[derive(Debug, Serialize)]
pub struct StatsReport {
    pub total_edges: u64,
    pub nodes: u64,
    pub distinct_src: u64,
    pub distinct_dst: u64,
    pub distinct_pairs: u64,
    pub t_min: i64,
    pub t_max: i64,
    pub events_per_event_type: Vec<(String, u64)>,
    /// Exact per-intent counts — free because label = cause (§11b).
    pub label_introspection: Vec<(String, u64)>,
    pub out_degree: Option<DegreeSummary>,
    pub in_degree: Option<DegreeSummary>,
    /// (node id, estimated in-event count), descending.
    pub top_in_nodes: Vec<(u64, u64)>,
    pub daily_events: Vec<u64>,
    pub inter_event_gap_s: MomentsSummary,
    /// Burstiness B = (σ−μ)/(σ+μ) of per-source inter-event times (Goh &
    /// Barabási 2008): ≈0 Poisson, →1 bursty. None when not measured (huge
    /// node counts). Makes temporal realism measurable, not asserted (§11b).
    pub burstiness_b: Option<f64>,
    /// Memory M = lag-1 correlation of consecutive inter-event times.
    pub memory_m: Option<f64>,
    /// Inter-event-time CDF as a log2 histogram (bin i = gaps in [2^i,2^{i+1}) s).
    pub inter_event_cdf_log2: Option<Vec<u64>>,
    /// Fraction of edges that repeat a previously-seen (src,dst) pair
    /// (TGB's surprise/recurrence statistic; HLL-estimated).
    pub repeat_edge_ratio: f64,
    pub numeric_attrs: Vec<(String, MomentsSummary)>,
    pub categorical_attrs: Vec<(String, Vec<(String, u64)>)>,
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Burstiness B must be ≈0 for a periodic (regular) source and rise toward
    /// 1 for a bursty one (Goh & Barabási 2008), with the correct sign.
    #[test]
    fn burstiness_distinguishes_regular_from_bursty() {
        // Regular: constant gaps → σ≈0 → B≈-1.
        let mut reg = Temporal::new(10, true);
        for k in 0..1000 {
            reg.observe(1, (k * 100) as i64);
        }
        let b_reg = reg.burstiness().unwrap();
        assert!(b_reg < -0.9, "constant-gap source should have B≈-1, got {b_reg}");

        // Bursty: mostly tiny gaps with rare huge ones → σ ≫ μ → B>0.
        let mut bur = Temporal::new(10, true);
        let mut t = 0i64;
        for k in 0..1000 {
            t += if k % 10 == 0 { 100_000 } else { 1 }; // 10% huge, 90% tiny
            bur.observe(1, t);
        }
        let b_bur = bur.burstiness().unwrap();
        assert!(b_bur > 0.4, "heavy-tailed-gap source should be bursty, got {b_bur}");
        assert!(b_bur > b_reg);
    }

    #[test]
    fn temporal_absent_when_not_exact() {
        // Streaming (non-exact) mode allocates no per-node arrays → no B.
        let mut t = Temporal::new(1_000, false);
        for k in 0..100 {
            t.observe(0, k);
        }
        assert!(t.burstiness().is_none());
    }
}
