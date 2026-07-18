//! Dry-run cost model (blueprint §11a, §12 `--dry-run`).
//!
//! Estimates wall time, peak RAM and disk for a requested generation run from
//! the host probe, and decides feasibility *before* anything is generated:
//! never a silent OOM or disk-full crash.
//!
//! The throughput constants below are deliberately conservative engineering
//! estimates; they are re-calibrated against the real engine as milestones
//! land (see `CALIBRATION` notes inline).

use serde::{Deserialize, Serialize};

use crate::probe::HostProbe;

/// What the user asked for, reduced to the quantities the cost model needs.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RunRequest {
    pub nodes: u64,
    pub edges: u64,
    /// Requested worker threads; `None` = auto (cores - 2).
    pub threads: Option<usize>,
    /// On-disk bytes per edge for the chosen output format.
    pub bytes_per_edge_disk: u64,
    /// RAM budget in bytes; `None` = auto (90% of available).
    pub mem_budget: Option<u64>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum ExecutionMode {
    /// Whole edge stream buffered in RAM, written at the end.
    InRam,
    /// Edges streamed to sharded files as they are generated.
    StreamToDisk,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case", tag = "status", content = "reason")]
pub enum Feasibility {
    Ok,
    /// Runnable but worth flagging (e.g. close to disk capacity).
    Warn(Vec<String>),
    /// Not runnable on this host; the reason includes a suggested scale.
    Infeasible(String),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CostEstimate {
    pub events: u64,
    pub threads: usize,
    pub mode: ExecutionMode,
    pub est_wall_time_s: f64,
    pub peak_ram_bytes: u64,
    pub disk_bytes: u64,
    pub feasibility: Feasibility,
    /// Human-readable assumptions behind the numbers.
    pub notes: Vec<String>,
}

/// Tunable model constants, with engine-anchored defaults.
#[derive(Debug, Clone)]
pub struct CostModel {
    /// Sustained events/s on one core.
    /// CALIBRATION (M6, finance, 32-core i9-13900K + NVMe): a single worker
    /// sustains ~4.3M ev/s end-to-end (generation + overlapped Parquet write).
    pub events_per_sec_per_core: f64,
    /// Parallel scaling efficiency. CALIBRATION: measured ~1.3x from 1→30
    /// workers — the per-window sort/encode pipeline and the single writer
    /// thread cap scaling well below linear (the [`aggregate_ceiling`] below
    /// is what actually binds at high core counts).
    pub parallel_efficiency: f64,
    /// Hard end-to-end throughput ceiling (ev/s) regardless of core count:
    /// the sequential per-window merge/sort + the single encode/writer thread.
    /// CALIBRATION (M6): ~6M ev/s for Parquet on the reference host. More
    /// cores past this point only speed the parallel generation fraction, not
    /// the pipeline — so the dry-run must not promise linear scaling.
    pub aggregate_ceiling: f64,
    /// Resident bytes of world state per entity (state + attributes + maps).
    pub bytes_per_node_state: u64,
    /// In-memory bytes per buffered edge (src, dst, t, type, attrs, label).
    pub bytes_per_edge_mem: u64,
    /// Sustained sequential write bandwidth assumed for the output disk.
    /// CALIBRATION: conservative NVMe figure; `agora doctor` may measure later.
    pub disk_write_bytes_per_sec: f64,
    /// Fixed startup overhead (topology skeleton build, rule compile).
    pub fixed_overhead_s: f64,
}

impl Default for CostModel {
    fn default() -> Self {
        CostModel {
            events_per_sec_per_core: 4_300_000.0,
            parallel_efficiency: 0.35,
            aggregate_ceiling: 6_000_000.0,
            bytes_per_node_state: 96,
            bytes_per_edge_mem: 48,
            disk_write_bytes_per_sec: 400.0 * 1024.0 * 1024.0,
            fixed_overhead_s: 2.0,
        }
    }
}

impl CostModel {
    pub fn estimate(&self, req: &RunRequest, host: &HostProbe) -> CostEstimate {
        let threads = req
            .threads
            .unwrap_or_else(|| host.default_threads())
            .clamp(1, host.cpu.logical_cores);
        let mem_budget = req
            .mem_budget
            .unwrap_or((host.mem.available_bytes as f64 * 0.9) as u64);

        let effective_eps = (self.events_per_sec_per_core * threads as f64 * self.parallel_efficiency)
            .min(self.aggregate_ceiling);
        let mut notes = vec![format!(
            "throughput ~{:.1}M ev/s ({} threads, capped at {:.0}M pipeline ceiling; calibrated M6)",
            effective_eps / 1e6,
            threads,
            self.aggregate_ceiling / 1e6
        )];
        let mut warnings = Vec::new();

        let events = req.edges; // one event emits one temporal edge
        let state_ram = req.nodes * self.bytes_per_node_state;
        let edge_buf_ram = req.edges.saturating_mul(self.bytes_per_edge_mem);
        let disk_bytes = req.edges.saturating_mul(req.bytes_per_edge_disk);

        // Mode selection: buffer in RAM only when it comfortably fits (≤70%
        // of the budget — at the boundary, streaming is strictly safer).
        let (mode, peak_ram) = if state_ram + edge_buf_ram <= mem_budget.saturating_mul(7) / 10 {
            (ExecutionMode::InRam, state_ram + edge_buf_ram)
        } else {
            // Streaming holds state + a bounded shard buffer (~256 MiB).
            let shard_buf = 256u64 << 20;
            notes.push("edge stream exceeds RAM budget: streaming shards to disk".into());
            (ExecutionMode::StreamToDisk, state_ram + shard_buf)
        };

        // Compute time + IO time. In streaming mode compute and IO overlap, so
        // wall time ≈ max(compute, io); in-RAM mode they are sequential.
        let compute_s = events as f64 / effective_eps;
        let io_s = disk_bytes as f64 / self.disk_write_bytes_per_sec;
        let wall_s = self.fixed_overhead_s
            + match mode {
                ExecutionMode::InRam => compute_s + io_s,
                ExecutionMode::StreamToDisk => compute_s.max(io_s),
            };
        if io_s > compute_s {
            notes.push("disk I/O dominates compute at this scale (expected, §10)".into());
        }

        // Hard guards. The disk guard can only fire when free space is
        // actually known: an unprobeable filesystem is not a full one, so it
        // must never block a run the user explicitly asked for (§11a).
        let hard_fail: Option<String> = if peak_ram > mem_budget {
            // Even streaming doesn't fit: node state alone exceeds budget.
            let max_nodes = mem_budget.saturating_sub(256u64 << 20) / self.bytes_per_node_state;
            Some(format!(
                "world state needs {} but the RAM budget is {}; reduce --nodes to ≤ {} or raise --mem-budget",
                human_bytes(peak_ram),
                human_bytes(mem_budget),
                max_nodes
            ))
        } else if host.disk.free_bytes.is_some_and(|free| disk_bytes > free.saturating_mul(95) / 100)
        {
            let free = host.disk.free_bytes.expect("checked by is_some_and above");
            Some(format!(
                "output needs ~{} but only {} is free on {}; reduce --edges to ≤ {} or change --out",
                human_bytes(disk_bytes),
                human_bytes(free),
                host.disk.location(),
                free.saturating_mul(90) / 100 / req.bytes_per_edge_disk.max(1)
            ))
        } else {
            None
        };

        // Soft checks.
        match host.disk.free_bytes {
            Some(free) if disk_bytes > free / 2 => warnings.push(format!(
                "output (~{}) will use more than half of the free space on {}",
                human_bytes(disk_bytes),
                host.disk.location()
            )),
            // Unknown free space: say so plainly and proceed. Naming the size
            // we need keeps the warning actionable without a bogus figure.
            None => warnings.push(format!(
                "could not determine free space on {}; skipping the disk-capacity check — this run needs ~{}",
                host.disk.location(),
                human_bytes(disk_bytes)
            )),
            _ => {}
        }
        if peak_ram > mem_budget.saturating_mul(80) / 100 {
            warnings.push(
                "peak RAM is within 20% of the budget; other processes may push the run over".into(),
            );
        }

        let feasibility = match hard_fail {
            Some(reason) => Feasibility::Infeasible(reason),
            None if warnings.is_empty() => Feasibility::Ok,
            None => Feasibility::Warn(warnings),
        };

        CostEstimate {
            events,
            threads,
            mode,
            est_wall_time_s: wall_s,
            peak_ram_bytes: peak_ram,
            disk_bytes,
            feasibility,
            notes,
        }
    }
}

/// 1024-based human bytes (shared by CLI output and error messages).
pub fn human_bytes(b: u64) -> String {
    const UNITS: [&str; 6] = ["B", "KiB", "MiB", "GiB", "TiB", "PiB"];
    let mut v = b as f64;
    let mut u = 0;
    while v >= 1024.0 && u < UNITS.len() - 1 {
        v /= 1024.0;
        u += 1;
    }
    if u == 0 {
        format!("{} {}", b, UNITS[u])
    } else {
        format!("{:.1} {}", v, UNITS[u])
    }
}

/// Human duration: "42 s", "12.5 min", "3.2 h".
pub fn human_duration(secs: f64) -> String {
    if secs < 90.0 {
        format!("{:.0} s", secs)
    } else if secs < 5400.0 {
        format!("{:.1} min", secs / 60.0)
    } else {
        format!("{:.1} h", secs / 3600.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::probe::{CpuInfo, DiskInfo, MemInfo};

    fn fake_host(ram_gb: u64, free_disk_gb: u64, cores: usize) -> HostProbe {
        HostProbe {
            hostname: "test".into(),
            os: "linux".into(),
            kernel: "6".into(),
            cpu: CpuInfo {
                physical_cores: cores,
                logical_cores: cores,
                model: "fake".into(),
                frequency_mhz: 3000,
            },
            mem: MemInfo {
                total_bytes: ram_gb << 30,
                available_bytes: ram_gb << 30,
            },
            gpus: vec![],
            numa_nodes: 1,
            disk: DiskInfo {
                mount_point: Some("/".into()),
                path: "/".into(),
                free_bytes: Some(free_disk_gb << 30),
                total_bytes: Some((free_disk_gb * 2) << 30),
            },
        }
    }

    /// A host whose output filesystem could not be measured — e.g. a network
    /// mount absent from the mount table. Unknown space, unknown mount point.
    fn host_with_unknown_disk(ram_gb: u64, cores: usize) -> HostProbe {
        let mut h = fake_host(ram_gb, 0, cores);
        h.disk = DiskInfo {
            mount_point: None,
            path: "/project/runs".into(),
            free_bytes: None,
            total_bytes: None,
        };
        h
    }

    fn req(nodes: u64, edges: u64) -> RunRequest {
        RunRequest {
            nodes,
            edges,
            threads: None,
            bytes_per_edge_disk: 24,
            mem_budget: None,
        }
    }

    #[test]
    fn small_run_fits_in_ram() {
        let est = CostModel::default().estimate(&req(100_000, 10_000_000), &fake_host(64, 500, 32));
        assert_eq!(est.mode, ExecutionMode::InRam);
        assert!(matches!(est.feasibility, Feasibility::Ok));
        assert!(est.est_wall_time_s < 60.0);
    }

    #[test]
    fn billion_edges_streams() {
        let est =
            CostModel::default().estimate(&req(100_000_000, 1_000_000_000), &fake_host(64, 500, 32));
        assert_eq!(est.mode, ExecutionMode::StreamToDisk);
        assert!(!matches!(est.feasibility, Feasibility::Infeasible(_)));
    }

    #[test]
    fn blueprint_laptop_guard_case() {
        // §11a: 10B edges on a 16 GB / 50 GB-free laptop must be refused.
        let est =
            CostModel::default().estimate(&req(1_000_000_000, 10_000_000_000), &fake_host(16, 50, 8));
        assert!(matches!(est.feasibility, Feasibility::Infeasible(_)));
    }

    /// Regression (measured on iTiger/Rocky 9.7): writing to an NFS mount the
    /// mount table does not list must not be refused. Unknown free space is
    /// not zero free space — the run proceeds with a warning.
    #[test]
    fn unknown_free_space_warns_but_does_not_block() {
        let est = CostModel::default().estimate(&req(1_000_000, 2_000_000), &host_with_unknown_disk(64, 32));
        match &est.feasibility {
            Feasibility::Warn(ws) => {
                assert!(
                    ws.iter().any(|w| w.contains("could not determine free space")),
                    "expected an unknown-space warning, got: {ws:?}"
                );
            }
            f => panic!("unknown free space must warn, not block; got: {f:?}"),
        }
    }

    /// The old code rendered unknown space as "0 B is free on ?". Whatever we
    /// print for an unmeasurable filesystem, it must not claim zero bytes free
    /// nor name the location "?".
    #[test]
    fn unknown_free_space_never_reports_zero_or_question_mark() {
        let est = CostModel::default().estimate(&req(1_000_000, 2_000_000), &host_with_unknown_disk(64, 32));
        let text = match &est.feasibility {
            Feasibility::Warn(ws) => ws.join(" "),
            f => format!("{f:?}"),
        };
        assert!(!text.contains("0 B is free"), "must not claim zero free: {text}");
        assert!(!text.contains(" on ?"), "must not name the location '?': {text}");
        // It should name the path it actually tried to measure.
        assert!(text.contains("/project/runs"), "should name the target path: {text}");
    }

    /// Unknown space disables only the *disk* guard; a run that cannot fit in
    /// RAM must still be refused on such a host.
    #[test]
    fn unknown_free_space_still_enforces_the_ram_guard() {
        let est = CostModel::default()
            .estimate(&req(100_000_000_000, 1_000_000), &host_with_unknown_disk(16, 8));
        assert!(
            matches!(est.feasibility, Feasibility::Infeasible(_)),
            "RAM guard must survive an unmeasurable disk; got: {:?}",
            est.feasibility
        );
    }

    #[test]
    fn infeasible_message_is_actionable() {
        let est =
            CostModel::default().estimate(&req(1_000_000, 100_000_000_000), &fake_host(64, 100, 32));
        if let Feasibility::Infeasible(msg) = &est.feasibility {
            assert!(msg.contains("reduce --edges"), "got: {msg}");
        } else {
            panic!("expected infeasible");
        }
    }
}
