//! Host capability probe: CPU, RAM, GPU, disk, NUMA, OS.
//!
//! Runs on `agora doctor` and at the start of every `agora generate`. Must be
//! fast (<100 ms), dependency-light and never fail: anything unprobeable
//! degrades to `None`/empty rather than an error.

use std::path::Path;
use std::process::Command;

use serde::{Deserialize, Serialize};
use sysinfo::{Disks, System};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CpuInfo {
    /// Physical core count (falls back to logical if undetectable).
    pub physical_cores: usize,
    pub logical_cores: usize,
    pub model: String,
    /// Base/advertised frequency of core 0 in MHz, if known.
    pub frequency_mhz: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemInfo {
    pub total_bytes: u64,
    pub available_bytes: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GpuInfo {
    pub name: String,
    pub vram_mb: u64,
    pub driver: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DiskInfo {
    /// Mount point hosting the probed output path, when it could be
    /// identified. `None` on filesystems absent from the mount table (network
    /// mounts, in particular) — the space figures below may still be known.
    pub mount_point: Option<String>,
    /// The existing path actually measured; always known, so callers have
    /// something truthful to name even when `mount_point` is `None`.
    pub path: String,
    /// Free bytes, or `None` if it could not be determined.
    ///
    /// `None` means *unknown*, never *zero*: callers must not treat an
    /// unprobeable filesystem as a full one (§11a).
    pub free_bytes: Option<u64>,
    /// Total bytes, or `None` if it could not be determined.
    pub total_bytes: Option<u64>,
}

impl DiskInfo {
    /// Best available human name for the measured location: the mount point
    /// when known, else the path itself. Never a placeholder.
    pub fn location(&self) -> &str {
        self.mount_point.as_deref().unwrap_or(&self.path)
    }
}

/// Everything AGORA knows about the machine it is running on.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HostProbe {
    pub hostname: String,
    pub os: String,
    pub kernel: String,
    pub cpu: CpuInfo,
    pub mem: MemInfo,
    pub gpus: Vec<GpuInfo>,
    pub numa_nodes: usize,
    /// Disk backing the requested output path.
    pub disk: DiskInfo,
}

impl HostProbe {
    /// Probe the host. `out_path` determines which disk is measured; it does
    /// not need to exist yet (the closest existing ancestor's mount is used).
    pub fn probe(out_path: &Path) -> Self {
        let mut sys = System::new();
        sys.refresh_cpu_all();
        sys.refresh_memory();

        let logical = sys.cpus().len().max(1);
        let physical = sys.physical_core_count().unwrap_or(logical);
        let (model, freq) = sys
            .cpus()
            .first()
            .map(|c| (c.brand().trim().to_string(), c.frequency()))
            .unwrap_or_else(|| ("unknown".into(), 0));

        HostProbe {
            hostname: System::host_name().unwrap_or_else(|| "unknown".into()),
            os: format!(
                "{} {}",
                System::name().unwrap_or_else(|| "unknown".into()),
                System::os_version().unwrap_or_default()
            )
            .trim()
            .to_string(),
            kernel: System::kernel_version().unwrap_or_else(|| "unknown".into()),
            cpu: CpuInfo {
                physical_cores: physical,
                logical_cores: logical,
                model,
                frequency_mhz: freq,
            },
            mem: MemInfo {
                total_bytes: sys.total_memory(),
                available_bytes: sys.available_memory(),
            },
            gpus: probe_gpus(),
            numa_nodes: probe_numa_nodes(),
            disk: probe_disk(out_path),
        }
    }

    /// Default worker thread count: cores - 2, floor 1 (blueprint §11a).
    pub fn default_threads(&self) -> usize {
        self.cpu.logical_cores.saturating_sub(2).max(1)
    }

    pub fn has_gpu(&self) -> bool {
        !self.gpus.is_empty()
    }
}

/// GPU probe via `nvidia-smi` (no NVML linkage: keeps the binary portable to
/// machines without the driver installed).
fn probe_gpus() -> Vec<GpuInfo> {
    let out = Command::new("nvidia-smi")
        .args([
            "--query-gpu=name,memory.total,driver_version",
            "--format=csv,noheader,nounits",
        ])
        .output();
    let Ok(out) = out else { return Vec::new() };
    if !out.status.success() {
        return Vec::new();
    }
    String::from_utf8_lossy(&out.stdout)
        .lines()
        .filter_map(|line| {
            let mut f = line.split(',').map(str::trim);
            Some(GpuInfo {
                name: f.next()?.to_string(),
                vram_mb: f.next()?.parse().ok()?,
                driver: f.next().unwrap_or("?").to_string(),
            })
        })
        .collect()
}

/// NUMA node count from sysfs (Linux); 1 elsewhere or on failure.
fn probe_numa_nodes() -> usize {
    let Ok(entries) = std::fs::read_dir("/sys/devices/system/node") else {
        return 1;
    };
    let n = entries
        .filter_map(Result::ok)
        .filter(|e| {
            let name = e.file_name();
            let name = name.to_string_lossy();
            name.starts_with("node") && name[4..].chars().all(|c| c.is_ascii_digit())
        })
        .count();
    n.max(1)
}

/// Free/total bytes of the filesystem backing `path`, via `statvfs(3)`.
///
/// This is a direct syscall on the target path rather than a mount-table
/// scan, which is what makes it correct on network filesystems (NFS, Lustre,
/// GPFS, CIFS): those are absent from sysinfo's physical-disk list, but the
/// kernel answers `statvfs` for them exactly as it does for a local disk.
///
/// Returns `None` — *unknown*, never *zero* — when the syscall fails or the
/// filesystem reports no blocks at all (a pseudo-fs, or a server declining to
/// report usage). Callers must not read `None` as "full".
#[cfg(unix)]
fn fs_space(path: &Path) -> Option<(u64, u64)> {
    use std::os::unix::ffi::OsStrExt;

    let c_path = std::ffi::CString::new(path.as_os_str().as_bytes()).ok()?;
    // SAFETY: `c_path` is a valid NUL-terminated C string that outlives the
    // call; `statvfs` only reads it and fills `st`, which is fully
    // initialised here and only read back on success (return 0).
    let mut st: libc::statvfs = unsafe { std::mem::zeroed() };
    if unsafe { libc::statvfs(c_path.as_ptr(), &mut st) } != 0 {
        return None;
    }

    // Block counts are expressed in f_frsize; some platforms leave it 0, in
    // which case f_bsize is the correct unit.
    let unit: u64 = if st.f_frsize > 0 {
        st.f_frsize as u64
    } else {
        st.f_bsize as u64
    };
    let total = (st.f_blocks as u64).checked_mul(unit)?;
    if total == 0 {
        // A real output filesystem always has blocks; zero means the fs did
        // not report usage. That is unknown, not full.
        return None;
    }
    // f_bavail (not f_bfree) is what an unprivileged writer can actually use:
    // it excludes the root-reserved blocks.
    let free = (st.f_bavail as u64).checked_mul(unit)?;
    Some((free, total))
}

#[cfg(not(unix))]
fn fs_space(_path: &Path) -> Option<(u64, u64)> {
    // No statvfs: fall back to the mount table in `probe_disk`.
    None
}

/// Whether two paths live on the same filesystem, by device id.
///
/// Used to check the mount table's answer before quoting it: a mount can be
/// the longest matching *prefix* of a path without hosting it (e.g. `/` vs a
/// tmpfs at `/dev/shm`), and naming the wrong filesystem is its own bug.
#[cfg(unix)]
fn same_filesystem(a: &Path, b: &Path) -> bool {
    use std::os::unix::fs::MetadataExt;
    match (std::fs::metadata(a), std::fs::metadata(b)) {
        (Ok(x), Ok(y)) => x.dev() == y.dev(),
        _ => false,
    }
}

#[cfg(not(unix))]
fn same_filesystem(_a: &Path, _b: &Path) -> bool {
    // No cheap device id: fall back to trusting the prefix match.
    true
}

/// Measure the filesystem backing `out_path` (resolved through the closest
/// existing ancestor, since the output dir need not exist yet).
///
/// Space comes from `statvfs` on that path; the mount table is consulted only
/// to *label* the location, never to gate the numbers — it omits network
/// filesystems, which is precisely where large runs are written.
fn probe_disk(out_path: &Path) -> DiskInfo {
    // Walk up until an existing ancestor, then canonicalize it.
    let mut anchor = out_path.to_path_buf();
    while !anchor.exists() {
        match anchor.parent() {
            Some(p) => anchor = p.to_path_buf(),
            None => break,
        }
    }
    let anchor = anchor.canonicalize().unwrap_or(anchor);

    let disks = Disks::new_with_refreshed_list();
    let best = disks
        .iter()
        .filter(|d| anchor.starts_with(d.mount_point()))
        .max_by_key(|d| d.mount_point().as_os_str().len());
    // Only quote the mount table when it really hosts the target: the longest
    // prefix can belong to a different filesystem entirely.
    let mount_point = best
        .filter(|d| same_filesystem(&anchor, d.mount_point()))
        .map(|d| d.mount_point().to_string_lossy().into_owned());

    let (free_bytes, total_bytes) = match fs_space(&anchor) {
        Some((free, total)) => (Some(free), Some(total)),
        // statvfs unavailable or failed: use the mount table if it happens to
        // know this path, otherwise report unknown and let the caller decide.
        None => match best {
            Some(d) if d.total_space() > 0 => (Some(d.available_space()), Some(d.total_space())),
            _ => (None, None),
        },
    };

    DiskInfo {
        mount_point,
        path: anchor.to_string_lossy().into_owned(),
        free_bytes,
        total_bytes,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn probe_returns_sane_values() {
        let p = HostProbe::probe(Path::new("."));
        assert!(p.cpu.logical_cores >= 1);
        assert!(p.cpu.physical_cores >= 1);
        assert!(p.mem.total_bytes > 0);
        assert!(p.numa_nodes >= 1);
        assert!(
            p.disk.total_bytes.is_some_and(|t| t > 0),
            "current dir must be on a real disk"
        );
        assert!(p.default_threads() >= 1);
    }

    #[test]
    fn disk_probe_handles_nonexistent_path() {
        let p = probe_disk(Path::new("/definitely/not/a/real/path/out"));
        // Walks up to "/" which must exist.
        assert!(p.total_bytes.is_some_and(|t| t > 0));
    }

    /// The output directory usually does not exist yet, and `statvfs` fails on
    /// a nonexistent path — so the probe must measure the closest existing
    /// ancestor and still return real numbers.
    #[test]
    fn disk_probe_measures_ancestor_of_unborn_output_dir() {
        let p = probe_disk(Path::new("/tmp/agora-does-not-exist-yet/run1/out"));
        assert!(p.free_bytes.is_some(), "free space of /tmp must be knowable");
        assert!(p.total_bytes.is_some_and(|t| t > 0));
    }

    /// `location()` must always name something real; the pre-fix code printed
    /// a bare "?" here.
    #[test]
    fn location_never_renders_a_placeholder() {
        let p = probe_disk(Path::new("."));
        assert!(!p.location().is_empty());
        assert_ne!(p.location(), "?");
    }

    /// A mount that is merely the longest *prefix* of the target must not be
    /// quoted as its location: `/` is a prefix of everything, but a tmpfs at
    /// `/dev/shm` is a different filesystem with different free space.
    #[cfg(unix)]
    #[test]
    fn does_not_mislabel_a_prefix_mount_as_the_location() {
        let shm = Path::new("/dev/shm");
        if !shm.exists() || same_filesystem(shm, Path::new("/")) {
            return; // not a separate fs on this host: nothing to check
        }
        let p = probe_disk(shm);
        assert_ne!(
            p.mount_point.as_deref(),
            Some("/"),
            "tmpfs at /dev/shm must not be labelled as /"
        );
        assert!(p.location().starts_with("/dev/shm"), "got: {}", p.location());
        // And the space reported must be the tmpfs's, not the root disk's.
        let (free, _) = fs_space(shm).expect("statvfs on /dev/shm");
        assert_eq!(p.free_bytes, Some(free));
    }

    /// statvfs must agree with the mount table on a plain local filesystem
    /// (the case the old mount-scan handled correctly) — free space within an
    /// order of magnitude, not an absurd unit-conversion error.
    #[cfg(unix)]
    #[test]
    fn fs_space_agrees_with_mount_table_on_local_fs() {
        let root = Path::new("/");
        let (free, total) = fs_space(root).expect("statvfs on / must succeed");
        assert!(total > 0 && free <= total, "free={free} total={total}");

        let disks = Disks::new_with_refreshed_list();
        if let Some(d) = disks.iter().find(|d| d.mount_point() == root) {
            let ratio = total as f64 / d.total_space().max(1) as f64;
            assert!(
                (0.5..2.0).contains(&ratio),
                "statvfs total {total} disagrees with mount table {} (ratio {ratio})",
                d.total_space()
            );
        }
    }
}
