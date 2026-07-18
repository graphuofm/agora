//! agora-host: probe the local machine and adapt AGORA's configuration to it.
//!
//! Two responsibilities (blueprint §11a):
//!   1. [`probe`] — measure CPU / RAM / GPU / disk / NUMA / OS.
//!   2. [`cost`] — estimate time, peak RAM and disk for a requested generation
//!      run *before* it starts, and guard against infeasible requests.

pub mod cost;
pub mod probe;

pub use cost::{CostEstimate, CostModel, ExecutionMode, Feasibility};
pub use probe::{CpuInfo, DiskInfo, GpuInfo, HostProbe, MemInfo};
