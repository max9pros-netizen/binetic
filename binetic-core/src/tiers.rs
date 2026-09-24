//! Storage tier abstraction.
//!
//! The fabric exposes a tier abstraction that maps to physical storage:
//! - tier_hot: CPU registers / L1 cache
//! - tier_warm: CPU L2/L3 / RAM
//! - tier_gpu: GPU shared memory / GPU RAM
//! - tier_cold: SSD / NVMe (mmap)
//! - tier_remote: remote node (future)
//!
//! Each tier has different latency, bandwidth, capacity, and energy characteristics.
//! The rotation scheduler and echo propagator use these to make placement and
//! propagation decisions.

use serde::{Deserialize, Serialize};
use std::fmt;
use std::iter;
use std::path::Path;
use std::sync::Arc;

/// A unique tier identifier.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct TierId(u8);

impl TierId {
    pub const HOT: TierId = TierId(0);
    pub const WARM: TierId = TierId(1);
    pub const GPU: TierId = TierId(2);
    pub const COLD: TierId = TierId(3);
    pub const REMOTE: TierId = TierId(4);

    pub fn from_u8(v: u8) -> Self {
        Self(if v > 4 { 4 } else { v })
    }

    pub const fn as_u8(&self) -> u8 {
        self.0
    }
}

impl fmt::Debug for TierId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self.0 {
            0 => write!(f, "Tier::Hot"),
            1 => write!(f, "Tier::Warm"),
            2 => write!(f, "Tier::Gpu"),
            3 => write!(f, "Tier::Cold"),
            4 => write!(f, "Tier::Remote"),
            _ => write!(f, "Tier::Unknown({})", self.0),
        }
    }
}

impl fmt::Display for TierId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Debug::fmt(self, f)
    }
}

/// Energy cost per access for a tier (approximate, in nanojoules).
///
/// These are rough estimates for illustration; real values depend on hardware.
pub const fn energy_per_access(tier: TierId) -> f64 {
    match tier {
        TierId::HOT => 0.0001,   // CPU register: ~0.1 pJ
        TierId::WARM => 0.1,     // CPU RAM: ~100 pJ
        TierId::GPU => 0.2,      // GPU memory: ~200 pJ
        TierId::COLD => 10.0,    // SSD mmap: ~10 nJ (page fault + SSD access)
        TierId::REMOTE => 100.0, // network: ~100 nJ
        _ => 1000.0,             // unknown: assume expensive
    }
}

/// Latency per access for a tier (approximate, in microseconds).
pub const fn latency_per_access(tier: TierId) -> f64 {
    match tier {
        TierId::HOT => 0.001,     // CPU register: ~1 ns
        TierId::WARM => 0.1,      // CPU RAM: ~100 ns
        TierId::GPU => 0.05,      // GPU memory: ~50 ns (but setup overhead)
        TierId::COLD => 100.0,    // SSD mmap: ~100 us (page fault)
        TierId::REMOTE => 1000.0, // network: ~1 ms
        _ => 2000.0,
    }
}

/// Bandwidth for a tier (approximate, in GB/s).
pub const fn bandwidth_per_second(tier: TierId) -> f64 {
    match tier {
        TierId::HOT => 1000.0,    // CPU register file: ~1 TB/s
        TierId::WARM => 50.0,     // CPU RAM: ~50 GB/s
        TierId::GPU => 200.0,     // GPU memory: ~200 GB/s
        TierId::COLD => 0.5,      // SSD: ~0.5 GB/s (sequential)
        TierId::REMOTE => 0.1, // network: ~0.1 GB/s
        _ => 0.01,
    }
}

/// Capacity for a tier (approximate, in GB).
///
/// These are typical values; actual capacity depends on the machine.
pub const fn typical_capacity(tier: TierId) -> u64 {
    match tier {
        TierId::HOT => 0,         // registers: negligible
        TierId::WARM => 32,       // RAM: 32 GB
        TierId::GPU => 8,         // GPU memory: 8 GB
        TierId::COLD => 1024,     // SSD: 1 TB
        TierId::REMOTE => 10000, // remote: 10 TB
        _ => 0,
    }
}

/// A storage tier in the fabric.
///
/// Each tier maps to a physical storage backend and has characteristics
/// (latency, bandwidth, energy, capacity) that inform scheduling decisions.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Tier {
    pub id: TierId,
    pub name: String,
    pub description: String,
    pub capacity_bytes: u64,
    pub used_bytes: u64,
    pub energy_per_access_nj: f64,
    pub latency_us: f64,
    pub bandwidth_gbps: f64,
    pub min_layer: u16,
    pub max_layer: u16,
}

impl Tier {
    pub fn new(
        id: TierId,
        name: impl Into<String>,
        description: impl Into<String>,
        capacity_bytes: u64,
        min_layer: u16,
        max_layer: u16,
    ) -> Self {
        Self {
            id,
            name: name.into(),
            description: description.into(),
            capacity_bytes,
            used_bytes: 0,
            energy_per_access_nj: energy_per_access(id),
            latency_us: latency_per_access(id),
            bandwidth_gbps: bandwidth_per_second(id),
            min_layer,
            max_layer,
        }
    }

    pub fn available_bytes(&self) -> u64 {
        self.capacity_bytes.saturating_sub(self.used_bytes)
    }

    pub fn utilization(&self) -> f64 {
        if self.capacity_bytes == 0 {
            0.0
        } else {
            self.used_bytes as f64 / self.capacity_bytes as f64
        }
    }

    /// Check if this tier contains a given layer.
    pub fn contains_layer(&self, layer: u16) -> bool {
        layer >= self.min_layer && layer <= self.max_layer
    }

    /// Estimate the energy cost to read N bytes from this tier.
    pub fn energy_to_read_bytes(&self, bytes: u64) -> f64 {
        // Approximate: cost per access × number of accesses
        let access_size = 64u64; // 64-byte cache line / page granularity
        let num_accesses = (bytes + access_size - 1) / access_size;
        self.energy_per_access_nj * num_accesses as f64
    }

    /// Estimate the energy cost to write N bytes to this tier.
    pub fn energy_to_write_bytes(&self, bytes: u64) -> f64 {
        // Writes are typically more expensive than reads
        self.energy_to_read_bytes(bytes) * 2.0
    }

    /// Estimate the time to read N bytes from this tier.
    pub fn time_to_read_bytes(&self, bytes: u64) -> f64 {
        let access_size = 64u64;
        let num_accesses = (bytes + access_size - 1) / access_size;
        // Time = latency per access + transfer time
        let transfer_time = bytes as f64 / (self.bandwidth_gbps * 1e9 / 8.0);
        self.latency_us * num_accesses as f64 + transfer_time * 1e6
    }
}

/// A RAM-based tier (CPU memory).
#[derive(Clone, Debug)]
pub struct RamTier {
    pub tier: Tier,
    pub base_address: *mut u8,
    pub size: usize,
}

impl RamTier {
    pub fn new(name: impl Into<String>, size_bytes: usize, min_layer: u16, max_layer: u16) -> Self {
        // In practice, this would map to actual memory; here we use a placeholder.
        let tier = Tier::new(
            TierId::WARM,
            name,
            "CPU RAM / unified memory",
            size_bytes as u64,
            min_layer,
            max_layer,
        );
        Self {
            tier,
            base_address: std::ptr::null_mut(), // placeholder
            size: size_bytes,
        }
    }
}

/// An mmap-based tier (SSD / NVMe / file-backed).
#[derive(Debug)]
pub struct MmapTier {
    pub tier: Tier,
    pub file_path: String,
    pub mmap: Option<Box<dyn std::any::Any>>,
    pub size: usize,
}

impl MmapTier {
    pub fn new(
        name: impl Into<String>,
        file_path: impl Into<String>,
        size_bytes: usize,
        min_layer: u16,
        max_layer: u16,
    ) -> Self {
        let tier = Tier::new(
            TierId::COLD,
            name,
            "SSD / NVMe mmap-backed storage",
            size_bytes as u64,
            min_layer,
            max_layer,
        );
        Self {
            tier,
            file_path: file_path.into(),
            mmap: None,
            size: size_bytes,
        }
    }

    /// Map the file into memory (call before using).
    pub fn map(&mut self) -> Result<(), String> {
        // Placeholder: actual implementation would mmap the file
        // For now, we just record that mapping is requested
        Ok(())
    }

    /// Unmap the file.
    pub fn unmap(&mut self) {
        self.mmap = None;
    }
}

/// A virtual tier that doesn't correspond to physical storage.
///
/// Used for the temporal register ring and other fabric-internal structures.
#[derive(Clone, Debug)]
pub struct VirtualTier {
    pub tier: Tier,
    pub size: usize,
}

impl VirtualTier {
    pub fn new_temporal(name: impl Into<String>, size_bytes: usize) -> Self {
        let tier = Tier::new(
            TierId::HOT,
            name,
            "Temporal register ring (active compute scratchpad)",
            size_bytes as u64,
            0,
            0,
        );
        Self { tier, size: size_bytes }
    }
}

/// Default tier configuration for a typical 8GB Mac.
pub fn default_tiers_mac_8gb() -> Vec<Tier> {
    vec![
        // Hot tier: CPU registers / L1 cache (very small, very fast)
        Tier::new(
            TierId::HOT,
            "hot",
            "CPU registers + L1 cache (ephemeral)",
            1 * 1024 * 1024, // 1 MB
            0,
            1,
        ),
        // Warm tier: unified memory (CPU + GPU share)
        Tier::new(
            TierId::WARM,
            "warm",
            "Unified memory (CPU + GPU share)",
            8 * 1024 * 1024 * 1024, // 8 GB
            2,
            12,
        ),
        // GPU tier: GPU memory portion of unified memory
        Tier::new(
            TierId::GPU,
            "gpu",
            "GPU memory (portion of unified memory)",
            2 * 1024 * 1024 * 1024, // 2 GB (GPU portion)
            3,
            10,
        ),
        // Cold tier: SSD (mmap)
        Tier::new(
            TierId::COLD,
            "cold",
            "SSD / NVMe mmap-backed storage",
            1 * 1024 * 1024 * 1024 * 1024, // 1 TB
            13,
            100,
        ),
    ]
}

/// Default tier configuration for a 16GB Linux machine with GPU.
pub fn default_tiers_linux_16gb() -> Vec<Tier> {
    vec![
        Tier::new(TierId::HOT, "hot", "CPU registers + L1 cache", 1 * 1024 * 1024, 0, 1),
        Tier::new(
            TierId::WARM,
            "warm",
            "CPU RAM",
            16 * 1024 * 1024 * 1024,
            2,
            15,
        ),
        Tier::new(
            TierId::GPU,
            "gpu",
            "GPU memory (discrete)",
            8 * 1024 * 1024 * 1024,
            3,
            12,
        ),
        Tier::new(
            TierId::COLD,
            "cold",
            "NVMe SSD",
            2 * 1024 * 1024 * 1024 * 1024,
            13,
            100,
        ),
    ]
}

/// Create tier configuration based on available resources.
///
/// This is a heuristic — in practice, you'd query the system for actual resources.
pub fn detect_tiers() -> Vec<Tier> {
    #[cfg(target_os = "macos")]
    {
        default_tiers_mac_8gb()
    }
    #[cfg(not(target_os = "macos"))]
    {
        default_tiers_linux_16gb()
    }
}

/// Memory mapping handle for a tier.
///
/// This abstracts over the actual memory mapping so that the fabric can
/// work with different backing stores uniformly.
pub struct MemoryMapping {
    pub tier_id: TierId,
    pub base: *mut u8,
    pub size: usize,
    pub mapped: bool,
}

impl MemoryMapping {
    pub fn new(tier_id: TierId, size: usize) -> Self {
        Self {
            tier_id,
            base: std::ptr::null_mut(),
            size,
            mapped: false,
        }
    }

    pub fn map(&mut self) -> Result<(), String> {
        // Placeholder: actual mapping depends on tier type
        self.mapped = true;
        Ok(())
    }

    pub fn unmap(&mut self) {
        self.mapped = false;
        self.base = std::ptr::null_mut();
    }

    pub fn as_ptr(&self) -> *const u8 {
        self.base
    }

    pub fn as_mut_ptr(&self) -> *mut u8 {
        self.base
    }
}

/// A collection of tiers with lookup by layer.
#[derive(Clone, Debug)]
pub struct TierSet {
    tiers: Vec<Arc<Tier>>,
}

impl TierSet {
    pub fn new(tiers: Vec<Tier>) -> Self {
        Self {
            tiers: tiers.into_iter().map(Arc::new).collect(),
        }
    }

    pub fn tier_for_layer(&self, layer: u16) -> Option<Arc<Tier>> {
        self.tiers.iter().find(|t| t.contains_layer(layer)).cloned()
    }

    /// Default tier configuration for an 8GB Mac.
    pub fn default_tiers_mac_8gb() -> Vec<Tier> {
        crate::tiers::default_tiers_mac_8gb()
    }

    pub fn all_tiers(&self) -> impl Iterator<Item = Arc<Tier>> + '_ {
        self.tiers.iter().cloned()
    }

    pub fn tier_by_id(&self, id: TierId) -> Option<Arc<Tier>> {
        self.tiers.iter().find(|t| t.id == id).cloned()
    }

    pub fn total_capacity(&self) -> u64 {
        self.tiers.iter().map(|t| t.capacity_bytes).sum()
    }

    pub fn total_used(&self) -> u64 {
        self.tiers.iter().map(|t| t.used_bytes).sum()
    }

    pub fn total_available(&self) -> u64 {
        self.total_capacity().saturating_sub(self.total_used())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tier_creation() {
        let tier = Tier::new(TierId::WARM, "test", "test tier", 1024, 2, 10);
        assert_eq!(tier.id, TierId::WARM);
        assert_eq!(tier.capacity_bytes, 1024);
        assert!(tier.contains_layer(5));
        assert!(!tier.contains_layer(1));
        assert!(!tier.contains_layer(11));
    }

    #[test]
    fn test_tier_energy() {
        let cold_tier = Tier::new(TierId::COLD, "cold", "cold tier", 1024 * 1024 * 1024, 13, 100);
        let energy = cold_tier.energy_to_read_bytes(1024 * 1024); // 1 MB read
        // Cold tier: ~10 nJ per access, 64-byte access size
        // 1 MB / 64 bytes = 16384 accesses
        // 16384 * 10 nJ = 163840 nJ = 0.16384 mJ
        assert!(energy > 0.0);
        assert!(energy < 1.0); // less than 1 mJ
    }

    #[test]
    fn test_tier_set() {
        let tiers = vec![
            Tier::new(TierId::HOT, "hot", "hot", 1024, 0, 1),
            Tier::new(TierId::WARM, "warm", "warm", 1024 * 1024, 2, 10),
            Tier::new(TierId::COLD, "cold", "cold", 1024 * 1024 * 1024, 11, 100),
        ];
        let set = TierSet::new(tiers);

        assert_eq!(set.tier_for_layer(0).unwrap().id, TierId::HOT);
        assert_eq!(set.tier_for_layer(5).unwrap().id, TierId::WARM);
        assert_eq!(set.tier_for_layer(50).unwrap().id, TierId::COLD);
        assert!(set.tier_for_layer(200).is_none());
    }
}
