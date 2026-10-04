//! The Fabric — the central abstraction of binetic.
//!
//! The fabric is the unified memory-and-compute fabric that ties together:
//! - Register addressing and geometry (4D sphere)
//! - Storage tiers (hot, warm, GPU, cold, remote)
//! - Self-describing registers with XOR lineage
//! - Bloom filters per sector for routing and echo gating
//! - Bitsliced lanes for SIMD-efficient computation
//! - Rotation scheduler (geometric scheduling, energy-aware)
//! - Echo propagation (minimal delta propagation, bloom-gated, rotation-damped)
//! - Backend integration (llama.cpp, MLX, native — no-copy compute-to-data)
//! - Self-optimization (observe access patterns, reorganize for locality)
//! - Energy-aware scheduling (battery, thermal)
//! - On-device adaptation (inject deltas into weight registers)
//!
//! # Design
//!
//! The fabric is the layer ABOVE inference backends. Backends (llama.cpp, MLX)
//! are plug-ins that execute compute on fabric memory. The fabric handles
//! memory management, paging, KV cache, eviction, routing, echo propagation,
//! rotation, and optimization — transparently.
//!
//! # Usage
//!
//! ```
//! let fabric = Fabric::new(Config::default());
//! let model = fabric.attach_model("llama3-8b-q4");
//! let token = fabric.compute(ComputeOp::sample_next_token(...));
//! ```

use crate::address::{RegisterAddress, ranges};
use crate::backend::{ArithmeticOp, Backend, BackendId, BackendSet, ComputeOp, ComputeResult, ComputeParams, NativeBackend, QuantizationLevel, RoutingPolicy};
use crate::bitslice::BitslicedLane;
use crate::bloom::{BloomRegistry, SectorKey};
use crate::echo::{EchoPolicy, EchoPropagator, StructuredEcho, EchoPurpose};
use crate::network::NetworkRegister;
use crate::register::{Register, CompoundRegister, CombiningMethod, Record};
use crate::rotation::{EnergyState, RotationPolicy, RotationResult, RotationScheduler, ThermalState};
use crate::tiers::{Tier, TierId, TierSet};
use parking_lot::RwLock;
use serde::{Deserialize, Serialize};
use std::ops::DerefMut;
use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::{Duration, Instant};

/// Configuration for the fabric.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct FabricConfig {
    /// Storage tiers.
    pub tiers: Vec<Tier>,
    /// Rotation policy.
    pub rotation: RotationPolicy,
    /// Echo policy.
    pub echo: EchoPolicy,
    /// Self-optimization policy.
    pub optimization: OptimizationPolicy,
    /// Integrity policy (validation, repair).
    pub integrity: IntegrityPolicy,
    /// Model storage directory (for cold tier model files).
    pub model_dir: Option<PathBuf>,
    /// Whether to log access patterns for self-optimization.
    pub log_access: bool,
}

/// Query result returned from query_scan.
#[derive(Clone, Debug, Default)]
pub struct QueryResult {
    pub match_count: usize,
    pub scanned_count: usize,
    pub elapsed_us: u64,
    pub match_addresses: Vec<RegisterAddress>,
}

impl Default for FabricConfig {
    fn default() -> Self {
        Self {
            tiers: TierSet::default_tiers_mac_8gb().into_iter().map(|t| t.clone()).collect(),
            rotation: RotationPolicy::default(),
            echo: EchoPolicy::default(),
            optimization: OptimizationPolicy::default(),
            integrity: IntegrityPolicy::default(),
            model_dir: Some(PathBuf::from("/tmp/binetic-models")),
            log_access: true,
        }
    }
}

/// Policy for self-optimization (layout adaptation).
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct OptimizationPolicy {
    /// Enable self-optimizing layout (reorganize based on access patterns).
    pub self_optimizing: bool,
    /// Observe and log access patterns.
    pub observe_access: bool,
    /// Reorganize during rotation cycles.
    pub reorganize_during_rotation: bool,
    /// Merge sectors that are similar (reduce fragmentation).
    pub merge_similar_sectors: bool,
    /// Split sectors that are too large or diverse.
    pub split_overloaded_sectors: bool,
    /// Create lanes for sequential access patterns.
    pub create_lanes_for_sequential: bool,
    /// Co-locate registers that are frequently accessed together.
    pub co_locate_frequent_pairs: bool,
    /// Maximum reorganization per rotation cycle (throttle).
    pub max_reorg_per_cycle: usize,
}

impl Default for OptimizationPolicy {
    fn default() -> Self {
        Self {
            self_optimizing: true,
            observe_access: true,
            reorganize_during_rotation: true,
            merge_similar_sectors: true,
            split_overloaded_sectors: true,
            create_lanes_for_sequential: true,
            co_locate_frequent_pairs: true,
            max_reorg_per_cycle: 100,
        }
    }
}

/// Policy for state integrity (validation, repair).
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct IntegrityPolicy {
    /// Validate register checksum on read.
    pub validate_on_read: bool,
    /// Validate XOR lineage on read.
    pub validate_lineage: bool,
    /// Attempt to repair corrupted registers from lineage or backup.
    pub repair_on_corruption: bool,
    /// Log integrity violations.
    pub log_violations: bool,
}

impl Default for IntegrityPolicy {
    fn default() -> Self {
        Self {
            validate_on_read: false,
            validate_lineage: false,
            repair_on_corruption: true,
            log_violations: true,
        }
    }
}

/// Statistics for the fabric.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct FabricStats {
    /// Number of registers in each tier.
    pub register_count_by_tier: HashMap<TierId, usize>,
    /// Number of sectors with bloom filters.
    pub sector_count: usize,
    /// Rotation stats.
    pub total_rotations: u64,
    pub total_echoes_processed: u64,
    pub total_prefetches: u64,
    /// Adaptation stats.
    pub adaptation_deltas_injected: u64,
    pub adaptation_echoes_propagated: u64,
    /// Energy stats.
    pub estimated_energy_per_token_nj: f64,
    pub total_energy_estimate_nj: f64,
    /// Access stats.
    pub total_reads: u64,
    pub total_writes: u64,
    pub total_computes: u64,
    /// Backend stats.
    pub backend_usage: HashMap<BackendId, u64>,
    /// Integrity stats.
    pub corruption_detected: u64,
    pub corruption_repaired: u64,
}

/// Model attachment — a model loaded into the fabric.
#[derive(Clone, Debug)]
pub struct ModelAttachment {
    pub name: String,
    pub base_address: RegisterAddress,
    pub weight_count: usize,
    pub kv_cache_count: usize,
    pub tier: TierId,
    pub quantization: QuantizationLevel,
    /// Whether this model is a delta from a base model.
    pub is_delta: bool,
    pub base_model_name: Option<String>,
}

/// The Binetic Fabric.
///
/// This is the main entry point. Create a fabric with `Fabric::new(config)`,
/// attach models, run compute, and observe stats.
pub struct Fabric {
    config: FabricConfig,
    /// The register store — maps addresses to registers.
    registers: Arc<RwLock<HashMap<RegisterAddress, Register>>>,
    /// Bloom registry for routing and echo gating.
    bloom_registry: Arc<RwLock<BloomRegistry>>,
    /// Rotation scheduler.
    rotation_scheduler: Arc<RwLock<RotationScheduler>>,
    /// Echo propagator.
    echo_propagator: Arc<RwLock<EchoPropagator>>,
    /// Backend set (llama.cpp, MLX, native, etc.).
    backends: Arc<RwLock<BackendSet>>,
    /// Access observation (for self-optimization).
    access_log: Arc<RwLock<Vec<AccessEntry>>>,
    /// Stats.
    stats: Arc<RwLock<FabricStats>>,
    /// Model attachments.
    models: Arc<RwLock<HashMap<String, ModelAttachment>>>,
    /// Network registers — the network as a register that grows recursively.
    networks: Arc<RwLock<HashMap<String, NetworkRegister>>>,
    /// Temporal register pool.
    temporal_pool: Arc<RwLock<Vec<Register>>>,
    /// Initialization state.
    initialized: Arc<RwLock<bool>>,
}

impl Fabric {
    /// Access the model attachments store.
    pub fn models(&self) -> &Arc<RwLock<HashMap<String, ModelAttachment>>> {
        &self.models
    }
}

/// An access entry for observation (self-optimization).
#[derive(Clone, Debug)]
pub struct AccessEntry {
    pub timestamp: u64,
    pub address: RegisterAddress,
    pub operation: AccessOperation,
    pub latency_ms: u64,
    pub tier_id: TierId,
}

#[derive(Clone, Debug)]
pub enum AccessOperation {
    Read,
    Write,
    Compute,
    Fold,
    Adapt,
}

impl Fabric {
    /// Create a new fabric with the given configuration.
    pub fn new(config: FabricConfig) -> Result<Self, String> {
        if config.tiers.is_empty() {
            return Err("At least one tier is required".to_string());
        }

        let tier_set = TierSet::new(config.tiers.clone());
        let bloom_registry = Arc::new(RwLock::new(BloomRegistry::new()));
        let bloom_registry_mutex = Arc::new(std::sync::Mutex::new(BloomRegistry::new()));
        let echo_propagator = Arc::new(RwLock::new(EchoPropagator::new(
            config.echo.clone(),
            Arc::clone(&bloom_registry_mutex),
        )));
        let echo_propagator_mutex = Arc::new(std::sync::Mutex::new(EchoPropagator::new(
            config.echo.clone(),
            Arc::clone(&bloom_registry_mutex),
        )));
        let rotation_scheduler = Arc::new(RwLock::new(RotationScheduler::new(
            config.rotation.clone(),
            tier_set.clone(),
            Arc::clone(&bloom_registry_mutex),
            Arc::clone(&echo_propagator_mutex),
        )));

        let mut backends = BackendSet::new();
        backends.add(Box::new(NativeBackend::new()));
        let backends = Arc::new(RwLock::new(backends));

        let mut temporal_pool = Vec::with_capacity(1024);
        for i in 0..1024 {
            let addr = RegisterAddress::new(0xF000, 0, 0, 0, 0, i as u16);
            temporal_pool.push(Register::new(addr, BitslicedLane::with_capacity(256)));
        }

        Ok(Self {
            config,
            registers: Arc::new(RwLock::new(HashMap::new())),
            bloom_registry,
            rotation_scheduler,
            echo_propagator,
            backends,
            access_log: Arc::new(RwLock::new(Vec::new())),
            stats: Arc::new(RwLock::new(FabricStats::default())),
            models: Arc::new(RwLock::new(HashMap::new())),
            networks: Arc::new(RwLock::new(HashMap::new())),
            temporal_pool: Arc::new(RwLock::new(temporal_pool)),
            initialized: Arc::new(RwLock::new(false)),
        })
    }

    /// Initialize the fabric (set up storage, map files, etc.).
    pub fn initialize(&self) -> Result<(), String> {
        let mut initialized = self.initialized.write();
        if *initialized {
            return Ok(());
        }

        if let Some(ref model_dir) = self.config.model_dir {
            std::fs::create_dir_all(model_dir).map_err(|e| format!("Failed to create model dir: {}", e))?;
        }

        let mut backends = self.backends.write();
        backends
            .with_each_backend_mut(|backend| {
                backend
                    .init()
                    .map_err(|e| format!("Backend init failed: {}", e))
            })
            .map_err(|e| e)?;

        *initialized = true;
        Ok(())
    }

    /// Read a register from the fabric.
    pub fn read(&self, address: RegisterAddress) -> Option<Register> {
        let start = Instant::now();

        let tier_id = {
            let tier_set = TierSet::new(self.config.tiers.clone());
            tier_set
                .tier_for_layer(address.layer())
                .map(|t| t.id)
                .unwrap_or(TierId::COLD)
        };

        let reg = self.registers.read().get(&address).cloned();

        // If this register belongs to a delta model, resolve via base ^ delta
        let reg = match reg {
            Some(reg) => {
                let models = self.models.read();
                let delta_model = models.values().find(|m| m.is_delta && m.base_model_name.is_some()).cloned();
                drop(models);
                if let Some(model) = delta_model {
                    let addr_bits = address.bits();
                    let base_bits = model.base_address.bits();
                    let end_bits = base_bits + model.weight_count as u128;
                    if addr_bits >= base_bits && addr_bits < end_bits {
                        let offset = (addr_bits - base_bits) as u128;
                        let delta_addr_bits = base_bits + model.weight_count as u128 + offset;
                        let delta_addr = RegisterAddress::from(delta_addr_bits);
                        if let Some(delta_reg) = self.registers.read().get(&delta_addr) {
                            let resolved_payload = BitslicedLane::xor(&reg.payload, &delta_reg.payload);
                            let mut resolved = reg.clone();
                            resolved.payload = resolved_payload;
                            Some(resolved)
                        } else {
                            Some(reg)
                        }
                    } else {
                        Some(reg)
                    }
                } else {
                    Some(reg)
                }
            }
            None => None,
        };
        let latency_ms = start.elapsed().as_millis() as u64;

        if self.config.log_access {
            let mut log = self.access_log.write();
            log.push(AccessEntry {
                timestamp: start.elapsed().as_millis() as u64,
                address,
                operation: AccessOperation::Read,
                latency_ms,
                tier_id,
            });
        }

        {
            let mut stats = self.stats.write();
            stats.total_reads += 1;
            *stats.register_count_by_tier.entry(tier_id).or_insert(0) += 1;
        }

        {
            let mut scheduler = self.rotation_scheduler.write();
            scheduler.record_access(address.layer());
        }

        reg
    }

    /// Read a register bypassing delta resolution — returns the raw stored value.
    /// Use this when you need the base register as-is, without XOR delta applied.
    pub fn read_raw(&self, address: RegisterAddress) -> Option<Register> {
        let start = Instant::now();

        let tier_id = {
            let tier_set = TierSet::new(self.config.tiers.clone());
            tier_set
                .tier_for_layer(address.layer())
                .map(|t| t.id)
                .unwrap_or(TierId::COLD)
        };

        let reg = self.registers.read().get(&address).cloned();

        let latency_ms = start.elapsed().as_millis() as u64;

        if self.config.log_access {
            let mut log = self.access_log.write();
            log.push(AccessEntry {
                timestamp: start.elapsed().as_millis() as u64,
                address,
                operation: AccessOperation::Read,
                latency_ms,
                tier_id,
            });
        }

        {
            let mut stats = self.stats.write();
            stats.total_reads += 1;
            *stats.register_count_by_tier.entry(tier_id).or_insert(0) += 1;
        }

        {
            let mut scheduler = self.rotation_scheduler.write();
            scheduler.record_access(address.layer());
        }

        reg
    }

    /// Explicitly resolve deltas for a register — base ^ delta at offset address.
    /// Returns None if no delta model is attached or no delta register exists.
    pub fn resolve(&self, address: RegisterAddress) -> Option<Register> {
        let reg = self.registers.read().get(&address).cloned();
        let reg = reg?;

        let models = self.models.read();
        let delta_model = models.values().find(|m| m.is_delta && m.base_model_name.is_some()).cloned();
        drop(models);

        if let Some(model) = delta_model {
            let addr_bits = address.bits();
            let base_bits = model.base_address.bits();
            let end_bits = base_bits + model.weight_count as u128;
            if addr_bits >= base_bits && addr_bits < end_bits {
                let offset = (addr_bits - base_bits) as u128;
                let delta_addr_bits = base_bits + model.weight_count as u128 + offset;
                let delta_addr = RegisterAddress::from(delta_addr_bits);
                if let Some(delta_reg) = self.registers.read().get(&delta_addr) {
                    let resolved_payload = BitslicedLane::xor(&reg.payload, &delta_reg.payload);
                    let mut resolved = reg.clone();
                    resolved.payload = resolved_payload;
                    return Some(resolved);
                }
            }
        }

        Some(reg)
    }

    /// Write a register to the fabric.
    pub fn write(&self, address: RegisterAddress, mut register: Register) -> Result<(), String> {
        let start = Instant::now();

        let tier_id = {
            let tier_set = TierSet::new(self.config.tiers.clone());
            tier_set.tier_for_layer(address.layer()).map(|t| t.id).unwrap_or(TierId::COLD)
        };

        let delta = if let Some(old_reg) = self.registers.read().get(&address) {
            Some(BitslicedLane::xor(&old_reg.payload, &register.payload))
        } else {
            None
        };

        {
            let mut registers = self.registers.write();
            registers.insert(address, register.clone());
        }

        if let Some(delta) = delta {
            let mut propagator = self.echo_propagator.write();
            let _ = propagator.queue_echo(delta, address, address.layer());
        }

        {
            let mut bloom = self.bloom_registry.write();
            let sector_key = SectorKey::new(
                address.spine(),
                address.layer(),
                address.ring(),
                address.sector(),
            );
            bloom.insert_into_sector(sector_key, address.into());
        }

        if self.config.log_access {
            let latency_ms = start.elapsed().as_millis() as u64;
            let mut log = self.access_log.write();
            log.push(AccessEntry {
                timestamp: start.elapsed().as_millis() as u64,
                address,
                operation: AccessOperation::Write,
                latency_ms,
                tier_id,
            });
        }

        {
            let mut stats = self.stats.write();
            stats.total_writes += 1;
        }

        {
            let mut scheduler = self.rotation_scheduler.write();
            scheduler.record_access(address.layer());
        }

        Ok(())
    }

    /// Read from the temporal register pool.
    pub fn read_temporal(&self, slot: u16) -> Option<Register> {
        let pool = self.temporal_pool.read();
        pool.get(slot as usize).cloned()
    }

    /// Write to the temporal register pool.
    pub fn write_temporal(&self, slot: u16, register: Register) {
        let mut pool = self.temporal_pool.write();
        if (slot as usize) < pool.len() {
            pool[slot as usize] = register;
        }
    }

    /// Execute a compute operation.
    pub fn compute(&self, op: ComputeOp) -> ComputeResult {
        let start = Instant::now();
        let mut result = ComputeResult::default();

        let operand_tiers: Vec<TierId> = op.operands.iter()
            .map(|addr| {
                let tier_set = TierSet::new(self.config.tiers.clone());
                tier_set.tier_for_layer(addr.layer()).map(|t| t.id).unwrap_or(TierId::COLD)
            })
            .collect();

        let backend_id: Option<BackendId> = {
            let backends = self.backends.read();
            match op.backend {
                Some(id) => Some(id),
                None => backends
                    .select_best(&op, &operand_tiers, self.config.rotation.energy_aware),
            }
        };

        let backend_used = backend_id;

        if let Some(id) = backend_id {
            // Look up the backend with the guard kept alive
            let backend_guard = self.backends.read();
            let backend = backend_guard.get(id);

            // Resolve operands from the register store
            let operand_guard = self.registers.read();
            let operands: Vec<BitslicedLane> = op
                .operands
                .iter()
                .filter_map(|addr| operand_guard.get(addr).map(|r| r.payload.clone()))
                .collect();
            drop(operand_guard);

            if let Some(backend) = backend {
                result = backend.execute(&op, &operands);
            }
            drop(backend_guard);

            // Write result back to the register store
            if result.success {
                let mut registers = self.registers.write();
                let reg = Register::new(op.result_address, result.payload.clone().unwrap_or_default());
                registers.insert(op.result_address, reg);
            }
        } else {
            result.success = false;
            result.error_message = Some("No suitable backend found".to_string());
        }

        result.latency_us = start.elapsed().as_micros() as u64;

        {
            let mut stats = self.stats.write();
            stats.total_computes += 1;
            if let Some(id) = backend_used {
                *stats.backend_usage.entry(id).or_insert(0) += 1;
            }
        }

        if self.config.log_access {
            let latency_ms = start.elapsed().as_millis() as u64;
            let mut log = self.access_log.write();
            for addr in &op.operands {
                log.push(AccessEntry {
                    timestamp: start.elapsed().as_millis() as u64,
                    address: *addr,
                    operation: AccessOperation::Compute,
                    latency_ms,
                    tier_id: TierId::COLD,
                });
            }
        }

        {
            let mut scheduler = self.rotation_scheduler.write();
            for addr in &op.operands {
                scheduler.record_access(addr.layer());
            }
        }

        result
    }

    /// Inject an adaptation delta into a weight register.
    pub fn inject_adapt(
        &self,
        target_address: RegisterAddress,
        delta: BitslicedLane,
        purpose: EchoPurpose,
    ) -> Result<(), String> {
        // Read raw register directly, bypassing delta resolution, so the
        // adaptation delta is applied to the base register only once.
        let reg = self
            .registers
            .read()
            .get(&target_address)
            .cloned()
            .ok_or_else(|| "Target register not found".to_string())?;

        let mut new_payload = reg.payload.clone();
        new_payload.xor_merge(&delta);

        let mut updated_reg = reg.clone();
        updated_reg.update_with_delta(&delta);

        self.write(target_address, updated_reg)?;

        {
            let mut propagator = self.echo_propagator.write();
            let echo = StructuredEcho {
                source_address: target_address,
                delta,
                target_layer: target_address.layer(),
                max_depth: 5,
                purpose,
            };
            propagator.inject_structured(echo)?;
        }

        {
            let mut propagator = self.echo_propagator.write();
            let _stats = propagator.propagate(
                |addr| self.read(addr),
                |addr, delta| {
                    if let Some(mut reg) = self.read(addr) {
                        let mut updated = reg.clone();
                        updated.update_with_delta(&delta);
                        let _ = self.write(addr, updated);
                    }
                },
                |addr| {
                    let mut deps = Vec::new();
                    if addr.shard() < 0xFFFE {
                        deps.push(addr.child(1));
                    }
                    deps
                },
            );
        }

        {
            let mut stats = self.stats.write();
            stats.adaptation_deltas_injected += 1;
        }

        Ok(())
    }

    /// Inject an adaptation delta into a raw register — bypasses delta resolution
    /// and skips echo propagation. Use this when you need full control over
    /// when and how deltas are applied (e.g., GA-driven adaptation where the
    /// caller manages the propagation schedule).
    pub fn inject_adapt_raw(
        &self,
        target_address: RegisterAddress,
        delta: BitslicedLane,
    ) -> Result<(), String> {
        {
            let mut registers = self.registers.write();
            let reg = registers.get(&target_address).cloned()
                .ok_or_else(|| "Target register not found".to_string())?;
            let mut updated = reg.clone();
            // update_with_delta already XOR-merges the payload — no separate xor_merge needed
            updated.update_with_delta(&delta);
            registers.insert(target_address, updated);
        }
        Ok(())
    }

    /// Attach a model to the fabric.
    pub fn attach_model(
        &self,
        name: &str,
        _weight_source: &str,
        quantization: QuantizationLevel,
        is_delta: bool,
        base_model_name: Option<&str>,
    ) -> Result<ModelAttachment, String> {
        let weight_count = 1000;
        let kv_cache_count = 100;

        let model = ModelAttachment {
            name: name.to_string(),
            base_address: ranges::weight_address(0, 0, 0),
            weight_count,
            kv_cache_count,
            tier: TierId::COLD,
            quantization,
            is_delta,
            base_model_name: base_model_name.map(String::from),
        };

        {
            let mut models = self.models.write();
            models.insert(name.to_string(), model.clone());
        }

        Ok(model)
    }

    /// Attach a model as an XOR delta from a base model.
    ///
    /// The delta model stores XOR differences at offset addresses from the base.
    /// When reading a delta register, the fabric computes: effective = base ^ delta.
    pub fn attach_delta_model(
        &self,
        name: &str,
        base_model_name: &str,
    ) -> Result<ModelAttachment, String> {
        let base_model = {
            let models = self.models.read();
            models
                .get(base_model_name)
                .ok_or_else(|| format!("Base model '{}' not found", base_model_name))?
                .clone()
        };

        let model = ModelAttachment {
            name: name.to_string(),
            base_address: base_model.base_address,
            weight_count: base_model.weight_count,
            kv_cache_count: base_model.kv_cache_count,
            tier: base_model.tier,
            quantization: base_model.quantization,
            is_delta: true,
            base_model_name: Some(base_model_name.to_string()),
        };

        {
            let mut models = self.models.write();
            models.insert(name.to_string(), model.clone());
        }

        Ok(model)
    }
    pub fn get_model(&self, name: &str) -> Option<ModelAttachment> {
        let models = self.models.read();
        models.get(name).cloned()
    }

    /// Attach a network register to the fabric — the network becomes a register
    /// that can grow recursively without slowdown.
    pub fn attach_network(&self, name: &str, address: RegisterAddress, num_nodes: usize) -> NetworkRegister {
        let mut network = NetworkRegister::new(address, num_nodes);
        {
            let mut networks = self.networks.write();
            networks.insert(name.to_string(), network.clone());
        }
        network
    }

    /// Get a network register by name.
    pub fn get_network(&self, name: &str) -> Option<NetworkRegister> {
        let networks = self.networks.read();
        networks.get(name).cloned()
    }

    /// Evaluate a network register — instant bisliced lane evaluation, O(1) per lane.
    pub fn eval_network(&self, name: &str) -> Option<BitslicedLane> {
        let networks = self.networks.read();
        networks.get(name).map(|n| n.evaluate())
    }

    /// Get the rotation scheduler.
    pub fn rotation_scheduler(&self) -> Arc<RwLock<RotationScheduler>> {
        Arc::clone(&self.rotation_scheduler)
    }

    /// Get the echo propagator.
    pub fn echo_propagator(&self) -> Arc<RwLock<EchoPropagator>> {
        Arc::clone(&self.echo_propagator)
    }

    /// Get the bloom registry.
    pub fn bloom_registry(&self) -> Arc<RwLock<BloomRegistry>> {
        Arc::clone(&self.bloom_registry)
    }

    /// Get the backend set.
    pub fn backends(&self) -> Arc<RwLock<BackendSet>> {
        Arc::clone(&self.backends)
    }

    /// Get current stats.
    pub fn stats(&self) -> FabricStats {
        self.stats.read().clone()
    }

    /// Advance the rotation scheduler by a given elapsed time.
    pub fn advance_rotation(&self, elapsed: Duration) -> Vec<RotationResult> {
        let mut scheduler = self.rotation_scheduler.write();
        let completed_layers = scheduler.advance(elapsed);

        let mut results = Vec::new();
        for &layer_id in &completed_layers {
            let result = scheduler.trigger_rotation(
                layer_id,
                |addr| self.read(addr),
                |addr, reg| { let _ = self.write(addr, reg); },
                |addr| {
                    let mut deps = Vec::new();
                    if addr.shard() < 0xFFFE {
                        deps.push(addr.child(1));
                    }
                    deps
                },
            );
            results.push(result);
        }

        {
            let mut stats = self.stats.write();
            stats.total_rotations += completed_layers.len() as u64;
            for result in &results {
                stats.total_echoes_processed += result.echoes_processed as u64;
                stats.total_prefetches += result.prefetches as u64;
            }
        }

        results
    }

    /// Set energy state for energy-aware scheduling.
    pub fn set_energy_state(&self, state: EnergyState) {
        let mut scheduler = self.rotation_scheduler.write();
        scheduler.set_energy_state(state);
    }

    /// Set thermal state for energy-aware scheduling.
    pub fn set_thermal_state(&self, state: ThermalState) {
        let mut scheduler = self.rotation_scheduler.write();
        scheduler.set_thermal_state(state);
    }

    /// Get the config.
    pub fn config(&self) -> FabricConfig {
        self.config.clone()
    }

    /// Get the register count.
    pub fn register_count(&self) -> usize {
        self.registers.read().len()
    }

    // ── Database query methods ──────────────────────────────────────────────────

    /// Scan registers in a sector range, applying a filter predicate.
    pub fn query_scan<F>(
        &self,
        spine: u16,
        layer: u16,
        sector_start: u16,
        sector_end: u16,
        filter: F,
        _output_address: RegisterAddress,
    ) -> QueryResult
    where
        F: Fn(RegisterAddress, &Register) -> bool,
    {
        let start = Instant::now();
        let mut matches = Vec::new();
        let mut scanned = 0usize;

        for sector in sector_start..=sector_end {
            let sector_key = SectorKey::new(spine, layer, 0, sector);

            {
                let bloom = self.bloom_registry.read();
                if let Some(bloom) = bloom.get_bloom(sector_key) {
                    if bloom.estimated_count() == 0 {
                        continue;
                    }
                }
            }

            let registers = self.registers.read();
            for (addr, reg) in registers.iter() {
                if addr.spine() == spine
                    && addr.layer() == layer
                    && addr.sector() == sector
                {
                    scanned += 1;
                    if filter(*addr, reg) {
                        matches.push(*addr);
                    }
                }
            }
        }

        let elapsed = start.elapsed();

        QueryResult {
            match_count: matches.len(),
            scanned_count: scanned,
            elapsed_us: elapsed.as_micros() as u64,
            match_addresses: matches,
        }
    }

    /// Insert a record (compound register) into the database.
    pub fn db_insert_record(&self, record: &Record) -> Result<(), String> {
        let entity_type = record.metadata.entity_type;

        for (field_idx, _field) in record.schema.iter().enumerate() {
            let addr = RegisterAddress::new(
                entity_type,
                5,
                0,
                field_idx as u16,
                entity_type,
                0,
            );

            let reg = Register::new(addr, BitslicedLane::with_capacity(256));
            self.write(addr, reg)?;
        }

        Ok(())
    }

    /// Update a record field (echo-propagated delta).
    pub fn db_update_record_field(
        &self,
        entity_type: u16,
        entity_id: u16,
        _field_name: &str,
        new_value: BitslicedLane,
    ) -> Result<(), String> {
        let field_idx = 0;

        let addr = RegisterAddress::new(
            entity_type,
            5,
            0,
            field_idx as u16,
            entity_id,
            0,
        );

        if let Some(existing) = self.read(addr) {
            let delta = BitslicedLane::xor(&existing.payload, &new_value);
            self.inject_adapt(addr, delta, EchoPurpose::RecordUpdate)?;
            Ok(())
        } else {
            Err(format!("Record not found: entity_type={}, entity_id={}", entity_type, entity_id))
        }
    }

    /// Delete a record (mark as deleted via a tombstone delta).
    pub fn db_delete_record(&self, entity_type: u16, entity_id: u16) -> Result<(), String> {
        let addr = RegisterAddress::new(
            entity_type,
            5,
            0,
            0xFFFF,
            entity_id,
            0,
        );

        let tombstone = Register::new(
            addr,
            BitslicedLane::from_bits(&[true]),
        );

        self.write(addr, tombstone)?;
        Ok(())
    }
}

impl Default for Fabric {
    fn default() -> Self {
        Self::new(FabricConfig::default()).expect("Failed to create default fabric")
    }
}

/// Helper functions for common compute operations.
impl Fabric {
    /// Sample the next token (convenience method).
    pub fn sample_next_token(
        &self,
        _model_name: &str,
        kv_cache_address: RegisterAddress,
        activations_address: RegisterAddress,
        temperature: f32,
        top_k: usize,
        top_p: f32,
    ) -> ComputeResult {
        let op = ComputeOp {
            operation: ArithmeticOp::SampleNextToken,
            operands: vec![kv_cache_address, activations_address],
            result_address: RegisterAddress::new(0xF000, 0, 0, 0, 0, 0),
            backend: None,
            routing: RoutingPolicy::ToData,
            params: ComputeParams {
                temperature: Some(temperature),
                top_k: Some(top_k),
                top_p: Some(top_p),
                ..Default::default()
            },
        };
        self.compute(op)
    }

    /// Matrix multiply (convenience method).
    pub fn matmul(
        &self,
        a_address: RegisterAddress,
        b_address: RegisterAddress,
        result_address: RegisterAddress,
    ) -> ComputeResult {
        let op = ComputeOp {
            operation: ArithmeticOp::MatMul,
            operands: vec![a_address, b_address],
            result_address,
            backend: None,
            routing: RoutingPolicy::ToData,
            params: ComputeParams::default(),
        };
        self.compute(op)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::address::ranges;

    #[test]
    fn test_fabric_creation() {
        let fabric = Fabric::new(FabricConfig::default()).expect("Failed to create fabric");
        assert_eq!(fabric.register_count(), 0);
        assert_eq!(fabric.stats().total_reads, 0);
    }

    #[test]
    fn test_fabric_write_read() {
        let fabric = Fabric::new(FabricConfig::default()).expect("Failed to create fabric");
        fabric.initialize().expect("Failed to initialize");

        let addr = RegisterAddress::new(0, 1, 2, 3, 4, 0);
        let payload = BitslicedLane::from_bits(&[true, false, true, false, true, false, true, false]);
        let reg = Register::new(addr, payload);

        fabric.write(addr, reg.clone()).expect("Failed to write");
        let read_reg = fabric.read(addr).expect("Failed to read");

        assert_eq!(read_reg.payload.len(), 8);
        assert_eq!(read_reg.payload.get_bit(0), true);
        assert_eq!(read_reg.payload.get_bit(1), false);
    }

    #[test]
    fn test_fabric_temporal() {
        let fabric = Fabric::new(FabricConfig::default()).expect("Failed to create fabric");

        let addr = RegisterAddress::new(0xF000, 0, 0, 0, 0, 0);
        let payload = BitslicedLane::from_bits(&[true; 64]);
        let reg = Register::new(addr, payload);

        fabric.write_temporal(0, reg.clone());
        let read = fabric.read_temporal(0).expect("Failed to read temporal");
        assert_eq!(read.payload.len(), 64);
        assert!(read.payload.all_set());
    }

    #[test]
    fn test_fabric_compute_add() {
        let fabric = Fabric::new(FabricConfig::default()).expect("Failed to create fabric");
        fabric.initialize().expect("Failed to initialize");

        let addr_a = ranges::weight_address(1, 0, 0);
        let addr_b = ranges::weight_address(2, 0, 0);
        let addr_result = ranges::weight_address(3, 0, 0);

        let payload_a = BitslicedLane::from_bits(&[true, false, true, false]);
        let payload_b = BitslicedLane::from_bits(&[false, true, false, true]);

        fabric.write(addr_a, Register::new(addr_a, payload_a)).unwrap();
        fabric.write(addr_b, Register::new(addr_b, payload_b)).unwrap();

        let result = fabric.compute(ComputeOp {
            operation: ArithmeticOp::Add,
            operands: vec![addr_a, addr_b],
            result_address: addr_result,
            backend: Some(BackendId::NATIVE),
            routing: RoutingPolicy::ToData,
            params: ComputeParams::default(),
        });

        assert!(result.success);

        let result_reg = fabric.read(addr_result).expect("Failed to read result");
        assert!(result_reg.payload.all_set());
    }

    #[test]
    fn test_fabric_adapt() {
        let fabric = Fabric::new(FabricConfig::default()).expect("Failed to create fabric");
        fabric.initialize().expect("Failed to initialize");

        let addr = ranges::weight_address(1, 0, 0);
        let payload = BitslicedLane::from_bits(&[true, false, true, false]);
        fabric.write(addr, Register::new(addr, payload)).unwrap();

        let delta = BitslicedLane::from_bits(&[false, true, false, false]);
        fabric.inject_adapt(addr, delta, EchoPurpose::WeightUpdate).unwrap();

        let updated = fabric.read(addr).unwrap();
        assert_eq!(updated.payload.get_bit(0), true);
        assert_eq!(updated.payload.get_bit(1), true);
        assert_eq!(updated.payload.get_bit(2), true);
        assert_eq!(updated.payload.get_bit(3), false);
    }

    #[test]
    fn test_xor_delta_model_sharing_with_adaptation() {
        let fabric = Fabric::new(FabricConfig::default()).expect("Failed to create fabric");
        fabric.initialize().expect("Failed to initialize");

        // Step 1: Attach base model
        let base_model = fabric
            .attach_model("base-v1", "", QuantizationLevel::Q4_0, false, None)
            .unwrap();

        // Step 2: Write base model weights into the fabric
        let base_addr = base_model.base_address;
        let base_payload = BitslicedLane::from_bits(&[true, false, true, false]);
        fabric.write(base_addr, Register::new(base_addr, base_payload)).unwrap();

        // Step 3: Attach delta model from base
        let delta_model = fabric
            .attach_delta_model("delta-v1", "base-v1")
            .unwrap();
        assert!(delta_model.is_delta);
        assert_eq!(delta_model.base_model_name.as_deref(), Some("base-v1"));

        // Step 4: Write delta register at offset address (XOR diff from base)
        // base = [1,0,1,0], delta = [0,1,0,0] → effective = [1,1,1,0]
        let base_bits = base_model.base_address.bits();
        let delta_bits = base_bits + delta_model.weight_count as u128;
        let delta_addr = RegisterAddress::from(delta_bits);
        let delta_payload = BitslicedLane::from_bits(&[false, true, false, false]);
        fabric.write(delta_addr, Register::new(delta_addr, delta_payload)).unwrap();

        // Step 5: Read via delta model — should resolve base ^ delta
        let resolved = fabric.read(base_addr).unwrap();
        // base ^ delta = [1^0, 0^1, 1^0, 0^0] = [1, 1, 1, 0]
        assert_eq!(resolved.payload.get_bit(0), true);
        assert_eq!(resolved.payload.get_bit(1), true);
        assert_eq!(resolved.payload.get_bit(2), true);
        assert_eq!(resolved.payload.get_bit(3), false);

        // Step 6: Inject adaptation delta into the delta model
        let adapt_delta = BitslicedLane::from_bits(&[false, false, true, false]);
        fabric
            .inject_adapt(base_addr, adapt_delta, EchoPurpose::WeightUpdate)
            .unwrap();

        // Step 7: Read after adaptation — effective = base ^ delta ^ adapt_delta
        let adapted = fabric.read(base_addr).unwrap();
        // base ^ delta ^ adapt = [1,1,1,0] ^ [0,0,1,0] = [1,1,0,0]
        assert_eq!(adapted.payload.get_bit(0), true);
        assert_eq!(adapted.payload.get_bit(1), true);
        assert_eq!(adapted.payload.get_bit(2), false);
        assert_eq!(adapted.payload.get_bit(3), false);

        // Stats verify
        let stats = fabric.stats();
        assert_eq!(stats.adaptation_deltas_injected, 1);
    }

    #[test]
    fn test_fabric_rotation_advance() {
        let fabric = Fabric::new(FabricConfig::default()).expect("Failed to create fabric");
        fabric.initialize().expect("Failed to initialize");

        let results = fabric.advance_rotation(Duration::from_millis(16));
        assert!(!results.is_empty());
    }

    #[test]
    fn test_fabric_energy_state() {
        let fabric = Fabric::new(FabricConfig::default()).expect("Failed to create fabric");
        fabric.initialize().expect("Failed to initialize");

        fabric.set_energy_state(EnergyState::OnBattery { level: 0.1 });
        fabric.set_thermal_state(ThermalState::Hot);

        let scheduler = fabric.rotation_scheduler();
        let sched = scheduler.read();
        assert!(matches!(sched.energy_state(), EnergyState::OnBattery { level: 0.1 }));
        assert!(matches!(sched.thermal_state(), ThermalState::Hot));
    }

    #[test]
    fn test_fabric_stats() {
        let fabric = Fabric::new(FabricConfig::default()).expect("Failed to create fabric");
        fabric.initialize().expect("Failed to initialize");

        let addr = RegisterAddress::new(0, 1, 2, 3, 4, 0);
        let payload = BitslicedLane::from_bits(&[true]);
        fabric.write(addr, Register::new(addr, payload)).unwrap();
        fabric.read(addr);
        fabric.compute(ComputeOp {
            operation: ArithmeticOp::Add,
            operands: vec![addr, addr],
            result_address: addr,
            backend: Some(BackendId::NATIVE),
            routing: RoutingPolicy::ToData,
            params: ComputeParams::default(),
        });

        let stats = fabric.stats();
        assert_eq!(stats.total_reads, 1);
        assert_eq!(stats.total_writes, 1);
        assert_eq!(stats.total_computes, 1);
    }

    #[test]
    fn test_fabric_db_insert_and_query() {
        let fabric = Fabric::new(FabricConfig::default()).expect("Failed to create fabric");
        fabric.initialize().expect("Failed to initialize");

        let fields = vec![
            ("id".to_string(), BitslicedLane::from_bytes(&[42])),
            ("name".to_string(), BitslicedLane::from_bytes(b"Alice")),
            ("age".to_string(), BitslicedLane::from_bytes(&[30])),
        ];
        let record = Record::new(0x1000, fields);

        fabric.db_insert_record(&record).expect("Failed to insert record");

        let result = fabric.query_scan(
            0x1000,
            5,
            0,
            0xFFFF,
            |_, reg| reg.payload.len() > 0,
            RegisterAddress::zero(),
        );

        assert_eq!(result.match_count, 3);
        assert!(result.scanned_count > 0);
    }

    #[test]
    fn test_fabric_db_update() {
        let fabric = Fabric::new(FabricConfig::default()).expect("Failed to create fabric");
        fabric.initialize().expect("Failed to initialize");

        let fields = vec![
            ("age".to_string(), BitslicedLane::from_bytes(&[30])),
        ];
        let record = Record::new(0x1000, fields);
        fabric.db_insert_record(&record).unwrap();

        let new_age = BitslicedLane::from_bytes(&[31]);
        fabric.db_update_record_field(0x1000, 0, "age", new_age).unwrap();

        let stats = fabric.stats();
        assert_eq!(stats.adaptation_deltas_injected, 1);
    }

    #[test]
    fn test_fabric_db_delete() {
        let fabric = Fabric::new(FabricConfig::default()).expect("Failed to create fabric");
        fabric.initialize().expect("Failed to initialize");

        let fields = vec![
            ("id".to_string(), BitslicedLane::from_bytes(&[42])),
        ];
        let record = Record::new(0x1000, fields);
        fabric.db_insert_record(&record).unwrap();

        fabric.db_delete_record(0x1000, 42).unwrap();

        let addr = RegisterAddress::new(0x1000, 5, 0, 0xFFFF, 42, 0);
        let reg = fabric.read(addr);
        assert!(reg.is_some());
        assert!(reg.unwrap().payload.get_bit(0));
    }

    #[test]
    fn test_compound_register_xor_chain() {
        use crate::register::CombiningMethod;

        let base0_addr = RegisterAddress::new(0, 1, 0, 0, 0, 0);
        let base1_addr = RegisterAddress::new(0, 1, 0, 0, 1, 0);

        let base0 = Register::new(base0_addr, BitslicedLane::from_bits(&[true, false, true, false]));
        let base1 = Register::new(base1_addr, BitslicedLane::from_bits(&[false, true, false, true]));

        let compound = CompoundRegister::new(CombiningMethod::XorChain, &[base0, base1]);

        assert_eq!(compound.base_count, 2);
        assert_eq!(compound.precision_bits, 128);
        assert!(compound.payload.all_set());
    }

    #[test]
    fn test_compound_register_sequential() {
        use crate::register::CombiningMethod;

        let base0 = Register::new(
            RegisterAddress::new(0, 1, 0, 0, 0, 0),
            BitslicedLane::from_bits(&[true, false]),
        );
        let base1 = Register::new(
            RegisterAddress::new(0, 1, 0, 0, 1, 0),
            BitslicedLane::from_bits(&[false, true]),
        );

        let b0 = base0.clone();
        let b1 = base1.clone();

        let compound = CompoundRegister::new(CombiningMethod::Sequential, &[b0, b1]);

        assert_eq!(compound.base_count, 2);
        assert_eq!(compound.precision_bits, 128);
        assert_eq!(compound.combining, CombiningMethod::Sequential);
        assert!(compound.is_consistent(|addr| {
            if addr == base0.address { Some(base0.clone()) }
            else if addr == base1.address { Some(base1.clone()) }
            else { None }
        }));
    }

    #[test]
    fn test_raw_vs_resolved_read() {
        let fabric = Fabric::new(FabricConfig::default()).expect("Failed to create fabric");
        fabric.initialize().expect("Failed to initialize");

        // Write a base register
        let addr = RegisterAddress::new(0, 1, 0, 0, 0, 0);
        let base_payload = BitslicedLane::from_bits(&[true, false, true, false]);
        fabric.write(addr, Register::new(addr, base_payload.clone())).unwrap();

        // read_raw returns the raw value
        let raw = fabric.read_raw(addr).unwrap();
        assert_eq!(raw.payload.get_bit(0), true);
        assert_eq!(raw.payload.get_bit(1), false);

        // resolve without a delta model returns the raw value
        let resolved = fabric.resolve(addr).unwrap();
        assert_eq!(resolved.payload.get_bit(0), true);
    }

    #[test]
    fn test_inject_adapt_raw() {
        let fabric = Fabric::new(FabricConfig::default()).expect("Failed to create fabric");
        fabric.initialize().expect("Failed to initialize");

        let addr = RegisterAddress::new(0, 1, 0, 0, 0, 0);
        let base_payload = BitslicedLane::from_bits(&[true, false, true, false]);
        fabric.write(addr, Register::new(addr, base_payload)).unwrap();

        // Apply delta via inject_adapt_raw — no echo propagation, no delta resolution
        let delta = BitslicedLane::from_bits(&[false, true, false, false]);
        fabric.inject_adapt_raw(addr, delta).unwrap();

        // read_raw returns the raw value with delta XOR-applied directly
        let raw = fabric.read_raw(addr).unwrap();
        // [true,false,true,false] XOR [false,true,false,false] = [true,true,true,false]
        assert_eq!(raw.payload.get_bit(0), true);
        assert_eq!(raw.payload.get_bit(1), true);
        assert_eq!(raw.payload.get_bit(2), true);
        assert_eq!(raw.payload.get_bit(3), false);
    }
}
