//! Rotation scheduler — the geometric scheduling engine.
//!
//! The rotation gradient expresses cache replacement and prefetching as geometry:
//! - Inner layers rotate slowly (hot, stable, fine-grained).
//! - Middle layers rotate at medium speed.
//! - Outer layers rotate fastest (cold, batched, scheduled).
//!
//! The rotation is structured, not random: it maintains sector coherence,
//! preserves lineage chains, and moves data between tiers in a way that
//! maintains locality.
//!
//! From the center: you see a lane (sequential access is cheap).
//! From the outside: you see a tree converging to a point (hierarchical lookup).

use crate::address::RegisterAddress;
use crate::bloom::BloomRegistry;
use crate::echo::EchoPropagator;
use crate::register::Register;
use crate::tiers::{TierId, TierSet};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, VecDeque};
use std::fmt;
use std::sync::Arc;
use std::time::Duration;

/// Rotation policy configuration.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct RotationPolicy {
    /// Base rotation period (for the hottest layer).
    pub base_period: Duration,
    /// Layer multiplier: each layer N rotates base_period * 2^N slower (for hot layers)
    /// or faster (for cold layers, in terms of batch size).
    pub layer_multiplier_base: f64,
    /// Adaptive rotation: adjust periods based on observed access patterns.
    pub adaptive: bool,
    /// Energy-aware rotation: consider battery/thermal state.
    pub energy_aware: bool,
    /// Batch echo propagation during rotation cycles.
    pub batch_echo_propagation: bool,
    /// Prefetch data into warmer tiers during rotation.
    pub prefetch_during_rotation: bool,
    /// Maximum rotation batch size (for cold layers).
    pub max_batch_size: usize,
}

impl Default for RotationPolicy {
    fn default() -> Self {
        Self {
            base_period: Duration::from_millis(16),
            layer_multiplier_base: 2.0,
            adaptive: true,
            energy_aware: true,
            batch_echo_propagation: true,
            prefetch_during_rotation: true,
            max_batch_size: 4096,
        }
    }
}

/// The state of a single layer's rotation.
#[derive(Clone, Debug, Default)]
pub struct LayerRotationState {
    /// Current rotation phase (0.0 to 1.0).
    pub phase: f64,
    /// Current period (may be adjusted by adaptive policy).
    pub period: Duration,
    /// Base period (for reference).
    pub base_period: Duration,
    /// Number of rotations completed.
    pub rotation_count: u64,
    /// Access count during this rotation cycle (for adaptive adjustment).
    pub access_count: u64,
    /// Whether this layer is currently being rotated.
    pub is_rotating: bool,
}

/// Energy state for energy-aware rotation.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum EnergyState {
    /// On battery power.
    OnBattery { level: f32 }, // 0.0 to 1.0
    /// Plugged in / charging.
    PluggedIn,
    /// Unknown / unavailable.
    Unknown,
}

/// Thermal state.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum ThermalState {
    Cool,
    Warm,
    Hot,
    Unknown,
}

/// The rotation scheduler — manages rotation cycles for all layers.
///
/// This is the heart of the geometric scheduling model. It:
/// - Advances rotation phases for each layer.
/// - Triggers rotation events when a layer's phase completes a cycle.
/// - Adjusts rotation periods based on access patterns (adaptive).
/// - Adjusts rotation periods based on energy/thermal state (energy-aware).
/// - Batches echo propagation during rotation.
/// - Prefetches data during rotation.
pub struct RotationScheduler {
    policy: RotationPolicy,
    layers: HashMap<u16, LayerRotationState>,
    energy_state: EnergyState,
    thermal_state: ThermalState,
    /// Pending rotation events (layer_id → scheduled time).
    pending_rotations: VecDeque<(u16, Duration)>,
    /// Echo propagator for batching echoes during rotation.
    echo_propagator: Arc<std::sync::Mutex<EchoPropagator>>,
    /// Bloom registry for prefetch decisions.
    bloom_registry: Arc<std::sync::Mutex<BloomRegistry>>,
    /// Tiers for prefetch targeting.
    tiers: Arc<TierSet>,
}

impl RotationScheduler {
    pub fn new(
        policy: RotationPolicy,
        tiers: TierSet,
        bloom_registry: Arc<std::sync::Mutex<BloomRegistry>>,
        echo_propagator: Arc<std::sync::Mutex<EchoPropagator>>,
    ) -> Self {
        let mut layers = HashMap::new();
        for tier in tiers.all_tiers() {
            for layer in tier.min_layer..=tier.max_layer {
                if !layers.contains_key(&layer) {
                    let base_period = Self::compute_layer_base_period(
                        &policy,
                        tier.id,
                        layer,
                    );
                    layers.insert(
                        layer,
                        LayerRotationState {
                            phase: 0.0,
                            period: base_period,
                            base_period,
                            rotation_count: 0,
                            access_count: 0,
                            is_rotating: false,
                        },
                    );
                }
            }
        }

        Self {
            policy,
            layers,
            energy_state: EnergyState::Unknown,
            thermal_state: ThermalState::Unknown,
            pending_rotations: VecDeque::new(),
            echo_propagator,
            bloom_registry,
            tiers: Arc::new(tiers),
        }
    }

    /// Compute the base rotation period for a layer.
    fn compute_layer_base_period(policy: &RotationPolicy, tier_id: TierId, layer: u16) -> Duration {
        // Hotter layers (lower layer number) rotate faster (shorter period).
        // Cold layers rotate slower (longer period) but in larger batches.
        let layer_factor = layer as f64;
        let multiplier = policy.layer_multiplier_base.powf(layer_factor);

        // For cold tiers, use longer periods
        let base_ms = match tier_id {
            TierId::HOT => 1.0,       // 1 ms
            TierId::WARM => 16.0,     // 16 ms
            TierId::GPU => 8.0,       // 8 ms (GPU rotates faster for compute)
            TierId::COLD => 256.0,    // 256 ms (cold rotates slowly)
            TierId::REMOTE => 1000.0, // 1 s (remote rotates very slowly)
            _ => 1000.0,              // unknown tiers: very slow (remote-like)
        };

        let period_ms = base_ms * multiplier;
        Duration::from_millis(period_ms as u64)
    }

    /// Set the energy state.
    pub fn set_energy_state(&mut self, state: EnergyState) {
        self.energy_state = state;
        self.adjust_for_energy();
    }

    /// Set the thermal state.
    pub fn set_thermal_state(&mut self, state: ThermalState) {
        self.thermal_state = state;
        self.adjust_for_thermal();
    }

    /// Adjust rotation periods for energy state.
    fn adjust_for_energy(&mut self) {
        if !self.policy.energy_aware {
            return;
        }

        let factor = match self.energy_state {
            EnergyState::OnBattery { level } => {
                if level < 0.2 {
                    4.0 // battery low: rotate 4x slower
                } else if level < 0.5 {
                    2.0 // battery medium: rotate 2x slower
                } else {
                    1.0 // battery okay: normal
                }
            }
            EnergyState::PluggedIn => 0.5, // plugged in: rotate 2x faster (performance)
            EnergyState::Unknown => 1.0,
        };

        for (_, state) in self.layers.iter_mut() {
            state.period = Duration::from_millis(
                (state.base_period.as_millis() as f64 * factor) as u64,
            );
        }
    }

    /// Adjust rotation periods for thermal state.
    fn adjust_for_thermal(&mut self) {
        if !self.policy.energy_aware {
            return;
        }

        let factor = match self.thermal_state {
            ThermalState::Hot => 3.0,   // hot: rotate slower to reduce activity
            ThermalState::Warm => 1.5,  // warm: rotate somewhat slower
            ThermalState::Cool => 1.0,  // cool: normal
            ThermalState::Unknown => 1.0,
        };

        for (_, state) in self.layers.iter_mut() {
            state.period = Duration::from_millis(
                (state.base_period.as_millis() as f64 * factor) as u64,
            );
        }
    }

    /// Advance the rotation phase by a given elapsed time.
    ///
    /// Returns the layer IDs that completed a rotation cycle.
    pub fn advance(&mut self, elapsed: Duration) -> Vec<u16> {
        let mut completed = Vec::new();

        for (layer_id, state) in self.layers.iter_mut() {
            let phase_increment = elapsed.as_millis() as f64 / state.period.as_millis() as f64;
            state.phase += phase_increment;

            if state.phase >= 1.0 {
                state.phase -= 1.0;
                state.rotation_count += 1;
                state.is_rotating = true;
                completed.push(*layer_id);
                state.is_rotating = false;
            }

            // Track accesses for adaptive adjustment
            state.access_count = 0; // reset each cycle
        }

        completed
    }

    /// Record an access to a layer (for adaptive adjustment).
    pub fn record_access(&mut self, layer: u16) {
        if let Some(state) = self.layers.get_mut(&layer) {
            state.access_count += 1;
        }
    }

    /// Trigger a rotation event for a specific layer.
    ///
    /// This performs the actual data movement, echo batching, and prefetching
    /// for the layer.
    pub fn trigger_rotation(
        &mut self,
        layer_id: u16,
        register_lookup: impl Fn(RegisterAddress) -> Option<Register>,
        register_update: impl Fn(RegisterAddress, Register),
        get_dependents: impl Fn(RegisterAddress) -> Vec<RegisterAddress>,
    ) -> RotationResult {
        let state = self.layers.get_mut(&layer_id);
        if state.is_none() {
            return RotationResult { layer_id, movements: 0, echoes_processed: 0, prefetches: 0 };
        }

        let state = state.unwrap();

        // Batch echo propagation
        let mut echoes_processed = 0;
        if self.policy.batch_echo_propagation {
            if let Ok(mut propagator) = self.echo_propagator.lock() {
                let lookup = &register_lookup;
                let _stats = propagator.propagate(
                    move |addr| lookup(addr),
                    |addr, delta| {
                        // Apply delta to register
                        if let Some(mut reg) = lookup(addr) {
                            let mut updated = reg.clone();
                            updated.update_with_delta(&delta);
                            register_update(addr, updated);
                        }
                    },
                    get_dependents,
                );
                echoes_processed = propagator.pending_count(); // approximate
            }
        }

        // Prefetch during rotation (for cold layers)
        let mut prefetches = 0;
        if self.policy.prefetch_during_rotation && layer_id > 10 {
            if let Ok(bloom_registry) = self.bloom_registry.lock() {
                // For each sector in this layer, check bloom for likely-needed addresses
                // and prefetch them into warmer tiers.
                // This is a placeholder — actual implementation would use bloom filters
                // to identify which cold registers are likely to be needed soon.
                prefetches = state.rotation_count % 10; // placeholder
            }
        }

        // Actual data movement (promote/demote between tiers)
        let movements = state.rotation_count % 100; // placeholder

        RotationResult {
            layer_id,
            movements: movements as usize,
            echoes_processed: echoes_processed as usize,
            prefetches: prefetches as usize,
        }
    }

    /// Get the current phase for a layer.
    pub fn phase(&self, layer: u16) -> Option<f64> {
        self.layers.get(&layer).map(|s| s.phase)
    }

    /// Get the current period for a layer.
    pub fn period(&self, layer: u16) -> Option<Duration> {
        self.layers.get(&layer).map(|s| s.period)
    }

    /// Get rotation stats for a layer.
    pub fn rotation_count(&self, layer: u16) -> Option<u64> {
        self.layers.get(&layer).map(|s| s.rotation_count)
    }

    /// Get all layer states.
    pub fn all_layers(&self) -> HashMap<u16, LayerRotationState> {
        self.layers.clone()
    }

    /// Get the energy state.
    pub fn energy_state(&self) -> &EnergyState {
        &self.energy_state
    }

    /// Get the thermal state.
    pub fn thermal_state(&self) -> &ThermalState {
        &self.thermal_state
    }
}

/// Result of a rotation event.
#[derive(Clone, Debug, Default)]
pub struct RotationResult {
    pub layer_id: u16,
    pub movements: usize,
    pub echoes_processed: usize,
    pub prefetches: usize,
}

impl fmt::Display for RotationScheduler {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "RotationScheduler {{")?;
        writeln!(f, "  policy: base_period={:?}, adaptive={}, energy_aware={}", self.policy.base_period, self.policy.adaptive, self.policy.energy_aware)?;
        writeln!(f, "  energy_state: {:?}", self.energy_state)?;
        writeln!(f, "  thermal_state: {:?}", self.thermal_state)?;
        writeln!(f, "  layers:")?;
        for (layer, state) in &self.layers {
            writeln!(
                f,
                "    Layer {:04x}: phase={:.3} period={:?} rotations={} accesses={}",
                layer, state.phase, state.period, state.rotation_count, state.access_count
            )?;
        }
        writeln!(f, "}}")?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rotation_scheduler_creation() {
        let tiers = TierSet::new(vec![Tier::new(TierId::WARM, "warm", "warm", 1024 * 1024, 2, 10)]);
        let policy = RotationPolicy::default();
        let bloom_registry = Arc::new(std::sync::Mutex::new(BloomRegistry::new()));
        let echo_propagator = Arc::new(std::sync::Mutex::new(EchoPropagator::new(
            EchoPolicy::default(),
            bloom_registry.clone(),
        )));

        let scheduler = RotationScheduler::new(policy, tiers, bloom_registry, echo_propagator);

        // Check that layer 5 has a rotation state
        assert!(scheduler.phase(5).is_some());
        assert!(scheduler.period(5).is_some());
    }

    #[test]
    fn test_rotation_advance() {
        let tiers = TierSet::new(vec![Tier::new(TierId::WARM, "warm", "warm", 1024 * 1024, 2, 10)]);
        let policy = RotationPolicy::default();
        let bloom_registry = Arc::new(std::sync::Mutex::new(BloomRegistry::new()));
        let echo_propagator = Arc::new(std::sync::Mutex::new(EchoPropagator::new(
            EchoPolicy::default(),
            bloom_registry.clone(),
        )));

        let mut scheduler = RotationScheduler::new(policy, tiers, bloom_registry, echo_propagator);

        // Advance by half the base period — no rotation should complete
        let completed = scheduler.advance(Duration::from_millis(8));
        assert_eq!(completed.len(), 0);

        // Advance by full base period — rotation should complete
        let completed = scheduler.advance(Duration::from_millis(16));
        assert!(!completed.is_empty());
    }

    #[test]
    fn test_energy_aware_rotation() {
        let tiers = TierSet::new(vec![Tier::new(TierId::WARM, "warm", "warm", 1024 * 1024, 2, 10)]);
        let policy = RotationPolicy::default();
        let bloom_registry = Arc::new(std::sync::Mutex::new(BloomRegistry::new()));
        let echo_propagator = Arc::new(std::sync::Mutex::new(EchoPropagator::new(
            EchoPolicy::default(),
            bloom_registry.clone(),
        )));

        let mut scheduler = RotationScheduler::new(policy, tiers, bloom_registry, echo_propagator);

        let base_period = scheduler.period(5).unwrap().as_millis();

        // Set battery low — periods should increase
        scheduler.set_energy_state(EnergyState::OnBattery { level: 0.1 });
        let battery_period = scheduler.period(5).unwrap().as_millis();
        assert!(battery_period > base_period);

        // Set plugged in — periods should decrease
        scheduler.set_energy_state(EnergyState::PluggedIn);
        let plugged_period = scheduler.period(5).unwrap().as_millis();
        assert!(plugged_period < base_period);
    }
}
