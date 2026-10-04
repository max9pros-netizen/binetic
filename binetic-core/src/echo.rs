//! Echo minimization — the propagation policy for the fabric.
//!
//! When a register changes, the change ripples through the fabric as an "echo".
//! Echo minimization ensures the ripple is exactly as large as it must be, no more.
//!
//! Mechanisms:
//! - Minimal delta propagation: changes travel as XOR deltas, not full state.
//! - Bloom gating: propagate only to sectors that might contain dependents.
//! - Rotation damping: cold layer changes are queued for the next rotation cycle.
//! - XOR echo cancellation: redundant paths cancel at merge points.

use crate::address::RegisterAddress;
use crate::bitslice::BitslicedLane;
use crate::bloom::{BloomRegistry, SectorKey};
use crate::register::Register;
use crate::tiers::TierId;
use serde::{Deserialize, Serialize};
use std::collections::VecDeque;
use std::fmt;
use std::sync::Arc;

/// Policy for echo propagation.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct EchoPolicy {
    /// Minimize echo: propagate deltas, not full state.
    pub minimal: bool,
    /// Gate propagation with bloom filters: skip sectors that can't contain dependents.
    pub bloom_gated: bool,
    /// Damp propagation for cold layers: queue for next rotation cycle.
    pub rotation_damped: bool,
    /// Cancel redundant echoes: if delta_A ^ delta_B == 0, stop propagating.
    pub xor_cancel: bool,
    /// Maximum propagation depth (0 = unlimited).
    pub max_depth: u8,
    /// Maximum echo queue size before dropping.
    pub max_queue_size: usize,
}

impl Default for EchoPolicy {
    fn default() -> Self {
        Self {
            minimal: true,
            bloom_gated: true,
            rotation_damped: true,
            xor_cancel: true,
            max_depth: 0, // unlimited
            max_queue_size: 1024,
        }
    }
}

/// A pending echo — a delta waiting to be propagated.
#[derive(Clone, Debug)]
pub struct PendingEcho {
    /// The delta (XOR difference from old to new state).
    pub delta: BitslicedLane,
    /// Source address (where the change originated).
    pub source_address: RegisterAddress,
    /// Current propagation depth.
    pub depth: u8,
    /// Target layer for this echo (if rotation-damped, echoes are queued per target layer).
    pub target_layer: u16,
    /// Timestamp (for ordering).
    pub timestamp: u64,
}

/// An echo that has been processed (for stats).
#[derive(Clone, Debug, Default)]
pub struct EchoStats {
    /// Number of echoes propagated.
    pub propagated: u64,
    /// Number of echoes canceled (XOR cancellation).
    pub canceled: u64,
    /// Number of echoes gated by bloom filter (skipped).
    pub bloom_gated: u64,
    /// Number of echoes damped by rotation (queued).
    pub rotation_damped: u64,
    /// Number of echoes dropped (queue full).
    pub dropped: u64,
    /// Total delta bytes propagated.
    pub bytes_propagated: u64,
}

/// The echo propagator — manages pending echoes and propagates them according to policy.
///
/// This is the core of echo minimization: it takes changes (deltas), queues them,
/// and propagates them to dependent registers with minimal overhead.
pub struct EchoPropagator {
    policy: EchoPolicy,
    pending: VecDeque<PendingEcho>,
    stats: EchoStats,
    bloom_registry: Arc<std::sync::Mutex<BloomRegistry>>,
}

impl EchoPropagator {
    pub fn new(policy: EchoPolicy, bloom_registry: Arc<std::sync::Mutex<BloomRegistry>>) -> Self {
        let max_queue = policy.max_queue_size;
        Self {
            policy,
            pending: VecDeque::with_capacity(max_queue),
            stats: EchoStats::default(),
            bloom_registry,
        }
    }

    /// Queue a delta for propagation.
    ///
    /// The delta is the XOR difference from the old state to the new state.
    /// The propagator will distribute this delta to dependent registers.
    pub fn queue_echo(
        &mut self,
        delta: BitslicedLane,
        source_address: RegisterAddress,
        target_layer: u16,
    ) -> Result<(), String> {
        if self.pending.len() >= self.policy.max_queue_size {
            self.stats.dropped += 1;
            return Err("Echo queue full".to_string());
        }

        self.pending.push_back(PendingEcho {
            delta,
            source_address,
            depth: 0,
            target_layer,
            timestamp: 0, // TODO: use actual timestamp
        });
        Ok(())
    }

    /// Process pending echoes.
    ///
    /// This propagates queued deltas to dependent registers, applying the echo
    /// minimization policy (minimal, bloom-gated, rotation-damped, XOR-cancel).
    ///
    /// `register_lookup`: a closure that given an address, returns the register at that address.
    /// `register_update`: a closure that updates a register with a delta.
    /// `get_dependents`: a closure that returns the addresses of registers that depend on a given address.
    pub fn propagate<F, G, H>(
        &mut self,
        mut register_lookup: F,
        mut register_update: G,
        mut get_dependents: H,
    ) -> EchoStats
    where
        F: FnMut(RegisterAddress) -> Option<Register>,
        G: FnMut(RegisterAddress, BitslicedLane),
        H: FnMut(RegisterAddress) -> Vec<RegisterAddress>,
    {
        let mut processed = VecDeque::new();

        while let Some(echo) = self.pending.pop_front() {
            // Check max depth
            if self.policy.max_depth > 0 && echo.depth >= self.policy.max_depth {
                continue;
            }

            // Get dependents of the source address
            let dependents = get_dependents(echo.source_address);

            for dep_addr in dependents {
                // Bloom gating: check if the dependent's sector might contain the source
                if self.policy.bloom_gated {
                    let dep_sector = SectorKey::new(
                        dep_addr.spine(),
                        dep_addr.layer(),
                        dep_addr.ring(),
                        dep_addr.sector(),
                    );
                    let source_bits = echo.source_address.into();
                    if let Ok(bloom_registry) = self.bloom_registry.lock() {
                        if !bloom_registry.sector_might_contain(dep_sector, source_bits) {
                            self.stats.bloom_gated += 1;
                            continue; // skip this dependent — bloom says it's not here
                        }
                    }
                }

                // Look up the dependent register
                if let Some(mut dep_reg) = register_lookup(dep_addr) {
                    // XOR cancellation: if delta ^ dep_reg's current delta == 0, cancel
                    if self.policy.xor_cancel {
                        let dep_delta = dep_reg.payload.clone();
                        let combined = BitslicedLane::xor(&echo.delta, &dep_delta);
                        if combined.none_set() {
                            self.stats.canceled += 1;
                            continue; // redundant echo, cancel
                        }
                    }

                    // Rotation damping: if target layer is cold, queue for later
                    if self.policy.rotation_damped && echo.target_layer > 12 {
                        self.stats.rotation_damped += 1;
                        // Re-queue with incremented depth (will be processed on next rotation)
                        if self.pending.len() < self.policy.max_queue_size {
                            processed.push_back(PendingEcho {
                                delta: echo.delta.clone(),
                                source_address: echo.source_address,
                                depth: echo.depth + 1,
                                target_layer: echo.target_layer,
                                timestamp: echo.timestamp,
                            });
                        }
                        continue;
                    }

                    // Propagate the delta to the dependent
                    register_update(dep_addr, echo.delta.clone());

                    self.stats.propagated += 1;
                    self.stats.bytes_propagated += echo.delta.len() as u64;
                }
            }
        }

        // Re-queue any processed echoes (for rotation-damped ones)
        self.pending = processed;

        self.stats.clone()
    }

    /// Get current stats.
    pub fn stats(&self) -> EchoStats {
        self.stats.clone()
    }

    /// Reset stats.
    pub fn reset_stats(&mut self) {
        self.stats = EchoStats::default();
    }

    /// Get the number of pending echoes.
    pub fn pending_count(&self) -> usize {
        self.pending.len()
    }

    /// Clear all pending echoes.
    pub fn clear(&mut self) {
        self.pending.clear();
    }
}

/// A structured echo injection — a deliberate, controlled propagation.
///
/// Used for:
/// - Cache coherence: ensure all nodes agree on a register's state.
/// - Routing refresh: propagate updated routing metadata.
/// - Synchronization pulse: coordinate a rotation cycle.
#[derive(Clone, Debug)]
pub struct StructuredEcho {
    /// Source address for the echo.
    pub source_address: RegisterAddress,
    /// The delta to propagate.
    pub delta: BitslicedLane,
    /// Target layer (how far the echo should propagate).
    pub target_layer: u16,
    /// Maximum propagation depth.
    pub max_depth: u8,
    /// Purpose of the echo (for logging / debugging).
    pub purpose: EchoPurpose,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum EchoPurpose {
    CacheCoherence,
    RoutingRefresh,
    SynchronizationPulse,
    AdaptationPropagate,
    WeightUpdate,
    GradientStep,
    KvCacheEviction,
    RecordUpdate,
    Custom(String),
}

impl EchoPropagator {
    /// Inject a structured echo.
    pub fn inject_structured(&mut self, echo: StructuredEcho) -> Result<(), String> {
        self.queue_echo(
            echo.delta,
            echo.source_address,
            echo.target_layer,
        )?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_echo_policy_default() {
        let policy = EchoPolicy::default();
        assert!(policy.minimal);
        assert!(policy.bloom_gated);
        assert!(policy.rotation_damped);
        assert!(policy.xor_cancel);
        assert_eq!(policy.max_depth, 0); // unlimited
    }

    #[test]
    fn test_echo_queue() {
        let policy = EchoPolicy::default();
        let bloom_registry = Arc::new(std::sync::Mutex::new(BloomRegistry::new()));
        let mut propagator = EchoPropagator::new(policy, bloom_registry);

        let delta = BitslicedLane::from_bits(&[true, false, true, false]);
        let addr = RegisterAddress::new(0, 1, 2, 3, 4, 0);

        assert!(propagator.queue_echo(delta.clone(), addr, 1).is_ok());
        assert_eq!(propagator.pending_count(), 1);

        // Queue too many
        for i in 0..1024 {
            let d = BitslicedLane::from_bits(&[true]);
            let a = RegisterAddress::new(0, 1, 2, 3, (i % 10) as u16, 0);
            let _ = propagator.queue_echo(d, a, 1);
        }
        assert!(propagator.pending_count() >= 1024);
    }

    #[test]
    fn test_echo_propagation_basic() {
        use std::cell::RefCell;
        use std::collections::HashMap;

        let policy = EchoPolicy::default();
        let bloom_registry = Arc::new(std::sync::Mutex::new(BloomRegistry::new()));
        let mut propagator = EchoPropagator::new(policy, bloom_registry);

        // Set up a simple register "database"
        let registers = RefCell::new(HashMap::<RegisterAddress, Register>::new());

        let addr_a = RegisterAddress::new(0, 1, 2, 3, 0, 0);
        let addr_b = RegisterAddress::new(0, 1, 2, 3, 1, 0); // dependent of A

        let payload_a = BitslicedLane::from_bits(&[true, false, true, false]);
        let payload_b = BitslicedLane::from_bits(&[false, true, false, true]);

        registers.borrow_mut().insert(addr_a, Register::new(addr_a, payload_a));
        registers.borrow_mut().insert(addr_b, Register::new(addr_b, payload_b));

        // Queue an echo from A
        let delta = BitslicedLane::from_bits(&[true, false, false, false]);
        propagator.queue_echo(delta, addr_a, 1).unwrap();

        // Propagate
        let stats = propagator.propagate(
            |addr| registers.borrow().get(&addr).cloned(),
            |addr, delta| {
                if let Some(mut reg) = registers.borrow_mut().get_mut(&addr) {
                    reg.update_with_delta(&delta);
                }
            },
            |addr| {
                // B depends on A
                if addr == addr_a {
                    vec![addr_b]
                } else {
                    vec![]
                }
            },
        );

        assert_eq!(stats.propagated, 1);
        assert!(stats.canceled == 0);

        // Check that B was updated
        let reg_b = registers.borrow().get(&addr_b).cloned().unwrap();
        // B's original payload was [false, true, false, true]
        // Delta was [true, false, false, false]
        // After XOR: [true, true, false, true]
        assert_eq!(reg_b.payload.get_bit(0), true);
        assert_eq!(reg_b.payload.get_bit(1), true);
        assert_eq!(reg_b.payload.get_bit(2), false);
        assert_eq!(reg_b.payload.get_bit(3), true);
    }
}
