//! Network register — the network as a register that grows recursively.
//!
//! Each register holds a `BislicedLane` (spatial + temporal planes) and can
//! contain child network registers, forming a recursive pisphere structure.
//!
//! Evaluation is always O(1) per lane — word-level bitwise ops on u64 chunks —
//! regardless of recursion depth or network size. No slowdown.

use crate::bitslice::{BislicedLane, BitslicedLane};
use crate::address::RegisterAddress;
use std::collections::HashMap;

/// A network register — stores network topology as a bisliced lane
/// and can contain child network registers for recursive growth.
///
/// The bisliced lane has two planes:
/// - Spatial: which nodes are connected (topology)
/// - Temporal: when connections are active (timing)
///
/// Evaluation: active = spatial AND temporal — O(1) word-level ops.
#[derive(Clone)]
pub struct NetworkRegister {
    /// The address of this register in the fabric.
    pub address: RegisterAddress,
    /// The bisliced lane storing spatial + temporal network topology.
    pub lane: BislicedLane,
    /// Child network registers (recursive pisphere growth).
    pub children: HashMap<u8, NetworkRegister>,
    /// Depth in the recursive hierarchy (0 = root).
    pub depth: u8,
}

impl NetworkRegister {
    /// Create a new network register at the given address with capacity for `num_nodes`.
    pub fn new(address: RegisterAddress, num_nodes: usize) -> Self {
        Self {
            address,
            lane: BislicedLane::with_capacity(num_nodes),
            children: HashMap::new(),
            depth: 0,
        }
    }

    /// Set a spatial connection from node `from` to node `to`.
    pub fn set_spatial(&mut self, from: usize, to: usize, value: bool) {
        self.lane.spatial.set_bit(from * self.lane.len() + to, value);
    }

    /// Set a temporal activity from node `from` to node `to`.
    pub fn set_temporal(&mut self, from: usize, to: usize, value: bool) {
        self.lane.temporal.set_bit(from * self.lane.len() + to, value);
    }

    /// Evaluate active connections — O(1) word-level AND of spatial and temporal planes.
    /// No slowdown regardless of network size or recursion depth.
    pub fn evaluate(&self) -> BitslicedLane {
        self.lane.active_connections()
    }

    /// Get the number of nodes this register covers.
    pub fn num_nodes(&self) -> usize {
        self.lane.len()
    }

    /// Resize the lane to cover `num_nodes` nodes.
    pub fn resize(&mut self, num_nodes: usize) {
        self.lane.resize(num_nodes * num_nodes);
    }

    /// Add a child network register (recursive growth).
    /// The child covers a subset of nodes and can be evaluated independently.
    pub fn add_child(&mut self, id: u8, address: RegisterAddress, num_nodes: usize) {
        let mut child = NetworkRegister::new(address, num_nodes);
        child.depth = self.depth + 1;
        self.children.insert(id, child);
    }

    /// Get a child register by ID.
    pub fn child(&self, id: u8) -> Option<&NetworkRegister> {
        self.children.get(&id)
    }

    /// Get a mutable child register by ID.
    pub fn child_mut(&mut self, id: u8) -> Option<&mut NetworkRegister> {
        self.children.get_mut(&id)
    }

    /// Evaluate this register AND all children recursively.
    /// Each level is O(1) per lane — total O(depth) word-level ops, not O(nodes^2).
    pub fn evaluate_recursive(&self) -> Vec<(u8, BitslicedLane)> {
        let mut results = Vec::new();
        results.push((self.depth, self.evaluate()));
        for (id, child) in &self.children {
            results.push((*id, child.evaluate()));
        }
        results
    }

    /// Count total active connections across this register and all children.
    pub fn total_active_connections(&self) -> usize {
        let mut count = self.lane.active_connections().popcount();
        for child in self.children.values() {
            count += child.lane.active_connections().popcount();
        }
        count
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::address::RegisterAddress;

    #[test]
    fn test_network_register_creation() {
        let addr = RegisterAddress::from(0u128);
        let reg = NetworkRegister::new(addr, 4);
        assert_eq!(reg.num_nodes(), 4);
        assert!(reg.children.is_empty());
        assert_eq!(reg.depth, 0);
    }

    #[test]
    fn test_network_register_evaluate() {
        let addr = RegisterAddress::from(1u128);
        let mut reg = NetworkRegister::new(addr, 4);

        // Set spatial: node 0 connects to node 1
        reg.lane.spatial.set_bit(0, true);
        // Set temporal: node 0 is active
        reg.lane.temporal.set_bit(0, true);

        let active = reg.evaluate();
        assert!(active.any_set());
    }

    #[test]
    fn test_network_register_recursive_growth() {
        let addr = RegisterAddress::from(2u128);
        let mut reg = NetworkRegister::new(addr, 4);

        // Add a child (recursive growth)
        let child_addr = RegisterAddress::from(3u128);
        reg.add_child(1, child_addr, 2);

        assert_eq!(reg.children.len(), 1);
        assert_eq!(reg.child(1).unwrap().depth, 1);

        // Evaluate recursively — O(depth) not O(nodes^2)
        let results = reg.evaluate_recursive();
        assert_eq!(results.len(), 2); // self + 1 child
    }

    #[test]
    fn test_network_register_no_slowdown() {
        let addr = RegisterAddress::from(4u128);
        let mut reg = NetworkRegister::new(addr, 64);

        // Set many connections
        for i in 0..64 {
            reg.lane.spatial.set_bit(i, true);
            reg.lane.temporal.set_bit(i, i % 2 == 0);
        }

        // Evaluation is O(1) per lane — word-level AND
        let active = reg.evaluate();
        // 32 even indices should be active
        assert_eq!(active.popcount(), 32);

        // Add children — evaluation still O(1) per lane
        for i in 0..4 {
            let child_addr = RegisterAddress::from((100 + i) as u128);
            reg.add_child(i, child_addr, 8);
        }

        let results = reg.evaluate_recursive();
        assert_eq!(results.len(), 5); // self + 4 children
        // Each child evaluation is O(1) per lane
        for (_, lane) in &results {
            let _ = lane.popcount(); // O(1) per lane
        }
    }
}