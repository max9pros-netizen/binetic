//! Register — the fundamental unit of the fabric.
//!
//! Each register carries:
//! - Payload (bitsliced lane)
//! - Self-describing metadata (address, version, lineage, bloom anchor)
//! - Echo counter (propagation depth tracking)
//! - Checksum anchor (for self-validation)

use crate::address::RegisterAddress;
use crate::bitslice::BitslicedLane;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::fmt;
use std::hash::{Hash, Hasher};

/// Schema version for self-description.
///
/// Version 0: initial format.
/// Future versions may add fields or change encoding.
pub const SCHEMA_VERSION: u8 = 0;

/// A register in the IPv6 data plane.
///
/// This is the core data structure of the fabric. Every piece of data —
/// weights, KV cache entries, activations, control metadata — lives in a register.
#[derive(Clone, Debug)]
pub struct Register {
    /// Self-describing position in the 4D sphere.
    pub address: RegisterAddress,
    /// The payload, stored in bitsliced format for SIMD efficiency.
    pub payload: BitslicedLane,
    /// XOR lineage fingerprint — chain of custody back to ancestor.
    ///
    /// `xor_lineage = parent.xor_lineage ^ hash(payload) ^ constant(address)`
    ///
    /// Used for:
    /// - Self-validation (detect corruption)
    /// - Lineage tracing (where did this register come from?)
    /// - Deterministic replay (reconstruct state from deltas)
    pub xor_lineage: u128,
    /// How many propagation steps this register has seen.
    ///
    /// Used for:
    /// - Echo damping (registers that have seen many echoes may be throttled)
    /// - Debugging (trace propagation paths)
    pub echo_counter: u8,
    /// Hash anchor for bloom filter attachment.
    ///
    /// Sectors attach bloom filters keyed by this anchor.
    /// Used for routing and echo gating.
    pub bloom_anchor: u64,
    /// Schema version — what operation set this register supports.
    pub version: u8,
    /// Phase within the rotation cycle (derived from ring, cached for convenience).
    pub phase: f64,
    /// Checksum anchor for self-validation.
    ///
    /// `checksum_anchor = hmac(payload, key)`
    /// Used with integrity policy to detect corruption on read.
    pub checksum_anchor: u64,
}

impl Register {
    /// Create a new register with the given address and payload.
    pub fn new(address: RegisterAddress, payload: BitslicedLane) -> Self {
        let bloom_anchor = Self::compute_bloom_anchor(&address, &payload);
        let checksum_anchor = Self::compute_checksum(&payload);
        let xor_lineage = Self::compute_initial_lineage(&address, &payload);

        Self {
            address,
            payload,
            xor_lineage,
            echo_counter: 0,
            bloom_anchor,
            version: SCHEMA_VERSION,
            phase: address.phase(),
            checksum_anchor,
        }
    }

    /// Create a register from raw bytes (for deserialization / FFI).
    pub fn from_bytes(address: RegisterAddress, data: &[u8]) -> Self {
        let payload = BitslicedLane::from_bytes(data);
        Self::new(address, payload)
    }

    /// Extract raw bytes from the register (for serialization / FFI).
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut result = Vec::with_capacity(256);
        result.extend_from_slice(&self.address.bits().to_le_bytes());
        result.extend_from_slice(self.payload.as_bytes());
        result.extend_from_slice(&self.xor_lineage.to_le_bytes());
        result.push(self.echo_counter);
        result.extend_from_slice(&self.bloom_anchor.to_le_bytes());
        result.push(self.version);
        result.extend_from_slice(&self.phase.to_le_bytes());
        result.extend_from_slice(&self.checksum_anchor.to_le_bytes());
        result
    }

    /// Compute the bloom anchor from address + payload.
    fn compute_bloom_anchor(address: &RegisterAddress, payload: &BitslicedLane) -> u64 {
        let mut hasher = std::collections::hash_map::DefaultHasher::new();
        address.hash(&mut hasher);
        payload.hash(&mut hasher);
        hasher.finish()
    }

    /// Compute the initial XOR lineage for a new register.
    fn compute_initial_lineage(address: &RegisterAddress, payload: &BitslicedLane) -> u128 {
        // lineage = hash(address) ^ hash(payload) ^ constant
        let addr_hash = hash_u128(address.bits());
        let payload_hash = hash_payload(payload);
        addr_hash ^ payload_hash ^ 0xDEADBEEF_CAFE_BABE_u128
    }

    /// Compute checksum from payload.
    fn compute_checksum(payload: &BitslicedLane) -> u64 {
        use sha2::Digest;
        let mut hasher = Sha256::new();
        hasher.update(payload.as_bytes());
        let result = hasher.finalize();
        u64::from_le_bytes(result[..8].try_into().unwrap())
    }

    /// Update the register's payload and minimally update lineage.
    ///
    /// This is the core of echo minimization: instead of replacing the entire
    /// register state, we XOR-merge the delta and update lineage minimally.
    pub fn update_with_delta(&mut self, delta: &BitslicedLane) {
        self.payload.xor_merge(delta);
        self.xor_lineage ^= hash_payload(delta) ^ 0xDEADBEEF_CAFE_BABE_u128;
        self.echo_counter = self.echo_counter.saturating_add(1);
        self.checksum_anchor = Self::compute_checksum(&self.payload);
        self.phase = self.address.phase(); // refresh phase from address
    }

    /// XOR-merge another register's payload into this one (fold operation).
    pub fn fold_with(&mut self, other: &Register) {
        self.update_with_delta(&other.payload);
        // Also merge lineage
        self.xor_lineage ^= other.xor_lineage;
    }

    /// Check if this register's lineage is consistent.
    ///
    /// Recomputes expected lineage from address and payload, compares.
    /// Returns true if consistent, false if corruption detected.
    pub fn lineage_consistent(&self) -> bool {
        let expected = Self::compute_initial_lineage(&self.address, &self.payload);
        self.xor_lineage == expected
    }

    /// Check if this register's checksum is valid.
    pub fn checksum_valid(&self) -> bool {
        self.checksum_anchor == Self::compute_checksum(&self.payload)
    }

    /// Get the effective payload as a byte slice.
    pub fn payload_bytes(&self) -> &[u8] {
        self.payload.as_bytes()
    }

    /// Check if this register is "hot" (recently accessed, low layer).
    pub fn is_hot(&self) -> bool {
        self.address.layer() < 4
    }

    /// Check if this register is "cold" (rarely accessed, high layer).
    pub fn is_cold(&self) -> bool {
        self.address.layer() > 12
    }

    /// Increment echo counter (called when this register receives a propagation).
    pub fn record_echo(&mut self) {
        self.echo_counter = self.echo_counter.saturating_add(1);
    }

    /// Reset echo counter (called after a rotation cycle or reorganization).
    pub fn reset_echo(&mut self) {
        self.echo_counter = 0;
    }
}

impl fmt::Display for Register {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "Register[{}] payload={}bytes lineage=0x{:032X} echo={} bloom=0x{:016X} v{}",
            self.address,
            self.payload.len(),
            self.xor_lineage,
            self.echo_counter,
            self.bloom_anchor,
            self.version
        )
    }
}

/// Hash a u128 value (for lineage computation).
fn hash_u128(value: u128) -> u128 {
    use std::hash::{Hash, Hasher};
    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    value.hash(&mut hasher);
    let result = hasher.finish();
    result as u128 | ((result as u128) << 64)
}

/// Hash a bitsliced payload (for lineage computation).
fn hash_payload(payload: &BitslicedLane) -> u128 {
    use std::hash::{Hash, Hasher};
    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    payload.as_bytes().hash(&mut hasher);
    let result = hasher.finish();
    result as u128 | ((result as u128) << 64)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bitslice::BitslicedLane;

    #[test]
    fn test_address_construction() {
        let addr = RegisterAddress::new(0x1234, 0x0001, 0x0002, 0x0003, 0x0004, 0x0005);
        assert_eq!(addr.spine(), 0x1234);
        assert_eq!(addr.layer(), 0x0001);
        assert_eq!(addr.ring(), 0x0002);
        assert_eq!(addr.sector(), 0x0003);
        assert_eq!(addr.shard(), 0x0004);
        assert_eq!(addr.temporal(), 0x0005);
    }

    #[test]
    fn test_address_distance() {
        let a = RegisterAddress::new(0, 0, 0, 0, 0, 0);
        let b = RegisterAddress::new(1, 0, 0, 0, 0, 0);
        assert_eq!(a.distance(&b), 1); // only spine differs

        let c = RegisterAddress::new(0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF);
        assert_eq!(a.distance(&c), 96); // all bits differ
    }

    #[test]
    fn test_address_parent_child() {
        let addr = RegisterAddress::new(0, 1, 2, 3, 5, 0);
        let parent = addr.parent();
        assert_eq!(parent.shard(), 4);
        assert_eq!(parent.sector(), 3);

        let child = addr.child(3);
        assert_eq!(child.shard(), 8);
    }

    #[test]
    fn test_register_creation() {
        let addr = RegisterAddress::new(0, 1, 2, 3, 4, 0);
        let payload = BitslicedLane::from_bytes(&[0xD3, 0x7A, 0xB2, 0xF1]);
        let reg = Register::new(addr, payload);

        assert_eq!(reg.address, addr);
        assert_eq!(reg.version, SCHEMA_VERSION);
        assert!(reg.lineage_consistent());
        assert!(reg.checksum_valid());
    }

    #[test]
    fn test_register_delta_update() {
        let addr = RegisterAddress::new(0, 1, 2, 3, 4, 0);
        let payload = BitslicedLane::from_bytes(&[0xD3, 0x7A, 0xB2, 0xF1]);
        let mut reg = Register::new(addr, payload);

        let old_lineage = reg.xor_lineage;
        let old_echo = reg.echo_counter;

        // Apply delta
        let delta = BitslicedLane::from_bytes(&[0x01, 0x00, 0x00, 0x00]);
        reg.update_with_delta(&delta);

        assert_eq!(reg.echo_counter, old_echo + 1);
        assert_ne!(reg.xor_lineage, old_lineage); // lineage changed
        assert!(reg.lineage_consistent()); // but still consistent
        assert!(reg.checksum_valid());
    }

    #[test]
    fn test_register_fold() {
        let addr_a = RegisterAddress::new(0, 1, 2, 3, 0, 0);
        let addr_b = RegisterAddress::new(0, 1, 2, 3, 1, 0);

        let payload_a = BitslicedLane::from_bytes(&[0xFF, 0xFF, 0xFF, 0xFF]);
        let payload_b = BitslicedLane::from_bytes(&[0x0F, 0x0F, 0x0F, 0x0F]);

        let mut reg_a = Register::new(addr_a, payload_a);
        let reg_b = Register::new(addr_b, payload_b);

        reg_a.fold_with(&reg_b);

        // After fold, reg_a's payload should be XOR of both
        let expected = BitslicedLane::from_bytes(&[0xF0, 0xF0, 0xF0, 0xF0]);
        assert_eq!(reg_a.payload.as_bytes(), expected.as_bytes());
    }
}

// ── Extended Precision Registers (Compound Registers) ──────────────────────
// Compound registers combine multiple base registers "the long way" to create
// wider registers with arbitrary precision (u256, u512, u1024, ...).
//
// See ARCHITECTURE.md section 22 for the full specification.

/// The method used to combine base registers into a compound register.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum CombiningMethod {
    /// XOR-chain: payload = base0 ^ base1 ^ ... ^ baseN
    /// Lineage = XOR of all base lineages.
    XorChain,
    /// Sequential: payload = base0 || base1 || ... || baseN (concatenation)
    Sequential,
    /// Interleaved: bit i comes from base(i % N) at position (i / N)
    Interleaved,
    /// Layered: base0 = high bits, base1 = low bits (significance-ordered)
    Layered,
}

impl CombiningMethod {
    /// Number of bases needed for a given precision (in bits).
    pub fn bases_for_precision(&self, precision_bits: u16) -> u8 {
        match self {
            CombiningMethod::XorChain => ((precision_bits + 63) / 64) as u8,
            CombiningMethod::Sequential => ((precision_bits + 63) / 64) as u8,
            CombiningMethod::Interleaved => ((precision_bits + 63) / 64) as u8,
            CombiningMethod::Layered => ((precision_bits + 63) / 64) as u8,
        }
    }

    /// Precision in bits for a given number of bases.
    pub fn precision_for_bases(&self, base_count: u8) -> u16 {
        (base_count as u16) * 64
    }
}

/// An extended precision register — a compound of multiple base registers.
///
/// This represents a u256, u512, u1024, or arbitrary-width value formed by
/// combining multiple 64-bit base registers.
#[derive(Clone, Debug)]
pub struct CompoundRegister {
    /// The combining method used.
    pub combining: CombiningMethod,
    /// Number of base registers.
    pub base_count: u8,
    /// Total precision in bits.
    pub precision_bits: u16,
    /// The primary base address (used for routing/lookup).
    pub primary_address: RegisterAddress,
    /// Addresses of all constituent bases.
    pub base_addresses: Vec<RegisterAddress>,
    /// The effective payload (computed from bases).
    pub payload: BitslicedLane,
    /// XOR lineage (for XOR-chain combining, this is the XOR of all base lineages).
    pub xor_lineage: u128,
    /// Schema version.
    pub version: u8,
}

impl CompoundRegister {
    /// Create a new compound register from a list of base registers.
    pub fn new(
        combining: CombiningMethod,
        base_registers: &[Register],
    ) -> Self {
        let base_count = base_registers.len() as u8;
        let precision_bits = combining.precision_for_bases(base_count);

        let primary_address = base_registers[0].address;
        let base_addresses: Vec<RegisterAddress> = base_registers.iter().map(|r| r.address).collect();

        // Compute the effective payload based on combining method
        let payload = match combining {
            CombiningMethod::XorChain => {
                let mut result = base_registers[0].payload.clone();
                for reg in &base_registers[1..] {
                    result.xor_merge(&reg.payload);
                }
                result
            }
            CombiningMethod::Sequential => {
                // Concatenate payloads
                let total_bits = precision_bits as usize;
                let mut result = BitslicedLane::with_capacity(total_bits);
                let mut bit_pos = 0;
                for reg in base_registers {
                    for i in 0..reg.payload.len() {
                        if result.get_bit(bit_pos) != reg.payload.get_bit(i) {
                            result.set_bit(bit_pos, reg.payload.get_bit(i));
                        }
                        bit_pos += 1;
                    }
                }
                result
            }
            CombiningMethod::Interleaved => {
                // Interleave: bit i from base(i % N)
                let total_bits = precision_bits as usize;
                let mut result = BitslicedLane::with_capacity(total_bits);
                for i in 0..total_bits {
                    let base_idx = i % base_count as usize;
                    let bit_in_base = i / base_count as usize;
                    if bit_in_base < base_registers[base_idx].payload.len() {
                        result.set_bit(i, base_registers[base_idx].payload.get_bit(bit_in_base));
                    }
                }
                result
            }
            CombiningMethod::Layered => {
                // Layered: base0 = high bits, baseN = low bits
                // For simplicity, we store as sequential (same as concatenation)
                // but the addressing semantics are significance-ordered
                let total_bits = precision_bits as usize;
                let mut result = BitslicedLane::with_capacity(total_bits);
                let mut bit_pos = 0;
                for reg in base_registers {
                    for i in 0..reg.payload.len() {
                        result.set_bit(bit_pos, reg.payload.get_bit(i));
                        bit_pos += 1;
                    }
                }
                result
            }
        };

        // Compute lineage (XOR-chain: XOR of all base lineages)
        let xor_lineage = match combining {
            CombiningMethod::XorChain => {
                base_registers.iter().fold(0u128, |acc, r| acc ^ r.xor_lineage)
            }
            _ => {
                // For non-XOR methods, lineage is computed from the payload
                let mut h = std::collections::hash_map::DefaultHasher::new();
                payload.hash(&mut h);
                h.finish() as u128 | ((h.finish() as u128) << 64)
            }
        };

        Self {
            combining,
            base_count,
            precision_bits,
            primary_address,
            base_addresses,
            payload,
            xor_lineage,
            version: crate::register::SCHEMA_VERSION,
        }
    }

    /// Create a compound register from base addresses (without loading the bases).
    /// Used for address-based routing without materializing the compound.
    pub fn from_addresses(
        combining: CombiningMethod,
        primary_address: RegisterAddress,
        base_addresses: Vec<RegisterAddress>,
    ) -> Self {
        let base_count = base_addresses.len() as u8;
        let precision_bits = combining.precision_for_bases(base_count);

        Self {
            combining,
            base_count,
            precision_bits,
            primary_address,
            base_addresses,
            payload: BitslicedLane::with_capacity(precision_bits as usize),
            xor_lineage: 0,
            version: crate::register::SCHEMA_VERSION,
        }
    }

    /// Get the primary address for routing/lookup.
    pub fn primary_address(&self) -> RegisterAddress {
        self.primary_address
    }

    /// Get all base addresses.
    pub fn base_addresses(&self) -> &[RegisterAddress] {
        &self.base_addresses
    }

    /// Check if this compound register is consistent (all bases exist and lineage is valid).
    pub fn is_consistent(&self, base_lookup: impl Fn(RegisterAddress) -> Option<Register>) -> bool {
        // Check all bases exist
        for addr in &self.base_addresses {
            if base_lookup(*addr).is_none() {
                return false;
            }
        }

        // For XOR-chain, verify lineage
        if self.combining == CombiningMethod::XorChain {
            let expected_lineage = self.base_addresses.iter()
                .filter_map(|addr| base_lookup(*addr))
                .fold(0u128, |acc, r| acc ^ r.xor_lineage);
            self.xor_lineage == expected_lineage
        } else {
            true
        }
    }

    /// Update a single base and recompute the compound.
    pub fn update_base(
        &mut self,
        base_index: usize,
        new_base: Register,
        combining: CombiningMethod,
    ) {
        if base_index >= self.base_addresses.len() {
            return;
        }

        self.base_addresses[base_index] = new_base.address;

        // Recompute payload
        let base_registers: Vec<Register> = std::iter::once(new_base.clone())
            .chain(self.base_addresses.iter().filter(|&a| a != &new_base.address)
                .filter_map(|a| None)) // Placeholder — real implementation would reload
            .collect();

        // For now, just update the payload directly (simplified)
        match combining {
            CombiningMethod::XorChain => {
                // XOR the delta into the compound payload
                let delta = BitslicedLane::xor(&self.payload, &new_base.payload);
                self.payload.xor_merge(&new_base.payload);
                self.xor_lineage ^= new_base.xor_lineage;
            }
            _ => {
                // For other methods, full recomputation is needed
                // This is a placeholder
            }
        }
    }
}

/// A database record — a compound register with self-describing fields.
///
/// This is the primary data structure for the database use case.
/// A record is a compound register where each base holds a field value.
#[derive(Clone, Debug)]
pub struct Record {
    /// The compound register holding the record data.
    pub compound: CompoundRegister,
    /// Field names and their base indices.
    pub schema: Vec<FieldSchema>,
    /// Record metadata (entity type, version, timestamps).
    pub metadata: RecordMetadata,
}

/// A field in a record schema.
#[derive(Clone, Debug)]
pub struct FieldSchema {
    /// Field name.
    pub name: String,
    /// Base index within the compound register.
    pub base_index: u8,
    /// Field type (derived from the combining method and base payload).
    pub field_type: FieldType,
}

/// Field types for record schema.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum FieldType {
    Integer(u16),      // integer of given bit width
    Float(u8),         // float of given bit width (32, 64)
    Text,              // text (hash or compressed)
    Boolean,           // single bit
    Blob,              // binary blob (sequential compound)
    Compound,          // nested compound register
}

/// Record metadata.
#[derive(Clone, Debug, Default)]
pub struct RecordMetadata {
    /// Entity type (encoded in spine).
    pub entity_type: u16,
    /// Schema version.
    pub schema_version: u8,
    /// Creation timestamp (unix epoch ms).
    pub created_at: u64,
    /// Last update timestamp.
    pub updated_at: u64,
    /// Record version (for optimistic concurrency).
    pub version: u64,
}

impl Record {
    /// Create a new record from field values.
    pub fn new(entity_type: u16, fields: Vec<(String, BitslicedLane)>) -> Self {
        let schema: Vec<FieldSchema> = fields.iter().enumerate().map(|(i, (name, _))| {
            FieldSchema {
                name: name.clone(),
                base_index: i as u8,
                field_type: FieldType::Blob, // default
            }
        }).collect();

        let base_registers: Vec<Register> = fields.iter().enumerate().map(|(i, (_, payload))| {
            let addr = RegisterAddress::new(
                entity_type,
                5,  // layer
                0,  // ring
                i as u16,  // shard = field index
                0,  // sector
                0,  // temporal
            );
            Register::new(addr, payload.clone())
        }).collect();

        let compound = CompoundRegister::new(CombiningMethod::Sequential, &base_registers);

        Self {
            compound,
            schema,
            metadata: RecordMetadata {
                entity_type,
                schema_version: 1,
                created_at: std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap()
                    .as_millis() as u64,
                updated_at: 0,
                version: 1,
            },
        }
    }

    /// Get a field value by name.
    pub fn get_field(&self, name: &str) -> Option<&BitslicedLane> {
        let field = self.schema.iter().find(|f| f.name == name)?;
        let base_idx = field.base_index as usize;
        if base_idx < self.compound.base_addresses.len() {
            // Return a slice of the compound payload for this field
            // (simplified — real implementation would extract the base payload)
            Some(&self.compound.payload)
        } else {
            None
        }
    }

    /// Update a field with a new value (returns a delta for echo propagation).
    pub fn update_field(&mut self, name: &str, new_value: BitslicedLane) -> Option<BitslicedLane> {
        let field = self.schema.iter().find(|f| f.name == name)?;
        let base_idx = field.base_index as usize;

        // Compute delta
        let old_value = self.compound.payload.clone(); // simplified
        let delta = BitslicedLane::xor(&old_value, &new_value);

        self.metadata.updated_at = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_millis() as u64;
        self.metadata.version += 1;

        Some(delta)
    }
}

#[cfg(test)]
mod compound_tests {
    use super::*;
    use crate::bitslice::BitslicedLane;

    #[test]
    fn test_xor_chain_compound() {
        let base0 = Register::new(
            RegisterAddress::new(0, 1, 0, 0, 0, 0),
            BitslicedLane::from_bits(&[true, false, true, false]),
        );
        let base1 = Register::new(
            RegisterAddress::new(0, 1, 0, 0, 1, 0),
            BitslicedLane::from_bits(&[false, true, false, true]),
        );

        let compound = CompoundRegister::new(CombiningMethod::XorChain, &[base0, base1]);

        assert_eq!(compound.base_count, 2);
        assert_eq!(compound.precision_bits, 128);
        assert_eq!(compound.combining, CombiningMethod::XorChain);

        // XOR of [true, false, true, false] and [false, true, false, true]
        // = [true, true, true, true]
        assert!(compound.payload.all_set());
    }

    #[test]
    fn test_sequential_compound() {
        let base0 = Register::new(
            RegisterAddress::new(0, 1, 0, 0, 0, 0),
            BitslicedLane::from_bits(&[true, false]),
        );
        let base1 = Register::new(
            RegisterAddress::new(0, 1, 0, 0, 1, 0),
            BitslicedLane::from_bits(&[false, true]),
        );

        let compound = CompoundRegister::new(CombiningMethod::Sequential, &[base0, base1]);

        assert_eq!(compound.base_count, 2);
        assert_eq!(compound.precision_bits, 128);
        assert_eq!(compound.combining, CombiningMethod::Sequential);
    }

    #[test]
    fn test_compound_consistency() {
        let base0 = Register::new(
            RegisterAddress::new(0, 1, 0, 0, 0, 0),
            BitslicedLane::from_bits(&[true]),
        );
        let base1 = Register::new(
            RegisterAddress::new(0, 1, 0, 0, 1, 0),
            BitslicedLane::from_bits(&[false]),
        );

        let compound = CompoundRegister::new(CombiningMethod::XorChain, &[base0, base1]);

        let lookup = |addr: RegisterAddress| -> Option<Register> {
            if addr == base0.address {
                Some(base0.clone())
            } else if addr == base1.address {
                Some(base1.clone())
            } else {
                None
            }
        };

        assert!(compound.is_consistent(lookup));
    }

    #[test]
    fn test_record_creation() {
        let fields = vec![
            ("id".to_string(), BitslicedLane::from_bytes(&[42])),
            ("name".to_string(), BitslicedLane::from_bytes(&["Alice".as_bytes()])),
            ("age".to_string(), BitslicedLane::from_bytes(&[30])),
        ];

        let record = Record::new(0x1000, fields);

        assert_eq!(record.metadata.entity_type, 0x1000);
        assert_eq!(record.schema.len(), 3);
        assert_eq!(record.compound.base_count, 3);
    }

    #[test]
    fn test_record_update() {
        let fields = vec![
            ("age".to_string(), BitslicedLane::from_bytes(&[30])),
        ];

        let mut record = Record::new(0x1000, fields);

        let new_age = BitslicedLane::from_bytes(&[31]);
        let delta = record.update_field("age", new_age);

        assert!(delta.is_some());
        assert_eq!(record.metadata.version, 2);
    }
}
