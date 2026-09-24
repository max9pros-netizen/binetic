//! Register addressing and geometry
//!
//! The address is the central abstraction: it encodes position, lineage, angle,
//! layer, sector, phase, and more. It is the lookup key, the routing hint, and
//! the lineage path — all in one.

use serde::{Deserialize, Serialize};
use std::fmt;
use std::hash::{Hash, Hasher};

/// A 128-bit register address encoding position in the 4D sphere.
///
/// Bits layout (MSB → LSB):
/// ```
/// ┌─────────────────────────────────────────────────────────────────────────┐
/// │                         RegisterAddress (128 bits)                      │
/// ├──────────┬──────────┬──────────┬──────────┬──────────┬──────────────────┤
/// │  Spine   │  Layer   │  Ring    │ Sector   │  Shard   │   Temporal       │
/// │  (16)    │  (16)    │  (16)    │  (16)    │  (16)    │   (16)           │
/// │          │          │          │          │          │                  │
/// │  Which   │  Depth   │  Rotation│  Angular │  Sub-    │  T-dimension:   │
/// │  fiber   │  from    │  position│  sector  │  division│  active compute │
/// │  ofCube  │  center  │  within  │          │  within  │  scratchpad     │
/// │          │  (0=hot, │  layer   │          │  sector  │  index          │
/// │          │   N=cold)│          │          │          │                  │
/// └──────────┴──────────┴──────────┴──────────┴──────────┴──────────────────┴──────────┘
/// ```
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct RegisterAddress {
    bits: u128,
}

// Bit field offsets (from LSB)
const SPINE_OFFSET: u32 = 0;
const LAYER_OFFSET: u32 = 16;
const RING_OFFSET: u32 = 32;
const SECTOR_OFFSET: u32 = 48;
const SHARD_OFFSET: u32 = 64;
const TEMPORAL_OFFSET: u32 = 80;
const FIELD_WIDTH: u32 = 16;

const MASK: u128 = 0xFFFF;

impl RegisterAddress {
    /// Create a new address from component fields.
    pub fn new(
        spine: u16,
        layer: u16,
        ring: u16,
        sector: u16,
        shard: u16,
        temporal: u16,
    ) -> Self {
        let bits = (spine as u128)
            | ((layer as u128) << LAYER_OFFSET)
            | ((ring as u128) << RING_OFFSET)
            | ((sector as u128) << SECTOR_OFFSET)
            | ((shard as u128) << SHARD_OFFSET)
            | ((temporal as u128) << TEMPORAL_OFFSET);
        Self { bits }
    }

    /// Zero address (invalid / null).
    pub fn zero() -> Self {
        Self { bits: 0 }
    }

    /// Decompose into component fields.
    pub fn spine(&self) -> u16 {
        ((self.bits >> SPINE_OFFSET) & MASK) as u16
    }

    pub fn layer(&self) -> u16 {
        ((self.bits >> LAYER_OFFSET) & MASK) as u16
    }

    pub fn ring(&self) -> u16 {
        ((self.bits >> RING_OFFSET) & MASK) as u16
    }

    pub fn sector(&self) -> u16 {
        ((self.bits >> SECTOR_OFFSET) & MASK) as u16
    }

    pub fn shard(&self) -> u16 {
        ((self.bits >> SHARD_OFFSET) & MASK) as u16
    }

    pub fn temporal(&self) -> u16 {
        ((self.bits >> TEMPORAL_OFFSET) & MASK) as u16
    }

    /// Raw u128 bit representation of the address.
    pub fn bits(&self) -> u128 {
        self.bits
    }

    /// Set individual fields (returns new address).
    pub fn with_spine(self, spine: u16) -> Self {
        let mut bits = self.bits;
        bits &= !(MASK << SPINE_OFFSET);
        bits |= (spine as u128) << SPINE_OFFSET;
        Self { bits }
    }

    pub fn with_layer(self, layer: u16) -> Self {
        let mut bits = self.bits;
        bits &= !(MASK << LAYER_OFFSET);
        bits |= (layer as u128) << LAYER_OFFSET;
        Self { bits }
    }

    pub fn with_ring(self, ring: u16) -> Self {
        let mut bits = self.bits;
        bits &= !(MASK << RING_OFFSET);
        bits |= (ring as u128) << RING_OFFSET;
        Self { bits }
    }

    pub fn with_sector(self, sector: u16) -> Self {
        let mut bits = self.bits;
        bits &= !(MASK << SECTOR_OFFSET);
        bits |= (sector as u128) << SECTOR_OFFSET;
        Self { bits }
    }

    pub fn with_shard(self, shard: u16) -> Self {
        let mut bits = self.bits;
        bits &= !(MASK << SHARD_OFFSET);
        bits |= (shard as u128) << SHARD_OFFSET;
        Self { bits }
    }

    pub fn with_temporal(self, temporal: u16) -> Self {
        let mut bits = self.bits;
        bits &= !(MASK << TEMPORAL_OFFSET);
        bits |= (temporal as u128) << TEMPORAL_OFFSET;
        Self { bits }
    }

    /// Structural distance between two addresses (XOR popcount).
    ///
    /// This is not Euclidean distance — it's a measure of how structurally
    /// related two addresses are. Addresses that are XOR-close are in related
    /// sectors/layers and route cheaply.
    pub fn distance(&self, other: &Self) -> u32 {
        (self.bits ^ other.bits).count_ones()
    }

    /// Check if two addresses are in the same sector (same spine, layer, ring, sector).
    pub fn same_sector(&self, other: &Self) -> bool {
        self.spine() == other.spine()
            && self.layer() == other.layer()
            && self.ring() == other.ring()
            && self.sector() == other.sector()
    }

    /// Check if two addresses are in the same layer.
    pub fn same_layer(&self, other: &Self) -> bool {
        self.layer() == other.layer()
    }

    /// Get the parent address (ancestor in the lineage chain).
    ///
    /// The parent is at the same spine/layer/ring/sector but one shard lower.
    /// If shard is 0, parent is in the previous sector (wrapping).
    pub fn parent(&self) -> Self {
        if self.shard() > 0 {
            self.with_shard(self.shard() - 1)
        } else if self.sector() > 0 {
            self.with_shard(0xFFFF).with_sector(self.sector() - 1)
        } else {
            // Root — no parent
            Self::zero()
        }
    }

    /// Get a child address (descendant in the lineage chain).
    pub fn child(&self, offset: u16) -> Self {
        let new_shard = self.shard().saturating_add(offset);
        if new_shard <= 0xFFFF {
            self.with_shard(new_shard)
        } else {
            self.with_shard(new_shard & 0xFFFF).with_sector(self.sector() + 1)
        }
    }

    /// Get a neighboring address in a given direction.
    ///
    /// `direction`: 0 = forward (spine+1), 1 = backward (spine-1), 2 = deeper (layer+1),
    /// 3 = shallower (layer-1), 4 = next ring, 5 = prev ring, 6 = next sector, 7 = prev sector.
    pub fn neighbor(&self, direction: u8) -> Self {
        match direction % 8 {
            0 => self.with_spine(self.spine().wrapping_add(1)),
            1 => self.with_spine(self.spine().wrapping_sub(1)),
            2 => self.with_layer(self.layer().wrapping_add(1)),
            3 => self.with_layer(self.layer().wrapping_sub(1)),
            4 => self.with_ring(self.ring().wrapping_add(1)),
            5 => self.with_ring(self.ring().wrapping_sub(1)),
            6 => self.with_sector(self.sector().wrapping_add(1)),
            7 => self.with_sector(self.sector().wrapping_sub(1)),
            _ => self.clone(),
        }
    }

    /// Derive angular position within the rotation from sector + ring.
    pub fn angle(&self) -> f64 {
        let sectors = self.sector() as f64;
        let rings = self.ring() as f64;
        let total_sectors = 0xFFFF as f64;
        let total_rings = 0xFFFF as f64;
        // Angle = sector fraction + ring modulation
        (sectors / total_sectors) * std::f64::consts::TAU + (rings / total_rings) * 0.1
    }

    /// Derive the phase within the rotation cycle.
    pub fn phase(&self) -> f64 {
        self.ring() as f64 / 0xFFFF as f64
    }

    /// Check if this is a temporal register (T dimension active).
    pub fn is_temporal(&self) -> bool {
        self.temporal() != 0
    }

    /// Check if this is a valid (non-zero) address.
    pub fn is_valid(&self) -> bool {
        self.bits != 0
    }
}

impl fmt::Debug for RegisterAddress {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("RegisterAddress")
            .field("spine", &self.spine())
            .field("layer", &self.layer())
            .field("ring", &self.ring())
            .field("sector", &self.sector())
            .field("shard", &self.shard())
            .field("temporal", &self.temporal())
            .field("is_temporal", &self.is_temporal())
            .field("angle", &format!("{:.4}", self.angle()))
            .field("phase", &format!("{:.4}", self.phase()))
            .field("hex", &format!("0x{:032X}", self.bits))
            .finish()
    }
}

impl fmt::Display for RegisterAddress {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "Addr(spine={:04x}, L{:04x}, R{:04x}, S{:04x}, Sh{:04x}, T{:04x})",
            self.spine(),
            self.layer(),
            self.ring(),
            self.sector(),
            self.shard(),
            self.temporal()
        )
    }
}

impl Hash for RegisterAddress {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.bits.hash(state);
    }
}

impl From<u128> for RegisterAddress {
    fn from(bits: u128) -> Self {
        Self { bits }
    }
}

impl From<RegisterAddress> for u128 {
    fn from(addr: RegisterAddress) -> u128 {
        addr.bits
    }
}

// ── Address ranges ──────────────────────────────────────────────────────────

/// Common address ranges for model components.
pub mod ranges {
    use super::RegisterAddress;

    /// Address range for model weights in a given layer.
    pub fn weight_address(layer: u16, head: u16, parameter: u16) -> RegisterAddress {
        RegisterAddress::new(0xD000, layer, 0, head, parameter, 0)
    }

    /// Address range for KV cache entries (token position → address).
    pub fn kv_address(token_pos: u32, layer: u16, head: u16) -> RegisterAddress {
        // Use spine to encode token position (avoid collision with weights)
        let spine = 0xE000u16 + (token_pos as u16 & 0xFFFF);
        RegisterAddress::new(spine, layer, 0, head, 0, 0)
    }

    /// Address for an activation in the temporal register ring.
    pub fn activation_address(slot: u16) -> RegisterAddress {
        RegisterAddress::new(0xF000, 0, 0, 0, 0, slot)
    }

    /// Address for the control plane (IPv4 metaphor).
    pub fn control_address(routing_key: u64) -> RegisterAddress {
        RegisterAddress::new(0xC000, 0, 0, (routing_key >> 32) as u16, (routing_key & 0xFFFF) as u16, 0)
    }

    /// Address range for cross-model deltas.
    pub fn delta_address(base_model: u16, variant: u16, layer: u16) -> RegisterAddress {
        RegisterAddress::new(0xA000 + base_model, layer, variant, 0, 0, 0)
    }
}
