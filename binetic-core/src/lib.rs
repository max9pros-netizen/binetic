//! Binetic Core — the self-describing, spatially-organized computation fabric
//!
//! # Architecture Overview
//!
//! Binetic is a metadata-rich, unified memory-and-compute fabric where:
//! - Every register's address encodes its position, lineage, angle, layer, phase.
//! - Data (IPv6 plane) and control (IPv4 plane) are separate, interlaced structures.
//! - XOR is the native operation: for lineage, fold, diff, routing distance, echo cancellation.
//! - Echo minimization: changes propagate as deltas, damped by rotation, gated by bloom filters.
//! - Compute routes to data, not data to compute — no unnecessary copies.
//! - The fabric self-optimizes: observes access patterns, reorganizes for locality.
//! - Energy is a first-class scheduling concern.

pub mod address;
pub mod network;
pub mod register;
pub mod tiers;
pub mod bloom;
pub mod bitslice;
pub mod rotation;
pub mod echo;
pub mod fabric;
pub mod backend;

pub use address::RegisterAddress;
pub use network::NetworkRegister;
pub use register::Register;
pub use tiers::{Tier, TierId, RamTier, MmapTier};
pub use bloom::SectorBloom;
pub use bitslice::{BislicedArray, BislicedLane};
pub use rotation::RotationScheduler;
pub use echo::{EchoPolicy, EchoPropagator};
pub use fabric::Fabric;
pub use backend::{Backend, BackendId};

// Extended precision and database types
pub use register::{CompoundRegister, CombiningMethod, Record, FieldSchema, FieldType, RecordMetadata};

// Re-exports for convenience
pub use sha2::Sha256;
pub use hmac::Hmac;
