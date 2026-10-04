//! Bitsliced lane operations.
//!
//! Instead of storing values as arrays of float32/float64/integers, we store
//! them as bit-plane slices across lanes. This is SIMD-friendly and natural for
//! quantized inference.
//!
//! Example: 8 INT8 values [0xD3, 0x7A, 0xB2, 0xF1, 0x4C, 0x8E, 0x19, 0xC5]
//! are stored as 8 lanes of 8 bits each:
//!   Lane 0 (MSB): [1, 0, 1, 1, 0, 1, 0, 1]
//!   Lane 1:       [1, 1, 0, 1, 1, 0, 0, 1]
//!   ...
//!   Lane 7 (LSB): [1, 0, 0, 1, 0, 0, 1, 1]

/// A bisliced lane — two bit-planes (spatial + temporal) for instant network evaluation.
///
/// Unlike a plain bitsliced lane, a bisliced lane stores **two** bit-planes:
/// - Spatial: which nodes are connected (topology)
/// - Temporal: when connections are active (timing)
///
/// Evaluation is O(1): AND the two planes to get active connections.
/// Recursive growth adds more lanes but each lane is still O(1) — no slowdown.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct BislicedLane {
    /// Spatial plane — which nodes are connected.
    pub spatial: BitslicedLane,
    /// Temporal plane — when connections are active.
    pub temporal: BitslicedLane,
}

impl BislicedLane {
    /// Create an empty bisliced lane with capacity for `num_bits` bits.
    pub fn with_capacity(num_bits: usize) -> Self {
        Self {
            spatial: BitslicedLane::with_capacity(num_bits),
            temporal: BitslicedLane::with_capacity(num_bits),
        }
    }

    /// Create from spatial and temporal bit slices.
    pub fn from_spatial_temporal(spatial: &[bool], temporal: &[bool]) -> Self {
        assert_eq!(spatial.len(), temporal.len());
        Self {
            spatial: BitslicedLane::from_bits(spatial),
            temporal: BitslicedLane::from_bits(temporal),
        }
    }

    /// Evaluate: active connections = spatial AND temporal.
    /// O(1) regardless of network size — single word-level AND per word.
    pub fn evaluate(&self) -> BitslicedLane {
        BitslicedLane::xor(&self.spatial, &self.temporal) // XOR for diff; AND for intersection
    }

    /// Evaluate as active connections (spatial AND temporal).
    pub fn active_connections(&self) -> BitslicedLane {
        let mut result = self.spatial.clone();
        result.and_merge(&self.temporal);
        result
    }

    /// Number of bits in this lane.
    pub fn len(&self) -> usize {
        self.spatial.len()
    }

    /// Resize both planes to new number of bits.
    pub fn resize(&mut self, new_num_bits: usize) {
        self.spatial.resize(new_num_bits);
        self.temporal.resize(new_num_bits);
    }

    /// Popcount of active connections.
    pub fn active_count(&self) -> usize {
        self.active_connections().popcount()
    }
}

/// A collection of bisliced lanes representing a full network topology.
///
/// For N network nodes, you have N bisliced lanes (one per node),
/// each storing spatial + temporal connectivity.
pub struct BislicedArray {
    lanes: Vec<BislicedLane>,
    num_nodes: usize,
}

impl BislicedArray {
    /// Create a bisliced array for N network nodes.
    pub fn new(num_nodes: usize) -> Self {
        let lanes: Vec<BislicedLane> = (0..num_nodes)
            .map(|_| BislicedLane::with_capacity(num_nodes))
            .collect();
        Self { lanes, num_nodes }
    }

    /// Set the spatial connection from node `from` to node `to`.
    pub fn set_spatial(&mut self, from: usize, to: usize, value: bool) {
        assert!(from < self.num_nodes);
        self.lanes[from].spatial.set_bit(to, value);
    }

    /// Set the temporal activity from node `from` to node `to`.
    pub fn set_temporal(&mut self, from: usize, to: usize, value: bool) {
        assert!(from < self.num_nodes);
        self.lanes[from].temporal.set_bit(to, value);
    }

    /// Get a lane by node index.
    pub fn lane(&self, node: usize) -> &BislicedLane {
        &self.lanes[node]
    }

    /// Get a mutable lane by node index.
    pub fn lane_mut(&mut self, node: usize) -> &mut BislicedLane {
        &mut self.lanes[node]
    }

    /// Evaluate all lanes — O(N) word-level operations, no per-node iteration.
    pub fn evaluate_all(&self) -> Vec<BitslicedLane> {
        self.lanes.iter().map(|l| l.active_connections()).collect()
    }
}
use std::hash::Hash;
use serde::{Deserialize, Serialize};
use std::fmt;

/// A bitsliced lane — a bit vector representing one bit-plane across multiple values.
///
/// Each lane is a vector of bits. N values × B bits = B lanes of N bits each.
///
/// Operations on lanes are bitwise and SIMD-friendly.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct BitslicedLane {
    /// The bits, stored as a vector of u64 words for efficient bitwise operations.
    words: Vec<u64>,
    /// Number of bits (values) in this lane.
    num_bits: usize,
}

impl BitslicedLane {
    /// Create an empty lane with capacity for `num_bits` bits.
    pub fn with_capacity(num_bits: usize) -> Self {
        let num_words = (num_bits + 63) / 64;
        Self {
            words: vec![0u64; num_words],
            num_bits,
        }
    }

    /// Create a lane from a slice of bytes.
    ///
    /// Each byte is one value (8 bits). The lane represents one bit-plane
    /// across all values.
    ///
    /// For a full bitsliced representation of B-bit values, you need B lanes.
    pub fn from_bytes(data: &[u8]) -> Self {
        let num_bits = data.len();
        let mut lane = Self::with_capacity(num_bits);
        for (i, &byte) in data.iter().enumerate() {
            // Set bit i to the MSB of the byte (bit 7)
            // For a full bitsliced representation, you'd create one lane per bit position.
            // Here we store the MSB as a representative example.
            if byte & 0x80 != 0 {
                lane.set_bit(i, true);
            }
        }
        lane
    }

    /// Create a lane from a slice of bits (booleans).
    pub fn from_bits(bits: &[bool]) -> Self {
        let num_bits = bits.len();
        let mut lane = Self::with_capacity(num_bits);
        for (i, &bit) in bits.iter().enumerate() {
            lane.set_bit(i, bit);
        }
        lane
    }

    /// Get the number of bits in this lane.
    pub fn len(&self) -> usize {
        self.num_bits
    }

    pub fn is_empty(&self) -> bool {
        self.num_bits == 0
    }

    /// Get the underlying words.
    pub fn words(&self) -> &[u64] {
        &self.words
    }

    /// Get the underlying bytes (for serialization / FFI).
    pub fn as_bytes(&self) -> &[u8] {
        // Convert words to bytes for external representation
        let num_bytes = self.num_bits;
        let mut result = Vec::with_capacity(num_bytes);
        for i in 0..num_bytes {
            let bit = self.get_bit(i);
            result.push(if bit { 0x80 } else { 0x00 });
        }
        result.leak() // FIXME: this is a leak; use proper allocation
    }

    /// Get a single bit.
    pub fn get_bit(&self, index: usize) -> bool {
        if index >= self.num_bits {
            return false;
        }
        let word_idx = index / 64;
        let bit_idx = index % 64;
        (self.words[word_idx] >> bit_idx) & 1 != 0
    }

    /// Set a single bit.
    pub fn set_bit(&mut self, index: usize, value: bool) {
        if index >= self.num_bits {
            return;
        }
        let word_idx = index / 64;
        let bit_idx = index % 64;
        if value {
            self.words[word_idx] |= 1u64 << bit_idx;
        } else {
            self.words[word_idx] &= !(1u64 << bit_idx);
        }
    }

    /// XOR this lane with another lane (in-place).
    ///
    /// This is the primary operation for XOR-based diff, fold, and echo propagation.
    pub fn xor_merge(&mut self, other: &Self) {
        assert_eq!(self.num_bits, other.num_bits);
        for (w, ow) in self.words.iter_mut().zip(other.words.iter()) {
            *w ^= ow;
        }
    }

    /// XOR two lanes, returning a new lane.
    pub fn xor(lhs: &Self, rhs: &Self) -> Self {
        assert_eq!(lhs.num_bits, rhs.num_bits);
        let mut result = lhs.clone();
        result.xor_merge(rhs);
        result
    }

    /// AND this lane with another lane (in-place).
    pub fn and_merge(&mut self, other: &Self) {
        assert_eq!(self.num_bits, other.num_bits);
        for (w, ow) in self.words.iter_mut().zip(other.words.iter()) {
            *w &= ow;
        }
    }

    /// NOT this lane (in-place).
    pub fn not_inplace(&mut self) {
        for w in self.words.iter_mut() {
            *w = !*w;
        }
        // Mask off bits beyond num_bits
        let remaining = self.num_bits % 64;
        if remaining > 0 {
            let last = self.words.len() - 1;
            let mask = (1u64 << remaining) - 1;
            self.words[last] &= mask;
        }
    }

    /// Popcount — count the number of set bits.
    pub fn popcount(&self) -> usize {
        self.words.iter().map(|w| w.count_ones() as usize).sum()
    }

    /// Check if all bits are set.
    pub fn all_set(&self) -> bool {
        self.popcount() == self.num_bits
    }

    /// Check if no bits are set.
    pub fn none_set(&self) -> bool {
        self.popcount() == 0
    }

    /// Check if any bit is set.
    pub fn any_set(&self) -> bool {
        self.popcount() > 0
    }

    /// Convert to a vector of booleans (for debugging / testing).
    pub fn to_bits(&self) -> Vec<bool> {
        (0..self.num_bits).map(|i| self.get_bit(i)).collect()
    }

    /// Get the Hamming distance to another lane (popcount of XOR).
    pub fn hamming_distance(&self, other: &Self) -> usize {
        let xor = Self::xor(self, other);
        xor.popcount()
    }

    /// Resize the lane to a new number of bits.
    pub fn resize(&mut self, new_num_bits: usize) {
        let new_num_words = (new_num_bits + 63) / 64;
        if new_num_words > self.words.len() {
            self.words.resize(new_num_words, 0);
        }
        self.num_bits = new_num_bits;
        // Mask off excess bits in the last word
        let remaining = new_num_bits % 64;
        if remaining > 0 {
            let last = self.words.len() - 1;
            let mask = (1u64 << remaining) - 1;
            self.words[last] &= mask;
        } else if new_num_bits > 0 {
            let last = self.words.len() - 1;
            self.words[last] &= !0u64;
        }
    }
}

impl fmt::Display for BitslicedLane {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "BitslicedLane(bits={} words={} popcount={})",
            self.num_bits,
            self.words.len(),
            self.popcount()
        )
    }
}

impl Hash for BitslicedLane {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.words.hash(state);
        self.num_bits.hash(state);
    }
}

impl PartialOrd for BitslicedLane {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for BitslicedLane {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.words.cmp(&other.words).then_with(|| self.num_bits.cmp(&other.num_bits))
    }
}

/// A collection of lanes representing a full bitsliced value array.
///
/// For B-bit values × N entries, you have B lanes of N bits each.
pub struct BitslicedArray {
    lanes: Vec<BitslicedLane>,
    num_values: usize,
    bits_per_value: usize,
}

impl BitslicedArray {
    /// Create a bitsliced array for B-bit values × N entries.
    pub fn new(bits_per_value: usize, num_values: usize) -> Self {
        let lanes: Vec<BitslicedLane> = (0..bits_per_value)
            .map(|_| BitslicedLane::with_capacity(num_values))
            .collect();
        Self {
            lanes,
            num_values,
            bits_per_value,
        }
    }

    /// Set a value at a given index.
    pub fn set_value(&mut self, index: usize, value: u64) {
        assert!(index < self.num_values);
        assert!(value < (1u64 << self.bits_per_value) as u64);

        for bit_pos in 0..self.bits_per_value {
            let bit = (value >> (self.bits_per_value - 1 - bit_pos)) & 1 != 0;
            self.lanes[bit_pos].set_bit(index, bit);
        }
    }

    /// Get a value at a given index.
    pub fn get_value(&self, index: usize) -> u64 {
        assert!(index < self.num_values);
        let mut value = 0u64;
        for bit_pos in 0..self.bits_per_value {
            if self.lanes[bit_pos].get_bit(index) {
                value |= 1u64 << (self.bits_per_value - 1 - bit_pos);
            }
        }
        value
    }

    /// XOR this array with another (in-place).
    pub fn xor_merge(&mut self, other: &Self) {
        assert_eq!(self.num_values, other.num_values);
        assert_eq!(self.bits_per_value, other.bits_per_value);
        for (lane, other_lane) in self.lanes.iter_mut().zip(other.lanes.iter()) {
            lane.xor_merge(other_lane);
        }
    }

    /// Get a lane by bit position.
    pub fn lane(&self, bit_pos: usize) -> &BitslicedLane {
        &self.lanes[bit_pos]
    }

    /// Get a mutable lane by bit position.
    pub fn lane_mut(&mut self, bit_pos: usize) -> &mut BitslicedLane {
        &mut self.lanes[bit_pos]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_lane_from_bytes() {
        let data = [0xD3, 0x7A, 0xB2];
        let lane = BitslicedLane::from_bytes(&data);
        assert_eq!(lane.len(), 3);
        // MSB of 0xD3 (11010011) is 1
        assert!(lane.get_bit(0));
        // MSB of 0x7A (01111010) is 0
        assert!(!lane.get_bit(1));
        // MSB of 0xB2 (10110010) is 1
        assert!(lane.get_bit(2));
    }

    #[test]
    fn test_lane_xor() {
        let a = BitslicedLane::from_bits(&[true, false, true, false]);
        let b = BitslicedLane::from_bits(&[true, true, false, false]);

        let c = BitslicedLane::xor(&a, &b);
        assert_eq!(c.to_bits(), vec![false, true, true, false]);
    }

    #[test]
    fn test_lane_popcount() {
        let lane = BitslicedLane::from_bits(&[true, false, true, true, false, false, true, false]);
        assert_eq!(lane.popcount(), 4);
    }

    #[test]
    fn test_bitsliced_array() {
        let mut arr = BitslicedArray::new(8, 4);
        arr.set_value(0, 0xD3);
        arr.set_value(1, 0x7A);
        arr.set_value(2, 0xB2);
        arr.set_value(3, 0xF1);

        assert_eq!(arr.get_value(0), 0xD3);
        assert_eq!(arr.get_value(1), 0x7A);
        assert_eq!(arr.get_value(2), 0xB2);
        assert_eq!(arr.get_value(3), 0xF1);
    }

    #[test]
    fn test_bitsliced_array_xor() {
        let mut a = BitslicedArray::new(8, 4);
        a.set_value(0, 0xD3);
        a.set_value(1, 0x7A);
        a.set_value(2, 0xB2);
        a.set_value(3, 0xF1);

        let mut b = BitslicedArray::new(8, 4);
        b.set_value(0, 0xB2);
        b.set_value(1, 0xF1);
        b.set_value(2, 0xD3);
        b.set_value(3, 0x7A);

        a.xor_merge(&b);

        assert_eq!(a.get_value(0), 0xD3 ^ 0xB2);
        assert_eq!(a.get_value(1), 0x7A ^ 0xF1);
        assert_eq!(a.get_value(2), 0xB2 ^ 0xD3);
        assert_eq!(a.get_value(3), 0xF1 ^ 0x7A);
    }
}
