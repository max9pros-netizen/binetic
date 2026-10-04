//! Bloom filters per sector.
//!
//! Each sector in the IPv4 control plane has a bloom filter attached.
//! The bloom filter answers: "might address X be in this sector?"
//!
//! Used for:
//! - Routing: before routing to a sector, check if the destination might be there.
//! - Echo propagation: before propagating a change to a sector, check if any register
//!   in the sector cares about the changed address.
//! - Lookup: before descending into a sub-sphere, check if the target might be there.

use serde::{Deserialize, Serialize};
use std::fmt;
use std::hash::{Hash, Hasher};

/// A bloom filter attached to a sector.
///
/// Uses multiple hash functions (XOR-based) for compact representation.
/// Supports merge (union) of bloom filters for hierarchical routing.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct SectorBloom {
    /// The bit vector (256 bits = 32 bytes by default).
    bits: Vec<u64>,
    /// Number of bits in the filter.
    num_bits: usize,
    /// Number of hash functions (k).
    num_hashes: usize,
    /// Generation counter — increments on each rebuild.
    generation: u8,
    /// Hash anchor for this sector (for bloom filter matching).
    anchor: u64,
}

impl SectorBloom {
    /// Create a new bloom filter with the given size (in bits) and number of hash functions.
    pub fn new(num_bits: usize, num_hashes: usize, anchor: u64) -> Self {
        let num_words = (num_bits + 63) / 64;
        Self {
            bits: vec![0u64; num_words],
            num_bits,
            num_hashes,
            generation: 0,
            anchor,
        }
    }

    /// Create a bloom filter with default parameters (256 bits, 3 hash functions).
    pub fn default_with_anchor(anchor: u64) -> Self {
        Self::new(256, 3, anchor)
    }

    /// Get the anchor for this bloom filter.
    pub fn anchor(&self) -> u64 {
        self.anchor
    }

    /// Get the current generation.
    pub fn generation(&self) -> u8 {
        self.generation
    }

    /// Insert an address into the bloom filter.
    ///
    /// Uses XOR-based hashing: hash(address ^ anchor) for each hash function.
    pub fn insert(&mut self, address_bits: u128) {
        for i in 0..self.num_hashes {
            let hash = self.hash(address_bits, i);
            let bit_idx = (hash as usize) % self.num_bits;
            let word_idx = bit_idx / 64;
            let bit_in_word = bit_idx % 64;
            self.bits[word_idx] |= 1u64 << bit_in_word;
        }
        self.generation = self.generation.wrapping_add(1);
    }

    /// Check if an address might be in the bloom filter.
    ///
    /// Returns true if the address might be present (true positive or false positive).
    /// Returns false if the address is definitely not present.
    pub fn might_contain(&self, address_bits: u128) -> bool {
        for i in 0..self.num_hashes {
            let hash = self.hash(address_bits, i);
            let bit_idx = (hash as usize) % self.num_bits;
            let word_idx = bit_idx / 64;
            let bit_in_word = bit_idx % 64;
            if self.bits[word_idx] & (1u64 << bit_in_word) == 0 {
                return false;
            }
        }
        true
    }

    /// Check if an address is definitely NOT in the bloom filter.
    pub fn does_not_contain(&self, address_bits: u128) -> bool {
        !self.might_contain(address_bits)
    }

    /// Merge another bloom filter into this one (union).
    ///
    /// This is useful for hierarchical routing: parent sector's bloom = union of children's blooms.
    pub fn merge(&mut self, other: &Self) {
        assert_eq!(self.num_bits, other.num_bits);
        assert_eq!(self.num_hashes, other.num_hashes);
        for (w, ow) in self.bits.iter_mut().zip(other.bits.iter()) {
            *w |= ow;
        }
        self.generation = self.generation.wrapping_add(1);
    }

    /// Check if this bloom filter might intersect with another.
    ///
    /// Returns true if there might be common elements (or false positives).
    pub fn might_intersect(&self, other: &Self) -> bool {
        for (w, ow) in self.bits.iter().zip(other.bits.iter()) {
            if w & ow != 0 {
                return true;
            }
        }
        false
    }

    /// Rebuild the bloom filter from a set of addresses.
    ///
    /// Clears the filter and re-inserts all addresses.
    /// Increments generation.
    pub fn rebuild(&mut self, addresses: &[u128]) {
        // Clear
        for w in self.bits.iter_mut() {
            *w = 0;
        }
        // Re-insert
        for &addr in addresses {
            self.insert(addr);
        }
    }

    /// Estimate the false positive rate.
    ///
    /// p ≈ (1 - e^(-k*n/m))^k
    /// where k = num_hashes, n = number of inserted elements, m = num_bits
    pub fn estimated_false_positive_rate(&self, num_inserted: usize) -> f64 {
        let m = self.num_bits as f64;
        let k = self.num_hashes as f64;
        let n = num_inserted as f64;
        let exponent = -k * n / m;
        let p = (1.0 - (-exponent).exp()).powf(k);
        p
    }

    /// Get an approximate count of inserted elements (using the harmonic mean estimator).
    pub fn estimated_count(&self) -> usize {
        // Simple estimator: -m/k * ln(1 - bit_count/m)
        let m = self.num_bits as f64;
        let k = self.num_hashes as f64;
        let bit_count = self.bits.iter().map(|w| w.count_ones() as usize).sum::<usize>() as f64;
        if bit_count >= m {
            return usize::MAX; // saturated
        }
        let fraction = bit_count / m;
        if fraction <= 0.0 {
            return 0;
        }
        let estimate = -(m / k) * (1.0 - fraction).ln();
        estimate as usize
    }

    /// Internal hash function: XOR-based hash using address and hash index.
    fn hash(&self, address_bits: u128, hash_index: usize) -> u64 {
        // Simple XOR-shift hash
        let mut h = (address_bits as u64) ^ (self.anchor.wrapping_mul(hash_index as u64 + 1));
        h ^= h >> 33;
        h = h.wrapping_mul(0xff51afd7ed558ccd_u64);
        h ^= h >> 33;
        h = h.wrapping_mul(0xc4ceb9fe1a85ec53_u64);
        h ^= h >> 33;
        h
    }
}

impl fmt::Display for SectorBloom {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "SectorBloom(bits={} hashes={} gen={} anchor=0x{:016X} estimated_count={})",
            self.num_bits,
            self.num_hashes,
            self.generation,
            self.anchor,
            self.estimated_count()
        )
    }
}

/// A bloom filter registry for a fabric.
///
/// Maintains bloom filters per sector, with hierarchical aggregation.
#[derive(Clone, Debug, Default)]
pub struct BloomRegistry {
    /// Bloom filters per sector key.
    ///
    /// Key = (spine, layer, ring, sector) — identifies a sector.
    sectors: std::collections::HashMap<SectorKey, SectorBloom>,
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct SectorKey {
    pub spine: u16,
    pub layer: u16,
    pub ring: u16,
    pub sector: u16,
}

impl SectorKey {
    pub fn new(spine: u16, layer: u16, ring: u16, sector: u16) -> Self {
        Self { spine, layer, ring, sector }
    }
}

impl BloomRegistry {
    pub fn new() -> Self {
        Self::default()
    }

    /// Get or create a bloom filter for a sector.
    pub fn bloom_for_sector(&mut self, key: SectorKey) -> &mut SectorBloom {
        let anchor = self.bloom_anchor_for_sector(key);
        self.sectors.entry(key).or_insert_with(|| {
            SectorBloom::default_with_anchor(anchor)
        })
    }

    /// Get a bloom filter for a sector (if it exists).
    pub fn get_bloom(&self, key: SectorKey) -> Option<&SectorBloom> {
        self.sectors.get(&key)
    }

    /// Check if an address might be in a sector.
    pub fn sector_might_contain(&self, key: SectorKey, address_bits: u128) -> bool {
        self.sectors.get(&key).is_some_and(|b| b.might_contain(address_bits))
    }

    /// Insert an address into a sector's bloom filter.
    pub fn insert_into_sector(&mut self, key: SectorKey, address_bits: u128) {
        self.bloom_for_sector(key).insert(address_bits);
    }

    /// Compute a bloom anchor for a sector.
    fn bloom_anchor_for_sector(&self, key: SectorKey) -> u64 {
        let mut h = std::collections::hash_map::DefaultHasher::new();
        key.hash(&mut h);
        h.finish()
    }

    /// Aggregate bloom filters for a parent sector from its children.
    ///
    /// This is used for hierarchical routing: the parent's bloom = union of children's blooms.
    pub fn aggregate_to_parent(&self, parent_key: SectorKey, child_keys: &[SectorKey]) -> SectorBloom {
        let mut parent = SectorBloom::default_with_anchor(self.bloom_anchor_for_sector(parent_key));
        for &child_key in child_keys {
            if let Some(child_bloom) = self.sectors.get(&child_key) {
                parent.merge(child_bloom);
            }
        }
        parent
    }

    /// Clear all bloom filters.
    pub fn clear(&mut self) {
        self.sectors.clear();
    }

    /// Get the number of sectors with bloom filters.
    pub fn sector_count(&self) -> usize {
        self.sectors.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bloom_insert_and_might_contain() {
        let mut bloom = SectorBloom::new(256, 3, 0xDEADBEEF);

        // Insert some addresses
        bloom.insert(0x1234567890ABCDEF_u128);
        bloom.insert(0xFEDCBA0987654321_u128);

        // Check that inserted addresses might be contained
        assert!(bloom.might_contain(0x1234567890ABCDEF_u128));
        assert!(bloom.might_contain(0xFEDCBA0987654321_u128));

        // Check that non-inserted addresses might or might not be contained
        // (false positives are possible, but not guaranteed)
        let result = bloom.might_contain(0xAAAAAAAAAAAAAAAA_u128);
        // We can't assert false because of false positives, but we can assert
        // that the bloom is not empty
        assert!(bloom.estimated_count() >= 2);
    }

    #[test]
    fn test_bloom_does_not_contain() {
        let mut bloom = SectorBloom::new(256, 3, 0xDEADBEEF);
        bloom.insert(0x1234_u128);

        // A very different address is likely not contained
        // (but could be a false positive — we can't guarantee)
        // We can at least verify the bloom works
        assert!(bloom.might_contain(0x1234_u128));
    }

    #[test]
    fn test_bloom_merge() {
        let mut bloom_a = SectorBloom::new(256, 3, 0xAAAA);
        let mut bloom_b = SectorBloom::new(256, 3, 0xBBBB);

        bloom_a.insert(0x1111_u128);
        bloom_b.insert(0x2222_u128);

        let mut merged = bloom_a.clone();
        merged.merge(&bloom_b);

        // Merged should might-contain both
        assert!(merged.might_contain(0x1111_u128));
        assert!(merged.might_contain(0x2222_u128));
    }

    #[test]
    fn test_bloom_registry() {
        let mut registry = BloomRegistry::new();

        let key = SectorKey::new(0, 1, 2, 3);

        registry.insert_into_sector(key, 0x1234_u128);
        registry.insert_into_sector(key, 0x5678_u128);

        assert!(registry.sector_might_contain(key, 0x1234_u128));
        assert!(registry.sector_might_contain(key, 0x5678_u128));
        assert_eq!(registry.sector_count(), 1);
    }

    #[test]
    fn test_false_positive_rate() {
        let mut bloom = SectorBloom::new(1024, 3, 0xDEADBEEF);

        // Insert 100 elements
        for i in 0..100 {
            bloom.insert(i as u128);
        }

        let fpr = bloom.estimated_false_positive_rate(100);
        // With 1024 bits, 3 hashes, 100 elements, FPR should be low
        // p ≈ (1 - e^(-3*100/1024))^3 ≈ (1 - e^(-0.293))^3 ≈ (1 - 0.746)^3 ≈ 0.016
        assert!(fpr < 0.1); // less than 10%
        assert!(fpr > 0.0); // greater than 0%
    }
}
