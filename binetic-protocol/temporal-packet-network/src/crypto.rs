use crate::primitives::Hash;
use sha2::{Digest, Sha256};

pub fn compute_sha256(data: &[u8]) -> Hash {
    let mut hasher = Sha256::new();
    hasher.update(data);
    let result = hasher.finalize();
    let mut hash = [0u8; 32];
    hash.copy_from_slice(&result);
    hash
}

pub fn xor_hashes(a: &Hash, b: &Hash) -> Hash {
    let mut result = [0u8; 32];
    for i in 0..32 {
        result[i] = a[i] ^ b[i];
    }
    result
}

pub fn verify_ticket(ticket_hash: &Hash, target_hash: &Hash, expected_key: &Hash) -> bool {
    let unlocked = xor_hashes(ticket_hash, target_hash);
    &unlocked == expected_key
}
