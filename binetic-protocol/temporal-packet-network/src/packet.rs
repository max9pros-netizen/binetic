use crate::crypto::compute_sha256;
use crate::primitives::{Guid, Hash, Rail, Timestamp};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PacketHeader {
    pub version: u8,
    pub chain_id: Guid,
    pub chain_rail: Rail,
    pub position: u8,
    pub issued_at: Timestamp,
    pub lifeterm: Timestamp,
    pub chain_hash: Hash,
    pub manifest_hash: Hash,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Packet {
    pub header: PacketHeader,
    pub payload: Vec<u8>,
    pub ticket: Option<Hash>,
}

impl Packet {
    pub fn new(
        chain_id: Guid,
        chain_rail: Rail,
        position: u8,
        issued_at: Timestamp,
        lifeterm: Timestamp,
        prev_hash: Hash,
        manifest_hash: Hash,
        payload: Vec<u8>,
    ) -> Self {
        let mut data_to_hash = Vec::new();
        data_to_hash.extend_from_slice(&chain_id);
        data_to_hash.push(match chain_rail { Rail::A => 0, Rail::B => 1 });
        data_to_hash.push(position);
        data_to_hash.extend_from_slice(&prev_hash);
        data_to_hash.extend_from_slice(&payload);
        let chain_hash = compute_sha256(&data_to_hash);

        Self {
            header: PacketHeader {
                version: 1,
                chain_id,
                chain_rail,
                position,
                issued_at,
                lifeterm,
                chain_hash,
                manifest_hash,
            },
            payload,
            ticket: None,
        }
    }

    pub fn with_ticket(mut self, ticket: Hash) -> Self {
        self.ticket = Some(ticket);
        self
    }
}
