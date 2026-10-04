use crate::crypto::xor_hashes;
use crate::primitives::{Address, GateStatus, Hash, Route, TemporalWindow, Timestamp};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TemporalGate {
    pub address: Address,
    pub window: TemporalWindow,
    pub on_early: Route,
    pub on_time: Route,
    pub on_late: Route,
    pub on_miss: Route,
    pub ticket_secret: Option<Hash>,
}

impl TemporalGate {
    pub fn new(address: Address, window: TemporalWindow, on_time: Route, on_miss: Route) -> Self {
        Self {
            address,
            window,
            on_early: on_miss.clone(),
            on_time,
            on_late: on_miss.clone(),
            on_miss,
            ticket_secret: None,
        }
    }

    pub fn with_ticket(mut self, secret: Hash) -> Self {
        self.ticket_secret = Some(secret);
        self
    }

    pub fn evaluate_arrival(&self, arrival: Timestamp, lifeterm: Timestamp) -> (GateStatus, &Route) {
        let status = self.window.evaluate(arrival, lifeterm);
        let route = match status {
            GateStatus::Early => &self.on_early,
            GateStatus::OnTime => &self.on_time,
            GateStatus::Late => &self.on_late,
            GateStatus::Missed => &self.on_miss,
        };
        (status, route)
    }

    pub fn verify_and_unlock(&self, ticket_hash: &Hash, control_hash: &Hash) -> bool {
        if let Some(expected_key) = &self.ticket_secret {
            let unlocked = xor_hashes(ticket_hash, control_hash);
            &unlocked == expected_key
        } else {
            true
        }
    }
}
