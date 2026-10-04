use crate::crypto::xor_hashes;
use crate::gate::TemporalGate;
use crate::packet::Packet;
use crate::primitives::{Address, GateStatus, Hash, Rail, Timestamp};
use std::cmp::Ordering;
use std::collections::{BinaryHeap, HashMap};
use thiserror::Error;

#[derive(Error, Debug)]
pub enum EngineError {
    #[error("Gate at address {0} not found")]
    GateNotFound(Address),
    #[error("Ticket unlock failed at gate {0}")]
    TicketUnlockFailed(Address),
    #[error("Dual rail desynchronization: hash mismatch")]
    DualRailDesync,
}

#[derive(Debug, Clone)]
pub struct ScheduledEvent {
    pub arrival_time: Timestamp,
    pub destination: Address,
    pub packet: Packet,
}

impl PartialEq for ScheduledEvent {
    fn eq(&self, other: &Self) -> bool {
        self.arrival_time == other.arrival_time
    }
}

impl Eq for ScheduledEvent {}

impl PartialOrd for ScheduledEvent {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for ScheduledEvent {
    fn cmp(&self, other: &Self) -> Ordering {
        other.arrival_time.cmp(&self.arrival_time)
    }
}

pub struct SimulationEngine {
    pub current_time: Timestamp,
    pub gates: HashMap<Address, TemporalGate>,
    pub event_queue: BinaryHeap<ScheduledEvent>,
    pub control_register: Option<Hash>,
    pub data_register: Option<Hash>,
    pub execution_log: Vec<String>,
}

impl SimulationEngine {
    pub fn new(start_time: Timestamp) -> Self {
        Self {
            current_time: start_time,
            gates: HashMap::new(),
            event_queue: BinaryHeap::new(),
            control_register: None,
            data_register: None,
            execution_log: Vec::new(),
        }
    }

    pub fn register_gate(&mut self, gate: TemporalGate) {
        self.gates.insert(gate.address, gate);
    }

    pub fn schedule_packet(&mut self, destination: Address, arrival_time: Timestamp, packet: Packet) {
        self.event_queue.push(ScheduledEvent {
            arrival_time,
            destination,
            packet,
        });
    }

    pub fn step(&mut self) -> Result<bool, EngineError> {
        if let Some(event) = self.event_queue.pop() {
            self.current_time = event.arrival_time;
            let gate = self.gates.get(&event.destination).ok_or(EngineError::GateNotFound(event.destination))?;
            let (status, route) = gate.evaluate_arrival(self.current_time, event.packet.header.lifeterm);

            self.execution_log.push(format!(
                "[T={}] Packet pos={} {:?} arrived at Gate 0x{:X} -> {:?}",
                self.current_time,
                event.packet.header.position,
                event.packet.header.chain_rail,
                gate.address,
                status
            ));

            if status == GateStatus::OnTime {
                if let Some(ticket) = event.packet.ticket {
                    if let Some(control_hash) = self.control_register {
                        if !gate.verify_and_unlock(&ticket, &control_hash) {
                            return Err(EngineError::TicketUnlockFailed(gate.address));
                        }
                    }
                }

                match event.packet.header.chain_rail {
                    Rail::A => self.control_register = Some(event.packet.header.chain_hash),
                    Rail::B => self.data_register = Some(event.packet.header.chain_hash),
                }

                if let (Some(a), Some(b)) = (self.control_register, self.data_register) {
                    let combined = xor_hashes(&a, &b);
                    self.execution_log.push(format!(
                        "  => Dual-Rail Synced! State digest: {}",
                        hex::encode(&combined[..8])
                    ));
                }
            }

            // Clone the route hops so we can mutably schedule new events without
            // conflicting with the immutable borrow of self.gates from above.
            let hops: Vec<Address> = route.hops.clone();
            for hop in &hops {
                self.schedule_packet(*hop, self.current_time + 10, event.packet.clone());
            }

            Ok(true)
        } else {
            Ok(false)
        }
    }

    pub fn run_until_idle(&mut self) -> Result<usize, EngineError> {
        let mut steps = 0;
        while self.step()? {
            steps += 1;
        }
        Ok(steps)
    }
}
