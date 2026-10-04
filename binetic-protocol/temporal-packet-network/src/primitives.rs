use serde::{Deserialize, Serialize};
use std::fmt;

pub type Timestamp = u64;
pub type Address = u128;
pub type Hash = [u8; 32];
pub type Guid = [u8; 16];

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Rail {
    A,
    B,
}

impl fmt::Display for Rail {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", match self { Rail::A => "Rail-A (Control)", Rail::B => "Rail-B (Data)" })
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum GateStatus {
    Early,
    OnTime,
    Late,
    Missed,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct TemporalWindow {
    pub t_min: Timestamp,
    pub t_max: Timestamp,
}

impl TemporalWindow {
    pub fn new(t_min: Timestamp, t_max: Timestamp) -> Self {
        assert!(t_min <= t_max, "Window t_min cannot exceed t_max");
        Self { t_min, t_max }
    }

    pub fn evaluate(&self, arrival: Timestamp, lifeterm: Timestamp) -> GateStatus {
        if arrival > lifeterm {
            GateStatus::Missed
        } else if arrival < self.t_min {
            GateStatus::Early
        } else if arrival <= self.t_max {
            GateStatus::OnTime
        } else {
            GateStatus::Late
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Route {
    pub hops: Vec<Address>,
}

impl Route {
    pub fn single(address: Address) -> Self {
        Self { hops: vec![address] }
    }

    pub fn new(hops: Vec<Address>) -> Self {
        Self { hops }
    }
}
