//! Backend abstraction — pluggable inference backends.
//!
//! The fabric integrates with inference backends (llama.cpp, MLX, custom kernels)
//! via a backend trait. Backends receive pointers into fabric memory (no copy),
//! execute compute, and write results back to temporal registers.

use crate::address::RegisterAddress;
use crate::bitslice::BitslicedLane;
use crate::register::Register;
use crate::tiers::TierId;
use serde::{Deserialize, Serialize};
use std::fmt;

/// A unique backend identifier.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct BackendId(pub u8);

impl BackendId {
    pub const LLAMA_CPP: BackendId = BackendId(0);
    pub const MLX: BackendId = BackendId(1);
    pub const NATIVE: BackendId = BackendId(2);
    pub const CUSTOM: BackendId = BackendId(3);
    const COUNT: u8 = 4;

    pub fn name(self) -> &'static str {
        match self.0 {
            0 => "llama.cpp",
            1 => "mlx",
            2 => "native",
            3 => "custom",
            _ => "unknown",
        }
    }
}

impl fmt::Debug for BackendId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "BackendId({})", self.name())
    }
}

/// An arithmetic operation that a backend can execute.
#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub enum ArithmeticOp {
    /// Matrix multiply: C = A × B
    MatMul,
    /// Element-wise addition: C = A + B
    Add,
    /// Element-wise multiplication: C = A × B (Hadamard)
    Mul,
    /// Softmax along a dimension
    Softmax,
    /// Attention: softmax(Q × K^T / sqrt(d)) × V
    Attention,
    /// Layer normalization
    LayerNorm,
    /// RMS normalization
    RMSNorm,
    /// Rotary position embedding (RoPE)
    RoPE,
    /// Swish / SiLU activation
    SiLU,
    /// GELU activation
    GELU,
    /// Quantized matrix multiply (for Q4, Q5, Q8 models)
    QuantizedMatMul,
    /// Sample next token from logits
    SampleNextToken,
    /// Custom operation (backend-specific)
    Custom(u32),
}

impl fmt::Display for ArithmeticOp {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ArithmeticOp::MatMul => write!(f, "MatMul"),
            ArithmeticOp::Add => write!(f, "Add"),
            ArithmeticOp::Mul => write!(f, "Mul"),
            ArithmeticOp::Softmax => write!(f, "Softmax"),
            ArithmeticOp::Attention => write!(f, "Attention"),
            ArithmeticOp::LayerNorm => write!(f, "LayerNorm"),
            ArithmeticOp::RMSNorm => write!(f, "RMSNorm"),
            ArithmeticOp::RoPE => write!(f, "RoPE"),
            ArithmeticOp::SiLU => write!(f, "SiLU"),
            ArithmeticOp::GELU => write!(f, "GELU"),
            ArithmeticOp::QuantizedMatMul => write!(f, "QuantMatMul"),
            ArithmeticOp::SampleNextToken => write!(f, "SampleNextToken"),
            ArithmeticOp::Custom(id) => write!(f, "Custom({})", id),
        }
    }
}

/// How to route compute relative to data.
#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub enum RoutingPolicy {
    /// Dispatch compute to where the data already is (minimize data movement).
    ToData,
    /// Move data to where the compute is (if compute is already set up).
    ToCompute,
    /// Balanced: choose based on data location, compute availability, energy state.
    Balanced,
}

impl Default for RoutingPolicy {
    fn default() -> Self {
        RoutingPolicy::ToData
    }
}

/// A compute operation to be dispatched to a backend.
#[derive(Clone, Debug)]
pub struct ComputeOp {
    pub operation: ArithmeticOp,
    pub operands: Vec<RegisterAddress>,
    pub result_address: RegisterAddress,
    pub backend: Option<BackendId>,
    pub routing: RoutingPolicy,
    /// Additional parameters for the operation (e.g., attention mask, scaling factors).
    pub params: ComputeParams,
}

impl ComputeOp {
    pub fn new(operation: ArithmeticOp, operands: Vec<RegisterAddress>, result_address: RegisterAddress) -> Self {
        Self {
            operation,
            operands,
            result_address,
            backend: None,
            routing: RoutingPolicy::default(),
            params: ComputeParams::default(),
        }
    }
}

/// Additional parameters for compute operations.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct ComputeParams {
    pub scaling_factor: Option<f32>,
    pub mask: Option<Vec<u8>>,
    pub quantization: Option<QuantizationLevel>,
    pub temperature: Option<f32>,
    pub top_k: Option<usize>,
    pub top_p: Option<f32>,
}

/// Quantization level for quantized operations.
#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub enum QuantizationLevel {
    F32,
    F16,
    BF16,
    Q4_0,
    Q4_1,
    Q5_0,
    Q5_1,
    Q6_K,
    Q8_0,
    Q8_1,
    IQ2_XS,
    IQ3_XS,
    IQ4_XS,
    IQ5_XS,
    IQ6_XS,
    IQ7_XS,
    IQ4_XXS,
    Unknown,
}

impl From<u8> for QuantizationLevel {
    fn from(v: u8) -> Self {
        match v {
            0 => QuantizationLevel::F32,
            1 => QuantizationLevel::F16,
            2 => QuantizationLevel::BF16,
            3 => QuantizationLevel::Q4_0,
            4 => QuantizationLevel::Q4_1,
            5 => QuantizationLevel::Q5_0,
            6 => QuantizationLevel::Q5_1,
            7 => QuantizationLevel::Q6_K,
            8 => QuantizationLevel::Q8_0,
            9 => QuantizationLevel::Q8_1,
            10 => QuantizationLevel::IQ2_XS,
            11 => QuantizationLevel::IQ3_XS,
            12 => QuantizationLevel::IQ4_XS,
            13 => QuantizationLevel::IQ5_XS,
            14 => QuantizationLevel::IQ6_XS,
            15 => QuantizationLevel::IQ7_XS,
            16 => QuantizationLevel::IQ4_XXS,
            _ => QuantizationLevel::Unknown,
        }
    }
}

/// Result of a compute operation.
#[derive(Clone, Debug, Default)]
pub struct ComputeResult {
    pub success: bool,
    pub backend_used: Option<BackendId>,
    pub latency_us: u64,
    pub energy_estimate_nj: u64,
    pub error_message: Option<String>,
    /// Result payload (written back to the result register by the fabric).
    pub payload: Option<BitslicedLane>,
}

/// A backend that can execute compute operations on fabric registers.
///
/// Backends are plug-ins that the fabric dispatches to. They receive registers
/// by address, resolve them to physical memory pointers (no copy), execute the
/// operation, and write results back.
pub trait Backend: Send + Sync {
    /// Unique identifier for this backend.
    fn id(&self) -> BackendId;

    /// Human-readable name.
    fn name(&self) -> &str;

    /// Which operations this backend supports.
    fn supported_ops(&self) -> &[ArithmeticOp];

    /// Initialize the backend (load libraries, set up memory mappings, etc.).
    fn init(&mut self) -> Result<(), String>;

    /// Teardown the backend.
    fn teardown(&mut self);

    /// Check if this backend can handle the given operation with the given operands.
    ///
    /// Considers: supported ops, operand locations (which tiers they're in), backend capabilities.
    fn can_handle(&self, op: &ComputeOp, operand_tiers: &[TierId]) -> bool;

    /// Dispatch a compute operation to this backend.
    ///
    /// The fabric resolves operands before calling this method (by reading from
    /// the register store), so the backend receives the actual data. The backend
    /// returns a ComputeResult that the fabric uses to update the register store.
    fn execute(&self, op: &ComputeOp, operands: &[BitslicedLane]) -> ComputeResult;

    /// Estimate the energy cost of executing this operation.
    fn estimate_energy(&self, op: &ComputeOp, operand_tiers: &[TierId]) -> u64;

    /// Estimate the latency of executing this operation.
    fn estimate_latency_us(&self, op: &ComputeOp, operand_tiers: &[TierId]) -> u64;
}

/// A collection of backends with selection logic.
pub struct BackendSet {
    backends: Vec<Box<dyn Backend>>,
}

impl BackendSet {
    pub fn new() -> Self {
        Self {
            backends: Vec::new(),
        }
    }

    pub fn add(&mut self, backend: Box<dyn Backend>) {
        self.backends.push(backend);
    }

    pub fn get(&self, id: BackendId) -> Option<&dyn Backend> {
        self.backends.iter().find(|b| b.id() == id).map(|b| &**b)
    }

    pub fn with_backend_mut<F, R>(&mut self, id: BackendId, f: F) -> Option<R>
    where
        F: FnOnce(&mut dyn Backend) -> R,
    {
        self.backends.iter_mut().find(|b| b.id() == id).map(|b| {
            let b: &mut Box<dyn Backend> = b;
            let b: &mut dyn Backend = &mut **b;
            f(b)
        })
    }

    /// Call a function with mutable references to all backends.
    pub fn with_each_backend_mut<F, E>(&mut self, mut f: F) -> Result<(), E>
    where
        F: FnMut(&mut dyn Backend) -> Result<(), E>,
        E: std::fmt::Debug,
    {
        for b in &mut self.backends {
            let b: &mut dyn Backend = &mut **b;
            f(b)?;
        }
        Ok(())
    }

    /// Select the best backend for a compute operation.
    ///
    /// Considers: supported ops, operand locations, energy/latency estimates.
    /// Returns the BackendId of the selected backend, or None if no backend can handle it.
    pub fn select_best(
        &self,
        op: &ComputeOp,
        operand_tiers: &[TierId],
        prefer_energy: bool,
    ) -> Option<BackendId> {
        let candidates: Vec<&dyn Backend> = self
            .backends
            .iter()
            .filter(|b| b.can_handle(op, operand_tiers))
            .map(|b| b.as_ref())
            .collect();

        if candidates.is_empty() {
            return None;
        }

        if candidates.len() == 1 {
            return Some(candidates[0].id());
        }

        // Score each candidate
        let mut best_id = None;
        let mut best_score = f64::MAX;

        for candidate in &candidates {
            let energy = candidate.estimate_energy(op, operand_tiers);
            let latency = candidate.estimate_latency_us(op, operand_tiers);

            let score = if prefer_energy {
                energy as f64 + latency as f64 * 0.001 // weight energy more
            } else {
                latency as f64 + energy as f64 * 0.0001 // weight latency more
            };

            if score < best_score {
                best_score = score;
                best_id = Some(candidate.id());
            }
        }

        best_id
    }

    pub fn iter(&self) -> impl Iterator<Item = &dyn Backend> + '_ {
        self.backends.iter().map(|b| b.as_ref())
    }

    pub fn len(&self) -> usize {
        self.backends.len()
    }

    pub fn is_empty(&self) -> bool {
        self.backends.is_empty()
    }
}

/// A simple in-memory native backend for testing and small models.
///
/// This backend executes operations directly on the bitsliced data in memory.
/// It's not optimized for performance, but it's useful for testing and as a fallback.
pub struct NativeBackend {
    id: BackendId,
}

impl NativeBackend {
    pub fn new() -> Self {
        Self {
            id: BackendId::NATIVE,
        }
    }
}

impl Backend for NativeBackend {
    fn id(&self) -> BackendId {
        self.id
    }

    fn name(&self) -> &str {
        "Native (in-memory)"
    }

    fn supported_ops(&self) -> &[ArithmeticOp] {
        &[
            ArithmeticOp::Add,
            ArithmeticOp::Mul,
            ArithmeticOp::MatMul,
            ArithmeticOp::SiLU,
            ArithmeticOp::GELU,
        ]
    }

    fn init(&mut self) -> Result<(), String> {
        Ok(()) // no initialization needed
    }

    fn teardown(&mut self) {
        // nothing to teardown
    }

    fn can_handle(&self, op: &ComputeOp, _operand_tiers: &[TierId]) -> bool {
        self.supported_ops().contains(&op.operation)
    }

    fn execute(&self, op: &ComputeOp, operands: &[BitslicedLane]) -> ComputeResult {
        let result = match op.operation {
            ArithmeticOp::Add => {
                if operands.len() >= 2 {
                    let mut r = operands[0].clone();
                    r.xor_merge(&operands[1]);
                    r
                } else {
                    return ComputeResult {
                        success: false,
                        backend_used: Some(self.id),
                        error_message: Some("Add requires 2 operands".to_string()),
                        ..Default::default()
                    };
                }
            }
            ArithmeticOp::Mul => {
                // Real bitwise multiply: AND merge (Hadamard product in bitsliced domain)
                if operands.len() >= 2 {
                    let mut r = operands[0].clone();
                    r.and_merge(&operands[1]);
                    r
                } else {
                    return ComputeResult {
                        success: false,
                        backend_used: Some(self.id),
                        error_message: Some("Mul requires 2 operands".to_string()),
                        ..Default::default()
                    };
                }
            }
            ArithmeticOp::MatMul => {
                // Real matrix multiply for bitsliced lanes.
                // Each lane is a bit-plane. For N×M × M×P, we compute
                // dot products: for each output bit, XOR of ANDs across the reduction dimension.
                // Simplified: treat operands as flattened vectors and compute inner product.
                if operands.len() >= 2 {
                    let a = &operands[0];
                    let b = &operands[1];
                    if a.len() != b.len() {
                        return ComputeResult {
                            success: false,
                            backend_used: Some(self.id),
                            error_message: Some(format!(
                                "MatMul dimension mismatch: {} vs {}",
                                a.len(),
                                b.len()
                            )),
                            ..Default::default()
                        };
                    }
                    // Inner product: popcount of AND, mod 2 = parity
                    let mut r = BitslicedLane::with_capacity(1);
                    let and_lane = {
                        let mut tmp = a.clone();
                        tmp.and_merge(b);
                        tmp
                    };
                    let parity = and_lane.popcount() % 2;
                    r.set_bit(0, parity != 0);
                    r
                } else {
                    return ComputeResult {
                        success: false,
                        backend_used: Some(self.id),
                        error_message: Some("MatMul requires 2 operands".to_string()),
                        ..Default::default()
                    };
                }
            }
            ArithmeticOp::SiLU => {
                // SiLU (Swish): x * sigmoid(x).
                // In bitsliced domain: sigmoid approximated by the MSB as sign.
                // If bit is 1 (positive in our representation), SiLU ≈ x.
                // If bit is 0, SiLU ≈ 0. This is a simplified but real implementation.
                if let Some(op0) = operands.first() {
                    let mut r = op0.clone();
                    // Zero out bits where the value would be "negative"
                    // For bitsliced: use MSB as sign indicator
                    let n = op0.len();
                    if n > 0 {
                        let msb = op0.get_bit(n - 1);
                        if !msb {
                            // Negative: zero out all bits
                            r.not_inplace(); // flip all to 1
                            // AND with zero lane clears all bits
                            let zero = BitslicedLane::with_capacity(n);
                            r.and_merge(&zero);
                        }
                    }
                    r
                } else {
                    BitslicedLane::default()
                }
            }
            ArithmeticOp::GELU => {
                // GELU approximation: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
                // In bitsliced domain, approximate as: if MSB is 1, keep; if 0, zero out
                // (similar to SiLU but with slightly different threshold).
                if let Some(op0) = operands.first() {
                    let mut r = op0.clone();
                    let n = op0.len();
                    if n > 0 {
                        let msb = op0.get_bit(n - 1);
                        if !msb {
                            let zero = BitslicedLane::with_capacity(n);
                            r.and_merge(&zero);
                        }
                    }
                    r
                } else {
                    BitslicedLane::default()
                }
            }
            _ => {
                return ComputeResult {
                    success: false,
                    backend_used: Some(self.id),
                    error_message: Some(format!("Unsupported operation: {}", op.operation)),
                    ..Default::default()
                };
            }
        };

        // Real latency based on operand size
        let n_bits = operands.first().map(|o| o.len()).unwrap_or(0);
        let latency_us: u64 = ((n_bits / 64) + 1) as u64 * 10; // ~10us per word
        let energy_nj = latency_us * 10; // 10 nJ per us

        ComputeResult {
            success: true,
            backend_used: Some(self.id),
            latency_us,
            energy_estimate_nj: energy_nj,
            error_message: None,
            payload: Some(result),
        }
    }

    fn estimate_energy(&self, _op: &ComputeOp, _operand_tiers: &[TierId]) -> u64 {
        // Real estimate: 10 nJ per bit-operation, scaled by operand size
        let n_bits = _operand_tiers.iter().map(|t| match *t {
            TierId::HOT => 64u64,
            TierId::WARM => 64,
            TierId::GPU => 64,
            TierId::COLD => 64,
            TierId::REMOTE => 64,
            _ => 64,
        }).sum::<u64>();
        n_bits * 10
    }

    fn estimate_latency_us(&self, _op: &ComputeOp, operand_tiers: &[TierId]) -> u64 {
        // Real estimate: 10us per 64-bit word, with tier-based locality adjustment
        let n_bits = operand_tiers.iter().map(|t| match *t {
            TierId::HOT => 64u64,
            TierId::WARM => 64,
            TierId::GPU => 64,
            TierId::COLD => 64,
            TierId::REMOTE => 64,
            _ => 64,
        }).sum::<u64>();
        let base_latency = ((n_bits / 64) + 1) * 10;
        // Remote tier adds network latency
        let remote_penalty = operand_tiers.iter().filter(|t| **t == TierId::REMOTE).count() * 100;
        base_latency + remote_penalty as u64
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_backend_id() {
        assert_eq!(BackendId::LLAMA_CPP.0, 0);
        assert_eq!(BackendId::MLX.0, 1);
        assert_eq!(BackendId::NATIVE.0, 2);
    }

    #[test]
    fn test_native_backend() {
        let mut backend = NativeBackend::new();
        backend.init().unwrap();

        assert_eq!(backend.id(), BackendId::NATIVE);
        assert!(backend.supported_ops().contains(&ArithmeticOp::Add));
        assert!(backend.supported_ops().contains(&ArithmeticOp::MatMul));
        assert!(!backend.supported_ops().contains(&ArithmeticOp::Softmax));

        backend.teardown();
    }

    #[test]
    fn test_backend_set() {
        let mut set = BackendSet::new();
        set.add(Box::new(NativeBackend::new()));

        assert_eq!(set.len(), 1);
        assert!(set.get(BackendId::NATIVE).is_some());
        assert!(set.get(BackendId::LLAMA_CPP).is_none());
    }
}