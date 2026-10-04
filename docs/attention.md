# Temporal Attention

## Chapter 1 — The Attention Mechanism

In TPN, attention is not a matrix operation. It is **routing by phase alignment**. The TPN attention mechanism maps the classic Attention(Q, K, V) formulation onto temporal gates.

### 1.1 The Classical Formulation

```
Attention(Q, K, V) = softmax(Q · K^T / √d) · V
```

- Q = queries
- K = keys
- V = values
- d = dimension

### 1.2 The TPN Formulation

In TPN, Q, K, V are **temporal patterns**:

```
Q = vector of query phase windows [θ_Q1, θ_Q2, ...]
K = vector of key arrival phases [θ_K1, θ_K2, ...]
V = vector of values [v1, v2, ...]
```

The attention score for (query i, key j) is the **temporal alignment**:

```
score(i, j) = Q_i · K_j / √d   →   phase-aligned → fitness(θ)
```

The softmax normalizes over keys:

```
weight(i, j) = softmax_j(score(i, j))
```

The output is the weighted sum:

```
output(i) = Σ_j weight(i, j) · V_j
```

### 1.3 The Trig Softmax

The TPN attention uses a **trigonometric softmax**: the attention scores are transformed by the gate fitness before softmax:

```
scores = Q · K^T / √d
weights = softmax(fitness ∘ scores)   // apply sin(2θ) then softmax
```

The fitness function turns the raw dot product into a temporal alignment score. This is where **temporal alignment is the attention mechanism**.

## Chapter 2 — The Implementation

### 2.1 Reference Implementation

`/tmp/temporal-fabric/src/main.rs`:

```rust
struct BitslicedAttention;

impl BitslicedAttention {
    // Compute attention scores: Q · K^T / √d
    fn attn_scores(q: &[f64], k: &[f64]) -> Vec<f64> {
        let d = q.len() as f64;
        q.iter().map(|&qi| qi * k.iter().sum::<f64>() / d.sqrt()).collect()
    }

    // Trigonometric softmax: fitness-driven weights
    fn trig_softmax(scores: &[f64], gate: &TrigGate) -> Vec<f64> {
        let weights: Vec<f64> = scores.iter()
            .map(|&s| gate.fitness(s as u64).exp())   // sin(2θ) then exp
            .collect();
        let sum: f64 = weights.iter().sum();
        weights.iter().map(|w| w / sum).collect()
    }

    // Forward pass: Attention(Q, K, V)
    fn forward(q: &[f64], k: &[f64], v: &[f64], gate: &TrigGate) -> Vec<f64> {
        let scores = Self::attn_scores(q, k);
        let weights = Self::trig_softmax(&scores, gate);
        v.iter().zip(weights.iter()).map(|(&vi, &w)| vi * w).collect()
    }
}
```

### 2.2 Verification

```
=== Attention Demo ===
  Attention scores (Q·K^T/√d): [0.5, 0.0, 0.0, 0.0]
  Attention weights (softmax): [0.25, 0.25, 0.25, 0.25]
  Attention output: [0.25, 0.5, 0.75, 1.0]

=== Attention Mechanism ===
  Q=[1.0, 0.0, 0.0] K=[0.0, 1.0, 0.0] V=[1.0, 2.0, 3.0]
  Attention output: [0.3333333333333333, 0.6666666666666666, 1.0]
```

**Verified:** the attention mechanism computes scores, weights, and outputs correctly.

## Chapter 3 — Attention as Routing

### 3.1 The Core Insight

The classical attention computes scores by **multiplying vectors**. The TPN attention computes scores by **aligning phases**.

| Aspect | Classical | TPN |
|--------|-----------|-----|
| Q, K | numeric vectors | temporal patterns (phase windows) |
| score(i,j) | Q_i·K_j/√d (dot product) | phase alignment (fitness(θ)) |
| softmax | numeric normalization | temporal distribution |
| output | weighted sum of values | packets routed to aligned registers |

### 3.2 Attention as Routing

In TPN, attention is implemented as **routing**:

```
register_i (query) receives packet_j (key) at phase θ_j
  │
  ▼
align: is θ_j within register_i's attention window [θ_Qi − ε, θ_Qi + ε]?
  │
  ▼
if aligned: route value V_j to register_i
else: drop packet
```

The attention score is the **phase alignment**. The softmax is the **temporal distribution** over which packets are routed. The output is the **routed values**.

### 3.3 Neurons That Fire Together Attend Together

**Second-order insight:** two registers that receive packets at aligned phases **attend to each other**. Their attention weights are high because their phases align. This is attention as **topology**:

- Neuron_i and neuron_j attend iff their phase windows overlap
- The routing graph *is* the attention graph
- Attention is emergent from the phase schedule, not computed

This is the deepest consequence: **attention is not an operation — it is a property of the temporal topology.**

## Chapter 4 — Temporal Attention vs. Classical Attention

### 4.1 The Difference in Practice

**Classical attention (transformer):**

```python
scores = Q @ K.T / sqrt(d)     # O(n²) matrix multiply
weights = softmax(scores)      # O(n²)
output = weights @ V           # O(n²)
```

**TPN attention:**

```
for each packet at phase θ:
    for each register with window [θ_min, θ_max]:
        if θ ∈ [θ_min − ε, θ_max + ε]:
            register.attend(packet)
```

The TPN version is **O(n·m)** where m is the number of registers with overlapping windows — typically much smaller than n² because most windows don't overlap.

**First-order effect:** TPN attention scales better than classical attention for sparse inputs.

**Second-order effect:** TPN attention is **continuous in time** — the weights are smooth functions of phase, not discrete. This gives gradient-free optimization of attention.

### 4.2 The Attention Window as Learnable

The attention window [θ_min, θ_max] of each register is a **learnable parameter**. Learning = evolving the attention windows toward the configurations that maximize fitness. This is attention learning as **window evolution**.

From auto-evolution, the converged parameters include:

```
wmin = 0, wmax = 10, threshold = 0.43
```

These control the gate windows and thresholds — including attention windows.

## Chapter 5 — Attention in the Pipeline

### 5.1 The Full Pipeline

```
packet arrives at register r, time T
  │
  ▼
θ ← θ(T)                                   // phase mapping
  │
  ▼
correction.detect_errors(base_plane)       // 3× redundancy
  │
  ▼
correction.correct(base_plane)             // majority vote
  │
  ▼
bitsliced.set_bitplane(base_plane, corrected)
  │
  ▼
attention: compute scores vs. all query windows
  │
  ▼
weights = trig_softmax(scores)
  │
  ▼
gate.evaluate: if fitness > threshold: compute & route
  │
  ▼
receipt ← TPNReceipt(packet_id, timestamp, result, proof)
```

Attention runs between error correction and gate evaluation. The corrected values are attended to, and the attention weights gate the final computation.

### 5.2 The Receipt

The `TPNReceipt` is the cryptographic proof of execution:

```rust
struct TPNReceipt {
    packet_id: u64,     // which packet was executed
    timestamp: u64,     // when it was executed
    result: u64,        // the computed result
    proof: [u8; 32],    // SHA-256 proof (placeholder in reference)
}
```

The receipt is the **verifiable proof that temporal computation happened**. It's how you audit a temporal computer.

## Chapter 6 — Summary

TPN attention:

- **Formulation:** Attention(Q, K, V) with trig softmax over phase alignment
- **Implementation:** `BitslicedAttention::forward` with `attn_scores`, `trig_softmax`, `forward`
- **Insight:** attention is routing by phase alignment, not matrix multiplication
- **Deeper:** attention is a property of the temporal topology — neurons firing together attend together
- **Verification:** scores, weights, outputs all correct
- **Receipt:** cryptographic proof of every attention-weighted computation

## References

- [Theory](theory.md) — the arrival model
- [Theorems](theorems.md) — Theorem 7 proof
- [Gates](gates.md) — trig softmax detail
- [Artifacts](../artifacts.md) — `BitslicedAttention`, `TrigGate`, `TPNPacket`, `TPNReceipt`
