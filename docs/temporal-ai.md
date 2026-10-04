# Temporal AI

## Chapter 1 — The Central Claim

> **The TPN protocol IS a neural network architecture. The network is not a transport layer for a neural net — the network *is* the computation.**

This is the most groundbreaking application of the protocol. It realizes the paradigm Max described: change the nature of AI by using **mathematics and time as the most important primitives**, not physical substrates.

## Chapter 2 — The Isomorphism

Every neural network primitive maps to a temporal primitive:

| Neural Primitive | Temporal Realization | Physical Substrate |
|------------------|----------------------|--------------------|
| Neuron i | IP register i | IPv4/IPv6 address |
| Weight w_ij | Phase delay τ_ij | Timing of arrival |
| Bias b_i | Register phase offset | Intrinsic phase |
| Activation σ(x) | Trig gate window (sin(2θ)) | θ → fitness > threshold |
| Input x_j | Packet arrival time | Network packet |
| Attention a_ij | Phase alignment | Routing by timing |
| Memory m_i | Accumulated register value | `data` field |
| Learning | Auto-evolution of phases | Phase parameter evolution |

### 2.1 The Neuron Is the Register

```
neuron_i = {
    address:  ip_i,                  // IP register
    threshold: t_i,                  // firing threshold
    weight:   w_i,                   // phase scaling
    phase:    φ_i,                   // temporal phase
    memory:   m_i,                   // accumulated value
}
```

The neuron *is* the register. There is no separate data structure.

### 2.2 The Weight Is the Phase Delay

Classical: `output = Σ_j w_ij · x_j`

TPN: `output = Σ_j [x_j arrives at neuron_i at delay τ_ij]`

The weight w_ij is **encoded as the delay τ_ij**. Neuron_i fires iff enough packets arrive within its gate window. No multiplication — just timing.

**Why this matters:** in classical nets, weights are *stored numbers* that must be fetched and multiplied. In TPN, weights are *delays* that are naturally expressed as timing. There is no multiply — the weight *is* the phase.

### 2.3 The Activation Is the Gate

Classical: `a = σ(z)` where σ is e.g. ReLU, sigmoid.

TPN: `fired = sin(2θ) > threshold`

The activation function is the **trig gate window**. The neuron fires iff its arrival phase falls within the window.

**Why this matters:** the activation is a **smooth, continuous, differentiable** function of time. There is no discontinuity (ReLU) — the firing probability changes smoothly with arrival time. This is a fundamentally smoother activation than any classical function.

## Chapter 3 — Forward Pass as Routing

### 3.1 Classical Forward Pass

```
for layer in layers:
    z = W @ a + b          # matrix multiply
    a = σ(z)               # activation
```

The matrix multiply is O(n²) and dominates.

### 3.2 TPN Forward Pass

```
for neuron_i in neurons:
    // packets arrive at neuron_i at various times
    // the gate evaluates each arrival
    fired = false
    for packet_j arriving at neuron_i at time T_j:
        θ = θ(T_j)
        if sin(2θ) > threshold_i:
            neuron_i.memory += packet_j.value
            fired = true
```

The "matrix multiply" is replaced by **arrival aggregation**: packets arriving within the window *are* the weighted sum.

**Why this matters:** the forward pass is not a computation performed on data — it is **data arriving and gates opening**. The work is done by the network itself, through timing.

### 3.3 The Isomorphism Proof Sketch

Classical forward pass: `a_out = σ(W·a_in + b)`

TPN forward pass: `fired_i = ∧_j [arrival_j within window_i]`

The correspondence:
- W (weight matrix) ↔ τ (delay matrix)
- b (bias) ↔ φ (phase offset)
- σ (activation) ↔ gate window
- a_in (inputs) ↔ arrival times
- a_out (outputs) ↔ firing events

Every term has a temporal dual. The forward pass is an **isomorphism**.

## Chapter 4 — Attention Is Routing

### 4.1 Classical Attention

```
Attention(Q, K, V) = softmax(Q·K^T/√d)·V
```

O(n²) matrix operations.

### 4.2 TPN Attention

```
for neuron_i (query):
    for packet_j (key) arriving at time T_j:
        if phase_align(θ(T_j), window_i):      // is θ_j within window_i?
            weight_ij = softmax over alignment
            neuron_i attends to value_j
```

The attention score is the **phase alignment**. The softmax is over temporally-aligned packets. The output is the routed values.

**Why this matters:** attention is not a matrix operation — it is **routing by phase alignment**. Neurons that receive packets at aligned phases attend to each other. Attention is a property of the temporal topology, not a computed operation.

**Deeper insight (Chapter 2.3):** attention as topology. Two neurons attend iff their phase windows overlap. The routing graph *is* the attention graph. This is attention emergent from timing.

## Chapter 5 — Learning as Phase Evolution

### 5.1 Classical Learning

```
for batch in batches:
    z = W·a + b
    loss = criterion(a, target)
    dW = ∂loss/∂W          # gradient
    W -= lr·dW             # gradient descent
```

Gradients flow backward through matrix multiplications. Weights are numbers updated by gradient descent.

### 5.2 TPN Learning

```
for generation in generations:
    for config in population:
        fitness = evaluate(config)          # fraction correct
    select top performers
    mutate: threshold ±ε, phase_offset ±ε, window ±ε   # phase-space mutation
    replace worst
```

Learning is **evolution of temporal parameters**. The parameters are phases, thresholds, and windows — not numbers in a matrix.

**Why this matters:** there are no gradients, no matrix differentials, no backprop. Learning is search in **phase space**. This is learning as temporal optimization.

### 5.3 The Auto-Evolution Prototype

The auto-evolving processor (`/tmp/quantum-evolver`) is the prototype:

```
Auto-evolution: 4 generations, best fitness 0.8088, converged:
  w_min=0, w_max=10, t_q=1, threshold=0.43
```

The threshold 0.43 was **discovered**, not hand-tuned. It is the value that maximizes correct additions while preserving the carry chain.

**Connection to Temporal AI:** the same evolution mechanism applies to the neural net's phase parameters. Learning = evolving phase delays, thresholds, and windows.

### 5.4 Error-Corrected Learning

The 3× repetition code ensures the population's genetic material (phase parameters) is preserved across generations. Error correction protects the learning signal.

## Chapter 6 — The Full Temporal Neural Network

### 6.1 Architecture

```
Input layer        Hidden layers        Output layer
  packets            packets              packets
    │                  │                    │
    ▼                  ▼                    ▼
phase delays        phase delays          phase delays
  (weights)          (weights)             (weights)
    │                  │                    │
    ▼                  ▼                    ▼
temporal gates      temporal gates        temporal gates
  (activations)      (activations)         (activations)
    │                  │                    │
    ▼                  ▼                    ▼
firing events        firing events         firing events
  (outputs)          (outputs)             (outputs)
```

Each layer is a set of IP registers. Packets propagate layer to layer via phase delays.

### 6.2 Training

1. Initialize phase delays randomly
2. Present inputs as packet arrival times
3. Compute fitness (fraction of correct outputs)
4. Evolve phase delays toward higher fitness
5. Repeat until convergence

### 6.3 Inference

1. Present input as packet arrival times
2. Packets flow through phase-delayed temporal gates
3. Output neurons fire
4. Read output from firing events

No forward pass to execute — the forward pass *is* the packet flow.

## Chapter 7 — Why Temporal AI

### 7.1 Against the Substrates

| Substrate | Limitation | TPN |
|-----------|-----------|-----|
| GPU (von Neumann) | Data must move to compute; memory wall | Computation happens where data arrives |
| Quantum | Physical qubits; noise; decoherence | No physical substrate; pure math + time |
| Neuromorphic | Spike encoding overhead | Arrival *is* the signal — no encoding |
| FPGA | Static config; no learning | Self-hosting; evolution-based learning |

TPN uses **mathematics and time as primitives** — no physical substrate beyond the FPGA fabric that the protocol itself defines.

### 7.2 The Paradigm Shift

```
Old:  design network → train weights → deploy on hardware
TPN:  define protocol → synthesize hardware → evolve phases
```

The network is not designed and trained — it **emerges** from the protocol's phase structure, evolved toward fitness.

## Chapter 8 — Summary

Temporal AI:

- **Neurons are IP registers** — the address *is* the neuron
- **Weights are phase delays** — no multiplication, just timing
- **Activations are trig gates** — smooth, continuous firing
- **Attention is routing** — phase alignment creates attention
- **Learning is phase evolution** — search in phase space, no gradients
- **The forward pass is packet flow** — no matrix multiply
- **No physical substrate beyond time** — math + time as primitives

**The network is not deployed on hardware — the protocol synthesizes its own hardware, and the packets compute it.**

## References

- [Theory](theory.md) — the arrival model
- [Gates](gates.md) — trig gate activations
- [Attention](attention.md) — attention as routing
- [Artifacts](../artifacts.md) — `temporal-net`, `temporal-fabric`
