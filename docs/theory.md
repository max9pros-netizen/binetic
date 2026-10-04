# The Theory of Temporal Packet Networks

## Chapter 1 — The Fundamental Premise

### 1.1 Computation as Temporal Alignment

The central thesis of the TPN model:

> **Computation is not something that happens at an address. Computation happens when signals align in time.**

In von Neumann computing, data is stored and fetched, and a clock tells the CPU *when* to operate. The clock is an external metronome, and data is passive. In TPN, **time itself is the data**. A packet carries no payload separate from its arrival time — the arrival *is* the operand, and the phase alignment *is* the operation.

This inverts the classical model:

```
Von Neumann:  data → [clock tells CPU to act] → result
TPN:          packet.arrives_at(T) → θ(T) → fitness(θ) → compute or not
```

### 1.2 Time as a Primitive, Not a Substrate

The TPN model explicitly rejects the assumption that computation requires a physical computational substrate (transistors, qubits, neurons) that *executes* logic. Instead:

- **Time is the substrate.** The phase angle θ is a real number in [0, π/2], continuous and exact.
- **Mathematics is the logic.** The gate is a trigonometric function evaluated over time.
- **Arrival is the trigger.** No clock cycle is needed — the arrival of the packet at its phase *is* the clock tick.

This is the paradigm Max described: change the nature of AI/computation by using **mathematics and time as the most important primitives**, not physical substrates like quantum computers.

### 1.3 The Three Primitive Operations

Everything in the TPN model reduces to three primitive operations:

1. **Encode** — map a value into a temporal address (IPv6 = identity, IPv4/phase = when)
2. **Route** — deliver the encoded value to its temporal register at its arrival time
3. **Evaluate** — the register's temporal gate fires iff the arrival phase is inside the gate window

Every computation, from addition to attention to backprop, is built from these three primitives.

## Chapter 2 — The Register as the Fundamental Unit

### 2.1 The U256 Temporal Register

```
┌────────────────────────────────────────────────────────┐
│  U256 Register:  [ id:32 | phase:64 | angle:64 | data:64 ]  │
├────────────────────────────────────────────────────────┤
│  id     — identity of the register (which computer)       │
│  phase  — temporal identity: when this register computes  │
│  angle  — position within the temporal window             │
│  data   — accumulated value (memory)                      │
└────────────────────────────────────────────────────────┘
```

The phase is a 64-bit integer in [0, 2⁶⁴). The angle is a 64-bit integer. This gives each register a **temporal address space of 2⁶⁴ × 2⁶⁴ possible temporal states** — enough that temporal collisions are astronomically unlikely.

### 2.2 The Register IS the Packet

The TPN packet structure mirrors the register exactly:

```
TPN Packet:  [ id:128 (IPv6) | phase:32 (IPv4) | angle:32 | value:64 ]
```

| Field | U256 Register | TPN Packet | Meaning |
|-------|--------------|------------|---------|
| id | id:32 | id:128 (IPv6) | Which register (spatial identity) |
| phase | phase:64 | phase:32 (IPv4) | When it computes (temporal identity) |
| angle | angle:64 | angle:32 | Position in the window |
| value/data | data:64 | value:64 | The accumulated value |

**Theorem (Identity):** The register and the packet are the same object in two states: *at rest* (register) and *in transit* (packet). This is the **Temporal/Spatial Duality**.

### 2.3 Temporal/Spatial Duality

The IPv4/IPv6 duality from the TPA-fpga spec is not a transport detail — it is the execution model:

- **IPv6 (128-bit id)** — spatial register: identifies *which* computer
- **IPv4 (32-bit phase)** — temporal register: identifies *when* it computes

A packet arrives at a **spatial address** (IPv6) at a **temporal address** (IPv4/phase). Computation happens at the **intersection** of the two. Routing is scheduling; latency is computation time.

## Chapter 3 — The Arrival Model

### 3.1 Arrival = Execution

```
     ┌─────────────┐
     │   Network   │  packet arrives at IPv6
     └──────┬──────┘
            ▼
     ┌─────────────┐   arrival time T → θ(T)
     │  Temporal   │   fitness = sin(2θ)
     │    Gate     │   if fitness > threshold: compute
     └──────┬──────┘
            ▼
     ┌─────────────┐
     │   Register  │   register.data = compute(register.data, input)
     └─────────────┘
```

There is no clock. There is no fetch. There is no decode. The packet's arrival *is* the clock tick; the arrival's phase *is* the instruction pointer; the register's value *is* the accumulator.

### 3.2 The Gate as Conditional Execution

A temporal gate is a conditional execution window:

```
gate = "{"
         "address"    : IPv6,
         "window"     : [T_min, T_max],
         "theta_min"  : f64,   // phase angle at window start
         "theta_max"  : f64,   // phase angle at window end
         "route_0"    : IPv6,  // if θ < π/4
         "route_1"    : IPv6,  // if π/4 ≤ θ < π/2
         ...
       "}"
```

When a packet arrives at `address` with arrival time `T`:
1. Map T → θ(T) via the theta mapping
2. Evaluate fitness(θ)
3. If fitness(θ) > threshold: **compute** — update the register
4. Else: **hold** — the register's value is preserved

The gate window [T_min, T_max] defines when the gate is "open." Outside the window, the gate is closed and the register holds its value (this is **memory**).

### 3.3 Fitness as the Computation Filter

The fitness function is the heart of TPN:

- It is a **filter**: determines whether computation happens
- It is **continuous**: smooth gradients for optimization
- It is **triangular**: 0 at window edges, 1 at center
- It is **self-normalizing**: always in [0, 1]

The corrected fitness `sin(2θ)` gives the triangular profile. (See [Theorems](theorems.md) for why `cos²(θ − π/4)` is wrong.)

## Chapter 4 — Bitsliced Phased Execution

### 4.1 The Bitsliced Array

A U256 register is 256 bits. In TPN, these bits are sliced into **bit-planes**, where each bit-plane represents a temporal slice:

```
bits[bit_plane][register_index] = bit_value

256 bit-planes × N registers
```

Each bit is replicated across **3 temporal bit-planes** for error correction. So 256 data bits → 768 physical bit-planes (256 data planes × 3 copies).

### 4.2 Bit-Planes as Temporal Slices

The bit-plane index encodes temporal order:

- bit-plane 0–255: the 256 data bits
- bit-planes 256–511: temporal copy 1
- bit-planes 512–767: temporal copy 2

Processing order across bit-planes is the **execution order**. The phase schedule determines *when* each bit-plane is processed. This is where **time drives computation** literally: bit-plane 0 is processed before bit-plane 1, which is processed before bit-plane 2, etc.

### 4.3 Carry Propagation Gated by Phase

In a bitsliced adder, the carry chain propagates from bit 0 to bit 255. In TPN, the carry propagation is **gated by phase**:

- The carry from bit-plane j is only available to bit-plane j+1 when the phase alignment allows it
- This means the adder's carry chain is *time-extended*: it completes over multiple phase slices rather than in one clock cycle
- **Second-order effect:** the adder's latency becomes a function of the gate window, not the technology node

This is a fundamental difference from classical bitsliced arithmetic: **carry propagation is temporal, not combinational.**

## Chapter 5 — The Synthesis Theorem

### 5.1 The Self-Hosting Property

The TPN protocol is **self-hosting**: the protocol's own structure defines the hardware it runs on.

Theorem (Self-Hosting Synthesis): Given a protocol with:
- N registers
- G gate definitions per register
- R register file capacity

The synthesized FPGA hardware contains:
- LUTs: 8 × N × G
- BRAMs: 1 × N (one register file per register)
- DSPs: 2 × G (one for each trig evaluation)

**Proof sketch:** Each gate is a boolean function over the phase bits → 8 LUTs. Each register is 256 bits of state → 1 BRAM. Each trig evaluation (sin(2θ)) requires a multiplier → 2 DSPs per gate.

### 5.2 No External Compute Needed

Because the protocol defines the hardware, **no external compute is needed to instantiate a computer.** The protocol's gate definitions, when synthesized, *create* the LUTs, BRAMs, and DSPs. This is the "create resources out of the synthesis" property:

```
protocol.gate_definitions → synthesis → LUTs/BRAMs/DSPs
```

No compiler. No binary. No runtime. The protocol *is* the hardware definition.

## Chapter 6 — Temporal AI as a Natural Consequence

### 6.1 The Neural Network Isomorphism

Every neural network primitive maps to a temporal operation:

| Neural Primitive | Temporal Realization |
|------------------|----------------------|
| Neuron | IP register |
| Weight w_ij | Phase delay τ_ij of the arriving packet |
| Bias b_i | Register's intrinsic phase offset |
| Activation σ(x) | Trig gate window (fitness > threshold) |
| Input x_i | Arrival time of the packet |
| Attention | Temporal proximity — neurons receiving at similar phases attend |
| Memory | Accumulated register value across arrivals |
| Learning | Auto-evolution of phase parameters |

### 6.2 Forward Pass as Routing

In a neural network, the forward pass computes:

```
a_i = σ(Σ_j w_ij · x_j + b_i)
```

In TPN, this is replaced by **routing**:

```
packet_j arrives at neuron_i after delay τ_ij → θ(T) → fitness(θ) > threshold → neuron fires
```

The weighted sum Σ_j w_ij · x_j becomes **temporal aggregation**: neuron_i fires iff enough packets arrive within its gate window. The weights are not multiplied — they are **encoded as delays**.

### 6.3 Attention as Temporal Proximity

The TPN attention mechanism (softmax over sin(2θ) fitness) becomes **attention**:

- Q, K, V are temporal patterns
- Attention scores are phase alignments
- softmax becomes the temporal distribution over which neurons fire
- Neurons that receive packets at aligned phases attend to each other

This is attention as **topology**, not as matrix multiplication.

### 6.4 Learning as Phase Evolution

Traditional learning updates numerical weights via gradient descent. TPN learning evolves **phases**:

```
phase_new = phase_old + ε · ∇fitness(phase)
```

Or via auto-evolution (genetic): mutate phase windows based on fitness, keep the best. The parameters are phases, so learning *is* phase-space exploration.

## Chapter 7 — Summary

The TPN theory rests on three core ideas:

1. **Time is a primitive.** θ is a real number; arrival is the trigger; no clock is needed.
2. **The register is the packet.** The boundary between data at rest and data in motion is erased; the protocol is the computer.
3. **The protocol synthesizes its own hardware.** Gate definitions → LUTs/BRAMs/DSPs; no external compute needed.

From these three ideas follow: bitsliced phased execution, temporal error correction, temporal attention, and temporal AI.

## References

- [Theorems & Proofs](theorems.md) — formal statements and proofs
- [Timing Semantics](timing.md) — the theta mapping and gate windows
- [Phases](phases.md) — phase schedules and bit-plane ordering
- [Component Relationships](relationships.md) — how the layers connect
