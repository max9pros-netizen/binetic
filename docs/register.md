# Register Encoding

## Chapter 1 — The U256 Temporal Register

The U256 register is the fundamental unit of computation in TPN. It encodes **identity, temporal position, and data** in a single 256-bit word.

### 1.1 Layout

```
┌────────────────────────────────────────────────────────────────┐
│   U256 = [ id:32 | phase:64 | angle:64 | data:64 ]             │
├────────────┬───────────────┬───────────────┬──────────────────┤
│  bits 0–31 │   bits 32–95  │   bits 96–159 │    bits 160–255  │
│            │               │               │                  │
│    id      │     phase     │     angle     │      data        │
│  (identity)(when to compute)(in-window   │   (value/          │
│            │               │    position)  │    memory)       │
└────────────┴───────────────┴───────────────┴──────────────────┘
```

### 1.2 Field Semantics

**id (32 bits)** — The spatial identity. Which computer is this? `u32` in [0, 2³²). In the network, this is the lower 32 bits of the IPv6 address.

**phase (64 bits)** — The temporal identity. When does this register compute? `u64` in [0, 2⁶⁴). This is the IPv4 address (lower 32 bits) plus the upper phase. The phase determines the arrival time T, which maps to the phase angle θ(T).

**angle (64 bits)** — The position within the temporal window. Given a gate window [T_min, T_max], the angle encodes where in that window this register's computation should occur. `angle` is the residual after phase resolution.

**data (64 bits)** — The accumulated value. Memory. This is what computation produces and consumes.

### 1.3 Construction

```rust
fn from_id_phase_angle(id: u32, phase: u64, angle: u64) -> U256 {
    let mut reg = U256::zero();
    reg.0[0] = ((id as u64) << 32) | (phase & 0xFFFFFFFF);   // id + phase low
    reg.0[1] = phase >> 32;                                  // phase high
    reg.0[2] = angle;                                        // angle
    // reg.0[3] = 0;  // data — loaded on arrival
    reg
}
```

### 1.4 Accessors

```rust
fn id(&self) -> u32        { (self.0[0] >> 32) as u32 }
fn phase(&self) -> u64     { (self.0[0] & 0xFFFFFFFF) | (self.0[1] << 32) }
fn angle(&self) -> u64     { self.0[2] }
```

## Chapter 2 — The Phase as Temporal Address

### 2.1 The Phase → Time Mapping

The 64-bit phase encodes an absolute arrival time T:

```
T(phase) = phase · τ
```

where τ is the fundamental time quantum (the smallest resolvable time step). With phase in [0, 2⁶⁴), the temporal address space spans 2⁶⁴ quanta.

### 2.2 The Phase → Angle Mapping

Given a gate window [θ_min, θ_max], the angle is computed by normalizing the phase within the window:

```
θ(phase) = θ_min + (phase - phase_min) / (phase_max - phase_min) · (θ_max - θ_min)
```

For a standard gate [T_min, T_max] → [0, π/2]:

```
θ(T) = (π/2) · (T - T_min) / (T_max - T_min)    for T_min ≤ T ≤ T_max
θ(T) = - (π/2) · (T_min - T) / (T_max - T_min)  for T < T_min
θ(T) = π/2 + (π/2) · (T - T_max) / (T_max - T_min) for T > T_max
```

### 2.3 The Angle as In-Window Position

The angle is the *residual* temporal position: once the phase resolves to a time T within [T_min, T_max], the angle specifies where in that window the computation should occur. This allows **sub-window scheduling**: multiple packets arriving at the same time can still be ordered by angle.

**Second-order insight:** the angle is how we get **parallelism within a single arrival**. Two packets with the same phase (same arrival) but different angles are processed in different temporal slots — this is the origin of time-sliced concurrency.

## Chapter 3 — The Register as Network Address

### 3.1 IPv6 = Spatial Identity

The register's `id` field is the lower 32 bits of the IPv6 address. The full 128-bit IPv6 address:

```
IPv6 = 0x2001:0DB8:0000:0000:0000:0000:0000:0000 | id
       ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^  ^
       network prefix                         id (register identity)
```

The prefix `2001:DB8::/32` is the documentation/routing prefix used by TPA-fpga. The remaining 96 bits are reserved; the id occupies the low 32.

### 3.2 IPv4 = Temporal Identity

The register's phase is the IPv4 address:

```
IPv4 = 0xC0A80100 | (phase & 0xFFFFFFFF)
       ^^^^^^^^    ^^^^^^^^^^^^^^^^^^^^^
       192.168.1.x   phase (arrival time)
```

The IPv4 phase field is 32 bits, so the *temporal address space* in the network is 2³² arrival slots. This matches the TPN gate window: with [T_min, T_max] spanning 2³² time slots, each IPv4 address is a unique arrival time.

### 3.3 The Dual Addressing

| Address | Meaning | Field |
|---------|---------|-------|
| IPv6 | **Which** register (spatial) | id |
| IPv4 | **When** it computes (temporal) | phase |

A packet is addressed to `IPv6:IPv4` = "register id, arriving at time phase." Computation happens when the packet reaches its address *and* its phase window.

## Chapter 4 — The Register IS the Packet

### 4.1 Structural Identity

```
U256 Register:  [ id:32 | phase:64 | angle:64 | data:64 ]
TPN Packet:     [ id:128 | phase:32 | angle:32 | value:64 ]
```

| Register Field | Packet Field | Bit width |
|----------------|--------------|-----------|
| id | id (IPv6) | 32 → 128 |
| phase | phase (IPv4) | 64 → 32 |
| angle | angle | 64 → 32 |
| data | value | 64 → 64 |

The data field is preserved exactly (64 bits). The id, phase, and angle fields are the same semantic values, just widened for network routing.

### 4.2 The Two States

A register exists in two states:

1. **At rest** — stored in a BRAM, holding accumulated value. This is the `U256 Register`.
2. **In transit** — transmitted over the network as a `TPN Packet`.

The packet is a **frozen snapshot** of a register: the same fields, carried to the destination. The destination register *unfreezes* the packet: reads the fields, evaluates the gate, updates the value.

### 4.3 The Implication

> **The protocol does not transport data to a computer. The protocol IS the computer.**

When a packet arrives, it is not "processed by" a computer — it **becomes** the computer's current state. The register at the destination address *is* the packet. This is why the TPN model can be self-hosting: the hardware the protocol defines *is* the hardware the packets execute on.

## Chapter 5 — Registers as Neurons (Temporal AI)

### 5.1 Each Register Is a Neuron

In the temporal AI model, every register is a neuron:

```
register = {
    id:     neuron_address,      // IP address
    phase:  firing_time,         // when the neuron fires
    angle:  firing_position,     // sub-window position
    data:   membrane_potential,  // accumulated input
}
```

The `data` field is the neuron's membrane potential. The `phase` is the spike time. The gate is the activation function.

### 5.2 Firing as Gate Evaluation

A neuron fires iff its temporal gate opens:

```
θ = θ(arrival_time)
fitness = sin(2θ)
if fitness > threshold:
    neuron fires → propagate to connected registers
else:
    neuron holds → no output
```

### 5.3 The Weight Is the Phase Delay

Traditional neural net: `output = σ(Σ w_ij · x_j + b_i)`

TPN neural net: `output = Σ [x_j arrives at neuron_i at time τ_ij]`

The weight w_ij is **encoded as the delay τ_ij**. Neuron_i fires iff enough packets arrive within its gate window. No multiplication — just timing.

## Chapter 6 — Verified Implementation

The register implementation is in `/tmp/temporal-fabric/src/main.rs` and `/tmp/tpn-simulator/src/main.rs`:

```rust
#[derive(Clone, Copy, Debug)]
struct U256([u64; 4]);

impl U256 {
    fn from_id_phase_angle(id: u32, phase: u64, angle: u64) -> Self { ... }
    fn id(&self) -> u32        { (self.0[0] >> 32) as u32 }
    fn phase(&self) -> u64     { (self.0[0] & 0xFFFFFFFF) | (self.0[1] << 32) }
    fn angle(&self) -> u64     { self.0[2] }
    fn bit(&self, i: usize) -> bool { (self.0[i / 64] >> (i % 64)) & 1 == 1 }
    fn set_bit(&mut self, i: usize, val: bool) { ... }
}
```

## References

- [Theory](theory.md) — the arrival model, the duality
- [Theorems](theorems.md) — register encoding theorems
- [Timing](timing.md) — the phase → time mapping
- [Artifacts](../artifacts.md) — verified code
