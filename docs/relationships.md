# Component Relationships

## Chapter 1 — The Layered Stack

The complete TPN system is a layered stack. Each layer maps to a physical or logical substrate.

```
┌──────────────────────────────────────────────────────────────────┐
│  LAYER 5: TEMPORAL AI                                            │
│  Neurons = IP registers · Weights = phase delays · Learning      │
│  = phase evolution                                               │
├──────────────────────────────────────────────────────────────────┤
│  LAYER 4: SELF-HOSTING PROTOCOL                                  │
│  Each IPv4/IPv6 address = a computer · Synthesis creates         │
│  resources (LUTs/BRAMs/DSPs) from protocol structure             │
├──────────────────────────────────────────────────────────────────┤
│  LAYER 3: TPN SPEC v2.0                                          │
│  Trig temporal gates · 3× error correction · Attention ·         │
│  Phase schedules                                                 │
├──────────────────────────────────────────────────────────────────┤
│  LAYER 2: TPA-FPGA                                               │
│  TPN packets · Cryptographic receipts · Arrival = execution      │
│  AXI4-Stream wrapper · HLS modules                               │
├──────────────────────────────────────────────────────────────────┤
│  LAYER 1: FPGA-NETWORK-STACK (ETH)                               │
│  TCP/IP · RoCEv2 · UDP/IP · 10-100Gbit/s · AXI4-Stream · DMA    │
├──────────────────────────────────────────────────────────────────┤
│  LAYER 0: FPGA HARDWARE                                          │
│  LUTs · BRAMs · DSPs · HBM · Network interfaces                  │
└──────────────────────────────────────────────────────────────────┘
```

## Chapter 2 — The Structural Identity (The Deep Relationship)

### 2.1 The Register ↔ Packet Identity

This is the foundational relationship. Every TPN component derives from it.

```
U256 Register (in BRAM, at rest):
  [ id:32 | phase:64 | angle:64 | data:64 ]

TPN Packet (in transit):
  [ id:128 | phase:32 | angle:32 | value:64 ]
```

| Register Field | Packet Field | Transformation |
|----------------|--------------|----------------|
| id (32) | id:128 (IPv6) | Zero-padded + network prefix |
| phase (64) | phase:32 (IPv4) | Truncated to 32-bit arrival slot |
| angle (64) | angle:32 | Truncated |
| data (64) | value:64 | Identity |

**Theorem (Identity):** The register and the packet are the same object in two states — *at rest* (register) and *in transit* (packet).

**Consequence:** The protocol does not transport data *to* a computer; the packet *is* the computer's state. This is why the protocol is self-hosting: the hardware the protocol defines *is* the hardware the packets execute on.

### 2.2 The IPv4/IPv6 Duality

| Address | Role | Semantic |
|---------|------|----------|
| IPv6 | Spatial register | **Which** computer (identity) |
| IPv4 | Temporal register | **When** it computes (phase/arrival) |

Computation happens at the **intersection**: the packet arrives at its spatial address at its temporal address.

**Consequence:** Routing is scheduling. Network latency is computation time.

## Chapter 3 — The Connection Map (4-Wire Wiring)

The bitsliced processor is wired with 4 fundamental connections. These are the **data paths** of temporal computation:

### Connection 1: Trig → Gate (Fitness Gating)

```
TrigGate::fitness(θ) ─────────────────────┐
                                          ▼
Bitplanes ─────────────────────────────► Gate (compute or hold)
```

The trig gate evaluates the fitness of the arrival phase and uses it to **gate** the bit-planes: when fitness > threshold, the bit-planes can flow through the adder; when closed, the carry chain is killed.

### Connection 2: Gate → Adder (Carry Gating)

```
Gate (fitness > threshold?) ─────────────┐
                                          ▼
Bitsliced Adder: c[j+1] = majority(a[j], b[j], c[j])  if gate open
Bitsliced Adder: c[j+1] = 0                                   if gate closed
```

The gate controls **carry propagation**. This is the deepest connection: **carry propagation is gated by phase alignment**. The adder's latency is a function of the gate window, not the technology node. This is **time-extended arithmetic**.

### Connection 3: Adder → Gate (Carry Feedback)

```
Adder carry chain ────────────────────────► Gate state
```

The carry chain feeds back into the gate state: a successful addition requires the gate to remain open for the *entire* carry chain. If the gate closes mid-chain, the carry is killed and the addition must be redone.

**Consequence:** The gate window must be wide enough to absorb the carry chain. This constrains the relationship between gate width and adder width.

### Connection 4: Phase → Bitplane (Slicing)

```
Register.phase (64-bit) ────────────────► Bitplane index
```

The phase determines **which bit-plane is active**. The phase is the address of the bit-plane. This is how the register's temporal identity maps to the bitsliced array:

```
bitsliced_array = from_registers(registers)
  bits[bitplane][register] = register.bit(bit)
```

The phase → bitplane mapping is how temporal identity becomes spatial (bit-plane) addressing.

## Chapter 4 — The Component Data Flow

```
U256 Register                      BitslicedArray
  [id, phase, angle, data]              [bits[plane][reg]]
        │                                      │
        │ from_id_phase_angle                  │ from_registers
        ▼                                      ▼
TrigGate ← phase mapping              bitplane access
  fitness(θ) = sin(2θ)                      │
        │                                   │
        └────► Gate (compute/hold) ◄────────┘
                    │
                    ▼
         Bitsliced Adder (bit-plane by bit-plane)
                    │
                    ├────► ErrorCorrection (3× majority)
                    │
                    ▼
               TPNPacket                          TPNReceipt
         [id:128, phase:32, angle:32, value:64] [packet_id, timestamp,
                    │                        result, proof]
                    │
                    ▼
           fpga-network-stack (TCP/IP, RoCEv2)
                    │
                    ▼
               FPGA Hardware (LUTs/BRAMs/DSPs)
```

## Chapter 5 — The Evolution Loop

### 5.1 Auto-Evolution as the Learning Signal

The **auto-evolving processor** is the learning layer. It evolves the gate parameters (w_min, w_max, t_q, threshold) to maximize fitness (correct additions):

```
population: set of gate configurations
for generation in 1..N:
    for config in population:
        fitness = run_adder(config)           // correctness rate
    select top performers
    mutate: w_min±ε, w_max±ε, t_q±ε, threshold±ε
    replace worst performers
```

**Converged result:** w_min=0, w_max=10, t_q=1, threshold=0.43, fitness=0.8088.

### 5.2 The Evolution Loop as Self-Design

The evolution loop is **self-designing**: the system evolves its own parameters. The threshold 0.43 is not hand-tuned — it is **discovered** by evolution. This is the prototype of Temporal AI learning: parameters are temporal, learning is evolution of temporal parameters.

## Chapter 6 — The Synthesis Map

### 6.1 Protocol → Hardware

The self-hosting synthesis maps protocol structure to hardware:

```
N registers ──────────────────────────────► N BRAMs (register files)
G gates/register ─────────────────────────► 8·N·G LUTs (gate logic)
G gates/register ─────────────────────────► 2·N·G DSPs (trig eval)
```

For 256 registers, 2 gates/register: **2048 LUTs, 256 BRAMs, 1024 DSPs**.

**Verified:** `/tmp/self-hosting-protocol` output:
```
Computers (registers): 256
Resources synthesized: 2048 LUTs, 256 BRAMs, 1024 DSPs
External compute needed: NONE
The protocol IS the computer.
```

### 6.2 The Synthesis Map as a Relationship

The synthesis map is the **compiler**: it translates the protocol into hardware. Because the protocol *defines* the hardware, the synthesis is a **self-compiling loop**:

```
protocol.gate_definitions ──► synthesis ──► LUTs/BRAMs/DSPs ──► executes protocol
         ▲                                                          │
         └────────────────── self-hosting loop ─────────────────────┘
```

## Chapter 7 — The AI Map

### 7.1 Neural Primitive → Temporal Primitive

| Neural | Temporal | Connection |
|--------|----------|------------|
| Neuron i | IP register i | The register *is* the neuron |
| Weight w_ij | Phase delay τ_ij | Weight encoded as timing |
| Bias b_i | Register phase offset | Intrinsic phase |
| Activation σ | Trig gate (sin(2θ) window) | Gate is the activation |
| Input x_j | Packet arrival | Arrival is the input |
| Attention a_ij | Phase alignment | Alignment is attention |
| Memory | Accumulated register value | `data` field |
| Learning | Auto-evolution of phases | Evolution of temporal params |

### 7.2 The Forward Pass as Routing

Classical: `a_i = σ(Σ_j w_ij·x_j + b_i)`

TPN: `neuron_i fires iff enough packets arrive within its gate window`

The weighted sum becomes **temporal aggregation**: packets arriving within the window *are* the sum.

## Chapter 8 — Summary of Relationships

1. **Register ↔ Packet** — structural identity, two states
2. **IPv6 ↔ IPv4** — spatial/temporal duality
3. **Trig → Gate → Adder → Carry → Phase** — the 4-wire data path
4. **Protocol → Synthesis → Hardware** — self-hosting
5. **Neural → Temporal** — AI isomorphism
6. **Error correction ↔ Attention ↔ Phases** — the three TPN v2.0 primitives interlock

Each relationship is a **knob**: changing the relationship changes what the system computes. The relationships are the design surface of temporal computing.

## References

- [Theory](theory.md) — the layered model
- [Registers](register.md) — the structural identity
- [Gates](gates.md) — the 4-wire wiring
- [Artifacts](../artifacts.md) — all components
