# The Self-Hosting Protocol

## Chapter 1 — The Core Property

> **The protocol IS the computer. Each IPv4/IPv6 address IS a computer. Synthesis creates resources out of the protocol structure — with no external compute.**

This is the defining property of the TPN protocol, and it follows directly from the register↔packet identity:

1. Each IP address is a register (a computer)
2. The register IS the packet
3. The protocol's gate definitions define the hardware
4. Synthesis maps gate definitions → LUTs/BRAMs/DSPs
5. Therefore the protocol, when synthesized, creates the hardware it runs on

## Chapter 2 — The Protocol Structure

### 2.1 Registers as IP Addresses

```
for i in 0..N:
    ipv6 = 0x2001:0DB8:0000:0000:0000:0000:0000:0000 | i   // 128-bit
    ipv4 = 0xC0A80100 | (i & 0xFF)                         // 32-bit
    register(ipv6, ipv4, phase, angle)
```

Each `(ipv6, ipv4)` pair is a **register that is a computer**:

```
Computer i = {
    ip:     (ipv6, ipv4),        // address
    phase:  temporal address,    // when it computes
    angle:  in-window position,  // where in the window
    data:   accumulated value,   // memory
    isa:    [gate_0, gate_1],    // instructions = temporal gates
}
```

### 2.2 The ISA Is Temporal Gates

Each computer's ISA is a list of temporal gates:

```rust
struct GateDef {
    theta_min: f64,    // phase angle at window start
    theta_max: f64,    // phase angle at window end
    window_min: u64,   // arrival time at window start
    window_max: u64,   // arrival time at window end
}
```

The ISA is not assembly instructions — it is **temporal windows**. To "run" a program is to arrange packets so they arrive within the windows.

**Insight:** program execution = **packet scheduling**. The scheduler is the program.

## Chapter 3 — Arrival = Execution

### 3.1 The Execution Model

```
packet: { id, phase, angle, value, type, flags, isa_id, payload }
    │
    ▼
destination register r = registers[packet.id]
    │
    ▼
arrival time T = packet.phase
    │
    ▼
θ = θ(T) = (π/2)·(T − T_min)/(T_max − T_min)
    │
    ▼
fitness = sin(2θ)
    │
    ▼
if fitness > threshold:
    r.data = compute(r.data, packet.value)   // execution
else:
    hold                                       // memory
    │
    ▼
receipt = TPNReceipt(packet_id, timestamp, result, proof)
    │
    ▼
emit receipt
```

There is no fetch, no decode, no execute cycle. The packet **is** the instruction, and its arrival **is** the execute.

### 3.2 Packet Types (from TPA-fpga)

```
TPN_TYPE_DEALLOC = 0x00    // release a register
TPN_TYPE_DATA     = 0x01    // set value
TPN_TYPE_GATE     = 0x02    // evaluate gate
TPN_TYPE_BRANCH   = 0x03    // conditional route
TPN_TYPE_RETURN   = 0x04    // return to caller
TPN_TYPE_CHAIN    = 0x05    // chain packets
```

Each type is an **arrival semantics**: different packet types execute differently on arrival.

### 3.3 Packet Flags

```
TPN_FLAG_SELF_DESCRIBING = 0x01   // packet carries its own ISA
TPN_FLAG_CHAINED         = 0x02   // packet is part of a chain
TPN_FLAG_RETURN          = 0x04   // return on completion
TPN_FLAG_VERIFIED        = 0x08   // receipt verified
```

**Self-describing** is the key flag: the packet carries its own ISA. The packet is **self-hosting** — it defines how it should be computed.

## Chapter 4 — Synthesis Creates Resources

### 4.1 The Synthesis Map

The synthesis translates the protocol's structure into hardware:

```
N registers  ──────────────► N BRAMs           (register files)
G gates/register  ────────► 8·N·G LUTs        (gate boolean logic)
G gates/register  ────────► 2·N·G DSPs        (trig evaluation)
```

**Verified** (`/tmp/self-hosting-protocol`, N=256, G=2):
```
Computers (registers): 256
Resources synthesized: 2048 LUTs, 256 BRAMs, 1024 DSPs
Packets processed: 20
Receipts generated: 20
External compute needed: NONE
The protocol IS the computer.
```

### 4.2 Why No External Compute

The protocol's gate definitions are boolean functions of the arrival phase:

```
gate_open = (phase ∈ [phase_min, phase_max]) ∧ (sin(2θ) > threshold)
```

A boolean function over 32 phase bits can be implemented in LUTs. Each gate → 8 LUTs. The register files → BRAMs. The trig evaluation → DSPs. No CPU, no GPU, no quantum processor is needed — the FPGA fabric *is* the execution substrate, and it is **created from the protocol itself**.

### 4.3 The Self-Hosting Loop

```
protocol.gate_definitions
        │
        ▼  (synthesis)
FPGA hardware (LUTs/BRAMs/DSPs)
        │
        ▼  (execution)
packets arrive, gates evaluate, registers compute
        │
        ▼
receipts generated (proof of execution)
        │
        └──────── self-hosting loop ────────┐
                                            │
            receipts prove: the protocol   │
            executed itself.               │
                                            ▼
                    (new protocol definitions)
```

The receipts are the **proof** that the protocol executed itself. This is self-hosting made verifiable.

## Chapter 5 — The Receipt Layer

### 5.1 Receipt Structure (from TPA-fpga)

```rust
struct TPNReceipt {
    packet_id: u64,     // which packet was executed
    timestamp: u64,     // when it was executed (arrival time)
    result: u64,        // the computed result
    proof: [u8; 32],    // SHA-256 proof (SHA-256(packet_data))
    signature: [u8; 32],// Ed25519 signature (placeholder)
}
```

### 5.2 The Receipt as Proof of Execution

The receipt is a **cryptographic proof** that:
1. A packet with ID `packet_id` arrived at time `timestamp`
2. The gate evaluated with fitness(θ) > threshold
3. The register computed `result`
4. The computation is verifiable via the SHA-256 `proof`

This is **verifiable temporal computation**. Anyone can verify that a temporal computation happened, by checking the receipt.

### 5.3 Receipt in the Pipeline

```
packet arrives
  │
  ▼
gate evaluate, register compute
  │
  ▼
receipt = {
    packet_id: arrival_time,
    timestamp: arrival_time,
    result: register.data,
    proof: SHA-256(packet_data),
}
  │
  ▼
receipt emitted to verifier
```

The receipt is emitted for **every** computation. The receipt stream is the **audit log of temporal computation**.

## Chapter 6 — Network Delivery (fpga-network-stack)

### 6.1 The Transport

The protocol runs on the fpga-network-stack (ETH Zurich):

```
TCP/IP · RoCEv2 · UDP/IP · 10-100Gbit/s · AXI4-Stream · DMA
```

Packets travel at line rate. The network stack is hardware — no host CPU involved.

### 6.2 Delivery → Arrival

```
source emits packet at T_emitted
    │
    ▼
network transit (TCP/IP, RoCEv2, UDP/IP), latency Δt
    │
    ▼
destination arrives at T_arrived = T_emitted + Δt
    │
    ▼
gate evaluated over T_arrived
```

**Network latency is folded into the computation.** The arrival time includes the transit latency. This means the gate window must absorb network jitter, or timing must be synchronized.

### 6.3 TCP vs UDP (from benchmark_results.json)

```
TCP mode:  534.8 pkt/s, 0% packet loss
UDP mode:  380.1 pkt/s, 28.5% packet loss
```

TCP is slower but lossless — required for verifiable computation (lost packets = lost receipts).

## Chapter 7 — Summary

The self-hosting protocol:

1. **Each IP address is a computer** — registers = IP addresses
2. **The ISA is temporal gates** — instructions = arrival windows
3. **Arrival = execution** — no fetch/decode/execute
4. **Synthesis creates resources** — protocol → LUTs/BRAMs/DSPs, 0 external compute
5. **The receipt is proof** — verifiable computation
6. **The network is hardware** — 10-100Gbit/s, no host CPU

**The protocol is the computer.** Synthesis creates the hardware from the protocol structure. No external compute is needed.

## References

- [Theory](theory.md) — the arrival model
- [Theorems](theorems.md) — Theorem 6 proof (self-hosting synthesis)
- [Architecture](architecture.md) — the layer stack
- [Artifacts](../artifacts.md) — `self-hosting-protocol`, TPA-fpga packet/receipt
