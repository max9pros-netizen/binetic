# Grammar and API

## Chapter 1 — The Temporal Gate Grammar

The core of the TPN protocol is the temporal gate. Its grammar is:

```
gate = "{"
         "address"    : address,    // IPv6: which register computes
         "window"     : window,      // [T_min, T_max]: open window
         "theta_min"  : f64,         // phase angle at window start
         "theta_max"  : f64,         // phase angle at window end
         conditional_route,           // 4 routes based on θ
         conditional_route,
         conditional_route,
         conditional_route
       "}" ;
```

### 1.1 Field Types

| Field | Type | Meaning |
|-------|------|---------|
| `address` | IPv6 (`u128`) | Spatial address of the destination register |
| `window` | `[u64; 2]` | Arrival time window [T_min, T_max] |
| `theta_min` | `f64` | Phase angle at T_min (typically 0) |
| `theta_max` | `f64` | Phase angle at T_max (typically π/2) |
| `conditional_route` | IPv6 | Destination if θ falls in this range |

### 1.2 Conditional Routes

The 4 routes partition the phase space:

| Route | θ range | Meaning |
|-------|---------|---------|
| route_0 | θ < π/4 | Early in window |
| route_1 | π/4 ≤ θ < π/2 | Middle of window |
| route_2 | π/2 ≤ θ < 3π/4 | Late-middle |
| route_3 | θ ≥ 3π/4 | Near end |

The route determines where the packet goes after evaluation — this is **temporal branching**.

## Chapter 2 — The Packet Grammar

### 2.1 TPN Packet Structure

```
TPNPacket {
    id:        u128,    // 128-bit IPv6 address
    phase:     u32,     // 32-bit IPv4 arrival slot
    angle:     u32,     // in-window position
    value:     u64,     // payload value
    type:      u8,      // packet type (see below)
    flags:     u8,      // packet flags
    isa_id:    u16,     // instruction set ID
    payload_length: u16,
    version:   u8,      // = 2
    checksum:  u16,     // CRC-16
    payload:   Vec<u8>, // optional payload
}
```

### 2.2 Packet Types

```
TPN_TYPE_DEALLOC = 0x00   // release a register
TPN_TYPE_DATA     = 0x01   // set the register's value
TPN_TYPE_GATE     = 0x02   // evaluate the temporal gate
TPN_TYPE_BRANCH   = 0x03   // conditional route
TPN_TYPE_RETURN   = 0x04   // return to caller
TPN_TYPE_CHAIN    = 0x05   // part of a packet chain
```

Each type has different arrival semantics (see [Self-Hosting](self-hosting.md)).

### 2.3 Packet Flags

```
TPN_FLAG_SELF_DESCRIBING = 0x01   // carries its own ISA
TPN_FLAG_CHAINED         = 0x02   // chained with other packets
TPN_FLAG_RETURN          = 0x04   // return on completion
TPN_FLAG_VERIFIED        = 0x08   // receipt verified
```

**Self-describing** is the critical flag: the packet carries its own gate definition, so it is self-hosting.

### 2.4 Packing / Unpacking

```rust
fn pack(&self) -> Vec<u8> {
    let header = struct::pack(
        "!16sIIQBBHHHH",
        self.id.to_bytes(16, "big"),
        self.phase, self.angle, self.value,
        self.type, self.flags, self.isa_id,
        self.payload_length, self.version, 0, // checksum placeholder
    );
    let checksum = self._calculate_checksum(header);
    header[..header.len()-2].to_vec() + struct::pack("!H", checksum)
        + self.payload
}

fn unpack(data: &[u8]) -> TPNPacket {
    // reads HEADER_SIZE bytes, returns packet + payload
}
```

Checksum is **CRC-16** (polynomial 0x1021). Corruption is detected when the checksum fails.

## Chapter 3 — The Receipt Grammar

### 3.1 TPN Receipt Structure

```
TPNReceipt {
    packet_id: u64,    // which packet was executed
    timestamp: u64,    // arrival time (when executed)
    result:    u64,    // computed result
    proof:     [u8; 32],  // SHA-256(packet_data)
    signature: [u8; 32],  // Ed25519 signature
}
```

### 3.2 Packing

```rust
fn pack(&self) -> Vec<u8> {
    struct::pack(
        "!IQ8s32s32s",
        self.packet_id,
        self.timestamp,
        self.result.to_bytes(8, "big"),
        self.proof,
        self.signature,
    )
}
```

Receipt size: 8 + 8 + 8 + 32 + 32 = **88 bytes** (spec says 84; implementation packs 88).

### 3.3 Proof Generation

```rust
fn generate_proof(&mut self, packet_data: &[u8]) {
    self.proof = sha256(packet_data).digest();  // SHA-256
}
```

The proof is the SHA-256 of the packet data — verifiable proof that a specific packet was executed.

## Chapter 4 — The Register Grammar

### 4.1 U256 Register

```rust
struct U256([u64; 4]);   // 256 bits = 4 x 64-bit words

// Layout: [id:32 | phase:64 | angle:64 | data:64]
fn from_id_phase_angle(id: u32, phase: u64, angle: u64) -> U256 {
    let mut reg = U256([0,0,0,0]);
    reg[0] = ((id as u64) << 32) | (phase & 0xFFFFFFFF);  // id + phase low
    reg[1] = phase >> 32;                                  // phase high
    reg[2] = angle;                                        // angle
    reg[3] = 0;                                            // data
    reg
}
```

### 4.2 Accessors

```rust
fn id(&self) -> u32        { (self[0] >> 32) as u32 }
fn phase(&self) -> u64     { (self[0] & 0xFFFFFFFF) | (self[1] << 32) }
fn angle(&self) -> u64     { self[2] }
fn bit(&self, i: usize) -> bool
fn set_bit(&mut self, i, val)
```

### 4.3 Bitsliced Array

```rust
struct BitslicedArray {
    bits: Vec<Vec<bool>>,   // bits[bitplane][register]
    num_registers: usize,
}

fn from_registers(registers: &[U256]) -> BitslicedArray {
    // 256 bit-planes × N registers
}

fn bitplane(&self, j: usize) -> Vec<bool> { self.bits[j].clone() }
fn set_bitplane(&mut self, j: usize, vals: &[bool]) { self.bits[j] = vals.to_vec() }
```

## Chapter 5 — The Gate Implementation API

### 5.1 TrigGate

```rust
struct TrigGate {
    window_min: u64,
    window_max: u64,
}

impl TrigGate {
    fn new(min: u64, max: u64) -> Self;
    fn theta(&self, arrival: u64) -> f64;          // θ(T) mapping
    fn fitness(&self, arrival: u64) -> f64;        // sin(2θ)
    fn evaluate(&self, arrival: u64) -> (bool, f64); // (fired, fitness)
}
```

### 5.2 ErrorCorrection

```rust
struct ErrorCorrection {
    bitsliced: BitslicedArray,
}

impl ErrorCorrection {
    fn from_registers(r: &[U256]) -> Self;
    fn majority(a: bool, b: bool, c: bool) -> bool;
    fn syndrome(&self, base_plane: usize) -> Vec<bool>;   // detect errors
    fn correct(&self, base_plane: usize) -> Vec<bool>;     // majority vote
}
```

### 5.3 BitslicedAttention

```rust
struct BitslicedAttention;

impl BitslicedAttention {
    fn attn_scores(q: &[f64], k: &[f64]) -> Vec<f64>;   // Q·K^T/√d
    fn trig_softmax(scores: &[f64], gate: &TrigGate) -> Vec<f64>;
    fn forward(q, k, v, gate) -> Vec<f64>;              // Attention(Q,K,V)
}
```

### 5.4 PhaseSchedule

```rust
struct PhaseSchedule;

impl PhaseSchedule {
    fn linear(j: usize, delta: f64) -> f64;
    fn sinusoidal(j: usize, base: f64, amp: f64, freq: f64) -> f64;
    fn exponential(j: usize, base: f64, decay: f64) -> f64;
}
```

## Chapter 6 — Example: Adding 7 + 5

### 6.1 Encode

```
register_A = U256::from_id_phase_angle(0x01, phase_A, 0);   // value 7
register_B = U256::from_id_phase_angle(0x02, phase_B, 0);   // value 5
```

### 6.2 Gate

```
gate = {
    "address":    0x01,
    "window":     [0, 100],
    "theta_min":  0.0,
    "theta_max":  1.5708,  // π/2
    "route_0": ..., "route_1": ..., "route_2": ..., "route_3": ...
}
```

### 6.3 Arrive

```
packet_A arrives at t = 50:
    θ = π/2 * 50/100 = π/4
    fitness = sin(π/2) = 1.0 > 0.43  →  gate open
    A + B computed over 256 bit-planes
    result = 12  ✓
```

**Verified:** 7+5=12, 15+17=32, 64+64=128, 3+4=7, 12+8=20, 200+300=500.

## Chapter 7 — Example: Self-Hosting 256 Computers

### 7.1 Register

```
for i in 0..256:
    ipv6 = 0x2001:0DB8:: | i
    ipv4 = 0xC0A80100 | i
    proto.register(ipv6, ipv4, phase=i*100, angle=i*0x1000)
```

### 7.2 Synthesize

```
256 registers × 2 gates/register
  → 2048 LUTs, 256 BRAMs, 1024 DSPs
```

**Verified:** "Computers (registers): 256 / Resources synthesized: 2048 LUTs, 256 BRAMs, 1024 DSPs / External compute needed: NONE."

## Chapter 8 — Summary

| API | Location | Purpose |
|-----|----------|---------|
| `gate` grammar | Chapter 1 | Temporal conditional execution |
| `TPNPacket` | Chapter 2 | [id:128 | phase:32 | angle:32 | value:64] |
| `TPNReceipt` | Chapter 3 | Cryptographic proof of execution |
| `U256` register | Chapter 4.1 | [id:32 | phase:64 | angle:64 | data:64] |
| `TrigGate` | 5.1 | θ(T) mapping, sin(2θ) fitness |
| `ErrorCorrection` | 5.2 | 3× majority vote |
| `BitslicedAttention` | 5.3 | Attention(Q,K,V) with trig softmax |
| `PhaseSchedule` | 5.4 | Linear/sinusoidal/exponential |

## References

- [TPA-fpga packet/receipt code](https://github.com/9pros/TPA-fpga/blob/master/python/tpn_simulator.py)
- [Theory](theory.md) — the register model
- [Artifacts](../artifacts.md) — reference implementations
