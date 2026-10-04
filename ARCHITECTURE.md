# Binetic — Architecture Specification

**Version:** 0.1.0  
**Status:** Architecture Definition  
**Target:** Local AI inference fabric for constrained hardware (8–16GB unified memory devices, scaling to clusters)  
**Language:** Rust (core fabric), C/Obj-C/Swift FFI (macOS backends), C API (llama.cpp, MLX integration)

---

## Table of Contents

1. [Vision & Problem Statement](#1-vision--problem-statement)
2. [Core Conceptual Model](#2-core-conceptual-model)
3. [Register Addressing & Geometry](#3-register-addressing--geometry)
4. [The Two Interlaced Planes](#4-the-two-interlaced-planes)
5. [The 4D Sphere Structure](#5-the-4d-sphere-structure)
6. [Rotation & Scheduling](#6-rotation--scheduling)
7. [XOR Operations & Lineage](#7-xor-operations--lineage)
8. [Echo Minimization](#8-echo-minimization)
9. [Self-Describing Registers](#9-self-describing-registers)
10. [Bloom Filters & Bitsliced Lanes](#10-bloom-filters--bitsliced-lanes)
11. [Storage Tiers & HardwareMapping](#11-storage-tiers--hardware-mapping)
12. [No-Copy Compute-to-Data Routing](#12-no-copy-compute-to-data-routing)
13. [Self-Optimizing Layout](#13-self-optimizing-layout)
14. [Energy-Aware Scheduling](#14-energy-aware-scheduling)
15. [Cross-Model XOR Deltas](#15-cross-model-xor-deltas)
16. [State Integrity & Provable Correctness](#16-state-integrity--provable-correctness)
17. [On-Device Adaptation](#17-on-device-adaptation)
18. [Backend Integration](#18-backend-integration)
19. [API Surface](#19-api-surface)
20. [Implementation Phases](#20-implementation-phases)
21. [Open Questions & Trade-offs](#21-open-questions--trade-offs)

---

## 1. Vision & Problem Statement

### 1.1 The Problem

Local AI inference on consumer hardware faces a fundamental tension:

| Constraint | Effect |
|---|---|
| RAM smaller than model | Model doesn't fit; must offload, quantize, or truncate |
| KV cache grows with context | Long contexts exhaust RAM; context window limited |
| Memory bandwidth is the bottleneck | Moving weights/activations between CPU/GPU/SSD kills throughput |
| Energy on battery-powered devices | Frequent wakeups, data movement, and redundant computation drain battery |
| Multiple models or variants | Each model duplicated in memory; effective capacity shrinks |
| Heterogeneous hardware | Different backends (llama.cpp, MLX, etc.) need separate integration |

Existing inference engines (llama.cpp, MLX, vLLM, etc.) solve these piecewise — separate allocators for weights vs. KV cache, static memory layouts, manual offloading, per-backend integration. They don't share a unified model of memory and compute.

### 1.2 The Vision

Binetic is a **self-describing, spatially-organized computation fabric** that unifies memory and compute into a single addressable 4D structure, where:

- **Every register's address encodes its position and lineage** — routing and lookup are implicit, not table-driven.
- **Data and control are separate planes** (IPv6 for payload, IPv4 for routing) — clean separation of concerns.
- **Propagation is minimal** — changes travel as deltas, damped by rotation, gated by bloom filters, canceled by XOR folding.
- **Compute routes to data, not data to compute** — no unnecessary copies.
- **The fabric is self-optimizing** — it learns access patterns and reorganizes for locality.
- **Energy is a first-class scheduling concern** — batched, rotation-aware, minimal propagation.

The result: an inference fabric that behaves like *more RAM than physically exists*, runs models larger than memory on consumer hardware, and scales from an 8GB Mac to a multi-GPU cluster with the same API.

### 1.3 Design Principles

1. **Address as geometry** — the address is the lookup key, the lineage, and the routing hint.
2. **XOR as the native operation** — for lineage, fold, diff, routing distance, and echo cancellation.
3. **Echo minimization as propagation policy** — changes propagate only as far as necessary.
4. **Rotation as scheduling** — the geometric rotation gradient maps directly to cache behavior and energy management.
5. **Self-description everywhere** — registers carry their own metadata; the fabric needs no external schema.
6. **Graceful degradation** — same binary, same API, across hardware tiers.
7. **Backend-agnostic core** — the fabric is the layer *above* llama.cpp, MLX, etc.; backends are plug-ins.

---

## 2. Core Conceptual Model

### 2.1 The Network Metaphor (Literal and Conceptual)

The fabric models the local machine as a **network of registers**, where:

- Each register is a node addressable like an IP endpoint.
- The **IPv4 control plane** carries routing, scheduling, and metadata.
- The **IPv6 data plane** carries payload, lineage, and computation state.
- The **network is local** — registers communicate within the same machine, but the model extends to cross-process and cross-machine (future).

This is *not* literally sending UDP packets per register access (that would be too slow). It is a **conceptual model** that produces a clean architecture:

- Routing is a first-class concern, not an afterthought.
- Control and data are separate, so you can change one without the other.
- The fabric can span processes, devices, and nodes using the same abstractions.

### 2.2 The Sphere as Memory Hierarchy

The sphere is a **4D geometric model of a multi-level memory hierarchy**:

```
                    Outer Layer (Cold)
    ┌─────────────────────────────────────┐
    │  SSD / NVMe / Network Storage       │
    │  mmap'd, compressed, slow           │
    │  Rotation: fastest (batched)        │
    │  Contains: cold weights, old KV     │
    └─────────────────────────────────────┘
                    │
                    ▼
    ┌─────────────────────────────────────┐
    │  Middle Layer (Warm)                │
    │  RAM (CPU) / GPU Memory             │
    │  Rotation: medium                   │
    │  Contains: warm weights, recent KV  │
    └─────────────────────────────────────┘
                    │
                    ▼
    ┌─────────────────────────────────────┐
    │  Inner Layer (Hot)                  │
    │  CPU cache / GPU shared memory      │
    │  Rotation: slowest (fine-grained)   │
    │  Contains: active weights, current  │
    │           KV entries, activations   │
    └─────────────────────────────────────┘
                    │
                    ▼
    ┌─────────────────────────────────────┐
    │  Temporal Registers (T dimension)   │
    │  Fast scratchpad / register file    │
    │  Ephemeral: current computation     │
    │  Rotation: none (direct access)     │
    └─────────────────────────────────────┘
```

From the **center** (hot inner layers), you see a **lane** — sequential access is cheap because adjacent registers are close in the fabric.

From the **outside** (cold outer layers), you see a **tree converging to a point** — lookups traverse the hierarchy inward, like a page table walk or B-tree descent.

### 2.3 Rotation as Cache Policy

The rotation gradient expresses cache replacement and prefetching as geometry:

- **Inner layers** rotate slowly — registers here are stable, accessed frequently, moved rarely. This is your L1/L2-equivalent behavior.
- **Middle layers** rotate at medium speed — registers are shuffled periodically to maintain locality, prefetched based on access patterns.
- **Outer layers** rotate fastest (in terms of batch size and frequency of bulk movement) — cold data is moved in large blocks, on a schedule that amortizes SSD/NVMe latency.

The rotation is **not random** — it's a structured shuffle that maintains the geometric relationships (sectors stay coherent, lineages remain valid) while moving data between tiers.

**From the center, you see a lane** — a sequential scan through the inner layers is a straight line through address space, no hopping.

**From the outside, you see a tree** — a lookup from the cold exterior descends through the hierarchy to find the data, converging at the center where the active data lives.

---

## 3. Register Addressing & Geometry

### 3.1 Address Structure

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         RegisterAddress (128 bits)                      │
├──────────┬──────────┬──────────┬──────────┬──────────┬──────────────────┤
│  Spine   │  Layer   │  Ring    │ Sector   │  Shard   │   Temporal       │
│  (16)    │  (16)    │  (16)    │  (16)    │  (16)    │   (16)           │
│          │          │          │          │          │                  │
│  Which   │  Depth   │  Rotation│  Angular │  Sub-    │  T-dimension:   │
│  fiber   │  from    │  position│  sector  │  division│  active compute │
│  ofCube  │  center  │  within  │          │  within  │  scratchpad     │
│          │  (0=hot, │  layer   │          │  sector  │  index          │
│          │   N=cold)│          │          │          │                  │
└──────────┴──────────┴──────────┴──────────┴──────────┴──────────────────┴──────────┘
```

**Spine** (16 bits): Which "fiber" of the 1024³ cube this register belongs to. A spine is a 1D thread through the 3D cube — a coherent access path.

**Layer** (16 bits): Depth from center. 0 = hottest (fastest storage, most frequently accessed). N = coldest (slowest storage, least recently accessed). The layer determines *which tier* the register maps to physically.

**Ring** (16 bits): Rotational position within a layer. Defines where the register is in the rotation cycle. Determines *when* it moves, *when* it's prefetched, *when* its echoes propagate.

**Sector** (16 bits): Angular sector within a ring. Groups registers that are conceptually related (e.g., same attention head, same layer of the model).

**Shard** (16 bits): Sub-division within a sector. Fine-grained addressing for individual registers within a sector.

**Temporal** (16 bits): Index into the temporal register ring (the T dimension). Temporal registers are the active compute scratchpad — distinct from the stored sphere.

### 3.2 Address as Lookup Key

Because the address encodes position, layer, ring, sector, and shard, **the address is the lookup key**. You don't need a separate hash table to find a register:

- To find "the weights for layer 3, head 2, token position 42," you compute the address from those logical components.
- The address tells you which tier the data lives in (layer field).
- The address tells you which sector to bloom-filter for routing.
- The address tells you the rotation phase (ring field) for scheduling.

This is the core efficiency gain: **no separate index, no separate lookup, no metadata overhead beyond the address itself.**

### 3.3 Address Arithmetic

The address supports meaningful arithmetic:

```
distance(A, B) = popcount(A XOR B)   // structural distance in the fabric
neighbor(A, direction) → Address     // adjacent register in a given direction
parent(A) → Address                  // ancestor in the lineage chain
child(A, offset) → Address           // descendant
```

`distance(A, B)` via XOR popcount gives a *structural* distance — not Euclidean, but meaningful for routing: addresses that are XOR-close are structurally related and route cheaply.

### 3.4 Address Ranges and Mapping

The 128-bit address space is vast (2^128). Practical ranges:

| Component | Bits | Range | Meaning |
|---|---|---|---|
| Spine | 16 | 0–65535 | Fibers in the cube |
| Layer | 16 | 0–65535 | 0 = hottest, 65535 = coldest |
| Ring | 16 | 0–65535 | Rotation phase |
| Sector | 16 | 0–65535 | Angular sector |
| Shard | 16 | 0–65535 | Sub-division |
| Temporal | 16 | 0–65535 | T-dimension index |

In practice, the fabric only uses a *portion* of this space — the rest is reserved for future expansion, cross-model addressing, and cluster spanning.

---

## 4. The Two Interlaced Planes

### 4.1 IPv6 Data Plane

The **IPv6 data plane** carries the actual payload and state:

```
IPv6DataRegister {
    address:        RegisterAddress,    // self-describing position
    payload:        BitslicedLane,      // the data (bit-plane sliced)
    xor_lineage:    u128,               // XOR fingerprint of lineage chain
    echo_counter:   u8,                 // propagation depth counter
    bloom_anchor:   u64,                // hash anchor for bloom attachment
    version:        u8,                 // schema version for self-description
    phase:          u8,                 // phase within rotation cycle
    checksum_anchor:u64,                // for self-validation
}
```

The payload is **bitsliced** — instead of storing values as `float32` arrays, you store bit-plane slices across lanes:

```
Value: 0xD3 (binary: 11010011)

Bitsliced representation (8 lanes for 8-bit values):
  Lane 0 (MSB): [1, 1, 0, 1, ...]
  Lane 1:       [1, 0, 0, 0, ...]
  Lane 2:       [0, 1, 1, 0, ...]
  ...
  Lane 7 (LSB): [1, 1, 1, 1, ...]
```

Each lane is a bit vector. Bitwise operations across lanes are SIMD-friendly (single instruction on Apple Silicon NEON/AMX).

### 4.2 IPv4 Control Plane

The **IPv4 control plane** carries routing, scheduling, and metadata:

```
IPv4ControlPacket {
    src_address:    RegisterAddress,
    dst_address:    RegisterAddress,
    operation:      ControlOp,          // fold, route, read, write, compute, echo
    metadata:       ControlMetadata,
    echo_depth:     u8,                 // how many hops this packet has traversed
    bloom_signature:u64,                // for bloom filter matching at each hop
    payload_ref:    Option<DataPointer>,// reference to data in IPv6 plane (not copied)
}
```

Control operations:

| Operation | Meaning |
|---|---|
| `fold` | Combine two registers via XOR (or other operation) |
| `route` | Route a packet to a destination address |
| `read` | Read a register's payload |
| `write` | Write/update a register's payload |
| `compute` | Dispatch compute to data location |
| `echo` | Propagate a delta through the structure |
| `rotate` | Trigger a rotation cycle for a layer |
| `adapt` | Inject an adaptation delta (on-device learning) |

The control plane **does not carry payload** — it carries references to the IPv6 plane. This separation means control packets are small, fast, and can be processed independently of data movement.

### 4.3 Interlacing

The two planes are interlaced: every IPv6 data register has a corresponding IPv4 control presence (its address is reachable via the control plane). The control plane knows where every data register is, can route to it, and can issue operations on it — but the data itself stays in the data plane.

This is the **separation of concerns** that makes the architecture clean:

- Data plane: storage, payload, lineage, bloom anchors.
- Control plane: routing, scheduling, operations, echo propagation.
- Neither plane needs to know the internals of the other.

---

## 5. The 4D Sphere Structure

### 5.1 1024 × 1024 × 1024 × 1024

The IPv6 data plane is organized as a 4D grid:

```
1024 (X) × 1024 (Y) × 1024 (Z) × 1024 (T)
```

- **X, Y, Z**: Three spatial dimensions of the cube. Each is 1024 registers wide.
- **T**: The temporal dimension — 1024 temporal registers for active computation.

Total IPv6 registers: 1024⁴ = 2^40 ≈ 1 trillion theoretical registers. In practice, only a tiny fraction are populated — the rest is sparse, gated by bloom filters.

### 5.2 IPv4 Routing Plane: Same 4D Grid

The IPv4 control plane mirrors the same 4D grid:

```
1024 (X) × 1024 (Y) × 1024 (Z) × 1024 (T)
```

But the IPv4 registers carry *routing metadata* per address, not payload. Each IPv4 control register at address A knows:

- Where the corresponding IPv6 data register is (tier, physical location).
- What operations are valid at this address.
- What bloom filters are attached.
- What the rotation schedule is for this layer.

### 5.3 Nested Spheres

Each register in the 1024³ cube contains a **full sub-sphere** with both IPv4 and IPv6 interlaced. This is recursive:

```
Sphere(X, Y, Z)
  ├── Register(X, Y, Z)
  │     ├── IPv6 payload + lineage
  │     └── IPv4 control presence
  ├── Sub-Sphere(X±1, Y±1, Z±1)   // recursive
  └── ...
```

In practice, the nesting is not infinite — it goes as deep as needed for the hierarchy (typically 2–4 levels: fabric → sector → shard → register).

The nesting enables **locality**: operations on a sector can be dispatched to the sector's sub-sphere without global routing.

### 5.4 Positional Encoding

Every register's address encodes its position, lineage, angle, row, section, group, phase:

```
Position:  (spine, layer, ring, sector, shard)
Lineage:   xor_lineage field (XOR chain back to ancestor)
Angle:     derived from sector + ring (angular position in rotation)
Row:       derived from spine + layer
Section:   derived from sector grouping
Group:     derived from shard grouping
Phase:     derived from ring (rotation phase)
```

This is not stored as separate fields — it's **derived from the address and the lineage**. The address *is* the encoding. This is what makes the structure self-describing: any register can compute its position and lineage from its address alone.

---

## 6. Rotation & Scheduling

### 6.1 Rotation Gradient

The rotation speed varies by layer:

```
Layer 0 (hottest):    rotation period = 1 cycle    (essentially static, fine-grained)
Layer 1:              rotation period = 4 cycles
Layer 2:              rotation period = 16 cycles
...
Layer N (coldest):    rotation period = 2^N cycles  (bulk movement, scheduled)
```

The rotation is **structured**, not random: it maintains sector coherence, preserves lineage chains, and moves data between tiers in a way that maintains locality.

### 6.2 Lane View (from Center)

From the center (hot inner layers), the rotation is slow enough that you see a **lane** — a coherent sequential path through the registers. A sequential scan (e.g., processing a sequence of tokens) walks down this lane with minimal disruption.

```
Center view (layer 0-1):
  Token 0 → Register A
  Token 1 → Register B (adjacent to A in the lane)
  Token 2 → Register C (adjacent to B)
  ...
  Sequential access = walking down the lane = cache-friendly
```

### 6.3 Tree View (from Outside)

From the outside (cold outer layers), the rotation is fast enough that you see a **tree converging to a point** — a lookup from the cold exterior descends through the hierarchy to find the data at the center.

```
Outside view (layer N):
  Lookup for address X:
    Start at outer layer (cold storage)
    Descend through layers (N → N-1 → ... → 0)
    Arrive at center where active data lives
  This is a hierarchical lookup, like a page table walk
```

### 6.4 Rotation Cycle

A rotation cycle is a scheduled event that:

1. **Moves data between tiers** — cold data moves inward (promoted), hot data moves outward (evicted), based on access patterns.
2. **Prefetches** — data likely to be needed soon is brought into warmer tiers.
3. **Damps echoes** — pending echo propagations are processed in batch, not individually.
4. **Updates bloom filters** — sectors update their bloom filters to reflect current contents.
5. **Reorganizes** — the self-optimizing layout can rearrange registers within a layer for better locality.

The rotation is **adaptive**: the periods adjust based on observed access patterns, energy state (battery vs. plugged in), and thermal state.

---

## 7. XOR Operations & Lineage

### 7.1 XOR as Native Operation

XOR is the fundamental operation of the fabric:

| Operation | XOR usage |
|---|---|
| **Lineage encoding** | `register.xor_lineage = parent.xor_lineage ^ register.payload ^ constant` |
| **Fold** | `fold(A, B) = A.payload ^ B.payload` (or other op with XOR as component) |
| **Diff** | `delta = new ^ old` — the minimal change between two states |
| **Distance** | `distance(A, B) = popcount(A.address ^ B.address)` — structural proximity |
| **Echo cancellation** | `delta_A ^ delta_B = 0` means redundant echoes cancel |
| **Checksum** | `checksum = xor_lineage ^ expected_anchor` — self-validation |

### 7.2 Lineage Chain

Each register carries a XOR lineage fingerprint:

```
register.xor_lineage = ancestor.xor_lineage ^ register.payload ^ lineage_constant
```

This creates a **chain of custody**: you can verify that a register's state is consistent with its ancestors by recomputing the chain. If the chain breaks, the register is corrupted or inconsistent.

The lineage is **self-describing**: any register can trace its lineage back through the chain without external metadata. The address encodes the lineage path (parents are at predictable addresses derived from the child's address).

### 7.3 XOR Folding

The `fold` operation combines two registers:

```
fold(A, B):
  result.payload = A.payload ^ B.payload          // XOR merge
  result.xor_lineage = A.xor_lineage ^ B.xor_lineage
  result.address = midpoint(A.address, B.address) // derived address
  return result
```

Fold is:
- **Self-inverse**: `fold(fold(A, B), B) = A` (with XOR, this holds)
- **Associative**: `fold(fold(A, B), C) = fold(A, fold(B, C))`
- **Schema-free**: no negotiation needed; the operation is defined by the addresses

### 7.4 XOR Diff Storage

For deduplication across the sphere:

```
canonical_register = store_once(payload)
duplicate_register = store_xor_diff(canonical, duplicate)
  // duplicate's effective payload = canonical.payload ^ duplicate.xor_diff
```

When a register's payload is XOR-close to an existing canonical register, store only the XOR diff. The effective payload is reconstructible by XORing the canonical payload with the diff.

This is especially powerful for:
- **KV cache** — nearby sequence positions often have similar activations; store one canonical + diffs.
- **Weight sharing** — multiple models or quantization levels share a base via XOR diffs.
- **Redundant activations** — repeated patterns across the batch collapse to diffs.

---

## 8. Echo Minimization

### 8.1 Echo Definition

An **echo** is the propagation of a change through the fabric. When register R changes, the change ripples to dependent registers, to the routing plane, to bloom filters, to related sectors.

**Echo minimization** = the propagation is exactly as large as it must be, no more.

### 8.2 Minimal Delta Propagation

When register R at address A changes from `old` to `new`:

```
Step 1: Compute delta
  delta = new.payload ^ old.payload

Step 2: Update R
  R.payload = new.payload
  R.xor_lineage = R.xor_lineage ^ delta   // lineage updates minimally
  R.echo_counter += 1

Step 3: Propagate delta to dependents
  For each dependent sector S:
    if bloom_filter(S).might_contain(A):
      for each register D in S that depends on A:
        D.payload = D.payload ^ delta       // XOR merge the delta
        D.xor_lineage = D.xor_lineage ^ delta
        D.echo_counter += 1
        if delta.affects(D) is false:       // delta doesn't change D's relevant bits
          stop propagating from D            // echo dies here
        else:
          continue propagation from D

Step 4: Rotation-aware damping
  if R.layer is cold (outer):
    queue delta for next rotation cycle     // don't propagate immediately
  else:
    propagate immediately                   // hot layer: fast propagation
```

### 8.3 Echo Damping by Rotation

The rotation gradient naturally damps echoes:

- **Hot inner layers**: changes propagate fully and immediately — these registers are the active compute set, echoes are necessary.
- **Cold outer layers**: changes are queued for the next rotation cycle — propagation is scheduled, not instant. This prevents cold-layer thrashing on every token.

### 8.4 Bloom-Gated Propagation

Before propagating to a sector, check the bloom filter:

```
if sector.bloom.might_contain(target_address):
  propagate
else:
  skip entirely  // echo dies at this boundary
```

This is especially powerful at the cold/outer layers, where most sectors are empty or irrelevant to a given change. The echo dies at the first empty boundary.

### 8.5 XOR Echo Cancellation

If two changes arrive at a register via different paths:

```
delta_A arrives via path 1
delta_B arrives via path 2

if delta_A ^ delta_B == 0:
  // Same change from different routes — echo cancels
  register sees no net change
  stop propagating
else:
  register sees delta_A ^ delta_B
  continue propagating if net change is non-zero
```

This prevents redundant propagation from thrashing the fabric.

### 8.6 Structured Echo Injection (Add Echo)

You can **add a controlled echo** — a deliberate, structured propagation for synchronization, cache coherence, or routing refresh:

```
inject_echo(source_address, delta, target_layer, propagation_depth):
  // Inject a deliberate echo at source_address
  // It propagates to target_layer, up to propagation_depth hops
  // Because everything is XOR-based, the echo merges cleanly
  // The echo is minimized: exactly the scope needed, no broadcast
```

Use cases:
- **Cache coherence** — ensure all nodes agree on a register's state after a write.
- **Routing refresh** — propagate updated routing metadata without a full rebuild.
- **Synchronization pulse** — coordinate a rotation cycle across sectors.

---

## 9. Self-Describing Registers

### 9.1 What "Self-Describing" Means

Every register carries enough information to be interpreted without external schema:

```
Register {
  address:        // position, lineage, angle, layer, sector, phase — encoded in the address
  version:       // schema version — what operation set this register supports
  payload:       // the data, with its type implied by the address context
  xor_lineage:   // lineage chain — where this register came from
  bloom_anchor:  // hash anchor — what bloom filters it belongs to
  checksum_anchor:// validation anchor — how to verify integrity
}
```

To interpret a register, you need:
1. The address (gives you position, lineage path, layer, sector).
2. The version (gives you the operation set).
3. The address context (gives you the payload type — e.g., address in "weights/layer3" means weights, address in "kv/token42" means KV cache entry).

No external registry, no schema lookup, no type table.

### 9.2 Self-Hosting

The fabric **hosts itself**: the control plane registers are themselves registers in the data plane. The routing metadata is stored in registers, accessible via the same address mechanism. The fabric is self-contained — no external configuration, no separate routing table.

### 9.3 Self-Processing

Operations are expressed as register interactions. A `fold` is two registers combining. A `compute` is a register dispatching to another register's location. The fabric processes itself through register operations — there's no separate "processor" abstraction.

---

## 10. Bloom Filters & Bitsliced Lanes

### 10.1 Bloom Filters per Sector

Each sector in the IPv4 control plane has a **bloom filter** attached:

```
SectorBloom {
    filter:      BitVector(256 bits),   // probabilistic membership
    anchor:      u64,                    // hash anchor for this sector
    generation:  u8,                     // increments on each rebuild
}
```

The bloom filter answers: **"might address X be in this sector?"**

- **True positive**: address is in the sector (or the bloom thinks it is — false positive possible).
- **True negative**: address is definitely not in the sector.

The bloom filter is used for:
- **Routing** — before routing to a sector, check if the destination might be there.
- **Echo propagation** — before propagating a change to a sector, check if any register in the sector cares about the changed address.
- **Lookup** — before descending into a sub-sphere, check if the target might be there.

### 10.2 Bloom Filter Operations

```
bloom.insert(address)      // add address to bloom
bloom.might_contain(address) → bool   // membership test
bloom.merge(other_bloom)   // union of two blooms (for sector aggregation)
bloom.rebuild(sector_contents)  // rebuild from actual contents (periodically)
```

Bloom filters are XOR-hashed for fast merging and compact representation.

### 10.3 Bitsliced Lanes

Payload is stored in **bitsliced** format:

```
For an array of N values, each B bits wide:

  Lane 0 (MSB of each value):  [bit0_of_val0, bit0_of_val1, ..., bit0_of_valN]
  Lane 1:                      [bit1_of_val0, bit1_of_val1, ..., bit1_of_valN]
  ...
  Lane B-1 (LSB):              [bitB-1_of_val0, bitB-1_of_val1, ..., bitB-1_of_valN]
```

Each lane is a bit vector (1 bit per value). Operations on lanes are bitwise:

| Operation | Lane operation |
|---|---|
| XOR two values | XOR corresponding lanes |
| AND two values | AND corresponding lanes |
| NOT a value | NOT each lane |
| Add (via bitwise) | More complex, but lane-parallel |
| Popcount | Popcount each lane, sum |

On Apple Silicon, lane operations map to NEON/AMX SIMD instructions — a single instruction processes 64+ bits in parallel.

### 10.4 Bitsliced Quantization

Bitslicing is a natural fit for quantized inference:

- **Binary networks** (1-bit): each lane is the entire value — trivial.
- **INT8** (8-bit): 8 lanes per value — each lane is a bit plane.
- **FP8** (8-bit floating point): similar, with exponent/mantissa lanes.
- **INT4** (4-bit): 4 lanes — very compact.

The bitsliced representation means quantized values are **already** in a SIMD-friendly layout — no conversion needed for bitwise operations.

---

## 11. Storage Tiers & Hardware Mapping

### 11.1 Tier Abstraction

The fabric exposes a **tier abstraction** that maps to physical storage:

```
Tier {
    id:              TierId,
    name:            String,           // "ram", "gpu", "ssd", "remote"
    storage:         StorageBackend,   // concrete backing store
    capacity:        Bytes,            // available capacity
    latency:          Duration,        // typical access latency
    bandwidth:        BytesPerSec,     // typical bandwidth
    energy_per_access: Joules,         // energy cost per access (for scheduling)
    layer_range:     Range<u16>,       // which layers map to this tier
}
```

### 11.2 Hardware Mapping

| Hardware | Tier | Layer Range | Characteristics |
|---|---|---|---|
| CPU registers / L1 cache | `tier_hot` | 0–1 | Fastest, smallest, ephemeral |
| CPU L2/L3 cache / RAM | `tier_warm` | 2–10 | Fast, medium capacity |
| GPU shared memory / GPU RAM | `tier_gpu` | 3–12 | Fast for compute, limited capacity |
| Unified memory (macOS) | `tier_unified` | 2–15 | CPU+GPU share; fabric manages coherence |
| SSD / NVMe (mmap) | `tier_cold` | 16–100 | Slow, high capacity, compressed |
| Remote node (network) | `tier_remote` | 100+ | Slowest, largest, distributed |

On an 8GB Mac:

```
tier_hot:    CPU registers + L1 (adaptive, small)
tier_warm:   Unified memory (RAM) — largest tier
tier_gpu:    GPU memory (shared from unified) — for compute
tier_cold:   SSD via mmap — model weights, old KV cache
```

### 11.3 Mmap-Based Cold Storage

The cold tier uses **mmap** to map SSD storage into the address space:

- Model weights are stored compressed on SSD, mmap'd into the cold layers.
- Pages are faulted in on demand (when the rotation brings them inward, or on explicit access).
- Dirty pages (updated weights, KV cache entries promoted to cold) are flushed back on eviction.

This is transparent to the fabric — the cold layer *is* the mmap region, addressed by the fabric's address space.

### 11.4 Compression

Cold tier data is compressed:

- **Weights**: quantized (Q4, Q5, Q8) + compressed (LZ4, ZSTD).
- **KV cache**: XOR-diffed against canonical entries + compressed.
- **Activations**: compressed on eviction to cold tier.

The fabric manages compression transparently — registers in the cold tier are compressed, registers in warm/hot tiers are uncompressed (or less compressed).

---

## 12. No-Copy Compute-to-Data Routing

### 12.1 The Principle

Instead of moving data to where the compute is, **move the compute to where the data is**.

```
Traditional:
  data (SSD) → copy to RAM → copy to GPU → compute → copy back

Binetic:
  compute dispatched to data location
  If data is in GPU memory: compute on GPU, no copy
  If data is in RAM: compute on CPU, no copy
  If data is mmap'd from SSD: fault page, compute on CPU, no explicit copy
```

### 12.2 Compute Dispatch

A compute operation specifies the registers involved and the operation:

```
ComputeOp {
    operation:    ArithmeticOp,   // matmul, add, softmax, attention, etc.
    operands:     [RegisterAddress; N],  // which registers (by address)
    result:       RegisterAddress,        // where to write result
    backend:      Option<BackendId>,      // which backend (llama.cpp, MLX, native)
    routing:      RoutingPolicy,          // to_data, to_compute, balanced
}
```

The fabric resolves each operand's address to its physical location:

```
operand_address → tier → physical location (GPU mem, CPU RAM, mmap region)
```

Then dispatches the compute to the location(s):

```
If all operands are in GPU memory:
  Dispatch to GPU backend, pointers into GPU memory

If operands are split (some GPU, some CPU):
  Choose the location that minimizes total movement
  Or split the operation if the backend supports it

If operands are in mmap region:
  Fault pages if needed, compute on CPU, keep result in place
```

### 12.3 Backend Integration

Backends (llama.cpp, MLX) receive **pointers to data in place** — no copying:

```
llama.cpp backend:
  receives: pointer to weight data (already in RAM or GPU)
            pointer to KV cache (already in place)
            pointer to activations (temporal registers)
  computes: in place, writes result back to temporal registers
  No copying — the fabric presented the data where it already is

MLX backend (Apple):
  receives: MLX arrays backed by the fabric's memory
            (the fabric's memory is already in the unified address space)
  computes: using MLX operations, in place
  No copying — MLX operates on the fabric's memory directly
```

### 12.4 Cross-Node Routing (Future)

In a cluster, the same principle extends:

```
Register A is on Node 1, Register B is on Node 2

ComputeOp { operands: [A, B], routing: to_data }

→ Dispatch compute to Node 1 (where A is) and Node 2 (where B is)
→ Or: route the smaller operand to the larger operand's node
→ Or: split the operation across nodes

The control plane routes the compute dispatch, not the data.
```

---

## 13. Self-Optimizing Layout

### 13.1 Observation

The fabric observes access patterns:

```
AccessLog {
    address:      RegisterAddress,
    timestamp:    Instant,
    operation:    AccessType,   // read, write, compute, fold
    latency:      Duration,     // how long the access took
    tier:         TierId,       // which tier was accessed
}
```

Access patterns are tracked per sector, per layer, per spine:

- **Hot registers** — accessed frequently, should be in warm/hot tiers.
- **Cold registers** — accessed rarely, can be in cold tier.
- **Co-accessed registers** — accessed together, should be close in the fabric.
- **Sequential access patterns** — walks down a spine or sector, should be lane-optimized.

### 13.2 Reorganization

Based on observed patterns, the fabric reorganizes:

```
Reorganization {
    move_hot_to_warm:      [Address; N]  // registers that got hot, move inward
    move_cold_to_cold:     [Address; N]  // registers that got cold, move outward
    co_locate:             [(Address, Address); N]  // registers accessed together, move closer
    create_lane:           [Address; N]  // registers in a sequential access pattern, organize into a lane
    split_sector:          SectorId       // sector too large/diverse, split
    merge_sectors:         [SectorId; N]  // sectors similar, merge
}
```

Reorganization is **gradual** — it happens during rotation cycles, not as a big stop-the-world event.

### 13.3 Adaptation to Workload

Different workloads produce different optimal layouts:

| Workload | Optimal layout |
|---|---|
| Single model, long context | KV cache hot in inner layers, weights in middle, cold weights on SSD |
| Multiple small models | Shared base in cold layer, model-specific diffs in warm layer |
| Batch inference | Co-locate batch entries in sectors for parallel access |
| Streaming inference (one token at a time) | Lane-optimized for sequential token access |
| Interactive (chat) | Recent conversation KV cache hottest, older entries colder |

The fabric adapts to the workload over time — it's not configured once, it learns.

---

## 14. Energy-Aware Scheduling

### 14.1 Energy as a Scheduling Constraint

The rotation scheduler considers energy, not just latency:

```
EnergyState {
    on_battery:       bool,
    battery_level:     Percentage,
    thermal_state:     ThermalLevel,   // cool, warm, hot
    idle_time:         Duration,       // how long since last active compute
}
```

Scheduling decisions adapt to energy state:

```
If on_battery && battery_level < 20%:
  - Reduce rotation frequency (fewer SSD wakeups)
  - Batch echo propagation more aggressively
  - Prefetch less aggressively (fewer page faults)
  - Prefer CPU compute over GPU (GPU draws more power)

If on_battery && battery_level > 50%:
  - Normal scheduling with energy awareness
  - Moderate batching, moderate prefetch

If plugged in:
  - Performance-oriented scheduling
  - Aggressive prefetch, frequent rotation, GPU preferred
```

### 14.2 Energy per Access

Each tier has an **energy per access** cost:

```
Tier energy costs (approximate, for illustration):

tier_hot (CPU registers):     ~0.1 pJ per access
tier_warm (CPU RAM):          ~100 pJ per access
tier_gpu (GPU memory):        ~200 pJ per access (plus GPU active power)
tier_cold (SSD mmap):         ~10 nJ per access (SSD wakeup + page fault)
tier_remote (network):        ~100 nJ per access (network + remote CPU)
```

The scheduler minimizes **total energy per token**, not just latency:

```
Energy per token = sum over all accesses of (access_count × tier_energy_per_access)
                  + energy_overhead(rotation, prefetch, propagation)
```

### 14.3 Batched Propagation

Echo propagation is batched to amortize energy:

```
Instead of:  propagate each delta immediately (N wakeups for N changes)
Do:          queue deltas, propagate on next rotation tick (1 batched wakeup)

Energy savings: proportional to batch size
```

On battery, batching is more aggressive — larger batches, less frequent rotation.

### 14.4 Prefetch Energy Trade-off

Prefetching brings data into warm tiers before it's needed, reducing latency. But prefetching costs energy (page faults, SSD wakeups, memory bandwidth).

```
If energy state is critical (battery low, thermal hot):
  - Reduce prefetch aggressiveness
  - Prefetch only on strong signals (high confidence the data will be needed)
  - Accept higher latency to save energy

If energy state is comfortable:
  - Aggressive prefetch
  - Prefetch on weak signals too
  - Optimize for latency
```

---

## 15. Cross-Model XOR Deltas

### 15.1 Model Relationship via XOR

Multiple models (or variants of the same model) can share storage via XOR deltas:

```
Model A (base):     stored in cold tier at address range [W0, W1]
Model B (variant):  stored as delta from A
  B.payload = A.payload ^ delta_B
  delta_B stored at address range [D0, D1] (much smaller than full model)

Model C (variant):  stored as delta from A (or from B)
  C.payload = A.payload ^ delta_C
```

### 15.2 Delta Computation

Deltas can be computed:

- **Offline** — during model download/preparation, compute delta from base model, store delta.
- **Online** — during on-device adaptation, compute delta from current model state, store delta.
- **Quantization delta** — Q4 model = Q8 model ^ quantization_delta (approximate).

### 15.3 Effective Capacity Multiplication

For an 8GB Mac running multiple models:

```
Without binetic:
  Model A (3B Q4):  2GB
  Model B (1B Q4):  0.7GB
  Total: 2.7GB (both in RAM)

With binetic:
  Model A (3B Q4) base:  2GB in cold tier (mmap from SSD)
  Model B (1B Q4) delta: 0.2GB in warm tier (XOR diff from A's layers where they overlap)
  Model C (0.5B Q4) delta: 0.1GB in warm tier
  Effective RAM usage: ~0.3GB for models B and C (plus shared base on SSD)
  Total capacity: all three models available, only ~2.3GB physical (2GB SSD + 0.3GB RAM)
```

The base model is on SSD (cold tier), faulted in as needed. The deltas are small and in RAM (warm tier). The fabric routes compute to the right tier for each model's weights.

### 15.4 Shared Base Across Tasks

If multiple models share architectural elements (same tokenizer, same embedding layer, similar attention patterns), the XOR deltas for those shared elements are tiny (or zero). The fabric automatically exploits this — if two registers are XOR-identical, they collapse to one canonical + zero-diff.

---

## 16. State Integrity & Provable Correctness

### 16.1 XOR Lineage as Chain of Custody

Each register's `xor_lineage` field is a chain of custody:

```
Register R at address A:
  R.xor_lineage = parent(R).xor_lineage ^ R.payload ^ lineage_constant(A)

Verification:
  expected = recompute_lineage(A)  // walk parent chain, XOR payloads
  if R.xor_lineage != expected:
    R is corrupted or inconsistent
```

This detects:
- **Bit flips** in RAM (cosmic rays, hardware faults).
- **Corruption** in SSD storage (bit rot, bad blocks).
- **Inconsistent state** — a register that doesn't match its lineage.

### 16.2 Self-Validation on Access

On every access, the fabric can validate:

```
On read(R):
  if R.checksum_anchor != compute_checksum(R):
    R is corrupted → fetch from backup tier or recompute from lineage
  if R.xor_lineage != recompute_lineage(R.address):
    R is inconsistent → flag, repair from lineage chain
  return R.payload
```

Validation is optional (can be disabled for performance), but available for correctness-critical inference.

### 16.3 Deterministic Replay

Given:
- Initial fabric state (all registers at time T0).
- Sequence of deltas (echo propagations, writes, adaptation injections).

You can **replay** the fabric state at any time T:

```
state_T = state_T0
for delta in deltas[T0..T]:
  apply_delta(state_T, delta)
```

This enables:
- **Debugging** — reproduce exact fabric state at the time of a bug.
- **Verifiable inference** — prove that a given inference run produced a given output from a given initial state.
- **Checkpointing** — save deltas, not full state; restore by replay.

### 16.4 Corruption Recovery

When corruption is detected:

```
If R.payload is corrupted:
  Try to repair from lineage: recompute R.payload from parent chain
  If lineage is intact: repair succeeds
  If lineage is also corrupted: escalate to backup tier (cold storage may have clean copy)

If R.xor_lineage is corrupted but payload is intact:
  Recompute lineage from payload and parent chain
  Repair succeeds if parents are intact
```

The XOR structure makes repair computable — you don't need a separate backup; the lineage chain *is* the backup.

---

## 17. On-Device Adaptation

### 17.1 Adaptation as Delta Injection

Fine-tuning or adaptation updates are injected as deltas into the fabric:

```
AdaptationDelta {
    target_address:   RegisterAddress,   // which weight register to update
    delta:            BitslicedLane,     // the update (XOR-mergeable)
    justification:    AdaptationJustification,  // why this update (gradient, heuristic, etc.)
    timestamp:        Instant,
    confidence:       f32,               // how confident in this update
}
```

The delta is injected via the control plane:

```
control_plane.inject_adapt(delta):
  target = resolve(delta.target_address)
  target.payload = target.payload ^ delta.delta    // XOR merge
  target.xor_lineage = target.xor_lineage ^ delta.delta
  propagate_echo(delta.delta, target.address, depth=adaptation_propagation_depth)
```

### 17.2 Adaptation Types

| Type | Mechanism | Use case |
|---|---|---|
| **Gradient-based** | Delta from gradient descent step | On-device fine-tuning on user data |
| **Heuristic** | Delta from heuristic rule | Personalization (language style, vocabulary) |
| **Quantization refinement** | Delta from error analysis | Per-layer quantization adjustment |
| **Context adaptation** | Delta from context statistics | Domain-specific adaptation (medical, legal, code) |

### 17.3 Echo-Propagated Adaptation

When a weight is updated via adaptation delta, the echo propagates to dependent registers:

```
Update weight W at layer 3:
  delta = new_W ^ old_W
  W.payload ^= delta
  propagate echo to:
    - Downstream weights that depend on W (next layer's inputs)
    - KV cache entries that used W (if they need recomputation)
    - Activation registers that were computed with old W (if they need refresh)
```

The echo propagation is **minimal** — only registers that actually depend on the changed weight are updated. Registers that don't depend on W are unaffected.

### 17.4 Privacy

Because adaptation happens on-device:

- **No user data leaves the device** — gradients are computed locally, deltas are injected locally.
- **No cloud round-trip** — adaptation is immediate, no network latency.
- **User-specific model** — the fabric carries a personalized model without exposing it.

This is a privacy win over cloud-based fine-tuning.

---

## 18. Backend Integration

### 18.1 Backend Abstraction

The fabric integrates with inference backends via a **backend abstraction**:

```
Backend {
    id:              BackendId,
    name:            String,           // "llama.cpp", "mlx", "native", ...
    supported_ops:   [ArithmeticOp],   // what operations this backend can execute
    memory_mapping:  MemoryMapping,    // how this backend maps to fabric tiers
    compute_dispatch: DispatchFn,      // how to dispatch compute to this backend
    init:            InitFn,
    teardown:        TeardownFn,
}
```

### 18.2 Llama.cpp Integration

```
LlamaCppBackend {
    init:
      Load llama.cpp library
      Map model weights into fabric's cold tier (mmap from disk)
      Set up KV cache in fabric's warm tier
      Set up activations in fabric's temporal registers

    compute_dispatch(compute_op):
      Resolve operand addresses to llama.cpp memory pointers
      (pointers into fabric memory — no copy)
      Call llama.cpp inference function with those pointers
      Write result back to temporal registers (in place)

    memory_mapping:
      Weights:  cold tier (mmap'd from model file)
      KV cache: warm tier (RAM, managed by fabric)
      Activations: temporal registers (fast scratchpad)
      sampler output: temporal registers
}
```

### 18.3 MLX Integration (Apple)

```
MLXBackend {
    init:
      Load MLX library
      Create MLX arrays backed by fabric's memory
      (MLX arrays point into fabric's unified memory — no copy)
      Set up model weights in fabric's cold/warm tiers

    compute_dispatch(compute_op):
      Resolve operand addresses to MLX array references
      Call MLX operations (matmul, softmax, etc.) on those arrays
      MLX operates in place on fabric memory
      Write result to temporal registers

    memory_mapping:
      Weights:  cold/warm tiers (MLX arrays backed by fabric memory)
      KV cache: warm tier (MLX arrays)
      Activations: temporal registers (MLX arrays)
      GPU memory: used automatically by MLX (unified memory on Mac)
}
```

### 18.4 Adding a New Backend

To add a new backend (e.g., CANDLE, bitsandbytes, a custom kernel):

```
1. Implement the Backend trait:
   - init: set up memory mapping
   - compute_dispatch: map fabric addresses to backend memory, dispatch compute
   - supported_ops: declare which operations you handle

2. Register the backend with the fabric:
   fabric.register_backend(my_backend)

3. The fabric routes compute ops to your backend based on:
   - Which tier the data is in
   - Which ops the backend supports
   - Energy/latency tradeoffs
```

No changes to the fabric core needed — backends are plug-ins.

---

## 19. API Surface

### 19.1 Core API (Rust)

```
// ── Fabric creation ──────────────────────────────────────────────────────

let fabric = Fabric::new(Config {
    tiers: vec![
        Tier::hot_cpu(),            // CPU registers, L1
        Tier::warm_ram(16 * GB),    // unified memory
        Tier::gpu(2 * GB),          // GPU memory (if available)
        Tier::cold_ssd("/Models"),  // SSD mmap region
    ],
    rotation: RotationPolicy {
        base_period: 16.ms(),
        adaptive: true,
        energy_aware: true,
    },
    echo: EchoPolicy {
        minimal: true,
        bloom_gated: true,
        rotation_damped: true,
    },
    optimization: OptimizationPolicy {
        self_optimizing: true,
        observe_access: true,
        reorganize_during_rotation: true,
    },
    integrity: IntegrityPolicy {
        validate_on_read: false,      // set true for correctness-critical
        validate_lineage: false,
        repair_on_corruption: true,
    },
});

// ── Model attachment ──────────────────────────────────────────────────────

// Attach a model to the fabric
// Weights go to cold tier (mmap'd), KV cache to warm, activations to temporal
let model = fabric.attach_model(ModelConfig {
    name: "llama3-8b-q4",
    weight_source: "https://models.example.com/llama3-8b-q4.gguf",
    weight_tier: Tier::cold_ssd,        // weights on SSD, faulted in as needed
    kv_cache_tier: Tier::warm_ram,      // KV cache in RAM
    activation_tier: Tier::temporal,    // activations in temporal registers
    quantization: Quantization::Q4,     // Q4_0 or custom
});

// Attach a second model as XOR delta from the first (if related)
let model_b = fabric.attach_delta_model(DeltaModelConfig {
    name: "llama3-3b-q4",
    base_model: "llama3-8b-q4",         // shares base with model A
    delta_source: "https://models.example.com/llama3-3b-q4.delta",
    delta_tier: Tier::warm_ram,         // deltas in RAM (small)
});

// ── Inference ─────────────────────────────────────────────────────────────

// Run inference: route compute to data, not data to compute
let output = fabric.compute(
    ComputeOp {
        operation: ArithmeticOp::SampleNextToken,
        operands: vec![
            model.scope("layers").register("attention/weights"),   // weight register
            model.scope("layers").register("attention/kv_cache"),  // KV cache register
            fabric.temporal().register("activations/current"),      // activation register
        ],
        result: fabric.temporal().register("output/token_logits"),
        backend: None,              // auto-select best backend for the data location
        routing: RoutingPolicy::ToData,  // dispatch compute to where data lives
    }
);

// The fabric handles:
// - Paging weights from SSD if needed (rotation or on-demand)
// - Managing KV cache (eviction, bloom gating, rotation)
// - Routing compute to the right backend (GPU if weights are there, CPU otherwise)
// - Echo propagation for any state changes
// - Energy-aware scheduling (battery state, thermal)

// ── Adaptation ────────────────────────────────────────────────────────────

// Inject an adaptation delta (on-device fine-tuning, personalization)
fabric.inject_adapt(
    AdaptDelta {
        target: model.scope("layers").register("attention/weights/layer3"),
        delta: compute_gradient_update(...),    // from local training step
        justification: AdaptationJustification::GradientStep,
        confidence: 0.95,
    }
);

// The delta is XOR-merged into the target register
// Echo propagates to dependent registers (minimal, rotation-damped)

// ── Observation & introspection ──────────────────────────────────────────

// Query fabric state
let stats = fabric.stats();
println!("Hot registers: {}", stats.hot_count);
println!("Warm registers: {}", stats.warm_count);
println!("Cold registers: {}", stats.cold_count);
println!("KV cache usage: {}%", stats.kv_cache_usage);
println!("Energy per token (avg): {}", stats.energy_per_token_avg);
println!("Adaptation deltas applied: {}", stats.adaptation_count);

// Query a specific register (self-describing)
let reg = fabric.read(model.scope("layers").register("attention/weights/layer3"));
println!("Address: {:?}", reg.address);
println!("Payload type: {:?}", reg.payload_type);      // inferred from address context
println!("Lineage: {:?}", reg.xor_lineage);
println!("Version: {:?}", reg.version);
println!("Bloom anchor: {:?}", reg.bloom_anchor);

// ── Cross-node (future) ──────────────────────────────────────────────────

// Extend fabric across nodes
let remote_fabric = Fabric::connect_to("node2.example.com");
fabric.merge_with(remote_fabric);   // unified address space across nodes

// Compute routes across nodes: compute dispatches to where data lives
// Data doesn't move; compute does
```

### 19.2 C API (for llama.cpp, MLX FFI)

```
// C API for backend integration

binetic_fabric_t* binetic_create(const binetic_config_t* config);
void              binetic_destroy(binetic_fabric_t* fabric);

binetic_model_t* binetic_attach_model(binetic_fabric_t* fabric, const binetic_model_config_t* config);
void              binetic_detach_model(binetic_fabric_t* fabric, binetic_model_t* model);

// Compute: dispatch to data location
binetic_result_t binetic_compute(
    binetic_fabric_t* fabric,
    const binetic_compute_op_t* op,
    binetic_compute_result_t* result
);

// Read/write registers
const uint8_t*   binetic_read_register(binetic_fabric_t* fabric, const binetic_address_t* addr);
void              binetic_write_register(binetic_fabric_t* fabric, const binetic_address_t* addr, const uint8_t* data, size_t len);

// Adaptation
void              binetic_inject_adapt(binetic_fabric_t* fabric, const binetic_adapt_delta_t* delta);

// Stats
binetic_stats_t   binetic_get_stats(const binetic_fabric_t* fabric);

// Backend registration (for llama.cpp, MLX, etc.)
void              binetic_register_backend(binetic_fabric_t* fabric, binetic_backend_t* backend);
```

### 19.3 API Design Principles

1. **Addresses are first-class** — everything is addressed via `binetic_address_t` / `RegisterAddress`.
2. **No explicit memory management** — the fabric handles tiers, paging, eviction.
3. **Backend-agnostic** — the same API works with llama.cpp, MLX, or any backend.
4. **Self-describing** — reading a register returns its address, version, lineage — no external schema.
5. **Energy-aware** — stats include energy per token; config includes energy policy.
6. **Minimal surface** — the core API is small; complexity is inside the fabric.

---

## 20. Implementation Phases

### Phase 1: Core Fabric (Rust)

**Goal:** A working register fabric with addressing, tiers, XOR lineage, echo minimization, bloom filters, bitsliced lanes.

Deliverables:
- `RegisterAddress` type with full address arithmetic.
- `Register` type with payload, lineage, bloom anchor, version.
- Tier abstraction with RAM and mmap SSD backends.
- Temporal register ring buffer.
- XOR lineage chain with self-validation.
- Echo minimization: delta propagation, bloom gating, rotation damping.
- Bloom filter per sector (insert, might_contain, merge, rebuild).
- Bitsliced lane operations (XOR, AND, NOT, popcount across lanes).
- Rotation scheduler (basic fixed-period, no adaptation yet).
- Self-describing register: address → position, lineage, phase derivation.

**Not yet:** self-optimization, energy awareness, cross-model deltas, backend integration, on-device adaptation.

### Phase 2: Backend Integration

**Goal:** Connect llama.cpp and MLX backends to the fabric.

Deliverables:
- Llama.cpp backend: map weights to cold tier, KV cache to warm, activations to temporal. Compute dispatch with no-copy pointers.
- MLX backend (Apple): map weights/activations to MLX arrays backed by fabric memory. Compute dispatch via MLX operations.
- C API for backend FFI.
- Basic inference loop: load model, run token generation, measure throughput.

**Verification:** Run a 7B Q4 model on an 8GB Mac via the fabric, measure tok/s vs. raw llama.cpp.

### Phase 3: Rotation & Energy

**Goal:** Adaptive rotation scheduling with energy awareness.

Deliverables:
- Adaptive rotation: periods adjust based on access patterns, not fixed.
- Energy-aware scheduling: battery state, thermal state influence scheduling.
- Batched echo propagation: queue deltas, propagate on rotation tick.
- Prefetch energy trade-off: aggressive vs. conservative based on energy state.
- Energy stats: energy per token, per tier, per operation.

**Verification:** Measure energy per token on battery vs. plugged in. Show improvement over non-energy-aware scheduling.

### Phase 4: Self-Optimization

**Goal:** Fabric reorganizes itself based on observed access patterns.

Deliverables:
- Access observation: log accesses per sector, layer, spine.
- Hot/cold detection: registers that are accessed frequently vs. rarely.
- Co-access detection: registers accessed together.
- Reorganization: move hot inward, cold outward, co-locate co-accessed registers.
- Lane creation: organize sequential access patterns into lanes.
- Sector split/merge: adapt sector granularity to workload.

**Verification:** Show that a worked-in fabric has better locality (fewer page faults, fewer SSD accesses) than a cold-start fabric.

### Phase 5: Cross-Model Deltas & Adaptation

**Goal:** Multiple models sharing storage via XOR deltas; on-device adaptation.

Deliverables:
- XOR delta model attachment: attach a model as delta from a base.
- Delta computation: offline (during download) and online (during adaptation).
- Adaptation injection: inject deltas into weight registers.
- Echo-propagated adaptation: dependent registers updated via minimal echo.
- Effective capacity measurement: show multiple models fitting in constrained RAM.

**Verification:** Run two models simultaneously on an 8GB Mac, show total RAM usage vs. running them separately.

### Phase 6: Cluster & Remote Tier (Future)

**Goal:** Fabric spans multiple nodes; compute routes across nodes.

Deliverables:
- Remote tier: registers on remote nodes, accessed via network.
- Cross-node compute routing: dispatch compute to remote node where data lives.
- Network protocol for control plane (IPv4 metaphor becomes literal).
- Coherence protocol: ensure consistent state across nodes.

**Not planned for initial versions** — single-machine first.

---

## 21. Diagrams

### 21.1 Overall Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                           APPLICATION                                    │
│  "Run inference on model X with context Y"                              │
└─────────────────────────────────────────────────────────────────────────┘
                                   │
                                   ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                        BINETIC FABRIC (Rust Core)                       │
│                                                                          │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │  Compute Engine                                                  │   │
│  │  - Routes compute to data (no-copy)                             │   │
│  │  - Dispatches to backend (llama.cpp, MLX, native)              │   │
│  │  - Manages temporal registers (active compute scratchpad)      │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                   │                                      │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │  Rotation Scheduler                                              │   │
│  │  - Adaptive rotation periods (layer-dependent)                  │   │
│  │  - Energy-aware (battery, thermal)                              │   │
│  │  - Batches echo propagation                                     │   │
│  │  - Triggers reorganization                                      │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                   │                                      │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │  Echo Propagation Engine                                         │   │
│  │  - Minimal delta propagation                                     │   │
│  │  - Bloom-gated (skip empty sectors)                             │   │
│  │  - Rotation-damped (cold layers: scheduled, not immediate)     │   │
│  │  - XOR echo cancellation (redundant paths cancel)               │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                   │                                      │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │  Self-Optimizing Layout                                          │   │
│  │  - Observes access patterns                                      │   │
│  │  - Moves hot inward, cold outward                               │   │
│  │  - Co-locates co-accessed registers                             │   │
│  │  - Creates lanes for sequential access                          │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                   │                                      │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │  Addressing & Geometry                                           │   │
│  │  - RegisterAddress: spine, layer, ring, sector, shard, temporal│   │
│  │  - Address arithmetic: distance, neighbor, parent, child        │   │
│  │  - Address → position/lineage/phase derivation                  │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                   │                                      │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │  Storage Tiering                                                 │   │
│  │  - Tier abstraction: hot, warm, gpu, cold, remote               │   │
│  │  - Mmap-based cold storage (SSD)                                │   │
│  │  - Compression (weights, KV cache, activations)                 │   │
│  │  - Paging: fault pages on demand, evict on rotation             │   │
│  └─────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────┘
                                   │
                                   ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                     BACKENDS (Llama.cpp, MLX, etc.)                     │
│                                                                          │
│  Llama.cpp ────► receives pointers into fabric memory (no copy)        │
│  MLX         ────► receives MLX arrays backed by fabric memory          │
│  Custom      ────► implements Backend trait, registers with fabric      │
└─────────────────────────────────────────────────────────────────────────┘
                                   │
                                   ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                      HARDWARE (Mac, Linux, Cluster)                     │
│                                                                          │
│  CPU registers │ CPU L1/L2/L3 │ CPU RAM │ GPU memory │ SSD/NVMe │ ...  │
│       (hot)       (warm)       (warm)     (gpu)        (cold)          │
└─────────────────────────────────────────────────────────────────────────┘
```

### 21.2 4D Sphere Structure

```
                    Z (depth)
                    ▼
    ┌─────────────────────────────────────────┐
    │  Z=1023 (coldest layer)                │
    │  ┌─────────────────────────────────┐   │
    │  │  Y=1023                          │   │
    │  │  ┌───────────────────────────┐   │   │
    │  │  │  X=1023 (coldest spine)   │   │   │
    │  │  │  X=0     X=512    X=1023  │   │   │
    │  │  │  ┌──────┬──────┬──────┐   │   │   │
    │  │  │  │ S0   │ S1   │ S2   │   │   │   │
    │  │  │  │Shard│Shard│Shard│   │   │   │   │
    │  │  │  └──────┴──────┴──────┘   │   │   │
    │  │  │  └───────────────────────────┘   │   │
    │  │  └─────────────────────────────────┘   │
    │  │  ... (Y=0 to Y=1022)                  │
    │  └─────────────────────────────────────────┘
    │  ... (Z=0 to Z=1022)
    └─────────────────────────────────────────┘

    Each cell (X, Y, Z) is a register with:
      - IPv6 payload (bitsliced lane)
      - IPv4 control presence (routing metadata)
      - Sub-sphere (recursive, for hierarchy)
      - Bloom filter attachment (per sector)

    T dimension (temporal): separate 1024-register ring for active compute
```

### 21.3 Rotation Gradient

```
Layer 0 (hottest):     ◉———◉———◉———◉———◉   slow rotation, lane view
                        │               │
Layer 1:               ◉———◉———◉———◉———◉   medium rotation
                        │               │
Layer 2:               ◉———◉———◉———◉———◉   faster rotation
                        │               │
...                     ...
                        │               │
Layer N (coldest):     ◉———◉———◉———◉———◉   fastest bulk rotation (scheduled)

    ◉ = register
    —— = rotation path within layer
    │ = tier boundary (data moves across on rotation)

From center (Layer 0): ◉→◉→◉→◉→◉   lane (sequential, cache-friendly)
From outside (Layer N): ◉           tree view (lookup descends inward)
                          └┬┘
                           ◉
                           │
                          ◉←◉←◉←◉←◉   converge to center
```

### 21.4 Echo Propagation

```
Change at register R (layer 2, sector A):

    R (layer 2)
    │
    ├─── Echo delta (XOR) ───► Dependent D1 (layer 2, sector A)  ✓ propagate
    │       D1.payload ^= delta
    │       D1.xor_lineage ^= delta
    │       if delta doesn't affect D1: STOP
    │
    ├─── Echo delta (XOR) ───► Dependent D2 (layer 2, sector B)  ✓ propagate
    │       (if bloom(B) might_contain(R.address))
    │
    ├─── Echo delta (XOR) ───► Dependent D3 (layer 3, sector A)  ✗ BLOOM GATE
    │       (if bloom(sector_A.layer3) does NOT contain R.address)
    │       → echo dies here, no propagation to D3
    │
    ├─── Echo delta (XOR) ───► Dependent D4 (layer 1, sector A)  ✓ propagate
    │       (hotter layer: propagate immediately)
    │
    └─── Echo delta (XOR) ───► Dependent D5 (layer 4, sector A)  ✗ ROTATION DAMP
            (colder layer: queue for next rotation cycle, don't propagate now)


Echo cancellation example:

    Path 1: delta = 0xAB ^ 0xCD = 0x66
    Path 2: delta = 0xAB ^ 0xCD = 0x66  (same change, different route)

    At merge point: delta1 ^ delta2 = 0x66 ^ 0x66 = 0x00
    → Echo cancels, no net change, stop propagating
```

### 21.5 No-Copy Compute Routing

```
Traditional (copy-heavy):

  SSD ──copy──► RAM ──copy──► GPU ──compute──► RAM ──copy──► CPU
                                            │
                                            ▼
                                       result

Binetic (no-copy, compute-to-data):

  Fabric resolves addresses:
    weight W → in GPU memory (tier_gpu)
    activation A → in CPU RAM (tier_warm)
    KV cache K → in GPU memory (tier_gpu)

  Compute dispatch:
    If W and K are on GPU, A is on CPU:
      Option 1: move A to GPU (one copy), compute on GPU, result on GPU
      Option 2: split compute — GPU does W×K part, CPU does A part, merge result
      Option 3: if A is small, option 1 is cheaper; if A is large, option 2

    Fabric chooses based on:
      - Which backend supports the split
      - Energy/latency tradeoff
      - Current energy state (battery: prefer less movement)

  Result: written to temporal registers (in place, no extra copy)


Cross-node (future):

  Register W on Node 1, Register A on Node 2

  ComputeOp { operands: [W, A], routing: to_data }

  → Dispatch compute kernel to Node 1 (W's location)
  → Send A's data to Node 1 (or: dispatch to Node 2 and send result)
  → Or: split operation across nodes

  Control plane routes the dispatch, not the data.
```

### 21.6 Self-Optimizing Layout

```
Before optimization (random layout):

  Sector A: [W1, W5, W3, W2, W4]   // weights for layer 1, random order
  Sector B: [K1, K3, K2, K4, K5]   // KV cache entries, random order
  Sector C: [A1, A3, A2, A4, A5]   // activations, random order

  Access pattern: W1 → W2 → W3 → W4 → W5 (sequential)
  Result: each access is to a random location → cache misses, page faults

After optimization (observed sequential access):

  Sector A: [W1, W2, W3, W4, W5]   // reorganized into lane order
  Sector B: [K1, K2, K3, K4, K5]   // reorganized
  Sector C: [A1, A2, A3, A4, A5]   // reorganized

  Access pattern: W1 → W2 → W3 → W4 → W5 (still sequential)
  Result: each access is adjacent → cache hit, no page fault

Co-access optimization:

  Observed: W1 and K1 are always accessed together
  Action: move W1 and K1 into the same sector or adjacent sectors
  Result: accessing W1 brings K1 into cache too → fewer accesses

Hot/cold optimization:

  Observed: W5 is rarely accessed (cold), W1-W4 are frequently accessed (hot)
  Action: move W5 to colder tier (SSD), keep W1-W4 in warm tier (RAM)
  Result: RAM holds only what's needed → more effective capacity
```

### 21.7 XOR Lineage Chain

```
Register chain: R0 → R1 → R2 → R3

R0:
  address: (spine=0, layer=0, ring=0, sector=0, shard=0, temporal=0)
  payload: P0
  xor_lineage: L0 = hash(P0) ^ constant(0)

R1 (child of R0):
  address: (spine=0, layer=0, ring=0, sector=0, shard=1, temporal=0)
  payload: P1
  xor_lineage: L1 = L0 ^ hash(P1) ^ constant(1)
              = hash(P0) ^ hash(P1) ^ constant(0) ^ constant(1)

R2 (child of R1):
  address: (spine=0, layer=0, ring=0, sector=0, shard=2, temporal=0)
  payload: P2
  xor_lineage: L2 = L1 ^ hash(P2) ^ constant(2)
              = hash(P0) ^ hash(P1) ^ hash(P2) ^ constant(0) ^ constant(1) ^ constant(2)

R3 (child of R2):
  address: (spine=0, layer=0, ring=0, sector=0, shard=3, temporal=0)
  payload: P3
  xor_lineage: L3 = L2 ^ hash(P3) ^ constant(3)
              = hash(P0) ^ hash(P1) ^ hash(P2) ^ hash(P3) ^ constants_sum


Verification of R3:
  expected_L3 = recompute_lineage(R3.address)
              = walk parent chain (R2 → R1 → R0)
              = hash(P0) ^ hash(P1) ^ hash(P2) ^ hash(P3) ^ constants_sum
  if R3.xor_lineage == expected_L3:
    ✓ R3 is consistent with its lineage
  else:
    ✗ R3 is corrupted or inconsistent


Repair R3 from lineage:
  R3.payload = recover_from_lineage(R3.address)
              = P3 = f(L3, L2, constants)
              (recompute P3 from lineage and parent payloads)
```

### 21.8 Bitsliced Lane Example

```
8 values, each 8-bit (INT8):

  Values: [0xD3, 0x7A, 0xB2, 0xF1, 0x4C, 0x8E, 0x19, 0xC5]
          [211,  122,  178,  241,  76,  142,  25,  197]

Binary:
  0xD3 = 11010011
  0x7A = 01111010
  0xB2 = 10110010
  0xF1 = 11110001
  0x4C = 01001100
  0x8E = 10001110
  0x19 = 00011001
  0xC5 = 11000101

Bitsliced lanes (8 lanes, one per bit position):

  Lane 0 (MSB, bit 7): [1, 0, 1, 1, 0, 1, 0, 1]   ← bit 7 of each value
  Lane 1 (bit 6):       [1, 1, 0, 1, 1, 0, 0, 1]
  Lane 2 (bit 5):       [0, 1, 1, 1, 0, 0, 0, 0]
  Lane 3 (bit 4):       [1, 1, 1, 1, 0, 0, 1, 0]
  Lane 4 (bit 3):       [0, 1, 0, 0, 1, 1, 1, 0]
  Lane 5 (bit 2):       [0, 0, 0, 0, 1, 1, 0, 1]
  Lane 6 (bit 1):       [1, 1, 1, 0, 0, 1, 0, 0]
  Lane 7 (LSB, bit 0):  [1, 0, 0, 1, 0, 0, 1, 1]


XOR two value arrays (A ^ B):

  A = [0xD3, 0x7A, ...]
  B = [0xB2, 0xF1, ...]

  Per-lane XOR:
    Lane 0: [1, 0, ...] ^ [1, 1, ...] = [0, 1, ...]
    Lane 1: [1, 1, ...] ^ [0, 1, ...] = [1, 0, ...]
    ...
    Lane 7: [1, 0, ...] ^ [0, 1, ...] = [1, 1, ...]

  Result: bitsliced representation of A ^ B

  On Apple Silicon: each lane XOR is a single NEON/AMX instruction
  processing 64+ bits in parallel → 8 lanes × 64 bits = 512 bits per instruction


Quantization mapping:

  INT4 values: 4 lanes (bits 3-0)
  INT8 values: 8 lanes (bits 7-0)
  FP8 values: 8 lanes (sign, exponent, mantissa split)
  Binary values: 1 lane (the bit itself)

  Fewer bits → fewer lanes → more compact → faster bitwise ops
```

---

## 22. Extended Precision Registers (u256, u512, u1024…)

### 22.1 The Concept

The base register payload is a `BitslicedLane` — a single bit vector. The natural extension: **compound registers** formed by combining multiple base IPv6 registers "the long way," creating wider registers with arbitrary precision.

A u256 register = multiple base registers combined. A u512 = more bases. A u1024 = even more. The precision is unbounded — you compound as many bases as needed.

This is not just "bigger registers." It's a **fractal extension mechanism**: each base register in the compound can itself be extended, recursively. The fabric supports arbitrary-depth precision without a separate bigint library or blob store.

### 22.2 Combining Methods

How you combine base registers determines what the extended register is good for:

#### XOR-Chain Combining

```
extended.payload = base0.payload ^ base1.payload ^ … ^ baseN.payload
extended.xor_lineage = base0.xor_lineage ^ base1.xor_lineage ^ … ^ baseN.xor_lineage
```

The extended register's lineage is the XOR of all base lineages. This is the most natural extension of the XOR-native fabric:

- **Fold stays free**: `fold(extended_A, extended_B)` = XOR of all constituent folds.
- **Self-validating**: the extended lineage is verifiable from the base lineages.
- **Echo-minimal**: a change in one base propagates as a delta from that base only — the other bases are unchanged.
- **Best for**: values where XOR semantics make sense (checksums, diffable data, cryptographic chains).

#### Sequential Combining

```
extended.payload = base0.payload || base1.payload || … || baseN.payload
  // Concatenation: base0 is the first chunk, base1 the next, etc.
```

The bases are concatenated in order. The extended register's payload is the concatenation of all base payloads.

- **Arbitrary-precision integers**: a u1024 integer is 16 × u64 bases, concatenated.
- **Large tensors**: a large matrix block spans multiple bases, each base holding a tile.
- **Blobs / large values**: a database record or file chunk is a sequential compound.
- **Best for**: values that need contiguous bit layout (bigint, tensors, blobs).

#### Interleaved Combining

```
extended.payload[i] = base(i % N).payload[i / N]
  // Bit i comes from base (i % N), at position (i / N) within that base
```

Bits are interleaved across bases. For N bases, bit 0 comes from base0, bit 1 from base1, …, bit N from base0 (next position), etc.

- **SIMD-parallel**: operations across bases are independent — you can process all N bases in parallel.
- **Vector-like**: each base holds one "lane" of the extended value; together they form a wide vector.
- **Best for**: SIMD-style parallel operations, vector values, quantized arrays where each base holds one quantization lane.

#### Layered Combining

```
extended.high_bits = base0.payload   // most significant chunk
extended.low_bits  = base1.payload   // least significant chunk
  // Each base holds a different significance layer
```

Bases are layered by significance. Base0 = most significant bits, base1 = next, etc. This is the multi-precision arithmetic layout.

- **Multi-precision arithmetic**: bigint add, multiply, divide operate on the layered bases with carry propagation between layers.
- **Best for**: bigint arithmetic, multi-word operations, values that need significance-ordered layout.

### 22.3 Extended Address

An extended register is addressable through its constituent bases:

```
ExtendedRegisterAddress {
    base_count:     u8,              // how many base registers compose this
    precision:      u16,             // total bits: 64, 128, 256, 512, 1024, 2048…
    combining:      CombiningMethod,  // XOR-Chain, Sequential, Interleaved, Layered
    base_addresses: [RegisterAddress; base_count],  // the constituents
}
```

The extended address is **derivable** from the base addresses — you don't lose self-description. Given the base addresses and the combining method, you can reconstruct the extended address.

For address-based routing and lookup, the extended register is addressed by its **primary base** (base0) — the first constituent. The other bases are reachable through the compound relationship.

### 22.4 Tier Spanning

A compound register can span tiers:

```
u1024 register = 16 × u64 bases

  base0  → tier_hot  (active, in CPU registers)
  base1  → tier_warm (recently accessed, in RAM)
  base2  → tier_warm
  ...
  base14 → tier_cold (rarely accessed, mmap'd from SSD)
  base15 → tier_cold
```

The compound register is a single logical value that physically lives across multiple tiers. The fabric handles accessing each base from its tier transparently — fault pages for cold bases, read warm bases directly, keep hot bases in registers.

This is how you store **large values in limited RAM**: the hot portion is in RAM, the cold portion is on SSD, and the compound register spans both. Accessing the full value faults in the cold bases on demand.

### 22.5 Self-Checking at Scale

Integrity verification extends to compound registers:

```
extended.xor_lineage = XOR of all base lineages

Verification:
  for each base_i:
    verify base_i.lineage_consistent()
  verify extended.xor_lineage == XOR of verified base lineages
```

If any base is corrupt, you detect it locally. If the corruption is in a cold base, you can repair from SSD backup or recompute from ancestors.

### 22.6 Use Cases

| Use case | Combining method | Why |
|---|---|---|
| Bigint arithmetic | Layered | Significance-ordered for carry propagation |
| Large matrix tiles | Sequential | Contiguous layout for matmul |
| Large model weights | Sequential | Contiguous across bases, tiered storage |
| Database rows / documents | Sequential or XOR-chain | Record as compound; XOR-chain for diffable records |
| Cryptographic values | XOR-chain | Lineage extends naturally; self-validating |
| SIMD vectors | Interleaved | Parallel ops across bases |
| Large blobs / files | Sequential | Contiguous, tiered, mmap-friendly |

### 22.7 Diagrams

#### XOR-Chain Compound (u256 = 4 × u64)

```
Register A (u256) = base0 ^ base1 ^ base2 ^ base3

  base0: [u64 value] ─┐
  base1: [u64 value] ─┼─ XOR-chain ──► [u256 value]
  base2: [u64 value] ─┤
  base3: [u64 value] ─┘

  lineage(A) = lineage(base0) ^ lineage(base1) ^ lineage(base2) ^ lineage(base3)

  Change in base1 only:
    delta = new_base1 ^ old_base1
    A.payload ^= delta         // XOR-merge delta into compound
    A.lineage ^= delta_lineage // update lineage minimally
    // base0, base2, base3 unchanged — no propagation from them
```

#### Sequential Compound (u512 = 8 × u64)

```
Register B (u512) = base0 || base1 || … || base7

  base0: [u64, bits 511..448]  ← most significant
  base1: [u64, bits 447..384]
  ...
  base7: [u64, bits 63..0]     ← least significant

  Physical layout:
    base0 in tier_hot  (hot, accessed frequently)
    base1 in tier_warm
    ...
    base6 in tier_cold (rarely accessed)
    base7 in tier_cold

  Access to full u512:
    Read base0 from hot (immediate)
    Read base1 from warm (fast)
    Fault base6, base7 from SSD (on demand)
    Assemble: base0 || base1 || … || base7
```

#### Fractal Extension

```
u64 base register (atomic)
  │
  ├── u128 = 2 × u64 (compound)
  │     │
  │     ├── u256 = 2 × u128 (compound of compounds)
  │     │     │
  │     │     └── u512 = 2 × u256
  │     │           │
  │     │           └── u1024 = 2 × u512
  │     │
  │     └── u1024 = 8 × u128 (direct compound)
  │
  └── u256 = 4 × u64 (direct compound)
```

Any register can be extended. Extensions can be nested. The fabric handles arbitrary depth.

---

## 23. Database Semantics — Binetic as a Database Engine

### 23.1 The Mapping

The binetic fabric is already most of a database engine. The mapping from database concepts to fabric mechanisms is direct:

| Database concept | Binetic equivalent |
|---|---|
| **Row / document / record** | A compound register (u256/u512/u1024+) with self-describing schema version in its `version` field |
| **Primary key** | The register address — spine encodes entity ID, layer encodes shard, sector encodes range |
| **Index** | Bloom filter per sector (probabilistic lookup) + ordered sector layout (range queries) + address arithmetic (exact lookup) |
| **Transaction** | Echo propagation with depth limit = isolation; XOR fold = atomicity; lineage = consistency; cold tier mmap = durability |
| **Query execution** | Compute-to-data routing: dispatch the query predicate to where the data lives, not vice versa |
| **Compaction** | Rotation: cold data moves outward, overlapping records merge, bloom filters update, old versions evict |
| **Replication** | Cross-node fabric: writes propagate via structured echo to replica nodes; bloom-gating prevents redundant propagation |
| **Schema** | Self-describing: each register carries its schema version; the database discovers structure from the data, no external catalog |
| **WAL / journal** | Echo log: the sequence of deltas *is* the journal; replay = deterministic reconstruction from initial state + deltas |
| **Partitioning** | Spine assignment: different spines = different partitions; routing goes to the right spine based on entity ID |
| **Backup** | Cold tier mmap = backup on SSD; lineage chain = recoverable from ancestors; echo log = replayable to any point in time |

### 23.2 ACID via the Fabric

#### Atomicity

A write is a delta injected into a compound register. The XOR fold is atomic — either the delta merges or it doesn't. There is no partial state. A transaction that writes multiple registers either commits all deltas or none (via echo batch + rollback on failure).

```
Transaction {
    deltas: [Delta; N],           // one delta per modified register
    addresses: [RegisterAddress; N],
    purpose: TransactionPurpose,
}

Commit:
  for each (address, delta) in transaction:
    inject_adapt(address, delta, purpose)  // XOR-merge
  // All or nothing: if any fails, roll back by injecting inverse deltas
```

#### Consistency

XOR lineage chain ensures a record's state is consistent with its history. If a record's lineage breaks, it's corrupt and repairable from ancestors. The database enforces constraints via echo propagation — a constraint violation generates a delta that propagates to dependent records (e.g., foreign key violations propagate to the referencing records).

#### Isolation

Echo propagation has a depth limit and scope. A transaction's echoes propagate only to dependents within its scope. Concurrent transactions on disjoint address ranges don't interfere. The bloom-gating mechanism ensures that a transaction's echoes don't reach registers outside its intended scope.

```
Isolation levels (via echo scope):

ReadUncommitted:  echoes propagate freely (no isolation)
ReadCommitted:    echoes propagate only to committed state (echo depth = 1)
RepeatableRead:   echoes propagate within transaction's address scope
Serializable:     echoes propagate only within a locked address range
```

#### Durability

Writes to the cold tier are mmap'd to SSD. The echo log (delta sequence) is written before acknowledging the transaction. Crash recovery = replay the echo log from the last checkpoint.

```
Write path:
  1. Compute delta = new_payload ^ old_payload
  2. Write delta to echo log (durability)
  3. Inject delta into target register (in-memory)
  4. Echo propagates to dependents
  5. Acknowledge write

Crash recovery:
  1. Load last checkpoint (cold tier mmap)
  2. Replay echo log from checkpoint to current
  3. Reconstruct in-memory state
  4. Resume operations
```

### 23.3 Indexing

The fabric provides multiple indexing mechanisms:

#### Primary Index (address-based)

The register address *is* the primary key. Lookup by address is O(1) — you compute the address from the entity ID and read the register. No separate index needed for primary key lookups.

```
Entity ID → RegisterAddress (via spine/sector encoding)
RegisterAddress → Register (via address lookup)
```

#### Bloom Index (probabilistic)

Each sector has a bloom filter. To find records matching a predicate:

1.Compute the bloom anchor for the predicate.
2.Check each sector's bloom: does it might_contain records matching this predicate?
3.For sectors that might contain matches, descend into the sector and scan.

This is a probabilistic index — false positives are possible (you scan a sector that has no matches), but false negatives are not (you never skip a sector that has matches).

#### Range Index (ordered)

Sectors are ordered within a layer. A range query (entity IDs from 0x1000 to 0x2000) scans the sectors in that range. The address arithmetic (neighbor, parent, child) supports ordered traversal.

#### Secondary Index (via compound registers)

A secondary index is a compound register that maps from index key to primary address:

```
SecondaryIndex {
    key:        RegisterAddress,   // index key (e.g., "age=30")
    value:      RegisterAddress,   // primary address of matching record
    lineage:    u128,              // for integrity
}
```

Updates to the primary record propagate to secondary indexes via echo — the delta is injected into the index register as well.

### 23.4 Query Execution

Queries are register operations dispatched via compute-to-data routing:

```
// Scan all records in a sector range where age > 30
let results = fabric.query(
    scope: SectorQuery {
        spine: 0x1000,           // entity type = users
        layer: 5,                // shard
        sector_range: 0x0000..=0xFFFF,
    },
    filter: |addr, reg| {
        // Decode the record's fields from its payload
        let age = reg.payload.get_field("age");
        age > 30
    },
    routing: ToData,  // push the filter to where the data is
    output: RegisterAddress::new(0xF000, 0, 0, 0, 0, 0),  // results go here
);
```

The fabric dispatches the scan to the tiers where the data lives. For cold data, it faults pages. For warm data, it reads directly. The filter runs where the data is — minimizing data movement.

### 23.5 Compaction

The rotation scheduler handles compaction:

- **Cold data moves outward**: records that haven't been accessed recently are moved to colder tiers (SSD).
- **Overlapping records merge**: if two records in the same sector are XOR-close (differ by a small delta), they merge into one canonical + one diff.
- **Old versions evict**: when a record is updated, the old version becomes cold and eventually evaporates (or stays as a delta for versioning).
- **Bloom filters update**: after compaction, bloom filters are rebuilt to reflect the new sector contents.

This is continuous compaction — it happens during rotation cycles, not as a separate process. No write amplification spikes, no read stalls.

### 23.6 Replication

The fabric extends across nodes. A write to a register on Node 1 propagates via structured echo to replica nodes:

```
Node 1: write register R
  → delta injected into R
  → structured echo sent to replica nodes (Node 2, Node 3)
  → replicas XOR-merge the delta into their copy of R
  → bloom-gating: replicas only process echoes for registers they host
```

Replication is asynchronous (echo-based) or synchronous (wait for echo acknowledgment), depending on the consistency requirement.

### 23.7 Schema-on-Read

There is no separate schema catalog. Each register carries its schema version in its `version` field. The database discovers the schema from the data:

```
Read register R:
  version = R.version
  schema = schema_for_version(version)  // derived from version, not stored externally
  fields = decode_payload(R.payload, schema)
```

Schema evolution is handled by version increment — new records get a new version, old records keep their version. The database can read both.

### 23.8 Use Cases

| Use case | Why binetic database fits |
|---|---|
| **Embedded database** (single machine, limited RAM) | Fabric scales from 8GB Mac to server; tiers handle capacity; compound registers handle large records |
| **Edge database** (low power, intermittent connectivity) | Energy-aware scheduling; echo-based replication; mmap-backed cold tier for SSD storage |
| **Time-series database** (append-heavy, temporal) | Temporal registers + rotation = natural time-series layout; bloom filters for recent-data lookups |
| **Document store** (schema-less, flexible) | Self-describing registers; schema-on-read; compound registers for large documents |
| **Key-value store** (simple, fast) | Address = key; O(1) lookup; bloom filters for existence checks; tiered storage for large values |
| **Graph database** (traversal-heavy) | Address arithmetic (neighbor, parent, child) for graph traversal; echo propagation for graph updates |
| **Distributed database** (multi-node) | Cross-node fabric; echo-based replication; compute-to-data routing across nodes |

### 23.9 Comparison to Existing Databases

| Feature | Binetic | LSM-tree (RocksDB) | B-tree (SQLite) | Document (MongoDB) |
|---|---|---|---|---|
| **Self-describing** | Yes — registers carry schema | No | No | Partial (BSON schema) |
| **Geometric indexing** | Yes — bloom per sector + address order | No (ordered log) | Yes (B-tree) | No (indexes separate) |
| **Built-in compaction** | Yes — rotation | Yes — compaction | No (VACUUM) | No (cleanup jobs) |
| **Built-in replication** | Yes — cross-node echo | No (separate) | No (separate) | Yes (replica sets) |
| **Tiered storage** | Yes — hot/warm/gpu/cold tiers | Yes — block cache | Limited | Limited |
| **Compound records** | Yes — u256/u512/etc. | No (fixed block) | No (page-based) | Yes (BSON) |
| **XOR lineage integrity** | Yes | No | No (checksums) | No |
| **Echo log = WAL** | Yes | Yes (WAL separate) | Yes (WAL) | Yes (oplog) |
| **Compute-to-data** | Yes | No | No | No |
| **Energy-aware** | Yes | No | No | No |

### 23.10 Diagrams

#### Record as Compound Register

```
Database row: User { id: 42, name: "Alice", age: 30, email: "alice@example.com" }

Encoded as compound register (u256 = 4 × u64 bases):
  base0: id (u64 = 42)
  base1: name (u64 hash of "Alice")
  base2: age (u64 = 30)
  base3: email (u64 hash of "alice@example.com")

  Address: RegisterAddress::new(spine=0x1000, layer=5, ring=0, sector=42, shard=0, temporal=0)
    // spine = entity type (users)
    // sector = entity ID (42)

  Schema version: encoded in register.version field

  Lookup: compute address from entity type + ID → read register → decode fields
```

#### Transaction as Echo Batch

```
Transaction: update user 42's age from 30 to 31

  delta_age = new_age ^ old_age = 31 ^ 30 = 1
  delta = BitslicedLane::from_bits([1])  // minimal delta

  Inject delta into user 42's age register:
    target: RegisterAddress::new(0x1000, 5, 0, 42, 2, 0)  // base2 = age field
    delta: delta_age

  Echo propagates to:
    - Secondary index on age (index register updates)
    - Any records that reference user 42 (foreign keys)
    - Audit log register (if attached)

  All via echo — minimal propagation, bloom-gated, rotation-damped
```

#### Index Lookup

```
Query: find all users with age > 30

  1. Compute bloom anchor for predicate "age > 30"
  2. Check each sector's bloom filter:
     Sector 0x0000: bloom.might_contain(predicate_anchor)? → maybe → scan
     Sector 0x0001: bloom.might_contain(predicate_anchor)? → no → skip
     Sector 0x0002: bloom.might_contain(predicate_anchor)? → maybe → scan
     ...
  3. For sectors that might contain matches, scan registers and apply filter
  4. Return matching records

  Bloom filter avoids scanning empty sectors — probabilistic index
```

#### Cross-Node Replication

```
Node 1 (primary)          Node 2 (replica)         Node 3 (replica)
     │                         │                         │
     ├─ write R (delta) ──────┼── echo delta ───────────┼── echo delta ──► apply
     │                         │                         │
     ├─ bloom-gated ──────────►│  (only if R exists)    │  (only if R exists)
     │                         │                         │
     └─ acknowledge ◄──────────┼── echo done ────────────┼── echo done ─────►
```

---

## Appendix A: Glossary

| Term | Definition |
|---|---|
| **Fabric** | The binetic runtime — the register space, tiers, scheduling, echo propagation, optimization. |
| **Register** | A single addressable unit in the fabric, carrying payload + self-describing metadata. |
| **RegisterAddress** | 128-bit address encoding spine, layer, ring, sector, shard, temporal. |
| **IPv6 data plane** | The payload plane — carries register data, lineage, bloom anchors. |
| **IPv4 control plane** | The routing plane — carries operations, routing metadata, echo propagation. |
| **Spine** | A 1D fiber through the 3D cube — a coherent access path. |
| **Layer** | Depth from center (0 = hottest, N = coldest) — determines storage tier. |
| **Ring** | Rotational position within a layer — determines when data moves, when echoes propagate. |
| **Sector** | Angular grouping within a ring — registers in a sector are conceptually related. |
| **Shard** | Sub-division within a sector — fine-grained addressing. |
| **Temporal register** | A register in the T dimension — active compute scratchpad, ephemeral. |
| **Rotation** | Scheduled movement of data between tiers, with layer-dependent periods. |
| **Echo** | Propagation of a change through the fabric — minimized via deltas, bloom gating, rotation damping, XOR cancellation. |
| **XOR lineage** | Chain of custody for a register — XOR fingerprint derived from parent lineage + own payload. |
| **Fold** | XOR-based combination of two registers — self-inverse, associative, schema-free. |
| **Bloom filter** | Probabilistic membership test per sector — gates routing and echo propagation. |
| **Bitsliced lane** | Bit-plane representation of values — SIMD-friendly, natural for quantization. |
| **Tier** | Abstract storage level (hot, warm, gpu, cold, remote) — maps to physical hardware. |
| **No-copy compute-to-data** | Dispatch compute to where data lives, rather than moving data to compute. |
| **Self-optimizing** | Fabric reorganizes based on observed access patterns for better locality. |
| **Energy-aware** | Scheduling considers battery state, thermal state, energy per access. |
| **Cross-model delta** | Model variant stored as XOR diff from base model — saves capacity. |
| **Adaptation delta** | On-device update injected as XOR delta into weight registers. |
|| **Compound register** | A register formed by combining multiple base registers "the long way" — u256, u512, u1024, etc. Supports XOR-chain, sequential, interleaved, and layered combining methods. |
|| **Combining method** | How base registers are combined into a compound: XOR-Chain (payload = XOR of bases), Sequential (concatenation), Interleaved (bit i from base i%N), Layered (significance-ordered). |
|| **Extended precision** | Arbitrary-width registers (u256, u512, u1024…) formed by compounding base registers. Unbounded precision without a separate bigint library. |
|| **Record** | A database record — a compound register with self-describing fields, schema version, and metadata. Each field is a base register in the compound. |
|| **Database engine** | Binetic applied to records instead of inference tensors. ACID via echo propagation (atomicity = XOR fold, consistency = lineage, isolation = echo scope, durability = mmap + echo log). Bloom-indexed, self-describing, tiered, replicated via cross-node echo. |
|| **Query scan** | Database query executed via compute-to-data routing: scan sectors, bloom-gating skips empty sectors, filter runs where data lives. |
|| **Tombstone** | A delete marker register — writing a tombstone register marks a record as deleted without removing the data (echo-propagated). |

---

## Appendix B: Open Questions & Trade-offs

### B.1 Address Space Size

The 128-bit address space (2^128 addresses) is vast. Practical question: how much of this space do we actually use, and how do we handle sparsity?

- **Current answer**: the fabric only populates a tiny fraction; bloom filters handle sparsity.
- **Open question**: do we need hierarchical address compression for very large fabrics (cluster scale)?

### B.2 Rotation Granularity

How fine-grained is rotation? Per-register? Per-sector? Per-layer?

- **Current answer**: rotation is per-layer (the whole layer rotates on its period), with intra-layer movement per-sector.
- **Open question**: does per-register rotation (finer) provide better locality at higher overhead?

### B.3 Echo Propagation Depth

How far do echoes propagate? Unlimited? Capped by layer? Capped by hop count?

- **Current answer**: echoes propagate to dependents, gated by bloom filters, damped by rotation, canceled by XOR. No explicit depth limit — the structure naturally limits propagation.
- **Open question**: do we need an explicit depth limit for pathological cases (e.g., a register that everything depends on)?

### B.4 Self-Optimization Overhead

Observing access patterns and reorganizing has overhead. When does it pay off?

- **Current answer**: observation is lightweight (sampling, not every access). Reorganization happens during rotation cycles (already happening). Payoff is in reduced page faults and better locality over time.
- **Open question**: what's the break-even point? How many accesses before reorganization is worth it?

### B.5 Compression vs. Compute

Compressed data in cold tier saves space but costs decompression on access. When is compression worth it?

- **Current answer**: cold tier is SSD-backed — space is cheap, but SSD latency is high. Compression reduces SSD I/O (fewer bytes to read), which can offset decompression cost. Worth it for large, rarely-accessed data (weights, old KV cache).
- **Open question**: what compression algorithm balances ratio vs. speed for this use case? LZ4? ZSTD? Something custom for bitsliced data?

### B.6 Backend Selection

When multiple backends can handle an operation (e.g., both CPU and GPU), how does the fabric choose?

- **Current answer**: based on data location (compute where data is), energy state (battery: prefer CPU), latency requirements (plugged in: prefer GPU), and backend capability (which ops each backend supports).
- **Open question**: do we need a more sophisticated cost model (predictive, learning-based)?

### B.7 Security

The fabric is a local system, but cross-model deltas and on-device adaptation touch model integrity. What guarantees do we need?

- **Current answer**: XOR lineage provides integrity verification. Adaptation deltas are local only. No network exposure in initial versions.
- **Open question**: do we need signed deltas (for verified model updates)? Do we need isolation between models (one model's deltas shouldn't affect another)?

---

## Appendix C: References & Influences

- **llama.cpp** — inference backend, model format (GGUF), quantization.
- **MLX** — Apple's inference framework, unified memory model, Apple Silicon optimization.
- **RDMA / NVMe-oF** — network-attached memory concepts (inspiration for the compute-to-data routing).
- **XOR-based deduplication** — rsync, binary diffing, delta encoding.
- **Bloom filters** — probabilistic data structures for membership testing.
- **Cache replacement policies** — LRU, ARC, CLOCK — inspiration for rotation as cache policy.
- **Self-describing formats** — Cap'n Proto, FlatBuffers — schema-less interpretation.
- **Bitsliced encryption** — bitsliced AES — inspiration for bitsliced compute.
- **Chain of custody / Merkle trees** — XOR lineage as a light-weight integrity chain.

---

*End of Architecture Specification v0.1.0*
