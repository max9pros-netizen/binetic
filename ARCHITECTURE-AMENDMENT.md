# Binetic — Strongest Possible Version Amendment
## Temporal Timers, Temporal Logic, Echo Differentiation, Nested Pispheres, Self-Hosting Network Protocol

**Amends:** ARCHITECTURE.md v0.1.0
**Status:** Architecture Amendment

---

## 1. Temporal Timers

### 1.1 Timer-Driven State Machine

Every register has an associated **temporal timer** — a countdown that triggers state transitions:

```
Timer {
    deadline:    Instant,        // when the timer fires
    period:      Duration,       // recurring interval (if periodic)
    on_fire:     TimerAction,    // what happens when timer expires
    repeat:      bool,           // one-shot vs periodic
    generation:  u8,             // which generation of this timer (for cancellation)
}
```

**Timer actions** (what fires):

| Action | Meaning |
|---|---|
| `Rotate(layer)` | Trigger rotation for a layer |
| `PropagateEcho(addr)` | Flush pending echo for a register |
| `RebuildBloom(addr)` | Rebuild bloom filter for a sector |
| `Promote(addr)` | Move register from cold → warm tier |
| `Demote(addr)` | Move register from warm → cold tier |
| `EvalEcho(addr)` | Evaluate whether an echo should continue propagating |
| `Checkpoint(layer)` | Save layer state to cold tier |
| `Adapt(reg)` | Inject adaptation delta for a register |

### 1.2 Epoch-Based Scheduling

Time is divided into **epochs** — fixed-duration windows during which the fabric makes scheduling decisions:

```
Epoch {
    number:        u64,
    duration:      Duration,       // e.g., 16ms (60Hz tick)
    start_time:    Instant,
    decisions:     Vec<ScheduledAction>,
    completed:     bool,
}
```

Epochs replace ad-hoc rotation triggers:

- **Epoch N**: schedule rotation for layers 0-3 (hot), echo propagation for layers 4-10
- **Epoch N+1**: schedule rotation for layers 4-7, bloom rebuild for outer layers
- **Epoch N+2**: checkpoint cold layers, adapt registers with high delta counts

The epoch counter is part of the **address** (temporal field), so every register knows which epoch it was last updated in.

### 1.3 Temporal Timeout as Echo Policy

Echoes that don't resolve within a timeout are **dropped**:

```
echo_timeout: Duration,   // default: 500ms per layer

On echo timeout:
  echo.dead = true
  echo.source.register.echo_counter -= 1
  If source.echo_counter == 0:
    source.register.state = Stable
```

This prevents echo storms from hanging the fabric.

---

## 2. Temporal Logic

### 2.1 Time-Dependent Register Validity

Registers carry **temporal predicates** that determine when they're valid:

```
TemporalPredicate {
    valid_from:  Option<Instant>,   // None = always valid
    valid_until: Option<Instant>,   // None = valid until overwritten
    version:     u64,               // monotonic version counter
    created_at:  Instant,
    modified_at: Instant,
}
```

A register is **live** if `now >= valid_from && now < valid_until`.

Expired registers are reclaimed by the rotation scheduler — they don't consume bloom filter space or echo propagation paths.

### 2.2 Causal Ordering via XOR Lineage

The XOR lineage chain encodes **causal order**:

```
If A.xor_lineage ^ B.xor_lineage == C.payload:
  // C was derived from A and B in this order
  // A happened before B (A is the base, B is the modifier)
```

This gives us **partial temporal ordering** without a clock — the lineage chain is a causal DAG encoded as XOR fingerprints.

### 2.3 Temporal Queries

```
fabric.query(TemporalQuery {
    before:    Some(Instant::now() - Duration::from_secs(60)),
    after:     Some(Instant::now() - Duration::from_secs(120)),
    layer:     Some(0..=3),           // hot layers only
    address:   None,
    lineage:   None,
}) → Vec<RegisterSnapshot>
```

Query results are temporally consistent — they represent a single fabric snapshot at a specific time, derived from the lineage chain.

### 2.4 Version Vectors for Cross-Node Consistency

When the fabric spans nodes, each node maintains a **version vector**:

```
VersionVector {
    node_id:   NodeId,
    versions:  HashMap<NodeId, u64>,  // each node's latest epoch
}
```

A register's version vector is part of its lineage. When merging state from another node, the version vectors determine which register is newer (causally) — standard vector clock comparison.

---

## 3. Echo Differentiation

### 3.1 Echo Types

Not all echoes are equal. Echoes are **typed** with different propagation semantics:

```
EchoType {
    Read    = 0,    // read-only: no mutation, no further propagation
    Write   = 1,    // mutation: propagates to dependents
    Fold    = 2,    // combine: merges with existing register via XOR
    Adapt   = 3,    // adaptation: triggers re-evaluation of dependent registers
    Rotate  = 4,    // rotation: moves registers between tiers
    Control = 5,    // control plane: routing metadata updates
    Diagnostic = 6, // diagnostic: logged but not propagated
}
```

### 3.2 Echo Differentiation Rules

| Echo Type | Propagation Depth | Bloom Gate | Rotation Damped | XOR Cancel |
|---|---|---|---|---|
| Read | 0 (local only) | N/A | N/A | N/A |
| Write | Full (all dependents) | Yes | Yes (cold layers) | Yes |
| Fold | Full | Yes | Yes | Yes |
| Adapt | Limited (depth=3) | Yes | Yes | Yes |
| Rotate | N/A (scheduler action) | N/A | N/A | N/A |
| Control | Full (routing changes) | Yes | No | Yes |
| Diagnostic | 0 (local log) | N/A | N/A | N/A |

### 3.3 Echo Priority

Echoes have **priority levels** that determine ordering when multiple echoes compete for the same register:

```
EchoPriority {
    Critical = 0,   // rotation, adaptation — must process immediately
    High     = 1,   // writes, folds — process within current epoch
    Normal   = 2,   // echo propagation — process next rotation tick
    Low      = 3,   // bloom rebuild, diagnostic — process when idle
}
```

Echoes are processed in priority order within each epoch. Critical echoes can preempt lower-priority work.

### 3.4 Echo Cancellation Refinement

XOR cancellation now considers echo type:

```
delta_A ^ delta_B == 0 AND A.type == B.type:
  // Same change, same type → full cancellation
  Echo dies

delta_A ^ delta_B == 0 AND A.type != B.type:
  // Same bits, different types → partial cancellation
  // The XOR result is zero but the types differ
  // Keep the higher-priority type, drop the lower

delta_A ^ delta_B != 0:
  // Different changes → merge
  Combined delta = delta_A ^ delta_B
  Combined type = max(A.type, B.type)  // higher priority wins for type
```

---

## 4. Nested Pispheres (Child Cubes)

### 4.1 From Distributed Nodes to Nested Nodes

Instead of "distributed nodes" (separate machines), each register **is** a node that contains a child pisphere:

```
Pisphere (root, 1024³ × 1024 temporal)
├── Register(X, Y, Z, T) — a node
│     ├── IPv6 payload + lineage
│     ├── IPv4 control presence
│     └── Child Pisphere (X', Y', Z', T') — nested within this register
│           ├── Register(X', Y', Z', T') — grandchild node
│           │     ├── Payload
│           │     └── Child Pisphere (X'', Y'', Z'', T'') — great-grandchild
│           │           └── ... (recursion depth 2-4)
│           └── ...
├── Register(X+1, Y, Z, T) — sibling node
│     └── Child Pisphere(...)
└── ...
```

**Nesting depth** is determined by the **shard field** of the address:
- Depth 0: root pisphere (the full 1024⁴ fabric)
- Depth 1: sector-level pisphere (1024³ registers per sector)
- Depth 2: shard-level pisphere (1024² registers per shard)
- Depth 3: register-level pisphere (1024 registers per register-group)

### 4.2 Recursive Addressing

Each nested pisphere has its own **local address space** that maps to the parent:

```
GlobalAddress = (spine, layer, ring, sector, shard, temporal)
LocalAddress  = (local_spine, local_layer, local_ring, local_sector)

GlobalAddress → LocalAddress:
  spine   → local_spine   (upper 6 bits)
  layer   → local_layer   (upper 6 bits)
  ring    → local_ring    (upper 6 bits)
  sector  → local_sector  (upper 6 bits)
  shard   → nesting depth + local shard
  temporal→ local_temporal
```

A register at global address (spine=0x3A2, layer=0x1B4, ring=0x0F7, sector=0x2D1, shard=0x03, temporal=0x4E8) contains a child pisphere whose local addresses are derived from the upper bits:

- Child local spine = 0x3A (upper 6 bits of parent spine)
- Child local layer = 0x1B (upper 6 bits of parent layer)
- Child local ring = 0x0F (upper 6 bits of parent ring)
- Child local sector = 0x2D (upper 6 bits of parent sector)

### 4.3 Self-Contained Nested Pispheres

Each child pisphere is **self-contained**:
- Has its own IPv4/IPv6 interlaced planes
- Has its own bloom filters, rotation schedule, echo engine
- Has its own tier mapping (the child's "hot" layer maps to the parent's register)
- Operates independently — the parent doesn't need to understand the child's internals

The parent only sees the child as a **register with a payload that happens to be another pisphere**. The child's self-description (version, checksum) tells the parent how to interpret it.

### 4.4 Cross-Nesting Communication

Registers in different nesting levels communicate via **cross-nesting echoes**:

```
Child register A → Parent fabric → Sibling child register B

Path:
  A (in child pisphere N)
    → echo up to N's parent register (echo crosses nesting boundary)
    → parent register's IPv4 control plane routes to sibling sector
    → echo down into sibling child pisphere M
    → target register B in M
```

Cross-nesting echoes are **damped more aggressively** than intra-nesting echoes (they traverse more boundaries, more cost).

---

## 5. Geometric Shard Non-Avalanche Hashing

### 5.1 Non-Avalanche Property

Standard cryptographic hashes (SHA3, Blake3) are **avalanching** — flipping one input bit flips ~50% of output bits. This is bad for spatial locality: close addresses get far-apart hashes.

**Non-avalanche hashing** preserves locality:

```
non_avalanche_hash(address):
  // Only the TOP bits of the hash change when input changes
  // Bottom bits stay stable — nearby addresses have nearby hashes
  top_bits = address >> 64      // high bits vary with address
  bottom_bits = address & 0xFFFFFFFF  // low bits are stable
  hash = (top_bits * prime) ^ bottom_bits
  return hash
```

**Properties:**
- `address A` and `address B` that are XOR-close → their hashes are XOR-close
- `address A` and `address B` that are far apart → their hashes may be far or close (unpredictable for distant addresses, predictable for nearby ones)
- The hash is **self-validating**: you can check if two addresses are related by comparing their hashes

### 5.2 Shard Mapping via Non-Avalanche Hash

Shards are assigned to storage tiers based on the non-avalanche hash:

```
shard_id = non_avalanche_hash(address)

Tier assignment:
  shard_id[0..4] == 0 → tier_hot    (1/16 of shards)
  shard_id[0..4] == 1 → tier_warm   (1/16 of shards)
  shard_id[0..4] == 2 → tier_gpu    (1/16 of shards)
  shard_id[0..4] >= 3 → tier_cold   (13/16 of shards)
```

Because the hash is non-avalanching, **adjacent shards map to adjacent tiers** — this creates natural locality bands in the storage hierarchy.

### 5.3 Multi-Level Non-Avalanche Hash

For different purposes, different hash granularities:

```
Address: [spine:16 | layer:16 | ring:16 | sector:16 | shard:16 | temporal:16]

Coarse hash (for tier mapping):   use bits [112..64] — spine + layer
Medium hash (for bloom filter):   use bits [63..32] — ring + sector  
Fine hash (for lineage anchor):   use bits [31..0] — shard + temporal
```

Each hash level is non-avalanching at its own granularity.

---

## 6. Multiple Interlaced Planes (On-Demand)

### 6.1 From 2 Planes to N Planes

The original spec has 2 interlaced planes (IPv6 data + IPv4 control). The strongest version supports **on-demand interlacing** — any number of planes, each with a different role:

```
Plane {
    id:          PlaneId,          // which plane (0=data, 1=control, 2=...)
    role:        PlaneRole,        // payload, routing, lineage, bloom, adaptation
    interleave:  InterleavePattern,// how this plane interleaves with others
    address_bits: u8,               // how many address bits this plane uses
}
```

**Plane roles:**

| Plane ID | Role | Content |
|---|---|---|
| 0 | Data | Payload, XOR lineage, version, checksum |
| 1 | Control | Routing, scheduling, echo engine state |
| 2 | Temporal | Timers, epoch counters, time-based state |
| 3 | Lineage | XOR lineage chain, causal ordering, version vectors |
| 4 | Bloom | Bloom filters per sector, membership proofs |
| 5 | Adaptation | Delta history, adaptation log, personalization state |
| 6 | Diagnostic | Access logs, performance metrics, error counters |

### 6.2 On-Demand Plane Activation

Planes are **sparse** — a register only has a plane if it's needed:

```
Register {
    address:       RegisterAddress,
    planes:        HashMap<PlaneId, PlaneData>,  // only populated planes
    plane_mask:    u8,                            // bitset of which planes are present
}
```

- A weight register might have planes 0 (data) + 1 (control) + 4 (bloom) — no temporal, no adaptation
- A temporal register might have planes 0 (data) + 2 (temporal) + 3 (lineage) — no bloom
- The **plane_mask** tells the fabric which planes to check on access — no wasted lookups

### 6.3 Interlacing Patterns

Different workloads use different interlacing patterns:

```
Inference workload:     planes [0, 1, 4] — data + control + bloom
Training workload:      planes [0, 1, 2, 3, 5] — add temporal + lineage + adaptation
Debugging workload:     planes [0, 1, 6] — add diagnostic
Cold storage workload:  planes [0, 4] — data + bloom only (minimal)
```

The interlacing pattern is part of the register's self-description — any node can determine which planes are present by reading the plane_mask.

### 6.4 Dynamic Plane Addition

Planes can be added to a register at runtime:

```
fabric.add_plane(register_address, PlaneId::Temporal):
  // Allocate temporal plane data for this register
  // Initialize timers, epoch counters
  // Set plane_mask bit for Temporal
  // Propagate echo to inform dependents of the new plane
```

Adding a plane is a **Write echo** — it propagates to registers that depend on this one, informing them that the plane structure changed.

---

## 7. Self-Hosting Network Protocol

### 7.1 Protocol Principles

The network protocol is **self-hosting**: the protocol description is carried within the protocol packets themselves. No external spec needed — a node can learn the protocol by inspecting packets.

**Four pillars:**

1. **Self-describing** — every packet describes its own format (version, type, field layout)
2. **Self-processing** — nodes can modify their own protocol behavior (adapt the packet parser)
3. **Self-organizing** — network topology adapts based on traffic patterns (not static routing)
4. **Self-routing** — packets route themselves based on embedded address information

### 7.2 Packet Structure

```
Packet {
    // Self-description (first 16 bytes — the "header that describes itself")
    protocol_version:  u8,              // protocol version (self-describing)
    packet_type:       PacketType,       // branch, hop, trace, route, eval
    header_length:     u8,              // length of this header (variable)
    payload_length:    u32,             // length of payload
    checksum:          u32,             // integrity check

    // Routing (next 16 bytes — embedded in the packet)
    src_address:       RegisterAddress,  // where this packet came from
    dst_address:       RegisterAddress,  // where this packet is going
    path_trace:        Vec<NodeId>,      // nodes this packet has visited (for trace)
    hop_count:         u8,               // how many hops taken
    max_hops:          u8,               // TTL — drop if exceeded

    // Branching (variable — depends on packet type)
    branch_count:      u8,               // how many sub-packets (0 = no branch)
    sub_packets:       Vec<Packet>,      // child packets (if branched)

    // Evaluation (embedded evaluation criteria)
    eval_criteria:     Option<EvalCriteria>, // conditions for packet acceptance
    eval_result:       Option<bool>,         // result of evaluation (set by receiver)

    // Payload (variable length)
    payload:           Bytes,             // the actual data
}
```

### 7.3 Packet Types

#### Branch Packet
A branch packet **splits into sub-packets** that travel different paths:

```
BranchPacket {
    branch_count: 3,
    sub_packets: [
        Packet { dst: RegisterAddress { shard: 0, ... }, payload: "sub-task-0" },
        Packet { dst: RegisterAddress { shard: 1, ... }, payload: "sub-task-1" },
        Packet { dst: RegisterAddress { shard: 2, ... }, payload: "sub-task-2" },
    ],
    merge_policy: MergeXOR,  // how to combine results: XOR, concat, priority
}
```

Use cases:
- **Parallel computation**: split a matmul across shards, branch to each shard, merge results via XOR
- **Echo fan-out**: branch an echo to multiple dependent sectors simultaneously
- **Model parallelism**: branch model layers across different tiers

#### Hop Packet
A hop packet **moves through the nested pisphere structure**, one level at a time:

```
HopPacket {
    current_depth: 1,       // currently at depth 1 (sector level)
    target_depth:  3,       // target depth 3 (register-group level)
    next_hop:      RegisterAddress,  // where to hop next
    path_so_far:   Vec<RegisterAddress>,  // audit trail
}
```

Each hop:
1. Packet arrives at a node (register)
2. Node reads the packet's `next_hop` field
3. Node forwards the packet to the next_hop register
4. Node records itself in `path_trace`
5. Node increments `hop_count`
6. If `hop_count > max_hops`, drop the packet (prevent infinite loops)

**Hop packets are the primary routing mechanism** — they don't use a routing table. The packet carries its own destination, and each node knows how to reach the next node based on the address geometry.

#### Trace Packet
A trace packet **records its path** for debugging, auditing, or causal analysis:

```
TracePacket {
    trace_type: TraceType,
    // TraceType::Path — record every node visited
    // TraceType::Bloom — record which bloom filters were checked
    // TraceType::Echo — record echo propagation path
    // TraceType::Rotation — record rotation events affecting this packet
    events: Vec<TraceEvent>,
}

TraceEvent {
    node_id:       NodeId,
    timestamp:     Instant,
    action:        TraceAction,  // received, forwarded, dropped, merged, echoed
    address:       RegisterAddress,
    echo_depth:    u8,
}
```

Trace packets are **diagnostic** — they don't carry payload data (or carry minimal payload). They're used to understand packet flow through the fabric.

#### Route Packet
A route packet **discovers a path** through the fabric:

```
RoutePacket {
    src: RegisterAddress,
    dst: RegisterAddress,
    discovered_path: Vec<RegisterAddress>,  // path found (empty if not found)
    path_cost: u32,                          // cost of the discovered path
    bloom_probes: u8,                        // how many bloom filters were checked
    hop_count: u8,                           // how many hops taken
}
```

Route packets are **query packets** — they don't carry application data. They're used to find a path from src to dst, returning the path and cost.

Route discovery works by **geometric descent**:
1. Start at src
2. At each node, check which neighbor is closest to dst (XOR distance)
3. Forward to that neighbor
4. Record the hop
5. Repeat until dst is reached or max_hops exceeded

The non-avalanche hash ensures that geometric descent is efficient — nearby addresses route through nearby nodes.

#### Eval Packet
An eval packet **carries evaluation criteria** — the receiver evaluates the payload against the criteria and returns a result:

```
EvalPacket {
    criteria: EvalCriteria,
    // EvalCriteria::Threshold { field: u8, min: u64, max: u64 }
    // EvalCriteria::Match { field: u8, expected: Bytes }
    // EvalCriteria::Range { field: u8, min: Instant, max: Instant }
    // EvalCriteria::XORCheck { expected_xor: u128 }
    payload:  Bytes,
    result:   Option<bool>,  // set by receiver after evaluation
}
```

Use cases:
- **Echo gate**: eval whether an echo should propagate (criteria: is the register still valid?)
- **Rotation trigger**: eval whether a layer should be rotated (criteria: access count > threshold?)
- **Adaptation decision**: eval whether an adaptation delta should be applied (criteria: confidence > 0.9?)
- **Self-test**: eval whether the fabric is healthy (criteria: all registers valid? all blooms consistent?)

### 7.4 Self-Hosting

The protocol **describes itself** in packets:

```
ProtocolDescriptor {
    version:            1,
    packet_types:       [Branch, Hop, Trace, Route, Eval],
    field_layout:       [protocol_version:8, packet_type:8, header_length:8, ...],
    address_structure:  [spine:16, layer:16, ring:16, sector:16, shard:16, temporal:16],
    echo_types:         [Read, Write, Fold, Adapt, Rotate, Control, Diagnostic],
    plane_roles:        [Data, Control, Temporal, Lineage, Bloom, Adaptation, Diagnostic],
    hash_function:      NonAvalancheHash,
    nesting_depth:      4,
}
```

A node receives an unknown packet type → reads the `protocol_version` and `header_length` → parses the self-description → knows how to handle the packet. **No external configuration needed.**

### 7.5 Self-Processing

Nodes can **modify their own protocol behavior** based on observed traffic:

```
// Node observes that most packets are Branch type
// It adapts by prioritizing branch processing
Node.adapt_protocol(ObservedTraffic {
    branch_ratio:   0.7,    // 70% branch packets
    hop_ratio:      0.2,    // 20% hop packets
    eval_ratio:     0.1,    // 10% eval packets
}):
  // Increase branch processing priority
  // Add bloom filter caching for branch destinations
  // Pre-compute common branch paths
```

Self-processing happens during the **control plane's Adapt echo type** — protocol adaptations are treated as first-class echoes that propagate through the fabric.

### 7.6 Self-Organizing Topology

The network **adapts its topology** based on traffic patterns:

```
TrafficPattern {
    high_traffic_nodes: Vec<NodeId>,     // nodes that handle lots of traffic
    low_traffic_nodes:  Vec<NodeId>,     // nodes that handle little traffic
    cross_traffic:      HashMap<NodeId, HashMap<NodeId, u64>>,  // traffic between node pairs
}

On observing traffic patterns:
  // Create shortcuts between high-traffic node pairs
  // Collapse low-traffic regions into deeper nesting
  // Expand high-traffic regions into shallower nesting
  // Rebalance bloom filters based on traffic density
```

The topology changes are propagated as **Control echoes** — they update the routing metadata in IPv4 control registers.

### 7.7 Branching, Hopping, Tracing, Routing, Evaluation

The five packet types map to five fundamental network operations:

```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│  Branch     │────▶│    Hop      │────▶│   Trace     │
│  Split work │     │ Move through│     │ Record path │
│  Parallel   │     │ nested cube │     │ Audit trail │
│  execution  │     │             │     │             │
└─────────────┘     └─────────────┘     └─────────────┘
         │                   │                   │
         ▼                   ▼                   ▼
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│   Route     │────▶│   Eval      │     │  Merge      │
│  Find path  │     │ Test criteria│    │ Combine     │
│  Geometric  │     │ Gate echo   │     │ results     │
│  descent    │     │ Decide      │     │ via XOR     │
└─────────────┘     └─────────────┘     └─────────────┘
```

**Full packet lifecycle:**

1. **Route** packet discovers path from src to dst (geometric descent via non-avalanche hash)
2. **Branch** packet splits work across multiple shards/nodes (parallel execution)
3. **Hop** packet traverses the nested pisphere structure (one level per hop)
4. **Eval** packet tests criteria at each hop (gate: should this echo continue?)
5. **Trace** packet records the path taken (audit trail for debugging/verification)
6. **Merge** combines results from branched sub-packets (XOR merge for lineage consistency)

---

## 8. Integration: Putting It All Together

### 8.1 Register with All Features

```
Register {
    address:           RegisterAddress,        // spine, layer, ring, sector, shard, temporal
    version:           u8,                      // self-description version
    payload:           BitslicedLane,           // data payload
    xor_lineage:       u128,                    // XOR fingerprint of lineage
    echo_counter:      u8,                      // propagation depth
    bloom_anchor:      u64,                     // hash anchor for bloom
    checksum_anchor:   u64,                     // self-validation anchor
    temporal_timer:    Option<Timer>,           // temporal timer (Section 1)
    temporal_predicate: TemporalPredicate,      // time-validity (Section 2)
    echo_type:         EchoType,                // differentiated echo (Section 3)
    echo_priority:     EchoPriority,            // echo priority (Section 3)
    plane_mask:        u8,                      // which planes are present (Section 6)
    nesting_depth:     u8,                      // child pisphere depth (Section 4)
    child_pisphere:    Option<ChildPisphere>,   // nested pisphere (Section 4)
    protocol_version:  u8,                      // self-hosting protocol (Section 7)
    plane_data:        HashMap<PlaneId, Bytes>, // plane-specific data (Section 6)
}
```

### 8.2 Packet Flow Example

**Scenario**: Adapt a weight register in a nested pisphere, with echo propagation across the fabric.

```
1. User injects adaptation delta for Register A (depth 2, shard 5)

2. Fabric creates Adapt Echo packet:
   Packet {
       type:        Hop,          // hops through nesting levels
       src:         Register A address,
       dst:         Register A address (same — propagation, not routing),
       hop_count:   0,
       max_hops:    4,            // max nesting depth
       echo_type:   Adapt,
       echo_priority: Critical,
       payload:     AdaptationDelta { target: A, delta: [...] },
       eval_criteria: Some(ConfidenceThreshold(0.9)),  // eval at each hop
       path_trace:  [],
   }

3. Hop 0: Packet enters root pisphere
   - Eval: confidence 0.95 > 0.9 → continue
   - Route: determine path to shard 5 at depth 2
   - Forward to child pisphere at depth 1

4. Hop 1: Packet enters depth-1 pisphere (sector level)
   - Trace: record node, action=forwarded
   - Eval: bloom filter check — shard 5 is in this sector? → yes
   - Forward to child pisphere at depth 2

5. Hop 2: Packet enters depth-2 pisphere (shard level)
   - Trace: record node, action=forwarded
   - Eval: register A exists in this shard? → yes
   - Apply adaptation delta to register A
   - Register A.echo_counter += 1
   - Register A.temporal_timer = Timer { on_fire: Adapt, repeat: false }

6. Echo propagates to dependents:
   - Branch packet to downstream registers (3 shards affected)
   - Each sub-packet hops to its target
   - Each target evaluates: does this delta affect me? → if yes, apply; if no, drop
   - XOR cancellation: if two deltas cancel, echo dies

7. Trace packet records full path:
   [root → depth1_sector3 → depth2_shard5 → registers_affected]

8. Route packet returns path cost for future routing decisions

9. Temporal timer fires after timeout → cleanup, mark echo as complete
```

### 8.3 Temporal Timer Lifecycle

```
Timer created (epoch N):
  deadline = now + period
  on_fire = PropagateEcho(register)

Epoch N+1 (timer not fired):
  Check deadline — not reached yet
  No action

Epoch N+5 (timer fires):
  on_fire action executed
  If repeat: deadline = now + period (reschedule)
  If one-shot: timer removed, register.echo_counter -= 1

Echo propagation completes:
  If echo_counter == 0:
    register.state = Stable
    If has temporal plane:
      Temporal predicate valid_until = now + TTL
```

### 8.4 Nested Pisphere Communication Pattern

```
Child Pisphere (depth 2, shard 5)
├── Register A (payload: weight data)
│     ├── Temporal timer: fires in 500ms
│     ├── Adapt echo received → payload updated
│     └── Echo propagated to:
│           ├── Child Pisphere (depth 3, shard 5.1) — branch
│           │     └── Register A.1 (downstream weight)
│           ├── Child Pisphere (depth 3, shard 5.2) — branch
│           │     └── Register A.2 (downstream weight)
│           └── Child Pisphere (depth 3, shard 5.3) — branch
│                 └── Register A.3 (downstream weight)
│
├── Bloom filter for shard 5 (planes: data, control, bloom)
├── Rotation schedule for shard 5
└── Temporal timers for shard 5
```

---

## 11. Bitslicing — Fundamental Computation Model

### 11.1 Everything Is Bitsliced

Bitslicing is not just a storage format — it is the **fundamental computation model**. Every operation in the fabric operates on bitsliced data:

```
BitslicedLane {
    bits:    Vec<u64>,    // one u64 per SIMD lane (64 values in parallel)
    width:   u8,           // bits per value (1=binary, 4=INT4, 8=INT8, 16=FP16)
    count:   u32,          // number of values in this lane
}
```

**All fabric operations are bitwise:**

| Operation | Bitsliced implementation |
|---|---|
| XOR lineage | XOR corresponding lanes — single instruction |
| Fold | XOR lanes — single instruction |
| Echo delta | XOR lanes — single instruction |
| Bloom filter | AND/OR across lanes — single instruction |
| Distance (popcount) | XOR lanes → popcount — single instruction |
| Quantization | Lane width determines quantization level |
| Rotation | Shift lanes by rotation offset — single instruction |

### 11.2 Multi-Slice Bitslicing

A single bitsliced lane can be **sliced** across multiple sub-lanes for different purposes:

```
BitslicedLane {
    bits:    Vec<u64>,
    width:   64,              // 64-bit SIMD width
    slices:  [                 // multi-slice decomposition
        Slice { purpose: Payload,    offset: 0,  width: 32 },   // lower 32 bits: payload
        Slice { purpose: Lineage,    offset: 32, width: 16 },   // upper 32 bits[0:16]: lineage
        Slice { purpose: BloomKey,   offset: 48, width: 16 },   // upper 32 bits[16:32]: bloom hash
    ],
}
```

Each slice operates independently within the same SIMD register — **one instruction, multiple purposes**:

```
// Single u64 operation simultaneously computes:
// - payload XOR (lower 32 bits)
// - lineage update (upper 16 bits)
// - bloom filter update (upper 16 bits)
result = lane.bits[0] ^ delta.bits[0]
// result.slice(Payload) = payload XOR delta
// result.slice(Lineage) = lineage XOR delta  
// result.slice(BloomKey) = bloom_hash XOR delta
```

**Multi-slice width allocation per register type:**

| Register Type | Slice 0 (Payload) | Slice 1 (Lineage) | Slice 2 (Bloom) | Slice 3 (Temporal) |
|---|---|---|---|---|
| Weight (A) | 32-bit FP | 16-bit lineage | 16-bit bloom | — |
| KV Cache (B) | 32-bit FP | 16-bit lineage | 16-bit position | — |
| Activation (C) | 16-bit FP | 16-bit sector | — | 32-bit timer |
| Control (D) | 8-bit opcode | 8-bit lineage | 8-bit bloom | 8-bit flags |
| Echo (E) | 32-bit delta | 32-bit source | — | 32-bit timestamp |
| Temporal (F) | 64-bit timer | 64-bit epoch | — | — |
| Container (G) | 32-bit child count | 32-bit depth | — | 64-bit nesting mask |
| Delta (H) | 32-bit XOR diff | 32-bit base addr | — | — |
| Bloom (I) | 256-bit filter | 64-bit anchor | — | 32-bit generation |
| Ticket (J) | 64-token seq | 64-token pos | — | — |
| History (K) | 64-bit prev addr | 64-bit prev lineage | — | — |
| Opcode (L) | 16-bit op | 16-bit dst | 16-bit src | 16-bit flags |
| Lineage (M) | 64-bit parent | 64-bit grandparent | — | — |
| Phase/Angle (N) | 16-bit phase | 16-bit angle | — | — |
| Logic (O) | 8-bit mode | 8-bit gate | 8-bit mask | 8-bit priority |

### 11.3 Fiber Bitslicing

A **fiber** (spine) is bitsliced across its entire length — each position in the fiber is a slice of the same bitsliced lane:

```
Fiber (spine = 0x3A2) = [
    BitslicedLane { bits: [v0, v1, v2, ..., v63] },  // position 0
    BitslicedLane { bits: [v0, v1, v2, ..., v63] },  // position 1
    ...                                              // position 1023
]
```

A single SIMD instruction processes **all 1024 positions in the fiber simultaneously** — one instruction, 1024 values.

**Fiber operations:**

```
fiber_xor(fiber_a, fiber_b) → fiber_c    // XOR all 1024 positions in one instruction
fiber_popcount(fiber) → u32               // count set bits across all positions
fiber_rotate(fiber, offset) → fiber       // rotate all positions by offset
fiber_bloom_insert(fiber, address)         // insert all positions into bloom filter
```

### 11.4 Bitsliced Inference

Inference operations on bitsliced data:

```
MatMul(bitsliced_weights, bitsliced_activations) → bitsliced_output
  // Each partial product is a bitwise AND
  // Each sum is a popcount + shift
  // No floating-point multiply — just AND + popcount

Attention(bitsliced_Q, bitsliced_K, bitsliced_V) → bitsliced_output
  // Q·K^T = popcount(Q_bits & K_bits)  // dot product via popcount
  // Softmax = approximated via bitsliced comparison
  // Output = weighted sum of V via bitsliced mux
```

On Apple Silicon, each NEON/AMX instruction processes 128 bits (2×64-bit lanes) — a single instruction handles 128 values simultaneously.

---

## 12. Full Fiber with Multi-Slice

### 12.1 Fiber as First-Class Abstraction

A **fiber** is a 1D coherent access path through the 3D cube. The full fiber spans all 1024 positions along one axis:

```
Fiber (along X axis, fixed Y=512, Z=256):
  Position 0: Register(0, 512, 256, T)
  Position 1: Register(1, 512, 256, T)
  ...
  Position 1023: Register(1023, 512, 256, T)
```

**Three fiber orientations:**

| Orientation | Fixed dims | Varies | Use case |
|---|---|---|---|
| X-fiber | Y, Z, T | X (0–1023) | Sequential token processing (attention heads) |
| Y-fiber | X, Z, T | Y (0–1023) | Layer-wise processing (feed-forward) |
| Z-fiber | X, Y, T | Z (0–1023) | Model parallelism (split model across Z) |
| T-fiber | X, Y, Z | T (0–1023) | Temporal sequence processing |

### 12.2 Multi-Slice Fiber Access

A fiber can be **sliced** at any boundary — you access a sub-fiber (a contiguous range of positions):

```
fiber.slice(start: u16, end: u16) → SubFiber
  // Returns positions [start, end) of the fiber
  // SubFiber is itself a bitsliced lane with (end - start) values

fiber.slice(0, 64)    → first 64 positions (warm layer access)
fiber.slice(512, 1024) → last 512 positions (cold layer access)
fiber.slice(100, 200)  → arbitrary range (sector access)
```

**Multi-slice** — access multiple fibers simultaneously at the same offset:

```
MultiSlice {
    fibers:    [FiberId; N],    // N fibers to slice together
    offset:    u16,               // same offset into all fibers
    width:     u16,               // slice width
}

multi_slice([X_fiber_512, Y_fiber_256, Z_fiber_768], offset: 100, width: 32)
  → [X_fiber_512[100..132], Y_fiber_256[100..132], Z_fiber_768[100..132]]
```

Multi-slice is the fundamental parallel access pattern — **one operation touches N fibers at the same position**, which is exactly how attention heads, model layers, and batch entries are accessed.

### 12.3 Fiber-Tree Hierarchy

Fibers form a **tree hierarchy**:

```
Root fiber (spine=0, full 1024³ cube)
├── X-fiber 0 (Y=0..255, Z=0..255, T=0..1023)
│     ├── X-fiber 0, Y-fiber 0 (Z=0..255, T=0..1023)
│     │     ├── X-fiber 0, Y-fiber 0, Z-fiber 0 (T=0..1023)
│     │     │     └── T-fiber (temporal register ring)
│     │     └── ...
│     └── ...
├── X-fiber 1
└── ...
```

Each level of the fiber tree corresponds to a nesting depth:
- **Depth 0**: root fiber (full cube)
- **Depth 1**: spine-level fiber (1024 positions along one axis)
- **Depth 2**: sector-level fiber (1024 positions within a sector)
- **Depth 3**: shard-level fiber (1024 positions within a shard)
- **Depth 4**: register-level fiber (1024 temporal registers)

Multi-slice can operate at any depth — you can slice 4 fibers at depth 2 (sector level) to access the same sector across 4 different spines.

### 12.4 Fiber-Bitsliced Operations

Combining fiber and bitslice:

```
// Process 64 positions across 4 fibers simultaneously, 32 values per position
fiber_bitsliced_op(
    fibers: [X_fiber, Y_fiber, Z_fiber, T_fiber],
    slice:  (100, 132),        // 32 positions
    operation: XOR,
) → [result_X, result_Y, result_Z, result_T]

// One instruction: 64 × 4 × 32 = 8192 values processed
// On Apple Silicon AMX: 128 × 4 × 32 = 16384 values per cycle
```

This is the **fundamental compute primitive** of the fabric — fiber-bitsliced operations replace scalar register-by-register processing.

---

## 13. Bell Timer Rails

### 13.1 Bell-Curve Timer Distribution

Timers don't fire at fixed intervals — they follow a **bell curve distribution** across the epoch:

```
BellTimerRail {
    epoch:        u64,               // which epoch this rail belongs to
    center:       f64,               // center of the bell curve (0.0–1.0 of epoch)
    sigma:        f64,               // spread of the bell curve
    count:        u32,               // number of timers on this rail
    timers:       Vec<BellTimer>,
}

BellTimer {
    id:           TimerId,
    bell_offset:  f64,               // offset from center (in sigma units)
    action:       TimerAction,
    priority:     EchoPriority,
    slice_id:     u16,               // which fiber slice this timer belongs to
}
```

**Bell distribution** — timers are denser near the center of the epoch and sparser at the edges:

```
Timer density across epoch:
  0.0         0.25        0.5         0.75        1.0
  |           |           |           |           |
  ░           ███         ███████     ███         ░
  (sparse)    (medium)    (dense)     (medium)    (sparse)

  Bell curve: density ∝ exp(-offset² / (2 * sigma²))
```

### 13.2 Why Bell Curves?

**Temporal locality** — most computations cluster around the center of an epoch (the "hot" period). Bell-distributed timers match this pattern:

- **Center of epoch**: high-frequency operations (attention, KV cache updates, activation computation)
- **Edges of epoch**: low-frequency operations (rotation, bloom rebuild, adaptation)
- **Bell curve**: natural fit for temporal locality — dense where work is concentrated, sparse at edges

**Echo damping** — echoes propagate during the epoch. Bell-distributed echo timers ensure:
- Early echoes (left side of bell) — propagate to nearby registers
- Center echoes — full propagation
- Late echoes (right side of bell) — damped, queued for next epoch

**Energy efficiency** — bell-distributed timers avoid the "sawtooth" pattern of fixed-interval timers (periodic bursts of activity followed by idle). Instead, activity is smooth and Gaussian — fewer sudden energy spikes.

### 13.3 Timer Rails

A **timer rail** is a collection of bell-distributed timers that share the same epoch and sigma:

```
TimerRail {
    rail_id:      RailId,
    epoch:        u64,
    sigma:        f64,               // shared spread
    center:       f64,               // shared center
    timers:       Vec<BellTimer>,    // timers on this rail
}
```

**Multiple rails per epoch** — each rail handles a different class of operations:

| Rail | Sigma | Center | Purpose |
|---|---|---|---|
| Hot rail | 0.1 | 0.5 | Fine-grained: KV cache, activations, attention |
| Warm rail | 0.2 | 0.5 | Medium-grained: weight fetching, bloom updates |
| Cold rail | 0.5 | 0.5 | Coarse-grained: rotation, tier promotion/demotion |
| Echo rail | 0.15 | 0.5 | Echo propagation: controlled echo dampening |
| Adapt rail | 0.3 | 0.5 | Adaptation: delta injection, model updates |
| Diag rail | 1.0 | 0.5 | Diagnostic: logging, tracing, auditing |

### 13.4 Rail Scheduling

Timers fire in **rail priority order** within each epoch:

```
Epoch N:
  1. Hot rail timers fire (center ± 0.1σ) — fine-grained compute
  2. Echo rail timers fire (center ± 0.15σ) — echo propagation
  3. Warm rail timers fire (center ± 0.2σ) — medium-grained updates
  4. Adapt rail timers fire (center ± 0.3σ) — adaptation
  5. Cold rail timers fire (center ± 0.5σ) — coarse-grained maintenance
  6. Diag rail timers fire (center ± 1.0σ) — diagnostics
```

**Rail crossing** — a timer can fire on one rail and schedule itself on another:

```
Hot rail timer fires:
  → Process attention head
  → If attention result is large: schedule Adapt rail timer (re-evaluate)
  → If attention result is small: schedule Cold rail timer (defer to next epoch)
```

### 13.5 Bell Timer Rails + Fiber Slicing

Timer rails combine with fiber slicing — each timer belongs to a specific fiber slice:

```
BellTimer {
    rail_id:      RailId,
    fiber_id:     FiberId,        // which fiber this timer belongs to
    slice_offset: u16,            // offset within the fiber slice
    bell_offset:  f64,            // offset from rail center
    action:       TimerAction,
}
```

When a hot rail timer fires for fiber X at slice offset 100:
- Process the bitsliced lane for fiber X, positions 100–131
- The bell curve determines WHEN within the epoch
- The fiber slice determines WHICH data to process
- The rail determines HOW (fine-grained vs. coarse-grained)

**Full fiber bell scheduling:**

```
All fibers on the hot rail fire in sequence:
  Fiber 0:    positions 0–63     (bell offset: -3σ)
  Fiber 1:    positions 0–63     (bell offset: -2σ)
  ...
  Fiber 1023: positions 0–63     (bell offset: +3σ)

Then next slice:
  Fiber 0:    positions 64–127   (bell offset: -3σ)
  ...
```

The bell curve schedules fibers — fibers near the center of the bell are processed first (hot), fibers at the edges are processed later (cold). This creates a **spatial-temporal scheduling** pattern where both position and time follow the bell curve.

### 13.6 Bell Curve Echo Propagation

Echoes follow the bell timer rail schedule:

```
Echo propagation timeline within an epoch:
  0.0         0.25        0.5         0.75        1.0
  |           |           |           |           |
  ░░░░░░░░░░░░████████████████████████████████░░░░░░░░░░░░░░░░░

  Left side (0.0–0.25):   Echo starts at source, propagates to immediate dependents
  Center (0.25–0.75):     Echo propagates to full depth, all dependent registers updated
  Right side (0.75–1.0):  Echo dampens, only critical echoes continue
```

**Bell-gated echo propagation:**
- Echoes arriving near the center of the epoch propagate fully
- Echoes arriving near the edges are damped (queued for next epoch)
- The bell curve acts as a **time-domain filter** on echo propagation

---

## 9. Summary of Changes from v0.1.0 (Updated)

| Feature | v0.1.0 | v1.0 (this amendment) |
|---|---|---|
| Temporal timers | ❌ Epoch-based only | ✅ Per-register timers with on-fire actions |
| Temporal logic | ❌ | ✅ Validity predicates, causal XOR ordering, version vectors |
| Echo differentiation | ✅ Minimal propagation | ✅ Typed echoes + priority levels + refined cancellation |
| Nested pispheres | ✅ 2-4 levels, abstract | ✅ Recursive child cubes, geometric addressing, cross-nesting echoes |
| Non-avalanche hash | ✅ Conceptual | ✅ Multi-level hash for tier/bloom/lineage, geometric locality |
| Multiple interlacing | ✅ 2 planes (IPv4/IPv6) | ✅ N planes on demand, dynamic activation |
| Network protocol | ❌ (cross-node "future") | ✅ Self-hosting, self-describing, self-processing, self-organizing |
| Packet types | ❌ | ✅ Branch, Hop, Trace, Route, Eval |
| Self-routing | ❌ | ✅ Geometric descent via non-avalanche hash |
| Self-organization | ❌ | ✅ Topology adapts to traffic patterns |
| **Bitslicing** | ✅ Storage format | ✅ **Fundamental computation model — all operations are bitsliced** |
| **Full fiber** | ✅ 1D thread concept | ✅ **First-class fiber abstraction with multi-slice access** |
| **Bell timer rails** | ❌ | ✅ **Bell-curve distributed timers across 6 rails per epoch** |

---

## 10. Open Questions

1. **Nesting depth limit**: 4 levels? Deeper = more overhead, shallower = less locality. Benchmark on target hardware.
2. **Plane count**: 7 planes (current proposal) vs. dynamic plane creation. Dynamic is more flexible but adds complexity.
3. **Timer resolution**: Epoch-based (coarse) vs. per-register timers (fine-grained). Trade-off: precision vs. overhead.
4. **Cross-node protocol**: This amendment covers single-machine nested pispheres. Cross-node (multi-machine) needs a separate protocol spec.
5. **Non-avalanche hash security**: Non-avalanche hashes are not collision-resistant. Fine for routing/locality, not for security. Use cryptographic hash for integrity, non-avalanche for routing.
6. **Echo storm prevention**: With nested pispheres, echoes can fan out exponentially. Need depth-limited propagation + bloom gating + timeout as safety valves.
