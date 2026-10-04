# Timing Semantics

## Chapter 1 — The Temporal Coordinate System

TPN has a single, unified notion of time: the **temporal coordinate** T, which is a 64-bit integer representing an absolute arrival time.

### 1.1 The Fundamental Time Quantum

The smallest resolvable time step is the **fundamental time quantum** τ. All times are multiples of τ:

```
T = n · τ,  n ∈ {0, 1, 2, ...}
```

The phase field (64-bit) encodes n. The IPv4 phase field (32-bit) encodes n mod 2³².

### 1.2 The Gate Window

A temporal gate is defined by its window:

```
window = [T_min, T_max]   ⊂ ℕ
window_width = T_max − T_min
```

The gate is **open** for T ∈ [T_min, T_max] and **closed** otherwise. This is the fundamental timing primitive: the gate is a temporal predicate on arrival time.

### 1.3 The Theta Mapping

The arrival time T is mapped to a phase angle θ via:

```
θ(T) = (π/2) · (T − T_min) / (T_max − T_min)        if T_min ≤ T ≤ T_max
θ(T) = −(π/2) · (T_min − T) / (T_max − T_min)       if T < T_min
θ(T) = π/2 + (π/2) · (T − T_max) / (T_max − T_min)  if T > T_max
```

For T outside the window, θ falls outside [0, π/2], and the fitness sin(2θ) is still defined (negative or >1 regions), but the gate is closed (fitness < threshold).

### 1.4 The Fitness Over Time

Substituting θ(T) into the fitness:

```
fitness(T) = sin(2θ(T)) = sin(π · (T − T_min) / (T_max − T_min))   for T ∈ [T_min, T_max]
```

The fitness as a function of T:

| T | u = (T−T_min)/w | θ | fitness = sin(π·u) |
|---|-----------------|---|-------------------|
| T_min | 0 | 0 | 0 |
| T_min + w/4 | 0.25 | π/8 | 0.707 |
| T_min + w/2 | 0.5 | π/4 | 1.0 |
| T_min + 3w/4 | 0.75 | 3π/8 | 0.707 |
| T_max | 1 | π/2 | 0 |

The fitness is a **half-sine wave** over the window: it starts at 0, peaks at the midpoint, and returns to 0.

## Chapter 2 — The Gate Open Interval

### 2.1 Effective Opening

With threshold θ, the gate is effectively open when:

```
sin(π · u) > threshold
```

Let α = arcsin(threshold) ∈ (0, π/2). Then:

```
sin(π·u) > threshold  ⇔  α < π·u < π − α  ⇔  (α/π) < u < 1 − (α/π)
```

So the effective open interval is:

```
T ∈ [T_min + (α/π)·w,  T_max − (α/π)·w]
```

The effective open width is:

```
w_eff = w · (1 − 2α/π)
```

**Second-order insight:** the threshold trims the window symmetrically. A higher threshold makes the gate narrower and more selective; a lower threshold makes it wider and more permissive.

### 2.2 The Evolved Threshold

From auto-evolution:

```
threshold = 0.43
α = arcsin(0.43) ≈ 0.445 rad
w_eff = w · (1 − 2·0.445/π) = w · (1 − 0.283) = 0.717 · w
```

So the gate is effectively open for ~72% of the window width, symmetric around the midpoint.

## Chapter 3 — The Phase Schedule

### 3.1 Definition

The **phase schedule** is the assignment of processing times to bit-planes:

```
schedule[j] = t_j   for j ∈ 0..(num_bitplanes − 1)
```

The three canonical schedules:

1. **Linear:** t_j = j · δ  (steady, predictable)
2. **Sinusoidal:** t_j = base + amp · sin(freq · j)  (sweeping, periodic)
3. **Exponential:** t_j = base · exp(decay · j)  (front-loaded)

### 3.2 Schedule Choice per Operation

| Operation | Schedule | Why |
|-----------|----------|-----|
| Bitsliced addition | Linear | The carry chain needs steady, monotonic progress from bit 0 to bit 255 |
| Attention | Sinusoidal | The attention focus should sweep across the input and return |
| Accumulation | Exponential | Early contributions matter more (like an exponential moving average) |
| Gate evaluation | Linear | The gate opens symmetrically over its window |

### 3.3 Bit-Plane Processing Order

In the bitsliced model, bit-planes are processed in order j = 0, 1, 2, ... The carry chain c[j] → c[j+1] flows in the same order. The schedule determines *when* each bit-plane is processed.

**Consequence for addition:** the carry must propagate from bit 0 to bit 255 through the bit-planes. With a linear schedule, the carry completes in 256 schedule steps. With a gated carry (gate closed at some bit-plane), the carry is killed and the addition must be redone.

## Chapter 4 — Timing of a Complete Operation

### 4.1 Single Gate Evaluation

```
t₀: packet arrives at address
t₁: θ ← θ(T)  (coordinate transform, ~0 cycles)
t₂: fitness ← sin(2θ)  (trig evaluation, ~2 DSP cycles)
t₃: compare fitness > threshold  (1 cycle)
t₄: if open: compute; if closed: hold
```

Total: ~3–5 cycles (dominated by the trig evaluation).

### 4.2 Bitsliced Addition

```
t₀: two packets arrive (operands A, B) at the adder
for j in 0..256:            // 256 bit-planes, carry chained
    if gate_open at j:
        s[j] = a[j] ⊕ b[j] ⊕ c[j]
        c[j+1] = majority(a[j], b[j], c[j])
    else:
        c[j+1] = 0          // gate kills the carry
t_end: result S = {s[j]} read out
```

Total: 256 schedule steps + setup + teardown. The carry chain is the critical path.

**Critical insight:** the adder's latency is now a function of the gate window, not the technology node. This is **time-extended arithmetic**.

### 4.3 Network Delivery

```
t₀: packet emitted at source
t₁..t_k: packet traverses the network (TCP/IP, RoCEv2, or UDP/IP)
t_k+1: packet arrives at destination address
t_k+2: θ(T) computed, gate evaluated
```

The network latency contributes to the arrival time T, which contributes to θ. **Network latency is folded into the computation.**

## Chapter 5 — The Arrival Model Timing Diagram

```
Source                         Network              Destination
  │                              │                      │
  │  emit(packet, T)             │                      │
  │─────────────────────────────▶│                      │
  │                              │  transit (Δt)        │
  │                              │─────────────────────▶│
  │                              │                      │  arrival at T' = T + Δt
  │                              │                      │  θ = θ(T')
  │                              │                      │  fitness = sin(2θ)
  │                              │                      │  gate open?
  │                              │                      │  ┌───────────┐
  │                              │                      │  │ register  │
  │                              │                      │  │ compute   │
  │                              │                      │  └───────────┘
```

Key relationship: **arrival time T' = emission time T + network latency Δt.** The gate evaluates over T', so network jitter affects computation. This is why **arrival = execution** requires precise timing: the gate window must be wide enough to absorb network jitter, or the timing must be synchronized.

**Second-order insight:** this is the trade-off. A wide gate window is robust to jitter but less selective. A narrow gate window is selective but fragile. The evolved threshold 0.43 balances this: ~72% of the window is effectively open.

## Chapter 6 — Verified Timing Numbers

From the reference implementation:

- Gate fitness at boundaries: sin(2·0) = 0, sin(2·π/2) = 0 ✓
- Gate fitness at center: sin(2·π/4) = sin(π/2) = 1 ✓
- Effective open width at threshold 0.43: 0.717 · w ✓
- Addition tests all pass: 7+5=12, 15+17=32, 64+64=128, 3+4=7, 12+8=20, 200+300=500 ✓
- Error correction: 4 errors detected, corrected by majority vote ✓
- Self-hosting synthesis: 256 registers → 2048 LUTs / 256 BRAMs / 1024 DSPs ✓

## References

- [Theory](theory.md) — the arrival model
- [Gates](gates.md) — gate window semantics
- [Phases](phases.md) — phase schedules
- [Artifacts](../artifacts.md) — `TrigGate::theta`, `TrigGate::fitness`
