# Temporal Gates

## Chapter 1 — The Trigonometric Temporal Gate

The temporal gate is the fundamental operation of TPN. It is a **conditional execution window** defined over time, implemented with trigonometric semantics.

### 1.1 The Gate Grammar

```
gate = "{"
         "address"    : IPv6,     // which register computes
         "window"     : [T_min, T_max],   // open window
         "theta_min"  : f64,     // phase angle at window start
         "theta_max"  : f64,     // phase angle at window end
         "route_0"    : IPv6,    // destination if θ < π/4
         "route_1"    : IPv6,    // destination if π/4 ≤ θ < π/2
         "route_2"    : IPv6,    // destination if π/2 ≤ θ < 3π/4
         "route_3"    : IPv6     // destination if θ ≥ 3π/4
       "}"
```

The gate has:
- A **destination** (`address`) — where the result goes
- A **window** (`[T_min, T_max]`) — when the gate is open
- **Conditional routes** — where to send the packet based on the phase angle (this is the temporal branch)

### 1.2 The Gate as Execution

When a packet arrives at `address` at time `T`:

1. Compute θ(T) — the phase angle at arrival
2. Compute fitness(θ) — the temporal alignment score
3. If fitness(θ) > threshold: **execute** — update the register, emit the result
4. Else: **hold** — the register's value is preserved (memory)

### 1.3 Fitness as the Filter

```
fitness(θ) = sin(2θ)      // CORRECTED: triangular 0→1→0
```

The fitness function determines *whether* computation happens. It is:
- **Triangular**: 0 at the window edges, 1 at the center
- **Smooth**: differentiable, for optimization
- **Self-normalized**: always in [0, 1]
- **Threshold-gated**: computation happens iff fitness > threshold

## Chapter 2 — The Theta Mapping

### 2.1 Definition

The arrival time T is mapped to a phase angle θ in [−π/2, 3π/2]:

```
θ(T) = (π/2) · (T − T_min) / (T_max − T_min)          if T_min ≤ T ≤ T_max
θ(T) = −(π/2) · (T_min − T) / (T_max − T_min)         if T < T_min
θ(T) = π/2 + (π/2) · (T − T_max) / (T_max − T_min)    if T > T_max
```

### 2.2 The Normalized Coordinate

Define the normalized time coordinate:

```
u(T) = (T − T_min) / (T_max − T_min)   ∈ [0, 1]   for T ∈ [T_min, T_max]
```

Then:

```
θ(T) = (π/2) · u(T)
```

So θ maps [T_min, T_max] → [0, π/2] linearly.

### 2.3 The Inverse Mapping

Given a phase angle θ, the arrival time that produces it is:

```
T(θ) = T_min + (θ / (π/2)) · (T_max − T_min)   for θ ∈ [0, π/2]
```

This is the **temporal address inverse**: given a desired phase angle, compute the arrival time that produces it.

## Chapter 3 — The Fitness Function

### 3.1 Definition

```
fitness(θ) = sin(2θ)   for θ ∈ [0, π/2]
```

### 3.2 Properties

| Property | Value |
|----------|-------|
| θ = 0 | fitness = sin(0) = 0 |
| θ = π/8 | fitness = sin(π/4) = 0.707 |
| θ = π/4 | fitness = sin(π/2) = 1 |
| θ = 3π/8 | fitness = sin(3π/4) = 0.707 |
| θ = π/2 | fitness = sin(π) = 0 |

The profile is **triangular** over [0, π/2]: 0 → 1 → 0.

### 3.3 Equivalence to the Time-Domain Definition

Since θ = (π/2) · u, we have 2θ = π · u, so:

```
fitness(T) = sin(π · (T − T_min) / (T_max − T_min))   for T ∈ [T_min, T_max]
```

This is the original time-domain definition. The θ-parametrization is just a change of coordinates.

### 3.4 Why `cos²(θ − π/4)` Is Wrong

The TPN Spec v2.0 claimed:

```
fitness(θ) = cos²(θ − π/4)
```

and "proved" (incorrectly) that this is equivalent to linear fitness. Let's evaluate:

| θ | cos²(θ − π/4) |
|---|---------------|
| 0 | cos²(−π/4) = 0.5 |
| π/4 | cos²(0) = 1.0 |
| π/2 | cos²(π/4) = 0.5 |

At the window boundaries, `cos²` gives 0.5, NOT 0. A gate with `cos²` fitness would be **half-open at both edges** — it would never fully close. This breaks the gate semantics: a packet arriving at T = T_min would still compute (with weight 0.5), violating the definition of the gate window.

**Theorem (Incompatibility):** `cos²(θ − π/4)` is incompatible with the TPN gate semantics, because it fails to reach 0 at the window boundaries. `sin(2θ)` is the correct triangular fitness.

See [Theorems](theorems.md) for the formal proof.

## Chapter 4 — Gate Execution

### 4.1 The Execution Sequence

```
packet arrives at address, time T
  │
  ▼
θ ← θ(T) = (π/2) · (T − T_min) / (T_max − T_min)
  │
  ▼
fitness ← sin(2θ)
  │
  ▼
if fitness > threshold:
    ┌─ execute: register.data = compute(register.data, input)
    └─ emit result packet to address
else:
    └─ hold: register.data unchanged
```

### 4.2 The Threshold

The threshold is a learned/evolved parameter. From auto-evolution, the converged threshold is:

```
threshold ≈ 0.43
```

This is the fitness value above which computation happens. With `sin(2θ)`, the threshold corresponds to:

```
sin(2θ) = 0.43  ⇒  2θ = arcsin(0.43) ≈ 0.445  ⇒  θ ≈ 0.222  rad
```

So the gate is effectively open in θ ∈ [0.222, π − 0.222] — i.e., the inner ~86% of the window.

**Second-order insight:** the threshold defines the **effective gate width**. A higher threshold narrows the gate (more selective); a lower threshold widens it (more permissive). Learning = finding the optimal threshold.

### 4.3 Conditional Routing (Temporal Branch)

The gate has 4 conditional routes based on θ:

```
θ < π/4          → route_0    (early in window)
π/4 ≤ θ < π/2    → route_1    (middle)
π/2 ≤ θ < 3π/4   → route_2    (late-middle)
θ ≥ 3π/4         → route_3    (near end)
```

This is a **temporal branch**: the routing decision depends on when the packet arrives. Early arrivals go one way, late arrivals another. This is how the gate implements control flow.

## Chapter 5 — Bitsliced Gate Execution

### 5.1 The Gate over Bit-Planes

In the bitsliced model, the gate is applied over all 256 bit-planes of the register array:

```
for bit_plane in 0..256:
    fitness = sin(2θ)
    if fitness > threshold:
        // gate open: bit-planes can flow
        adder(bitsliced.bitplane(bit_plane))
    else:
        // gate closed: carry is killed
        carry = 0
```

### 5.2 Carry Propagation Gated by Phase

In a bitsliced adder, the carry chain propagates from bit 0 to bit 255. In TPN, the carry propagation is gated by the gate fitness:

```
if gate_open:
    carry_out = majority(carry_in, a_bit, b_bit)
else:
    carry_out = 0          // gate kills the carry chain
```

**Consequence:** when the gate is closed, the carry chain is broken. This means a bitsliced adder can only complete its carry chain when the gate is open for the *entire* chain.

**Second-order effect:** the adder's latency is now a function of the gate window. If the gate opens for only a fraction of the bit-plane range, the adder requires multiple phase slices to complete. This is **time-extended arithmetic**: the carry chain completes over time, not in one combinational step.

### 5.3 Verified Addition

The DataAwareProcessor implements this. Verified test cases:

| Input | Expected | Gate-Corrected Result |
|-------|----------|----------------------|
| 7 + 5 | 12 | **12** ✓ |
| 15 + 17 | 32 | **32** ✓ |
| 64 + 64 | 128 | **128** ✓ |
| 3 + 4 | 7 | **7** ✓ |
| 12 + 8 | 20 | **20** ✓ |
| 200 + 300 | 500 | **500** ✓ |

Before the correction (using `cos²`), the results were wrong (7+5=10, 15+17=30) because the gate killed the carry chain prematurely.

## Chapter 6 — Phase Schedules for Gates

### 6.1 The Gate Schedule

The gate's window [T_min, T_max] defines a **phase schedule**: when the gate is open over time. The phase schedule can be:

1. **Linear**: θ(T) increases linearly across the window (the standard)
2. **Sinusoidal**: θ(T) = base + amp · sin(freq · T)
3. **Exponential**: θ(T) = base · exp(decay · T)

### 6.2 Schedule Selection

Different computations need different schedules:
- **Addition**: linear — the carry chain needs steady progress
- **Attention**: sinusoidal — the focus should sweep and return
- **Accumulation**: exponential — early arrivals matter more

The PhaseScheduleExplorer generates and compares these.

## Chapter 7 — Summary

The temporal gate is a trigonometric conditional execution window:

- **Grammar**: gate with address, window, theta bounds, 4 routes
- **Theta mapping**: T → θ(T) = (π/2) · (T − T_min)/(T_max − T_min)
- **Fitness**: sin(2θ) — triangular, 0→1→0, corrected from cos²
- **Execution**: compute iff fitness > threshold
- **Bitsliced**: gate gates the carry chain across bit-planes
- **Schedule**: linear/sinusoidal/exponential phase schedules

## References

- [Theory](theory.md) — the gate model
- [Theorems](theorems.md) — the fitness theorems
- [Timing](timing.md) — the theta mapping
- [Artifacts](../artifacts.md) — `tpn-simulator` (TrigGate), `temporal-fabric` (TrigGate + ErrorCorrectingBitsliced)
