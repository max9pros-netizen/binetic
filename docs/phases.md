# Phase Schedules

## Chapter 1 — The Phase Schedule Concept

The **phase schedule** determines *when* each bit-plane is processed. It is the temporal orchestration of computation.

### 1.1 Definition

```
schedule: ℕ → ℝ
schedule(j) = t_j   // the time at which bit-plane j is processed
```

For a linear schedule with quantum δ:

```
t_j = j · δ
```

### 1.2 Why Phase Schedules Matter

In the bitsliced model, computation proceeds bit-plane by bit-plane. The phase schedule determines the **timing of this progression**. This is crucial for:

1. **Carry propagation:** the carry c[j] → c[j+1] must flow in order
2. **Gate alignment:** each bit-plane's computation must occur when its gate is open
3. **Temporal attention:** which bit-planes fire together determines what "attends" to what
4. **Scheduling concurrency:** overlapping phases enable parallelism

## Chapter 2 — The Three Canonical Schedules

### 2.1 Linear Schedule

```
t_j = j · δ
```

| j | t_j |
|---|-----|
| 0 | 0 |
| 1 | δ |
| 2 | 2δ |
| 3 | 3δ |
| ... | ... |

**Characteristics:** steady, predictable, monotonic. The ideal for carry chains, where steady progress from bit 0 to bit 255 is required.

### 2.2 Sinusoidal Schedule

```
t_j = base + amp · sin(freq · j)
```

| j | t_j |
|---|-----|
| 0 | base |
| 1 | base + amp·sin(freq) |
| 2 | base + amp·sin(2·freq) |
| ... | ... |

**Characteristics:** sweeping, periodic, non-monotonic. The ideal for attention, where the focus should sweep across the input and return.

From the reference implementation:

```
Plane   Linear   Sinusoidal  Exponential
    0      0.00         5.00         1.00
    1      1.00         6.44         1.22
    2      2.00         7.52         1.49
    3      3.00         7.99         1.82
    4      4.00         7.73         2.23
    5      5.00         6.80         2.72
    6      6.00         5.42         3.32
    7      7.00         3.95         4.06
    8      8.00         2.73         4.95
    9      9.00         2.07         6.05
   10     10.00         2.12         7.39
   11     11.00         2.88         9.03
   12     12.00         4.16        11.02
   13     13.00         5.65        13.46
   14     14.00         6.97        16.44
   15     15.00         7.81        20.09
```

**Characteristics:** front-loaded then compressed. The ideal for accumulation, where early contributions matter more.

### 2.4 Choosing a Schedule

The schedule is a **design choice** that affects behavior:

| Schedule | Best for | Effect on computation |
|----------|----------|----------------------|
| Linear | Addition, gates | Steady, predictable processing |
| Sinusoidal | Attention | Sweeping focus, periodic attention |
| Exponential | Accumulation | Early contributions weighted more |

**Second-order insight:** the schedule is not just an implementation detail — it is a **computational primitive**. Changing the schedule changes what the system computes. This is schedule-driven computation: the timing *is* the algorithm.

## Chapter 3 — Schedule as Algorithm

### 3.1 The Schedule Encodes the Algorithm

Because the schedule determines when each bit-plane fires, the schedule encodes the algorithm:

- **Addition:** linear schedule, carry chained
- **Attention:** sinusoidal schedule, sweeping focus
- **Accumulation:** exponential schedule, front-loaded

The algorithm is not in the gates — it is in the **phases**. The gates are uniform; the schedule is the program.

### 3.2 Dynamic Scheduling

The schedule can be **dynamic**: computed at runtime based on the data:

```
t_j = f(data, j)
```

For example, the DataAwareProcessor uses an **exponential decay gate** relative to the register phase: the schedule adapts to the data. This is data-driven scheduling.

### 3.3 Phase Schedule Explorer

The reference implementation (`/tmp/temporal-fabric/src/main.rs`) includes a `PhaseScheduleExplorer` that generates and compares all three schedules:

```rust
fn linear(j: usize, delta: f64) -> f64 { j as f64 * delta }
fn sinusoidal(j: usize, base: f64, amp: f64, freq: f64) -> f64 { base + amp * (j as f64 * freq).sin() }
fn exponential(j: usize, base: f64, decay: f64) -> f64 { base * (j as f64 * decay).exp() }
```

## Chapter 4 — Phase Alignment and Concurrency

### 4.1 Phase Alignment

Two bit-planes are **phase-aligned** if their processing times coincide (or are close):

```
phase_aligned(j, k) ⇔ |t_j − t_k| < ε
```

Phase-aligned bit-planes fire together and can communicate. This is the basis of **parallelism** and **attention**: bit-planes that fire together form a cluster.

### 4.2 Temporal Clusters

A **temporal cluster** is a set of bit-planes that fire close together in time. The cluster structure determines:

- What computes together (parallelism)
- What attends to what (attention)
- How carries propagate (locality)

With a linear schedule, clusters are single bit-planes (no concurrency). With a sinusoidal schedule, clusters form at the peaks and troughs (the sweep creates groups). With an exponential schedule, clusters form early (front-loaded).

### 4.3 Schedule-Driven Parallelism

**Second-order insight:** the schedule *creates* parallelism. By choosing a non-monotonic schedule (sinusoidal), you create clusters of bit-planes that fire together, enabling parallel computation. The sinusoidal schedule's peaks create clusters of high-activity bit-planes.

This is a fundamental departure from classical parallelism (which splits work across cores): TPN parallelism is **emergent from the timing**.

## Chapter 5 — The Phase Field

### 5.1 The Phase Field as Schedule

The register's `phase` field (64-bit) is itself a schedule: it assigns each register an arrival time. Across the network, all registers together form a **global phase schedule**:

```
schedule(register_i) = phase_i
```

The IPv4 phase field (32-bit) is the network-visible part of this schedule.

### 5.2 Phase Encoding

From `U256::from_id_phase_angle`:

```rust
reg.0[0] = ((id as u64) << 32) | (phase & 0xFFFFFFFF);   // id + phase low
reg.0[1] = phase >> 32;                                  // phase high
reg.0[2] = angle;                                        // angle
```

The phase is split across reg.0[0] and reg.0[1]. The angle is reg.0[2]. This encoding makes the phase and angle directly accessible from the register word.

## Chapter 6 — Summary

The phase schedule is the temporal orchestration layer:

- **Definition:** t_j = when bit-plane j is processed
- **Three canonical schedules:** linear (steady), sinusoidal (sweeping), exponential (front-loaded)
- **The schedule encodes the algorithm** — timing is computation
- **Schedule choice affects behavior** — it is a design knob
- **The register phase is itself a schedule** — global coordination
- **Schedule-driven parallelism** — concurrency emerges from timing

## References

- [Theory](theory.md) — bitsliced phased execution
- [Timing](timing.md) — the temporal coordinate system
- [Gates](gates.md) — gate window timing
- [Artifacts](../artifacts.md) — `PhaseScheduleExplorer`
