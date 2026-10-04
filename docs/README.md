# Temporal Packet Network (TPN)

**A temporal computing protocol where time is a primitive, computation is arrival, and the protocol is the computer.**

## Table of Contents

- [1. Overview](#1-overview)
- [2. The Theory](docs/theory.md)
- [3. Register Encoding](docs/register.md)
- [4. Temporal Gates](docs/gates.md)
- [5. Theorems & Proofs](docs/theorems.md)
- [6. Timing Semantics](docs/timing.md)
- [7. Phase Schedules](docs/phases.md)
- [8. Error Correction](docs/error-correction.md)
- [9. Temporal Attention](docs/attention.md)
- [10. Component Relationships](docs/relationships.md)
- [11. Self-Hosting Protocol](docs/self-hosting.md)
- [12. Temporal AI](docs/temporal-ai.md)
- [13. Architecture](docs/architecture.md)
- [4. Grammar & API](docs/grammar.md)
- [15. Verified Artifacts](docs/artifacts.md)

---

## 1. Overview

### The Premise

**Change the nature of computation by using mathematics and time as the most important primitives — not physical substrates.**

Traditional computing models:
- **Von Neumann**: fetch → decode → execute. Clock-driven. Data moves to compute.
- **Quantum**: prepare → evolve → measure. Physical qubits. Probabilistic.
- **FPGA**: configure → run. Static hardware.

The TPN model:
- **TPN**: packet arrives → phase maps to θ → fitness gates → compute happens. **Arrival = execution.** No clock. No fetch. No decode. The data *is* the trigger.

### The Core Insight

> **The register format IS the packet format.**
>
> - U256 Register: `[id:32 | phase:64 | angle:64 | data:64]`
> - TPN Packet: `[id:128 (IPv6) | phase:32 (IPv4) | angle:32 | value:64]`
>
> A packet IS a register that has been transmitted. A register IS a packet waiting to arrive. The boundary between *data at rest* and *data in motion* is erased.

### The Four Pillars

1. **Temporal Gates** — computation triggered by arrival-time phase alignment (`sin(2θ)` fitness)
2. **Bitsliced Phased Execution** — 256-bit registers sliced into temporal bit-planes
3. **Self-Hosting Synthesis** — the protocol's gate definitions auto-instantiate LUTs, BRAMs, DSPs
4. **Temporal AI** — neural computation as temporal routing where synapses are phase delays

### Verified Artifacts

| Component | Description | Status |
|-----------|-------------|--------|
| `tpn-simulator` | TPN v2.0 primitives | ✅ Compiled, running |
| `temporal-fabric` | TPN + bitsliced + FPGA layer | ✅ Compiled, running |
| `self-hosting-protocol` | IP registers = computers; synthesis creates resources | ✅ Compiled, running |
| `temporal-net` | Temporal neurons with auto-evolving phases | ✅ Compiled, running |

---

## The Critical Correction

The TPN Specification v2.0 claimed `cos²(θ − π/4)` ≡ linear fitness. **This is mathematically false.** It yields 0.5 at both window boundaries instead of 0, breaking gate equivalence.

We corrected the fitness function to **`sin(2θ)`**, which yields the correct triangular profile: 0 → 1 → 0 across the gate window. This correction is foundational to every theorem in this document.

| Time | `cos²(θ−π/4)` (WRONG) | `sin(2θ)` (CORRECT) |
|------|----------------------|---------------------|
| T = T_min | 0.5 | **0** |
| T = midpoint | 1.0 | **1.0** |
| T = T_max | 0.5 | **0** |

See [Theorems & Proofs](docs/theorems.md) for the formal proof.

---

## The Most Groundbreaking Opportunity

**The TPN protocol IS a neural network architecture.**

| Neural Primitive | Temporal Realization |
|------------------|----------------------|
| Neuron | IP register (temporal address) |
| Synaptic weight | Phase delay of arrival |
| Activation | Trig gate window (`sin(2θ)`) |
| Input | Packet arrival time |
| Attention | Temporal proximity |
| Learning | Auto-evolution of phases |
| Memory | Accumulated register value |
| Hardware | Synthesized from protocol structure |

- **Communication IS computation**
- **No matrix multiplies**
- **The hardware is synthesized from the protocol — self-hosting**

---

## References

- [TPN Specification v2.0](https://github.com/9pros/TPA-fpga) — protocol grammar, trigonometric semantics
- [TPA-fpga](https://github.com/9pros/TPA-fpga) — FPGA integration, packets, receipts
- [fpga-network-stack](https://github.com/fpgasystems/fpga-network-stack) — 10-100Gbit/s TCP/IP/RoCEv2 FPGA stack
- `/tmp/tpn-simulator/`, `/tmp/temporal-fabric/`, `/tmp/self-hosting-protocol/`, `/tmp/temporal-net/` — verified Rust artifacts

---

*Version 1.0 — Temporal Packet Network Specification*
*For: binetic AI — temporal computing as a paradigm shift*
