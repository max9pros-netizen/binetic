# binetic

A self-describing, spatially-organized computation fabric for local AI inference
on constrained hardware (8-16GB unified memory), scaling to clusters.

**Status: pre-alpha. Recovered from session history 2026-09-24. Build has 11
known compile errors in binetic-core (u64/usize mismatches in backend.rs);
see RECOVERY.md.**

See [ARCHITECTURE.md](ARCHITECTURE.md) for the full specification (v0.1.0,
sections 1-23 + appendices).

## Workspace layout

| Crate | Purpose |
|---|---|
| `binetic-core` | The fabric: addressing, registers, bitsliced lanes, bloom filters, echo propagation, rotation scheduling, storage tiers, backend abstraction |
| `binetic-llama-cpp` | llama.cpp CPU backend (vendored under `backends/llama-cpp`, re-clone — see .gitignore) |
| `binetic-mlx` | MLX backend for Apple Silicon (macOS-only, cfg-gated) |
| `binetic-cli` | Command-line interface |

## Build

```bash
# clone backends first (not committed):
git clone --depth 1 https://github.com/ggerganov/llama.cpp backends/llama-cpp
git clone --depth 1 https://github.com/ml-explore/mlx backends/mlx

cargo build
cargo test
```

## Core ideas

1. 128-bit RegisterAddress encodes spine/layer/ring/sector/shard/temporal -
   the address IS the index, lineage path, tier assignment, and rotation phase.
2. Two interlaced planes: IPv6 data plane (bitsliced payloads + XOR lineage),
   IPv4 control plane (ops, routing, echo propagation).
3. 4D sphere geometry: 1024^3 spatial cube + temporal registers, nested
   sub-spheres.
4. Rotation gradient as cache policy (layer 0 hot/static -> layer N cold/bulk).
5. XOR as the native algebra: lineage, fold, diff, distance, echo cancellation.
6. Echo minimization: delta propagation, bloom-gated, rotation-damped.
7. Bitsliced lanes: bit-plane storage, SIMD-native, natural for quantization.
8. Tiered storage (hot/warm/gpu/cold-mmap/remote) with no-copy
   compute-to-data routing and energy-aware scheduling.
9. Compound registers (u256/u512/u1024) and database semantics (sections 22-23).
