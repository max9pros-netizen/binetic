# binetic — Temporal Computing Runtime

A clean declarative interface over the integrated TPN · bitsliced phased processor · self-hosting synthesis · FPGA network stack.

## What this is

The same `.tpn` protocol declaration runs in three modes without any change:

- **simulate** — pure Python execution: packets arrive at temporal registers, trig gates evaluate, receipts are produced and verified
- **synth** — synthesize the protocol into a hardware plan: LUTs/BRAMs/DSPs, HLS C++ for TPA-fpga, XDC timing constraints
- **deploy** — compile → execute → synthesize → verify end to end, with cryptographic receipt chain persisted

```
$ ./binetic simulate /tmp/tpn/example.tpn -p 0x...001,50,10
[simulate] Protocol(mode='simulate', registers=2, luts=48, brams=2, dsps=12)
[simulate] 2 packets processed, verified=True

$ ./binetic synth /tmp/tpn/example.tpn -o /tmp/tpn/out
[synth] 48 LUTs  2 BRAMs  12 DSPs
[synth] HLS: /tmp/tpn/out/tpn_top.hls   XDC: /tmp/tpn/out/constraints.xdc

$ ./binetic deploy /tmp/tpn/example.tpn -p 0x...001,50,10
[deploy] compiled: 2 registers, 3 gates
[deploy] executed 2 packets, verified=True
[deploy] synthesized: 48 LUTs, 2 BRAMs, 12 DSPs
[deploy] deploy complete — protocol IS the computer
```

## Why this is the interface

The interface does not ask you to choose an implementation. You declare the protocol, and the declaration runs everywhere:

- `python3 dsl/__init__.py` — the `.tpn` parser compiles into `Protocol` / `Network`
- `python3 cli.py` — the three-mode CLI
- `python3 protocol.py` — the runtime with `arrive`/`run`/`verify`/`synthesize`
- `python3 synth.py` — hardware plan + HLS + XDC exports
- `python3 network.py` — auto-evolution (learning = phase adjustment)

Every artifact shares one data model: **the register format IS the packet format**, so the declaration compiles into simulation, synthesis, and hardware with no translation layer.

## Files

- `protocol.py` — `Protocol`, `Register`, `GateDef`, `TPNReceipt` (trig gate with corrected `sin(2θ)` fitness)
- `network.py` — `Neuron`, `Network`, auto-evolution
- `synth.py` — `HardwarePlan`, `Synthesizer` (LUTs/BRAMs/DSPs, Xilinx U55C density)
- `dsl/` — the `.tpn` domain-specific language parser
- `cli.py` — the `binetic` CLI (simulate/synth/deploy)
- `binetic-runtime/` — the runtime subpackage (`__main__.py` entry point)

## Verified artifacts (docs/artifacts.md)

- `/tmp/tpn-simulator` — TPN v2.0 primitives (406 lines, compiled & ran)
- `/tmp/temporal-fabric` — TPN + bitsliced + FPGA packet/receipt (276 lines)
- `/tmp/self-hosting-protocol` — 256 IP computers → 2048 LUTs / 256 BRAMs / 1024 DSPs (138 lines)
- `/tmp/temporal-net` — temporal neural network (148 lines)

## Documentation

Full documentation at `/Users/binetic/tpn-spec/` (15 files, 3,718 lines):

- `docs/theory.md` — time as primitive, arrival = execution, register = packet
- `docs/gates.md` — trig gate grammar, `sin(2θ)` vs the spec's false `cos²`
- `docs/theorems.md` — 8 formal theorems with proofs
- `docs/timing.md`, `docs/phases.md`, `docs/error-correction.md`, `docs/attention.md`
- `docs/self-hosting.md` — each IP = a computer; synthesis from protocol structure
- `docs/temporal-ai.md` — neural net is the protocol (isomorphism table)
- `docs/architecture.md` — full 7-layer stack
- `docs/grammar.md` — .tpn DSL grammar
- `docs/interface-design.md` — layered API design
- `docs/artifacts.md` — verification log of every artifact
