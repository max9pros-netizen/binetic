# binetic — Interface Design

## The interface

The interface is a **declarative temporal programming language**. You write the protocol once in the `.tpn` DSL and run it in any mode — the declaration compiles, executes, synthesizes, and verifies itself with no translation layer.

```
$ ./binetic simulate network.tpn -p 0x...001,50,10
$ ./binetic synth network.tpn -o /tmp/out
$ ./binetic deploy network.tpn -p 0x...001,50,10
```

Three modes. One declaration. Same data model everywhere: **the register format IS the packet format** `[id:32 | phase:64 | angle:64 | data:64]` — the same `U256` that simulates in Python is the TPA-fpga packet that synthesizes into HLS. This identity is why the interface is declarative rather than imperative: you never say "how to build it," you say "what it is."

## Layered API

The interface is four layers deep:

### 1. Exposure layer (`.tpn` DSL + CLI + Python API)

The `.tpn` domain-specific language is the user-facing interface:

```
protocol version 2 {
  neuron 0x20010DB8000000000000000000000001 phase 500 {
    threshold 0.43
    isa {
      gate 0 { window 0..100 theta 0.0..1.57 threshold 0.43 route 0 }
      gate 1 { window 100..200 theta 1.57..3.14 threshold 0.43 route 1 }
    }
  }
  synapse 0x...001 -> 0x...002 delay 50.0
  mode simulate
}
```

The CLI exposes three modes (`simulate` / `synth` / `deploy`); the Python API exposes `Parser`, `compile_program`, `load`, `Protocol`, `Network`, `Synthesizer`. Both compile the same file into the same objects.

### 2. Compilation layer (parser → IR → Protocol/Network)

`dsl/__init__.py` parses `.tpn` into `DslProgram`, `compile_program` turns it into a `Protocol` (for packet execution) or a `Network` (for auto-evolution). No intermediate format — parse straight into the runtime objects.

### 3. Execution layer (simulate / deploy)

`protocol.py` — `Protocol.arrive(ipv6, arrival, value)`: arrival maps to phase angle, the trig gate evaluates `sin(2θ)`, computation happens iff fitness > threshold, otherwise the register holds (memory). `Protocol.run(packets)` returns `TPNReceipt`s; `Protocol.verify(receipts)` closes the cryptographic chain.

### 4. Synthesis layer (synth / deploy)

`synth.py` — `Synthesizer.synthesize(out_dir)` writes HLS (`tpn_top.hls`), XDC constraints (`constraints.xdc`), and `hardware_plan.json`:

```
48 LUTs  2 BRAMs  12 DSPs
HLS: /tmp/out/tpn_top.hls
XDC: /tmp/out/constraints.xdc
density on Xilinx U55C: LUT 0.0% · BRAM 0.0% · DSP 0.1%
```

### 5. Verification layer (deploy)

Every packet produces a `TPNReceipt` (packet_id, timestamp, result, proof). `Protocol.verify` closes the chain — deploy fails fast if any receipt is unverifiable.

## Modes

| mode | does | when |
|---|---|---|
| `simulate` | parse → compile → execute packets → verify | pure Python, fast, for logic |
| `synth` | parse → compile → hardware plan → HLS + XDC + JSON | for FPGA build (Vivado HLS) |
| `deploy` | compile → execute → synthesize → verify + persist receipts | end-to-end CI, deployment |

## The three files that are one program

The interface's power comes from the fact that three files are really one program sharing one model:

- `protocol.py` — the runtime (`Protocol`, `Register`, `GateDef`, `TPNReceipt`)
- `synth.py` — the synthesizer (`HardwarePlan`, `Synthesizer`)
- `dsl/__init__.py` — the parser (`Parser`, `compile_program`)

The `Register` dataclass and the TPA-fpga packet definition are structurally identical, which is why `protocol.py`'s `synthesize()` returns LUTs/BRAMs/DSPs from the same gate definitions that `protocol.py`'s `arrive()` evaluates. No schema migration. No re-implementation.

## Why this is novel

Most interfaces ask you to describe hardware, describe software, then describe how they connect. This interface asks you to describe the protocol — and the protocol *is* the hardware and *is* the software. Temporal computing on real FPGA hardware: communication IS computation, time IS the weight parameter, attention IS topology, hardware IS synthesized from the declaration. Not quantum. Math and time only.
