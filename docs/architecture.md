# Architecture

## Chapter 1 — The Layered Stack

```
┌──────────────────────────────────────────────────────────────────┐
│  LAYER 5: TEMPORAL AI                                            │
│  Neurons = IP registers · Weights = phase delays · Learning      │
│  = phase evolution                                               │
├──────────────────────────────────────────────────────────────────┤
│  LAYER 4: SELF-HOSTING PROTOCOL                                  │
│  Each IPv4/IPv6 address = a computer · Synthesis creates         │
│  resources (LUTs/BRAMs/DSPs) from protocol structure             │
├──────────────────────────────────────────────────────────────────┤
│  LAYER 3: TPN SPEC v2.0                                          │
│  Trig temporal gates · 3× error correction · Attention ·         │
│  Phase schedules                                                 │
├──────────────────────────────────────────────────────────────────┤
│  LAYER 2: TPA-FPGA                                               │
│  TPN packets · Cryptographic receipts · Arrival = execution      │
│  AXI4-Stream wrapper · HLS modules · Gate encoding               │
├──────────────────────────────────────────────────────────────────┤
│  LAYER 1: FPGA-NETWORK-STACK (ETH Zurich)                        │
│  TCP/IP · RoCEv2 · UDP/IP · 10-100Gbit/s · AXI4-Stream · DMA    │
├──────────────────────────────────────────────────────────────────┤
│  LAYER 0: FPGA HARDWARE                                          │
│  LUTs · BRAMs · DSPs · HBM · Network interfaces                  │
└──────────────────────────────────────────────────────────────────┘
```

## Chapter 2 — Layer 5: Temporal AI

### 2.1 The Neural Network as Protocol

- **Neurons** = IP register addresses (Layer 4)
- **Weights** = phase delays of packet arrival (Layer 3 timing)
- **Activations** = trig gate windows (Layer 3 gates)
- **Attention** = phase alignment routing (Layer 3 attention)
- **Learning** = auto-evolution of phases (evolution loop)

### 2.2 The Neural Net Is the Protocol

The neural network is not "on top of" the protocol — it **is** the protocol running with a particular phase configuration. Different phase configurations = different networks.

## Chapter 3 — Layer 4: The Self-Hosting Protocol

### 3.1 The Protocol

- Each IP address = a computer (register)
- Arrival time = phase → gate evaluation
- The protocol's gate definitions define the hardware
- Synthesis maps gate definitions → LUTs/BRAMs/DSPs
- No external compute needed

### 3.2 Key Components

- `U256 Register` — id/phase/angle/data encoding
- `TPN Packet` — [id:128 | phase:32 | angle:32 | value:64]
- `TPN Receipt` — [packet_id | timestamp | result | proof]
- Gate grammar — [address | window | theta_min | theta_max | 4 routes]

## Chapter 4 — Layer 3: TPN Spec v2.0

### 4.1 Trigonometric Temporal Gates

- θ(T) = (π/2)·(T−T_min)/(T_max−T_min)
- fitness(θ) = sin(2θ) — triangular 0→1→0 (corrected from cos²)
- Gate fires iff fitness > threshold

### 4.2 3× Error Correction

- Each bit stored in 3 temporal copies
- Syndrome detection: copies disagree → error
- Majority voting: correct single errors

### 4.3 Temporal Attention

- Attention(Q, K, V) = softmax(Q·K^T/√d)·V mapped to phase alignment
- Trig softmax over fitness
- Attention as routing by phase alignment

### 4.4 Phase Schedules

- Linear: t_j = j·δ
- Sinusoidal: t_j = base + amp·sin(freq·j)
- Exponential: t_j = base·exp(decay·j)

## Chapter 5 — Layer 2: TPA-fpga

### 5.1 Repository

https://github.com/9pros/TPA-fpga — "Temporal Packet Network FPGA Integration: Self-hosting network protocol for on-demand computing resources"

### 5.2 Structure

```
TPA-fpga/
├── hdl/                    # SystemVerilog modules
│   ├── tpn_top.sv         # Top-level TPN module
│   ├── tpn_packet_parser.sv
│   ├── tpn_gate_executor.sv
│   ├── tpn_receipt_generator.sv
│   └── tpn_testbench.sv
├── hls/tpn/               # HLS C++ modules (Vivado HLS)
├── python/                # Python simulator & tests
│   ├── tpn_simulator.py
│   └── test_tpn.py
├── constraints/           # FPGA constraints
├── scripts/               # Build & test scripts
├── CMakeLists.txt
└── README.md
```

### 5.3 Key Features

- **Self-hosting packets** — each packet carries its own ISA
- **Arrival = execution** — packet arrival triggers computation
- **Cryptographic receipts** — verifiable proof of execution
- **IPv4/IPv6 duality** — temporal/spatial registers
- **TCP mode**: 534.8 pkt/s, 0% packet loss (UDP: 380.1 pkt/s, 28.5% loss)

### 5.4 Quick Start

```bash
# Python simulator
cd python && python3 tpn_simulator.py

# FPGA build
mkdir build && cd build
cmake .. -DFNS_PLATFORM=xilinx_u55c_gen3x16_xdma_3_202210_1 \
         -DFNS_DATA_WIDTH=64
make
```

## Chapter 6 — Layer 1: fpga-network-stack (ETH Zurich)

### 6.1 Repository

https://github.com/fpgasystems/fpga-network-stack

Scalable Network Stack supporting TCP/IP, RoCEv2, UDP/IP at 10-100Gbit/s.

- 969 stars, 312 forks
- BSD 3-Clause License
- Created 2016-09-15

### 6.2 Supported Boards

- Xilinx VC709
- Xilinx VCU118
- Alpha Data ADM-PCIE-7V3

### 6.3 Interfaces

All interfaces use **AXI4-Stream**. Includes DMA to host memory. Supports:
- TCP/IP stack (with configurable MSS, fast retransmit, Nagle)
- RoCEv2 stack (configurable queue pairs)
- UDP/IP stack

### 6.4 Quick Start

```bash
git clone --recurse-submodules <repo>
mkdir build && cd build
cmake .. -DFNS_PLATFORM=xilinx_u55c_gen3x16_xdma_3_202210_1 \
         -DFNS_DATA_WIDTH=64
make ip    # compile HLS modules, install to IP repo
```

### 6.5 HLS Modules

```
cd hls/<module>
mkdir build && cd build
cmake .. -DFNS_PLATFORM=<platform> -DFNS_DATA_WIDTH=<width>
make csim   # C simulation
make synth  # synthesis (csynth_design)
make cosim  # co-simulation (cosim_design)
make ip     # export IP (export_design)
```

## Chapter 7 — Layer 0: FPGA Hardware

### 7.1 Resources

- **LUTs** — lookup tables for gate boolean logic (8 per gate per register)
- **BRAMs** — block RAM for register files (1 per register)
- **DSPs** — DSP slices for trig evaluation (2 per gate)
- **HBM** — high-bandwidth memory for large register arrays
- **Network interfaces** — 10-100Gbit/s Ethernet

### 7.2 The Synthesis Map

For N registers, G gates per register:

```
LUTs:  8·N·G
BRAMs: 1·N
DSPs:  2·N·G
```

Verified (N=256, G=2): **2048 LUTs, 256 BRAMs, 1024 DSPs**.

## Chapter 8 — Data Flow Through the Stack

```
┌─────────────────────────────────────────────────────┐
│  Temporal AI (Layer 5)                               │
│  Packets arrive with phase delays = weights          │
├─────────────────────────────────────────────────────┤
│  Self-Hosting Protocol (Layer 4)                     │
│  Packet.id → register; packet.phase → arrival time   │
├─────────────────────────────────────────────────────┤
│  TPN Spec v2.0 (Layer 3)                             │
│  θ(T) → sin(2θ) → gate opens; 3× correction;        │
│  attention alignment; phase schedule                 │
├─────────────────────────────────────────────────────┤
│  TPA-fpga (Layer 2)                                  │
│  Packet parser → gate executor → receipt generator   │
│  AXI4-Stream wrapper; HLS modules                    │
├─────────────────────────────────────────────────────┤
│  fpga-network-stack (Layer 1)                        │
│  TCP/IP, RoCEv2, UDP/IP at 10-100Gbit/s, AXI4-Stream │
├─────────────────────────────────────────────────────┤
│  FPGA Hardware (Layer 0)                             │
│  LUTs/BRAMs/DSPs synthesized from protocol           │
└─────────────────────────────────────────────────────┘
```

Packets flow down (emission) and receipts flow up (verification).

## Chapter 9 — The Design Surface

The architecture offers many design knobs — each relationship is a tunable parameter:

| Knob | Effect |
|------|--------|
| Gate window [T_min, T_max] | How wide the gate is |
| Threshold | How selective the gate is (evolved: 0.43) |
| Phase schedule | Linear/sinusoidal/exponential — changes behavior |
| Error correction copies | 3× (more = more robust, less throughput) |
| Attention windows | Which neurons attend to which |
| Phase delays (weights) | The neural network's weights |
| Synthesis mapping | LUTs/BRAMs/DSPs per component |

The architecture is **programmable through timing**: change the phases, change the computation.

## Chapter 10 — Summary

The TPN architecture:

1. **Layer 5: Temporal AI** — the neural network is the protocol
2. **Layer 4: Self-hosting protocol** — each IP = computer, synthesis creates hardware
3. **Layer 3: TPN Spec v2.0** — trig gates, error correction, attention, schedules
4. **Layer 2: TPA-fpga** — packets, receipts, AXI4-Stream, HLS
5. **Layer 1: fpga-network-stack** — TCP/IP, RoCEv2, 10-100Gbit/s
6. **Layer 0: FPGA hardware** — LUTs/BRAMs/DSPs synthesized from the protocol

The protocol is self-hosting: it defines the hardware it runs on. No external compute needed. The neural network is the protocol evolved toward fitness.

## References

- [TPA-fpga](https://github.com/9pros/TPA-fpga)
- [fpga-network-stack](https://github.com/fpgasystems/fpga-network-stack)
- [Theory](theory.md) — the layered model
- [Artifacts](../artifacts.md) — verified implementations
