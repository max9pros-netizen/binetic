"""
binetic DSL package — the .tpn domain-specific language.

Write the temporal protocol in a declarative DSL; the parser compiles it
into Protocol / Network objects. The same file runs in all three modes
(simulate / synth / deploy).

Example network.tpn:

    protocol version 2 {
      neuron 0x20010DB8000000000000000000000001 phase 500 {
        threshold 0.43
        isa {
          gate 0 { window 0..100 theta 0.0..1.57 threshold 0.43 route 0 }
          gate 1 { window 100..200 theta 1.57..3.14 threshold 0.43 route 1 }
        }
      }
      synapse 0x...001 -> 0x...002 delay 50
      mode simulate
    }
"""

from __future__ import annotations

import re
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Union

try:
    from .protocol import Protocol, GateDef
except ImportError:
    import sys

    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from protocol import Protocol, GateDef

try:
    from .network import Network, Neuron
except ImportError:
    import sys

    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from network import Network, Neuron

PI = 3.141592653589793


@dataclass
class DslGate:
    index: int
    window_min: int
    window_max: int
    theta_min: float
    theta_max: float
    threshold: float
    route: int


@dataclass
class DslNeuron:
    ip: int
    phase: int
    threshold: float = 0.43
    gates: List[DslGate] = field(default_factory=list)


@dataclass
class DslProgram:
    version: int = 2
    neurons: List[DslNeuron] = field(default_factory=list)
    synapses: List[Tuple[int, int, float]] = field(default_factory=list)
    mode: str = "simulate"


class DslParseError(Exception):
    pass


class Parser:
    """Parse a .tpn file into a DslProgram using line-oriented parsing."""

    def parse_file(self, path: str) -> DslProgram:
        with open(path) as f:
            text = f.read()
        return self.parse(text)

    def parse(self, text: str) -> DslProgram:
        program = DslProgram()
        current_neuron: Optional[DslNeuron] = None
        current_gate: Optional[DslGate] = None
        in_gate = False

        for raw_line in text.splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue

            # protocol block: protocol version N {
            m = re.match(r"^protocol\s+v(?:ersion\s+)?(\d+)(?:\s*\{$)?\s*$", line)
            if m:
                program.version = int(m.group(1))
                continue

            # mode
            m = re.match(r"^mode\s+(simulate|synth|deploy)\s*$", line)
            if m:
                program.mode = m.group(1)
                continue

            # neuron declaration
            m = re.match(r"^neuron\s+(0x[0-9a-fA-F]+)(?:\s+phase\s+(\d+))?\s*\{$", line)
            if m:
                neuron_ip = int(m.group(1), 16)
                neuron_phase = int(m.group(2) or "0")
                current_neuron = DslNeuron(ip=neuron_ip, phase=neuron_phase)
                in_gate = False
                program.neurons.append(current_neuron)
                continue

            # gate declaration: gate N { window ... } (one line or spread)
            m = re.match(r"^gate\s+(\d+)\s*\{(.*)$", line)
            if m:
                idx = int(m.group(1))
                rest = m.group(2).strip()
                current_gate = DslGate(
                    index=idx,
                    window_min=0,
                    window_max=100,
                    theta_min=0.0,
                    theta_max=PI / 2.0,
                    threshold=0.43,
                    route=0,
                )
                if rest:
                    if self._parse_gate_props(rest, current_gate):
                        current_neuron.gates.append(current_gate)
                        current_gate = None
                        in_gate = False
                        continue
                    in_gate = True
                else:
                    in_gate = True
                continue

            if in_gate and current_gate is not None:
                m = re.match(r"^window\s+(\d+)\.\.(\d+)\s*$", line)
                if m:
                    current_gate.window_min = int(m.group(1))
                    current_gate.window_max = int(m.group(2))
                    continue
                m = re.match(r"^theta\s+([0-9.]+)\.\.([0-9.]+)\s*$", line)
                if m:
                    current_gate.theta_min = float(m.group(1))
                    current_gate.theta_max = float(m.group(2))
                    continue
                m = re.match(r"^threshold\s+([0-9.]+)\s*$", line)
                if m:
                    current_gate.threshold = float(m.group(1))
                    continue
                m = re.match(r"^route\s+(\d+)\s*$", line)
                if m:
                    current_gate.route = int(m.group(1))
                    continue
                if line == "}":
                    if current_neuron is not None:
                        current_neuron.gates.append(current_gate)
                    current_gate = None
                    in_gate = False
                    continue

            if current_neuron is not None:
                m = re.match(r"^threshold\s+([0-9.]+)\s*$", line)
                if m:
                    current_neuron.threshold = float(m.group(1))
                    continue
                m = re.match(r"^isa\s*\{$", line)
                if m:
                    continue
                if line == "}":
                    # end of neuron block (includes closing isa block)
                    continue

            # synapse
            m = re.match(r"^synapse\s+(0x[0-9a-fA-F]+)\s*->\s*(0x[0-9a-fA-F]+)\s+delay\s+([0-9.]+)\s*$", line)
            if m:
                pre = int(m.group(1), 16)
                post = int(m.group(2), 16)
                delay = float(m.group(3))
                program.synapses.append((pre, post, delay))
                continue

            raise DslParseError(f"cannot parse line: {line!r}")

        return program

    def _parse_gate_props(self, rest: str, gate: DslGate) -> bool:
        """Parse gate properties from the remainder of a line like
        'window 0..100 theta 0.0..1.57 threshold 0.43 route 0'."""
        props = re.findall(
            r"(\bwindow\b)\s+([0-9]+)\.\.([0-9]+)|"
            r"(\btheta\b)\s+([0-9.]+)\.\.([0-9.]+)|"
            r"(\bthreshold\b)\s+([0-9.]+)|"
            r"(\broute\b)\s+(\d+)",
            rest,
        )
        if not props:
            return False
        for p in props:
            if p[0]:
                gate.window_min = int(p[1])
                gate.window_max = int(p[2])
            elif p[3]:
                gate.theta_min = float(p[4])
                gate.theta_max = float(p[5])
            elif p[6]:
                gate.threshold = float(p[7])
            elif p[8]:
                gate.route = int(p[9])
        return True


# --------------------- Compilation: DSL -> binetic Protocol/Network ---------------------

import sys

_binetic_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _binetic_dir not in sys.path:
    sys.path.insert(0, _binetic_dir)

from protocol import Protocol, GateDef

try:
    from network import Network, Neuron
except ImportError:
    from network import Network, Neuron


def compile_program(prog: DslProgram, kind: str = "protocol") -> Union[Protocol, Network]:
    """Compile a DslProgram into a binetic Protocol (or Network)."""
    if kind == "protocol":
        proto = Protocol(mode=prog.mode)
        for n in prog.neurons:
            proto.register(ipv6=n.ip, ipv4=n.ip & 0xFFFFFFFF, phase=n.phase)
            for g in n.gates:
                proto.gate(
                    address=n.ip,
                    window=(g.window_min, g.window_max),
                    threshold=g.threshold,
                    theta_min=g.theta_min,
                    theta_max=g.theta_max,
                )
        return proto
    else:
        net = Network()
        for n in prog.neurons:
            net.add_neuron(ip=n.ip, phase=n.phase, threshold=n.threshold)
        for pre, post, delay in prog.synapses:
            net.synapse(pre, post, delay=delay)
        return net


def load(path: str, kind: str = "protocol") -> Union[Protocol, Network]:
    """Load and compile a .tpn file."""
    parser = Parser()
    prog = parser.parse_file(path)
    return compile_program(prog, kind=kind)


def demo() -> None:
    """Demonstrate the DSL parser with a sample .tpn program."""
    print("=" * 64)
    print("   binetic DSL — .tpn parser")
    print("   Declarative temporal protocol")
    print("=" * 64)
    print()

    sample = """
    # network.tpn — a temporal neural network
    protocol version 2 {
      neuron 0x20010DB8000000000000000000000001 phase 500 {
        threshold 0.43
        isa {
          gate 0 { window 0..100 theta 0.0..1.57 threshold 0.43 route 0 }
          gate 1 { window 100..200 theta 1.57..3.14 threshold 0.43 route 1 }
        }
      }
      neuron 0x20010DB8000000000000000000000002 phase 550 {
        threshold 0.5
        isa {
          gate 0 { window 0..100 theta 0.0..1.57 threshold 0.5 route 0 }
        }
      }
      synapse 0x20010DB8000000000000000000000001 -> 0x20010DB8000000000000000000000002 delay 50.0
      mode simulate
    }
    """

    sample_path = "/tmp/tpn/network.tpn"
    os.makedirs("/tmp/tpn", exist_ok=True)
    with open(sample_path, "w") as f:
        f.write(sample)

    parser = Parser()
    prog = parser.parse_file(sample_path)

    print("--- 1. Parsed .tpn program ---")
    print(f"   version: {prog.version}, mode: {prog.mode}")
    print(f"   neurons: {len(prog.neurons)}, synapses: {len(prog.synapses)}")
    for n in prog.neurons:
        print(f"     neuron 0x{n.ip:032x} phase={n.phase} threshold={n.threshold}")
        for g in n.gates:
            print(f"       gate {g.index}: window=({g.window_min},{g.window_max}) "
                  f"theta=({g.theta_min:.2f},{g.theta_max:.2f}) thr={g.threshold}")
    for pre, post, delay in prog.synapses:
        print(f"     synapse 0x{pre:x} -> 0x{post:x} delay={delay}")
    print()

    proto = compile_program(prog, kind="protocol")
    print("--- 2. Compiled to binetic Protocol ---")
    print(f"   {proto}")
    print()

    packets = [
        (0x20010DB8000000000000000000000001, 50, 10),
        (0x20010DB8000000000000000000000002, 55, 20),
    ]
    receipts = proto.run(packets)
    print("--- 3. Execute packets ---")
    for p, r in zip(packets, receipts):
        print(f"   packet->0x{p[0]:x} arrival={p[1]} value={p[2]} result=0x{r.result:x}")
    print(f"   verified: {proto.verify(receipts)}")
    print()

    print("--- 4. Synthesize ---")
    luts, brams, dsps = proto.synthesize()
    print(f"   -> {luts} LUTs, {brams} BRAMs, {dsps} DSPs")
    print()

    print("   .tpn DSL parses, compiles, executes, verifies, synthesizes.")


if __name__ == "__main__":
    demo()
