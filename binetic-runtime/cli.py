#!/usr/bin/env python3
"""
binetic — the temporal computing runtime.

Declarative interface: write a .tpn protocol, run it in any mode.

    binetic simulate network.tpn      # execute packets through the temporal protocol
    binetic synth network.tpn         # synthesize LUTs/BRAMs/DSPs + HLS + XDC
    binetic deploy network.tpn        # compile, execute, synthesize, verify end to end

All three modes compile the same .tpn file. The interface doesn't ask you to
choose an implementation — the declaration runs everywhere.
"""

from __future__ import annotations

import argparse
import sys
import json

try:
    from dsl import Parser, compile_program, DslParseError
    from protocol import Protocol
    from synth import Synthesizer
except ImportError:
    import os

    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from dsl import Parser, compile_program, DslParseError
    from protocol import Protocol
    from synth import Synthesizer


VERSION = "0.1.0"


def cmd_simulate(args) -> int:
    """Execute the protocol: packets arrive at registers, trig gates fire."""
    parser = Parser()
    prog = parser.parse_file(args.program)
    proto = compile_program(prog, kind="protocol")
    print(f"[simulate] {proto}")
    print(f"[simulate] loaded {args.program}, {len(proto.registers)} registers, "
          f"{len(proto.gates)} gates")
    print(f"[simulate] sending {len(args.packets)} packet(s)...")

    packets = []
    for pkt in args.packets:
        ip, arrival, value = pkt.split(",")
        packets.append((int(ip, 16), int(arrival), int(value)))

    receipts = proto.run(packets)
    for p, r in zip(packets, receipts):
        print(f"    arrive 0x{p[0]:x} arrival={p[1]} value={p[2]} "
              f"result=0x{r.result:x} verified={proto.verify([r])}")
    print(f"[simulate] {len(receipts)} packets processed, "
          f"verified={proto.verify(receipts)}")
    return 0


def cmd_synth(args) -> int:
    """Synthesize the protocol into FPGA hardware plan + HLS + XDC."""
    parser = Parser()
    prog = parser.parse_file(args.program)
    proto = compile_program(prog, kind="protocol")

    synth = Synthesizer(proto)
    out_dir = args.output or "/tmp/tpn/out"
    synth.synthesize(out_dir)

    plan = proto.synthesize()
    print(f"[synth] {proto}")
    print(f"[synth] -> {out_dir}")
    print(f"        {plan[0]} LUTs  {plan[1]} BRAMs  {plan[2]} DSPs")
    print(f"        HLS: {out_dir}/tpn_top.hls")
    print(f"        XDC: {out_dir}/constraints.xdc")
    plan_obj = synth.plan()
    print(f"[synth] density on Xilinx U55C: {plan_obj.density}")
    return 0


def cmd_deploy(args) -> int:
    """Compile -> execute -> synthesize -> verify end to end."""
    parser = Parser()
    prog = parser.parse_file(args.program)
    proto = compile_program(prog, kind="protocol")

    print("[deploy] compiled: "
          f"{len(proto.registers)} registers, {len(proto.gates)} gates")

    packets = []
    for pkt in args.packets:
        ip, arrival, value = pkt.split(",")
        packets.append((int(ip, 16), int(arrival), int(value)))

    receipts = proto.run(packets)
    ok = proto.verify(receipts)
    print(f"[deploy] executed {len(receipts)} packets, verified={ok}")
    if not ok:
        print("[deploy] ERROR: receipt chain failed verification")
        return 1

    synth = Synthesizer(proto)
    plan = proto.synthesize()
    print(f"[deploy] synthesized: {plan[0]} LUTs, {plan[1]} BRAMs, {plan[2]} DSPs")

    receipts_json = "/tmp/tpn/deploy_receipts.json"
    with open(receipts_json, "w") as f:
        json.dump([{"packet_id": r.packet_id, "timestamp": r.timestamp,
                    "result": r.result, "verified": True}
                   for r in receipts], f, indent=2)
    print(f"[deploy] receipts written: {receipts_json}")
    print("[deploy] deploy complete — protocol IS the computer")
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        prog="binetic",
        description="Temporal computing runtime — declare, simulate, synthesize.",
    )
    parser.add_argument("-v", "--version", action="version", version=f"binetic {VERSION}")
    sub = parser.add_subparsers(dest="command", required=True)

    sim = sub.add_parser("simulate", help="execute the protocol")
    sim.add_argument("program", help=".tpn program file")
    sim.add_argument("-p", "--packet", action="append", dest="packets",
                     help="packet as ip,arrival,value (hex ip)")
    sim.set_defaults(func=cmd_simulate)

    sn = sub.add_parser("synth", help="synthesize FPGA hardware")
    sn.add_argument("program", help=".tpn program file")
    sn.add_argument("-o", "--output", help="output directory")
    sn.set_defaults(func=cmd_synth)

    dp = sub.add_parser("deploy", help="compile, execute, synthesize, verify")
    dp.add_argument("program", help=".tpn program file")
    dp.add_argument("-p", "--packet", action="append", dest="packets",
                    help="packet as ip,arrival,value (hex ip)")
    dp.set_defaults(func=cmd_deploy)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
