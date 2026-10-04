"""binetic-runtime — the temporal computing runtime.

Declarative interface: write a .tpn protocol and run it — simulate,
synthesize, or deploy.

Usage:
    python3 -m binetic_runtime simulate network.tpn -p IP,arrival,value
    python3 -m binetic_runtime synth    network.tpn -o OUTDIR
    python3 -m binetic_runtime deploy   network.tpn -p IP,arrival,value

Or install with `pip install .` and call `binetic <command> <program> ...`
directly from anywhere.
"""

from __future__ import annotations

import os
import sys

# Make the runtime package's modules (cli.py, protocol.py, dsl/) importable
# when invoked as `python3 -m binetic_runtime` from this directory.
_binetic_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _binetic_root not in sys.path:
    sys.path.insert(0, _binetic_root)

from cli import main

__version__ = "0.1.0"
__all__ = ["main", "__version__"]
