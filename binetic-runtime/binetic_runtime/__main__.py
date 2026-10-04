import os
import sys

# Make the runtime's modules (cli.py, protocol.py, dsl/) importable
# when invoked as `python3 -m binetic_runtime` from this directory.
_binetic_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _binetic_root not in sys.path:
    sys.path.insert(0, _binetic_root)

from cli import main

if __name__ == "__main__":
    sys.exit(main())
