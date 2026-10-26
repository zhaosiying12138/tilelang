"""CPU-only tests; no CUDA initialization, compilation, or kernel execution."""

from pathlib import Path
import sys

# benchmark/z3_cost_model/operators/tests -> repository root.
REPOSITORY_ROOT = Path(__file__).resolve().parents[4]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))
