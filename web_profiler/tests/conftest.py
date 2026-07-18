"""Shared fixtures for the VoxelSim Workbench test suite.

Session-scoped ``app``/``client``/``index`` fixtures build the results index
once (~2 s on the real data set) and share it across all test modules.
"""

import sys
from pathlib import Path

import pytest

# Make the project root importable regardless of the pytest invocation dir.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from web_profiler.server import create_app          # noqa: E402
from web_profiler.server import classify, parsers   # noqa: E402
from web_profiler.server.index import get_index     # noqa: E402

# Verified on the real data set: compile cache (all_configs_dict.json) exists
# and its op count matches the parsed operators of this log.
KNOWN_CLASSIFIABLE = (
    "results/logs/dit-xl/bs_32/core_256/decode/sa_32-vu_32/"
    "sram_2048-drambw_12288_PLACEHOLDER/topo_1-nocbw16/best/"
    "output_cg_8_row_8192.log")


@pytest.fixture(scope="session")
def app():
    """Flask app without the prewarm thread (tests stay synchronous)."""
    return create_app(prewarm=False)


@pytest.fixture(scope="session")
def client(app):
    return app.test_client()


@pytest.fixture(scope="session")
def index():
    """The shared ResultsIndex singleton over the real result roots."""
    return get_index()


@pytest.fixture(scope="session")
def real_id(index):
    """A real result id; the smallest log keeps op-level endpoints cheap."""
    return min(index.all(), key=lambda c: c["size"])["id"]


@pytest.fixture(scope="session")
def classifiable_id(index):
    """A result id for which classify.op_breakdown returns a breakdown.

    Falls back to a bounded scan (small logs first) when the known result is
    not indexed; skips when no classifiable result exists at all.
    """
    cfg = index.get(KNOWN_CLASSIFIABLE)
    if cfg is not None:
        ops = parsers.parse_operators_file(cfg["log_file"])
        if classify.op_breakdown(cfg, ops) is not None:
            return cfg["id"]
    tried = 0
    for cand in sorted(index.all(), key=lambda c: c["size"]):
        if classify.find_configs_dict(cand) is None:
            continue
        tried += 1
        if tried > 60:
            break
        ops = parsers.parse_operators_file(cand["log_file"])
        if classify.op_breakdown(cand, ops) is not None:
            return cand["id"]
    pytest.skip("no classifiable result in the index "
                "(compile cache all_configs_dict.json unavailable)")


@pytest.fixture(scope="session")
def unclassifiable_id(index):
    """A result id whose operators parse but cannot be classified."""
    tried = 0
    for cand in sorted(index.all(), key=lambda c: c["size"]):
        tried += 1
        if tried > 60:
            break
        ops = parsers.parse_operators_file(cand["log_file"])
        if ops and classify.op_breakdown(cand, ops) is None:
            return cand["id"]
    pytest.skip("no unclassifiable result found (everything classifies?)")
