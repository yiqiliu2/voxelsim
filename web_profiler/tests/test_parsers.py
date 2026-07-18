"""parsers.py: summary/operator parsing on real logs, plus synthetic
overlap / top_power / downsample coverage (pure functions only)."""

import pytest

from web_profiler.server import parsers

SUMMARY_KEYS = ("total_time", "total_energy_mj", "static_energy_mj",
                "dynamic_energy_mj", "avg_power_w", "overall_util",
                "dram_r_util", "dram_w_util", "sa_util", "vu_util",
                "noc_util", "mm_gflops", "vu_gflops")

DUR_KEYS = ("dur_ld", "dur_bcast", "dur_comp", "dur_shift", "dur_reduce",
            "dur_store")


@pytest.fixture(scope="module")
def sample_cfgs(index):
    """One real log from the primary root and one from a pareto root."""
    by_root = {}
    for cfg in index.all():
        by_root.setdefault(cfg["root"], cfg)
    chosen = [by_root[r] for r in ("logs", "pareto_decode") if r in by_root]
    assert chosen, "no sample logs found in the index"
    return chosen


# ---------------------------------------------------------------------------
# summary parsing on real logs
# ---------------------------------------------------------------------------

def test_summary_real_logs(sample_cfgs):
    for cfg in sample_cfgs:
        s = parsers.parse_summary_file(cfg["log_file"])
        assert s is not None, cfg["id"]
        for key in SUMMARY_KEYS:
            assert key in s, f"{key} missing from {cfg['id']}"
        assert isinstance(s["total_time"], int) and s["total_time"] > 0
        assert s["total_energy_mj"] > 0
        assert s["static_energy_mj"] >= 0 and s["dynamic_energy_mj"] >= 0
        # static + dynamic should approximately equal the total energy
        total = s["total_energy_mj"]
        assert abs(s["static_energy_mj"] + s["dynamic_energy_mj"]
                   - total) / total < 0.05
        assert s["avg_power_w"] > 0
        assert 0.0 <= s["overall_util"] <= 1.0
        for key in ("dram_r_util", "dram_w_util", "sa_util", "vu_util",
                    "noc_util", "mm_gflops", "vu_gflops"):
            assert s[key] >= 0.0, key


def test_summary_file_bad_input(tmp_path):
    assert parsers.parse_summary_file(tmp_path / "missing.log") is None
    junk = tmp_path / "junk.log"
    junk.write_text("not a simulation log\n")
    assert parsers.parse_summary_file(junk) is None


def test_summary_text_minimal():
    text = ("EXE time (total, fused): 12345,Energy (mJ): 10.0, "
            "Static: 4.0 mJ, Dyn.: 6.0 mJ\n")
    s = parsers.parse_summary_text(text)
    assert s["total_time"] == 12345
    assert s["total_energy_mj"] == 10.0
    assert s["static_energy_mj"] == 4.0
    assert s["dynamic_energy_mj"] == 6.0


# ---------------------------------------------------------------------------
# operator parsing on real logs
# ---------------------------------------------------------------------------

def test_operators_real_logs(sample_cfgs):
    for cfg in sample_cfgs:
        ops = parsers.parse_operators_file(cfg["log_file"])
        assert len(ops) > 0, cfg["id"]
        assert ops[0]["op_id"] == 0
        for prev, cur in zip(ops, ops[1:]):
            assert cur["op_id"] > prev["op_id"], "op_id must increase"
        for op in ops:
            for key in DUR_KEYS:
                assert op[key] >= 0, f"{key} negative in op {op['op_id']}"
            assert op["start_finish"] >= op["start_ld"]
            assert op["write_bytes"] >= 0 and op["read_bytes"] >= 0
            assert op["avg_power_w"] >= 0.0
            assert op["mm_util"] >= 0.0 and op["vu_util"] >= 0.0


# ---------------------------------------------------------------------------
# overlap / top_power / downsample (synthetic)
# ---------------------------------------------------------------------------

_OVERLAP_TEXT = """\
Interval: cycles=[0, 100] units = {'dram_r', 'comp_sa'}
Dynamic Power (W): 12.5
Interval: cycles=[100, 250] units = {'noc', 'comp_sa', 'dram_w'}
Dynamic Power (W): 20.0
"""


def test_parse_overlap_text():
    intervals = parsers.parse_overlap_text(_OVERLAP_TEXT)
    assert len(intervals) == 2
    first, second = intervals
    assert first["t_start"] == 0 and first["t_end"] == 100
    assert first["units"] == ["comp_sa", "dram_r"]
    assert first["power_w"] == 12.5
    assert second["power_w"] == 20.0
    assert second["units"] == ["comp_sa", "dram_w", "noc"]


def test_parse_top_power_text():
    intervals = parsers.parse_top_power_text(_OVERLAP_TEXT)
    assert [i["power_w"] for i in intervals] == [12.5, 20.0]


def test_downsample_intervals_merges():
    intervals = [{"t_start": i * 10, "t_end": (i + 1) * 10,
                  "units": ["comp"], "power_w": float(i)}
                 for i in range(100)]
    merged = parsers.downsample_intervals(intervals, max_points=10)
    assert len(merged) <= 10
    assert merged[0]["t_start"] == 0
    assert merged[-1]["t_end"] == 1000
    # short list passes through untouched
    assert parsers.downsample_intervals(intervals[:5], 10) == intervals[:5]


def test_parse_top_power_file_real(index):
    """parse_top_power_file parses a real top_power_cg_*.log from disk."""
    cfg = next(c for c in index.all() if c["has_overlap"])
    intervals = parsers.parse_top_power_file(cfg["top_power_file"])
    assert intervals, cfg["top_power_file"]
    for i in intervals:
        assert i["t_end"] >= i["t_start"]
        assert i["power_w"] >= 0.0
        assert isinstance(i["units"], list)
