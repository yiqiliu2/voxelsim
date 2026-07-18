"""classify.py: operator category state machine (synthetic sequences),
busy-interval time attribution, and op_breakdown on a real result whose
compile cache lines up (plus the None contract when it does not)."""

import pytest

from web_profiler.server import classify, parsers


# ---------------------------------------------------------------------------
# classify_ops state machine (synthetic)
# ---------------------------------------------------------------------------

def test_classify_ops_attn_then_ffn():
    types = ["Dot", "SoftmaxBasic", "Dot", "Sum", "Dot", "Gelu", "Dot",
             "Sum"]
    cats = classify.classify_ops(types, flash=False)
    assert len(cats) == len(types)
    assert all(c in classify.CAT_LABEL for c in cats)
    assert cats[0] == "attn_qkv"        # Dot ahead of attention defs
    assert cats[1] == "attn_score"      # SoftmaxBasic
    assert cats[2] == "attn_av"         # Dot after softmax
    assert cats[3] == "other"           # Sum resets the phase
    assert cats[5] == "ffn_up"          # FFN activation
    assert cats[6] == "ffn_down"        # Dot inside the FFN phase


def test_classify_ops_flash_mode():
    types = ["Dot", "SoftmaxBasic", "BatchMatMul", "Sum"]
    cats = classify.classify_ops(types, flash=True)
    assert cats == ["attn_flash", "attn_flash", "attn_flash", "other"]


def test_classify_ops_no_attention():
    types = ["Dot", "Sum", "Dot"]
    cats = classify.classify_ops(types, flash=False)
    assert cats == ["other", "other", "other"]


# ---------------------------------------------------------------------------
# attribute_op_times
# ---------------------------------------------------------------------------

def test_attribute_op_times_overlap_split():
    # [0,10) and [5,15): 5 cycles solo each + 5 shared -> 7.5 apiece
    assert classify.attribute_op_times([(0, 10), (5, 15)]) == [7.5, 7.5]


def test_attribute_op_times_degenerate_span():
    # finish <= start is clamped to a 1-cycle span
    assert classify.attribute_op_times([(5, 5)]) == [1.0]
    assert classify.attribute_op_times([]) == []


# ---------------------------------------------------------------------------
# op_breakdown on real data
# ---------------------------------------------------------------------------

def test_op_breakdown_real_result(index, classifiable_id):
    cfg = index.get(classifiable_id)
    ops = parsers.parse_operators_file(cfg["log_file"])
    bd = classify.op_breakdown(cfg, ops)
    assert bd is not None
    cats = bd["categories"]
    assert cats, "expected at least one category"
    for c in cats:
        assert c["key"] in classify.CAT_LABEL
        assert c["label"] == classify.CAT_LABEL[c["key"]]
        assert c["cycles"] >= 0.0
        assert c["op_count"] >= 1
    assert sum(c["cycles"] for c in cats) > 0.0
    assert len(bd["op_categories"]) == len(ops)
    assert len(bd["op_types"]) == len(ops)
    assert len(bd["op_times"]) == len(ops)
    assert all(t >= 0.0 for t in bd["op_times"])
    assert bd["source"].endswith("all_configs_dict.json")


def test_op_breakdown_unavailable_returns_none(index, unclassifiable_id):
    cfg = index.get(unclassifiable_id)
    ops = parsers.parse_operators_file(cfg["log_file"])
    assert ops, "fixture must pick a log with operators"
    assert classify.op_breakdown(cfg, ops) is None


def test_find_configs_dict_prefers_mode_batch(index, classifiable_id):
    cfg = index.get(classifiable_id)
    path = classify.find_configs_dict(cfg)
    assert path is not None and path.endswith("all_configs_dict.json")
