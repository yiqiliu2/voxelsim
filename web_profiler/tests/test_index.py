"""index.py: singleton behavior, index size/fields on real data, root
distribution, and the summaries()/time_ms contract."""

import pytest

REQUIRED_FIELDS = ("id", "model", "mode", "batch_size", "num_cores",
                   "sa_size", "vu_size", "sram_kb", "dram_bw", "dram_name",
                   "noc_topo", "noc_bw", "impl", "core_group", "row",
                   "root", "seq_length", "log_file", "overlap_file",
                   "top_power_file", "pickle_file", "has_overlap",
                   "has_pickle", "mtime", "size")


def test_get_index_singleton():
    from web_profiler.server.index import get_index
    assert get_index() is get_index()


def test_index_total(index):
    # 3080 on the current data set; keep the bound loose but meaningful.
    assert len(index.all()) > 2500


def test_entry_required_fields(index):
    for cfg in index.all():
        for field in REQUIRED_FIELDS:
            assert field in cfg, f"{field} missing from {cfg.get('id')}"
        assert cfg["mode"] in ("decode", "prefill")
        assert cfg["id"].endswith(".log")
        assert cfg["size"] > 0 and cfg["mtime"] > 0


def test_root_distribution(index):
    roots = {c["root"] for c in index.all()}
    assert "logs" in roots
    assert any(r.startswith("logs_seq") for r in roots)
    assert {"pareto_decode", "pareto_prefill"} <= roots
    declared = {r["name"] for r in index.roots}
    assert roots <= declared
    for r in index.roots:
        assert r["seq_length"] in (1024, 2048, 4096)


def test_get_roundtrip_and_unknown(index, real_id):
    cfg = index.get(real_id)
    assert cfg is not None and cfg["id"] == real_id
    assert index.get("results/logs/no/such/result.log") is None


def test_models(index):
    models = index.models()
    assert "llama2-13" in models
    assert models == sorted(models)


def test_filter_values(index):
    fv = index.filter_values()
    for key in ("root", "model", "mode", "batch_size", "num_cores",
                    "sa_size", "sram_kb", "dram_bw", "noc_topo", "noc_bw",
                    "core_group", "impl", "row", "seq_length"):
        assert key in fv
    assert set(fv["mode"]) == {"decode", "prefill"}
    assert "logs" in fv["root"]


def test_summaries_contract(index, real_id):
    """summaries() returns raw metrics keyed by id; time_ms is NOT derived
    here (it is added by the API layer — covered in test_api_results)."""
    cfg = index.get(real_id)
    out = index.summaries([cfg])
    assert set(out) == {real_id}
    metrics = out[real_id]
    assert metrics["total_time"] > 0
    assert "time_ms" not in metrics
    # cached second call returns identical content
    assert index.summaries([cfg])[real_id] == metrics
    assert index.summary(cfg) == metrics


def test_summary_missing_file(index, real_id):
    cfg = dict(index.get(real_id))
    cfg["id"] = "synthetic/missing.log"
    cfg["log_file"] = "/nonexistent/output_cg_1_row_1.log"
    cfg["mtime"] = 1
    cfg["size"] = 1
    assert index.summary(cfg) is None
