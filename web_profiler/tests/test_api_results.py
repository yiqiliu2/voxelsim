"""api_results.py endpoints: overview/filters/metrics_meta, the /api/results
browser (pagination, filters, q, sort, with_metrics), and every per-result
sub-endpoint against a real result id."""

import json as jsonlib

import pytest

BAD_ID = "results/logs/no/such/result.log"


# ---------------------------------------------------------------------------
# browsing endpoints
# ---------------------------------------------------------------------------

def test_overview(client):
    r = client.get("/api/overview")
    assert r.status_code == 200
    d = r.get_json()
    assert d["total"] > 2500
    for key in ("by_model", "by_mode", "by_impl", "by_root", "by_cores"):
        assert isinstance(d[key], dict) and d[key]
    assert set(d["by_mode"]) == {"decode", "prefill"}
    assert isinstance(d["roots"], list) and d["roots"]
    assert 0 < len(d["latest"]) <= 12
    assert all("id" in row for row in d["latest"])


def test_filters(client):
    r = client.get("/api/filters")
    assert r.status_code == 200
    d = r.get_json()
    assert set(d["mode"]) == {"decode", "prefill"}
    assert "llama2-13" in d["model"]
    assert 2048 in d["seq_length"]


def test_metrics_meta(client):
    r = client.get("/api/metrics_meta")
    assert r.status_code == 200
    d = r.get_json()
    assert "total_time" in d
    for meta in d.values():
        assert set(meta) == {"label", "unit", "better"}
        assert meta["better"] in ("lower", "higher")


def test_results_pagination(client):
    r = client.get("/api/results?page=1&page_size=50")
    d = r.get_json()
    assert r.status_code == 200
    assert d["total"] > 2500
    assert d["page"] == 1 and d["page_size"] == 50
    assert len(d["rows"]) == 50
    ids_p1 = {row["id"] for row in d["rows"]}
    d2 = client.get("/api/results?page=2&page_size=50").get_json()
    assert len(d2["rows"]) == 50
    assert not ids_p1 & {row["id"] for row in d2["rows"]}
    # page beyond the end -> empty rows, still 200
    d3 = client.get("/api/results?page=999999&page_size=50").get_json()
    assert d3["rows"] == []


def test_results_filters(client):
    d = client.get("/api/results?mode=decode&page_size=100").get_json()
    assert d["total"] > 0
    assert all(row["mode"] == "decode" for row in d["rows"])
    # multi-value filter
    d = client.get("/api/results?noc_topo=1,2&page_size=100").get_json()
    assert all(row["noc_topo"] in (1, 2) for row in d["rows"])


def test_results_q_search(client):
    d = client.get("/api/results?q=llama2-13&page_size=20").get_json()
    assert d["total"] > 0
    assert all("llama2-13" in row["id"] for row in d["rows"])


def test_results_sort(client):
    d = client.get("/api/results?sort=dram_bw&order=desc"
                   "&page_size=100").get_json()
    bws = [row["dram_bw"] for row in d["rows"]]
    assert bws == sorted(bws, reverse=True)
    # metrics sort on a small filtered subset (fast even on a cold cache)
    d = client.get("/api/results?model=dit-xl&root=logs&mode=decode"
                   "&sort=metrics.total_time&order=asc"
                   "&with_metrics=1&page_size=50").get_json()
    rows = [row for row in d["rows"] if "metrics" in row]
    assert len(rows) > 1
    ts = [row["metrics"]["total_time"] for row in rows]
    assert ts == sorted(ts)


def test_results_with_metrics(client):
    d = client.get("/api/results?page=1&page_size=5&with_metrics=1").get_json()
    assert len(d["rows"]) == 5
    for row in d["rows"]:
        assert "metrics" in row, row["id"]
        m = row["metrics"]
        assert m["total_time"] > 0
        # time_ms derived from cycles at the default NPU frequency
        assert m["time_ms"] == pytest.approx(m["total_time"] / 1.5e6)


# ---------------------------------------------------------------------------
# per-result detail
# ---------------------------------------------------------------------------

def test_result_detail(client, real_id):
    r = client.get(f"/api/result/detail?id={real_id}")
    assert r.status_code == 200
    d = r.get_json()
    assert d["id"] == real_id
    assert d["impl_label"]
    # absolute paths must not leak into API responses
    for k in ("log_file", "overlap_file", "top_power_file", "pickle_file"):
        assert k not in d
    assert d["metrics"]["total_time"] > 0
    assert d["metrics"]["time_ms"] == pytest.approx(
        d["metrics"]["total_time"] / 1.5e6)


def test_result_detail_bad_id(client):
    r = client.get(f"/api/result/detail?id={BAD_ID}")
    assert r.status_code == 404
    assert "error" in r.get_json()


def test_result_operators(client, real_id):
    r = client.get(f"/api/result/operators?id={real_id}")
    assert r.status_code == 200
    d = r.get_json()
    assert isinstance(d["classified"], bool)
    ops = d["operators"]
    assert len(ops) > 0
    assert {"op_id", "dur_comp", "dur_ld", "start_ld", "start_finish"} \
        <= set(ops[0])


def test_result_operators_classified(client, classifiable_id):
    d = client.get(f"/api/result/operators?id={classifiable_id}").get_json()
    assert d["classified"] is True
    assert all("category" in op and "op_type" in op
               for op in d["operators"])


def test_result_op_energy(client, real_id):
    r = client.get(f"/api/result/op_energy?id={real_id}")
    assert r.status_code == 200
    ops = r.get_json()["operators"]
    assert len(ops) > 0
    for op in ops:
        assert op["energy_total_mj"] >= 0.0
        assert op["t_finish"] >= op["t_start"]


def test_result_top_power(client, real_id):
    r = client.get(f"/api/result/top_power?id={real_id}")
    assert r.status_code == 200
    intervals = r.get_json()["intervals"]
    assert len(intervals) > 0
    powers = [i["power_w"] for i in intervals]
    assert powers == sorted(powers, reverse=True)
    assert all(i["t_end"] >= i["t_start"] for i in intervals)


def test_result_overlap(client, real_id):
    r = client.get(f"/api/result/overlap?id={real_id}")
    assert r.status_code == 200
    d = r.get_json()
    assert d["count"] == len(d["intervals"]) > 0
    for i in d["intervals"]:
        assert i["t_end"] >= i["t_start"]
        assert isinstance(i["units"], list)
        assert i["power_w"] >= 0.0


def test_result_reproduce(client, real_id, index):
    r = client.get(f"/api/result/reproduce?id={real_id}")
    assert r.status_code == 200
    d = r.get_json()
    assert "icbm_launch.py" in d["command"]
    cfg = index.get(real_id)
    assert cfg["model"] in d["argv"]
    if cfg["mode"] == "prefill":
        assert "--prefill" in d["argv"]
    else:
        assert "--prefill" not in d["argv"]


def test_result_op_breakdown_classifiable(client, classifiable_id):
    r = client.get(f"/api/result/op_breakdown?id={classifiable_id}")
    assert r.status_code == 200
    d = r.get_json()
    cats = d["categories"]
    assert cats
    for c in cats:
        assert {"key", "label", "cycles", "op_count", "fraction",
                "time_ms"} <= set(c)
        assert c["cycles"] >= 0.0
    assert sum(c["fraction"] for c in cats) == pytest.approx(1.0)


def test_result_op_breakdown_maybe_404(client, unclassifiable_id):
    """Compile cache unavailable/mismatched -> 404 with a legal error body."""
    r = client.get(f"/api/result/op_breakdown?id={unclassifiable_id}")
    assert r.status_code == 404
    assert "error" in r.get_json()


def test_result_export_csv(client, real_id):
    r = client.get(f"/api/result/export.csv?id={real_id}")
    assert r.status_code == 200
    assert r.content_type.startswith("text/csv")
    assert "attachment" in r.headers.get("Content-Disposition", "")
    first_line = r.get_data(as_text=True).splitlines()[0]
    assert "op_id" in first_line
    r = client.get(f"/api/result/export.csv?id={real_id}&kind=summary")
    assert r.status_code == 200
    assert "total_time" in r.get_data(as_text=True)
    r = client.get(f"/api/result/export.csv?id={real_id}&kind=bogus")
    assert r.status_code == 400


def test_result_trace_json(client, real_id):
    r = client.get(f"/api/result/trace.json?id={real_id}")
    assert r.status_code == 200
    data = jsonlib.loads(r.get_data(as_text=True))
    events = data["traceEvents"]
    complete = [e for e in events if e.get("ph") == "X"]
    assert complete, "expected complete (X) events"
    for e in complete:
        assert e["dur"] > 0 and e["ts"] >= 0
        assert {"name", "cat", "pid", "tid"} <= set(e)
    assert any(e.get("name") == "thread_name" for e in events)


@pytest.mark.parametrize("sub", ["detail", "operators", "op_energy",
                                 "top_power", "overlap", "reproduce",
                                 "op_breakdown", "export.csv", "trace.json"])
def test_result_subendpoints_bad_id(client, sub):
    r = client.get(f"/api/result/{sub}?id={BAD_ID}")
    assert r.status_code == 404
    assert "error" in r.get_json()


def test_index_rebuild(client, index):
    """POST /api/index/rebuild rescans all roots and returns fresh stats."""
    r = client.post("/api/index/rebuild")
    assert r.status_code == 200
    d = r.get_json()
    assert {"total", "roots", "built_at"} <= set(d)
    assert d["total"] == len(index.all()) > 2500
    assert isinstance(d["roots"], list) and d["roots"]
    assert "logs" in d["roots"]
    assert {"pareto_decode", "pareto_prefill"} <= set(d["roots"])
    assert d["built_at"] > 0
    # the rebuilt index still serves real lookups
    assert len(index.all()) == d["total"]
    assert index.get(index.all()[0]["id"]) is not None
