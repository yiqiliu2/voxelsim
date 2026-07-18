"""api_system.py endpoints: host status, RAM timeline, project log listing
and the guarded log viewer (path escape / suffix / .env refusal)."""

import pytest


def test_system_host(client):
    r = client.get("/api/system/host")
    assert r.status_code == 200
    d = r.get_json()
    assert d["cpu_count"] > 0
    assert d["mem"]["total_mb"] > 0
    assert 0 <= d["mem"]["used_mb"] <= d["mem"]["total_mb"]
    assert d["disk"]["total_gb"] > 0
    assert d["disk"]["free_gb"] >= 0
    assert d["results"]["roots"], "expected at least one result root"
    for root in d["results"]["roots"]:
        assert {"name", "path"} <= set(root)


def test_ram_timeline(client):
    r = client.get("/api/system/ram_timeline")
    assert r.status_code == 200
    d = r.get_json()
    assert d["source"] in ("dram_usage.log", "dram_monitor.log", None)
    if d["source"] is None:
        pytest.skip("no dram usage/monitor log in the project root")
    assert len(d["t"]) > 0
    n = len(d["t"])
    assert len(d["used_mb"]) == len(d["available_mb"]) == \
        len(d["total_mb"]) == n
    assert all(u >= 0 for u in d["used_mb"])


def test_ram_timeline_max_points(client):
    d = client.get("/api/system/ram_timeline?max_points=50").get_json()
    # decimation keeps at most ~max_points (+1 for the pinned last sample)
    assert len(d["t"]) <= 51


def test_logs_list(client):
    r = client.get("/api/logs/list")
    assert r.status_code == 200
    logs = r.get_json()["logs"]
    assert logs
    for entry in logs:
        assert {"category", "name", "path", "size", "mtime"} <= set(entry)
        assert entry["size"] >= 0
        assert entry["category"] in ("test_logs", "root", "dse", "thermal")


def test_logs_view_ok(client):
    logs = client.get("/api/logs/list").get_json()["logs"]
    path = logs[0]["path"]
    r = client.get(f"/api/logs/view?path={path}&length=128")
    assert r.status_code == 200
    d = r.get_json()
    assert d["path"] == path
    assert d["total_size"] > 0
    assert d["offset"] == 0
    assert 0 < d["length"] <= 128
    assert isinstance(d["text"], str)
    # offset beyond EOF -> empty window but still 200
    d2 = client.get(f"/api/logs/view?path={path}"
                    f"&offset={d['total_size'] + 1000}").get_json()
    assert d2["length"] == 0 and d2["eof"] is True
    # tail window ends at EOF
    d3 = client.get(f"/api/logs/view?path={path}&length=64&tail=1") \
        .get_json()
    assert d3["offset"] + d3["length"] == d3["total_size"]
    assert d3["eof"] is True


def test_logs_view_traversal_403(client):
    r = client.get("/api/logs/view?path=../../etc/passwd")
    assert r.status_code == 403


def test_logs_view_disallowed_suffix_403(client):
    # .pdf is not in the allowed suffix set (checked before existence)
    r = client.get("/api/logs/view?path=test_logs/some_file.pdf")
    assert r.status_code == 403


def test_logs_view_env_refused_403(client):
    r = client.get("/api/logs/view?path=.env")
    assert r.status_code == 403
    r = client.get("/api/logs/view?path=config/.env.secret")
    assert r.status_code == 403


def test_logs_view_missing_404(client):
    r = client.get("/api/logs/view?path=no_such_log_file.log")
    assert r.status_code == 404


def test_logs_view_no_path_400(client):
    assert client.get("/api/logs/view").status_code == 400
