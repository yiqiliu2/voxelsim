"""api_catalog.py endpoints: hw_configs (list/detail/create with strict
validation), model library + op browser, NoC distance tables, and
llmservingsim profiles (including the path-escape guard)."""

import uuid

import pytest

from web_profiler.server import commands
from web_profiler.server.config import HW_CONFIG_DIR, PROJECT_ROOT


# ---------------------------------------------------------------------------
# hardware configs
# ---------------------------------------------------------------------------

def test_hwconfigs_list(client):
    r = client.get("/api/hwconfigs")
    assert r.status_code == 200
    configs = r.get_json()["configs"]
    # 70 at suite-writing time; web-launched runs may legitimately add more.
    assert len(configs) >= 70
    names = [c["name"] for c in configs]
    assert names == sorted(names)
    for c in configs:
        assert "parse_error" not in c, c["name"]
        assert {"name", "size", "mtime", "sa", "dram_bw", "noc_topo",
                "noc_bw", "row", "freq_mhz", "trp"} <= set(c)


def test_hwconfig_detail(client):
    r = client.get("/api/hwconfigs/sa_128_vu_128_drambw_12288_"
                   "noc_1_16_trcd_14_trp_14")
    assert r.status_code == 200
    d = r.get_json()
    assert set(d) == {"compute", "noc", "dram"}
    assert d["compute"]["mm_pad_shape"] == [128, 128, 128]
    assert d["dram"]["bandwidth_GBps"] == 12288


def test_hwconfig_detail_unknown_404(client):
    r = client.get("/api/hwconfigs/definitely_not_a_config")
    assert r.status_code == 404


def test_hwconfig_detail_invalid_name_400(client):
    r = client.get("/api/hwconfigs/bad%20name%21")
    assert r.status_code == 400


def test_hwconfig_create_conflict_badschema_and_cleanup(client):
    """201 on first create, 409 on duplicate, 400 on bad schema/name.
    The created file uses a random name and is always removed afterwards."""
    name = f"pytest_tmp_{uuid.uuid4().hex[:12]}"
    path = HW_CONFIG_DIR / f"{name}.json"
    cfg = commands.make_hw_config_dict(sa=16, noc_topo=1, noc_bw=8,
                                       dram_bw=1234, row=512, trp=14)
    try:
        r = client.post("/api/hwconfigs",
                        json={"name": name, "config": cfg})
        assert r.status_code == 201
        assert r.get_json()["created"] == name
        assert path.is_file()
        # detail round-trip returns exactly what was posted
        d = client.get(f"/api/hwconfigs/{name}").get_json()
        assert d == cfg
        # duplicate -> 409, file not overwritten
        r = client.post("/api/hwconfigs",
                        json={"name": name, "config": cfg})
        assert r.status_code == 409
        # bad schema -> 400 (missing dram section)
        r = client.post("/api/hwconfigs",
                        json={"name": name + "_bad",
                              "config": {"compute": cfg["compute"]}})
        assert r.status_code == 400
        assert not (HW_CONFIG_DIR / f"{name}_bad.json").exists()
        # invalid name -> 400
        r = client.post("/api/hwconfigs",
                        json={"name": "bad name!", "config": cfg})
        assert r.status_code == 400
        # non-dict body -> 400
        r = client.post("/api/hwconfigs", json=[1, 2, 3])
        assert r.status_code == 400
    finally:
        path.unlink(missing_ok=True)
    assert not path.exists()


def test_hwconfig_create_never_touches_existing(client):
    """POSTing a name that already exists must not modify the file."""
    name = client.get("/api/hwconfigs").get_json()["configs"][0]["name"]
    existing = HW_CONFIG_DIR / f"{name}.json"
    before = existing.read_bytes()
    r = client.post("/api/hwconfigs",
                    json={"name": name, "config": {}})
    assert r.status_code in (400, 409)
    assert existing.read_bytes() == before


# ---------------------------------------------------------------------------
# models
# ---------------------------------------------------------------------------

def test_models_list(client):
    r = client.get("/api/models")
    assert r.status_code == 200
    models = r.get_json()["models"]
    assert len(models) == 24
    by_name = {m["name"]: m for m in models}
    assert "llama2-13" in by_name
    m = by_name["llama2-13"]
    assert m["has_parsed"] and m["op_count"] > 0
    assert m["op_types"]


def test_model_ops_browse(client):
    r = client.get("/api/models/llama2-13/ops?page=1&page_size=50")
    assert r.status_code == 200
    d = r.get_json()
    assert d["total"] > 50
    assert len(d["rows"]) == 50
    for row in d["rows"]:
        assert {"id", "type", "expr", "shapes", "inputs"} <= set(row)
    # page 2 returns different ops
    d2 = client.get("/api/models/llama2-13/ops?page=2&page_size=50") \
        .get_json()
    assert d2["page"] == 2
    assert d2["rows"][0]["id"] != d["rows"][0]["id"]


def test_model_ops_filters(client):
    d = client.get("/api/models/llama2-13/ops?page_size=500").get_json()
    a_type = d["rows"][0]["type"]
    dt = client.get(f"/api/models/llama2-13/ops?type={a_type}").get_json()
    assert 0 < dt["total"] < d["total"] or dt["total"] == d["total"]
    assert all(row["type"] == a_type for row in dt["rows"])
    # q substring search narrows results
    needle = d["rows"][0]["expr"][:6].lower()
    dq = client.get(f"/api/models/llama2-13/ops?q={needle}").get_json()
    assert 0 < dq["total"] <= d["total"]


def test_model_ops_unparsed_404(client):
    models = client.get("/api/models").get_json()["models"]
    unparsed = [m["name"] for m in models if not m["has_parsed"]]
    name = unparsed[0] if unparsed else "definitely_not_a_model"
    r = client.get(f"/api/models/{name}/ops")
    assert r.status_code == 404


def test_model_ops_invalid_name_400(client):
    r = client.get("/api/models/bad%20name/ops")
    assert r.status_code == 400


# ---------------------------------------------------------------------------
# NoC distance tables
# ---------------------------------------------------------------------------

def test_noc_tables_list(client):
    r = client.get("/api/noc/tables")
    assert r.status_code == 200
    tables = r.get_json()["tables"]
    assert len(tables) == 12
    assert {"topo": "MESH", "n": 256} in tables
    assert {"topo": "MESH", "n": 1024} in tables
    assert {"topo": "ALL", "n": 256} in tables
    assert {"topo": "TORUS", "n": 256} in tables


def test_noc_table_mesh_256(client):
    r = client.get("/api/noc/tables/mesh/256")
    assert r.status_code == 200
    d = r.get_json()
    assert d["topo"] == "MESH" and d["n"] == 256
    assert d["stats"]["avg_hops"] > 0
    assert d["stats"]["max_hops"] > 0
    assert "downsampled" not in d
    assert len(d["matrix"]) == 256
    assert len(d["matrix"][0]) == 256


def test_noc_table_mesh_1024_downsampled(client):
    r = client.get("/api/noc/tables/mesh/1024")
    assert r.status_code == 200
    d = r.get_json()
    assert d["downsampled"] is True
    assert d["block"] == 4
    assert len(d["matrix"]) == 256
    assert d["stats"]["avg_hops"] > 0


def test_noc_table_bad_topo_400(client):
    assert client.get("/api/noc/tables/hyperx/256").status_code == 400


def test_noc_table_missing_404(client):
    assert client.get("/api/noc/tables/mesh/999").status_code == 404


# ---------------------------------------------------------------------------
# llmservingsim profiles
# ---------------------------------------------------------------------------

def test_serving_profiles(client):
    r = client.get("/api/serving/profiles")
    assert r.status_code == 200
    profiles = r.get_json()["profiles"]
    assert profiles, "expected at least one serving profile"
    for p in profiles:
        assert {"path", "model", "tp", "files", "meta"} <= set(p)
        assert p["files"]


def test_serving_profile_ok(client):
    profiles = client.get("/api/serving/profiles").get_json()["profiles"]
    path = profiles[0]["path"]
    r = client.get(f"/api/serving/profile?path={path}")
    assert r.status_code == 200
    d = r.get_json()
    assert d["path"] == path
    assert d["files"], "expected at least one CSV"
    for csv in d["files"].values():
        assert csv["columns"]
        assert csv["rows"]


def test_serving_profile_path_escape_403(client):
    r = client.get("/api/serving/profile?path=../../etc")
    assert r.status_code == 403
    # a real project dir outside llmservingsim is refused too
    r = client.get("/api/serving/profile?path=results")
    assert r.status_code == 403


def test_serving_profile_missing_404(client):
    r = client.get("/api/serving/profile?path=llmservingsim/no_such_dir")
    assert r.status_code == 404


def test_serving_profile_no_path_400(client):
    assert client.get("/api/serving/profile").status_code == 400
