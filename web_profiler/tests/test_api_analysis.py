"""api_analysis.py endpoints: /api/sweep, /api/compare, /api/pareto and the
paper-figure builders /api/paper/fig10..fig20 (404 tolerated per figure but
recorded; the unknown-figure contract is strict)."""

import pytest

PAPER_FIGS = ["fig10", "fig11", "fig12", "fig13", "fig15", "fig17",
              "fig18", "fig19", "fig20"]

_PANEL_CONTENT_KEYS = ("series", "components", "categories")


# ---------------------------------------------------------------------------
# sweep
# ---------------------------------------------------------------------------

def test_sweep_basic(client):
    r = client.get("/api/sweep?x=dram_bw&metric=total_time&mode=decode")
    assert r.status_code == 200
    d = r.get_json()
    assert d["x_key"] == "dram_bw" and d["metric"] == "total_time"
    assert d["count"] > 0
    assert len(d["x"]) > 1
    assert d["series"]
    for series in d["series"]:
        assert series["name"]
        for p in series["points"]:
            assert {"x", "y", "ymin", "ymax", "n", "ids"} <= set(p)
            assert p["ymin"] <= p["y"] <= p["ymax"]
            assert p["n"] == len(p["ids"]) >= 1
    assert isinstance(d["x_log"], bool)


def test_sweep_grouped(client):
    d = client.get("/api/sweep?x=num_cores&metric=total_energy_mj"
                   "&group=model&mode=decode&root=logs").get_json()
    assert d["group_key"] == "model"
    assert len(d["series"]) >= 1


def test_sweep_bad_x(client):
    r = client.get("/api/sweep?x=not_a_dimension")
    assert r.status_code == 400
    assert "error" in r.get_json()


def test_sweep_bad_group(client):
    r = client.get("/api/sweep?x=dram_bw&group=not_a_dimension")
    assert r.status_code == 400


def test_sweep_time_ms_derived_metric(client):
    d = client.get("/api/sweep?x=dram_bw&metric=time_ms&mode=decode"
                   "&root=logs").get_json()
    assert d["series"], "time_ms should derive from total_time"


# ---------------------------------------------------------------------------
# compare
# ---------------------------------------------------------------------------

def test_compare_two_ids(client, index):
    ids = [c["id"] for c in index.all()[:2]]
    r = client.get("/api/compare?ids=" + ",".join(ids))
    assert r.status_code == 200
    items = r.get_json()["items"]
    assert len(items) == 2
    for it in items:
        assert {"id", "config", "metrics"} <= set(it)
        assert it["metrics"]["time_ms"] > 0


def test_compare_one_id_400(client, real_id):
    assert client.get(f"/api/compare?ids={real_id}").status_code == 400


def test_compare_too_many_ids_400(client, index):
    ids = [c["id"] for c in index.all()[:13]]
    assert client.get("/api/compare?ids=" + ",".join(ids)).status_code == 400


def test_compare_all_fake_ids_404(client):
    r = client.get("/api/compare?ids=fake/a.log,fake/b.log")
    assert r.status_code == 404


def test_compare_mixed_ids_keeps_known(client, real_id):
    r = client.get(f"/api/compare?ids={real_id},fake/b.log")
    assert r.status_code == 200
    assert len(r.get_json()["items"]) == 1


# ---------------------------------------------------------------------------
# pareto
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("mode", ["decode", "prefill"])
def test_pareto(client, mode):
    r = client.get(f"/api/pareto?mode={mode}")
    assert r.status_code == 200
    d = r.get_json()
    assert d["all_points"], "expected DSE points on the real data set"
    assert d["pareto_front"]
    assert d["num_points"] == len(d["all_points"])
    assert "fixed_params" in d


def test_pareto_bad_mode(client):
    assert client.get("/api/pareto?mode=bogus").status_code == 400


# ---------------------------------------------------------------------------
# paper figures
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("fig", PAPER_FIGS)
def test_paper_fig(client, fig):
    r = client.get(f"/api/paper/{fig}")
    if r.status_code == 404:
        # Allowed when a figure's slice of the data set is empty; recorded.
        print(f"\n[note] /api/paper/{fig} -> 404 "
              f"({r.get_json().get('error')})")
        return
    assert r.status_code == 200
    d = r.get_json()
    assert d["fig"] == fig
    assert isinstance(d["title"], str) and d["title"]
    assert isinstance(d["panels"], list) and d["panels"]
    for panel in d["panels"]:
        assert panel.get("title")
        assert any(k in panel for k in _PANEL_CONTENT_KEYS), \
            f"panel without plottable content in {fig}"


def test_paper_fig_with_mode_param(client):
    for fig in ("fig11", "fig15", "fig18", "fig19"):
        r = client.get(f"/api/paper/{fig}?mode=prefill")
        assert r.status_code in (200, 404)
        if r.status_code == 200:
            d = r.get_json()
            assert d["fig"] == fig and d["panels"]


def test_paper_fig99_unknown(client):
    r = client.get("/api/paper/fig99")
    assert r.status_code == 404
    d = r.get_json()
    assert "error" in d
    assert "fig10" in d["known"]
