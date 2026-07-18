"""api_jobs.py + jobs.py: submission validation, queue control API, and the
JobManager lifecycle.

Two safety properties hold throughout:
- API tests pin api_jobs.get_manager to a JobManager rooted at tmp_path with
  zero worker slots, so nothing ever launches and runtime/jobs is untouched.
- Lifecycle tests submit only /bin/echo, /bin/sh and /bin/sleep — never a
  real simulation.
"""

import time

import pytest

from web_profiler.server import api_jobs as api_jobs_mod
from web_profiler.server import jobs as jobs_mod


# ---------------------------------------------------------------------------
# fixtures / helpers
# ---------------------------------------------------------------------------

@pytest.fixture()
def queued_manager(tmp_path, monkeypatch):
    """JobManager with 0 worker slots pinned into the jobs API.

    Jobs accepted through the API stay queued forever; no process is ever
    spawned and all state lives under tmp_path.
    """
    monkeypatch.setattr(jobs_mod, "max_workers", lambda: 0)
    mgr = jobs_mod.JobManager(jobs_dir=tmp_path / "jobs")
    monkeypatch.setattr(api_jobs_mod, "get_manager", lambda: mgr)
    return mgr


@pytest.fixture()
def running_manager(tmp_path, monkeypatch):
    """A real (2-slot) JobManager on tmp_path for echo/sleep lifecycle tests."""
    monkeypatch.setattr(jobs_mod, "max_workers", lambda: 2)
    mgr = jobs_mod.JobManager(jobs_dir=tmp_path / "run")
    yield mgr
    for j in mgr.list_jobs():
        if j["status"] in ("queued", "running"):
            try:
                mgr.cancel(j["id"])
            except ValueError:
                pass


def _wait_status(mgr, job_id, wanted, timeout=20.0):
    if isinstance(wanted, str):
        wanted = (wanted,)
    deadline = time.time() + timeout
    while time.time() < deadline:
        job = mgr.get(job_id)
        if job["status"] in wanted:
            return job
        time.sleep(0.1)
    raise AssertionError(
        f"job {job_id} stuck at {mgr.get(job_id)['status']!r}, "
        f"wanted {wanted}")


def _submit(client, jtype, params):
    return client.post("/api/jobs", json={"type": jtype, "params": params})


# ---------------------------------------------------------------------------
# defs + sim submission validation
# ---------------------------------------------------------------------------

def test_job_defs(client):
    r = client.get("/api/jobs/defs")
    assert r.status_code == 200
    d = r.get_json()
    assert {"models", "tiers", "defaults", "impls", "modes",
            "max_workers", "notes"} <= set(d)
    assert d["modes"] == ["decode", "prefill"]
    assert "llama2-13" in d["models"]
    assert d["defaults"]["model"] == "llama2-13"
    assert d["max_workers"] >= 1


def test_submit_sim_ok(client, queued_manager):
    r = _submit(client, "sim", {"model": "llama2-13", "mode": "decode"})
    assert r.status_code == 201
    job = r.get_json()
    assert job["type"] == "sim" and job["status"] == "queued"
    assert len(job["units"]) == 1
    argv = job["units"][0]["argv"]
    assert "icbm_launch.py" in argv
    assert "--prefill" not in argv
    assert job["units"][0]["expected_output"].endswith(".log")
    # nothing was launched: the unit never left the queue
    assert job["units"][0]["status"] == "queued"
    assert queued_manager.get(job["id"])["status"] == "queued"


def test_submit_sim_prefill_forces_batch_size_1(client, queued_manager):
    r = _submit(client, "sim", {"mode": "prefill", "batch_size": 32})
    assert r.status_code == 201
    job = r.get_json()
    assert job["params"]["batch_size"] == 1
    assert "--prefill" in job["units"][0]["argv"]


@pytest.mark.parametrize("params", [
    {"model": "not-a-model"},
    {"mode": "sideways"},
    {"noc_topo": 7},
    {"noc_topo": 0},
    {"bogus_param": 1},
    {"sa_size": [16, 32]},          # lists are batch-only
    {"num_cores": 0},
    {"num_cores": True},
])
def test_submit_sim_bad_params_400(client, queued_manager, params):
    r = _submit(client, "sim", params)
    assert r.status_code == 400
    assert "error" in r.get_json()


def test_submit_bad_body_400(client, queued_manager):
    assert client.post("/api/jobs", json=[1, 2, 3]).status_code == 400
    assert client.post("/api/jobs",
                       json={"type": "sim", "params": [1]}).status_code == 400
    assert client.post("/api/jobs",
                       json={"type": "warp", "params": {}}).status_code == 400


# ---------------------------------------------------------------------------
# batch submission
# ---------------------------------------------------------------------------

def test_submit_batch_cartesian(client, queued_manager):
    r = _submit(client, "batch", {"sa_size": [16, 32], "noc_bw": [4, 8]})
    assert r.status_code == 201
    job = r.get_json()
    assert job["type"] == "batch"
    assert len(job["units"]) == 4
    # batch responses strip unit argv
    for unit in job["units"]:
        assert "argv" not in unit
        assert unit["status"] == "queued"
    assert job["progress"]["total"] == 4
    assert job["progress"]["queued"] == 4


def test_submit_batch_max_boundary(client, queued_manager):
    params = {"sa_size": [16, 32, 64, 128], "noc_bw": [4, 8, 16, 32],
              "core_group": [1, 2, 4, 8]}  # 4*4*4 = 64 = MAX_BATCH_UNITS
    r = _submit(client, "batch", params)
    assert r.status_code == 201
    assert len(r.get_json()["units"]) == 64


def test_submit_batch_too_many_400(client, queued_manager):
    params = {"sa_size": [16, 32, 64, 128], "noc_bw": [4, 8, 16, 32, 64],
              "core_group": [1, 2, 4, 8]}  # 4*5*4 = 80 > 64
    r = _submit(client, "batch", params)
    assert r.status_code == 400
    assert queued_manager.list_jobs() == []  # rejected before submission


# ---------------------------------------------------------------------------
# dse / thermal submission
# ---------------------------------------------------------------------------

def test_submit_dse_ok(client, queued_manager):
    r = _submit(client, "dse", {"mode": "decode", "models": ["llama2-13"],
                                "num_sweeps": 2})
    assert r.status_code == 201
    job = r.get_json()
    assert job["type"] == "dse" and job["status"] == "queued"
    argv = job["units"][0]["argv"]
    assert "dse_pareto.py" in argv


@pytest.mark.parametrize("params", [
    {"mode": "sideways"},
    {"models": ["not-a-model"]},
    {"models": []},
    {"num_sweeps": 0},
    {"num_sweeps": 101},
    {"bogus": 1},
])
def test_submit_dse_bad_params_400(client, queued_manager, params):
    assert _submit(client, "dse", params).status_code == 400


def test_submit_thermal_ok(client, queued_manager):
    r = _submit(client, "thermal", {"models": "llama2-13", "modes": "decode",
                                    "backends": "simple"})
    assert r.status_code == 201
    job = r.get_json()
    assert job["type"] == "thermal"
    argv = job["units"][0]["argv"]
    assert "tsim_thermal.cli" in argv


@pytest.mark.parametrize("params", [
    {"backends": "not-a-backend"},
    {"models": "not-a-model"},
    {"results_dir": "../escape"},
    {"out_dir": "bad dir!"},
    {"dram_layers": 0},
    {"bogus": 1},
])
def test_submit_thermal_bad_params_400(client, queued_manager, params):
    assert _submit(client, "thermal", params).status_code == 400


# ---------------------------------------------------------------------------
# queue queries / cancel / delete (nothing ever runs)
# ---------------------------------------------------------------------------

def test_job_list_and_detail(client, queued_manager):
    job = _submit(client, "sim", {}).get_json()
    r = client.get("/api/jobs")
    assert r.status_code == 200
    jobs = r.get_json()["jobs"]
    assert any(j["id"] == job["id"] for j in jobs)
    for j in jobs:
        assert "progress" in j
        for unit in j["units"]:
            assert "argv" not in unit  # list responses strip argv
    r = client.get(f"/api/jobs/{job['id']}")
    assert r.status_code == 200
    d = r.get_json()
    assert d["progress"]["total"] == 1
    assert d["units"][0]["argv"]  # detail keeps argv
    assert client.get("/api/jobs/deadbeef").status_code == 404


def test_job_log_window(client, queued_manager):
    job = _submit(client, "sim", {}).get_json()
    r = client.get(f"/api/jobs/{job['id']}/log")
    assert r.status_code == 200
    d = r.get_json()
    assert {"total_size", "offset", "length", "eof", "text"} <= set(d)
    assert d["total_size"] == 0 and d["eof"] is True
    assert client.get("/api/jobs/deadbeef/log").status_code == 404


def test_cancel_then_cancel_conflict(client, queued_manager):
    job = _submit(client, "sim", {}).get_json()
    r = client.post(f"/api/jobs/{job['id']}/cancel")
    assert r.status_code == 200
    assert r.get_json()["status"] == "cancelled"
    # terminal jobs cannot be cancelled again
    r = client.post(f"/api/jobs/{job['id']}/cancel")
    assert r.status_code == 409
    assert client.post("/api/jobs/deadbeef/cancel").status_code == 404


def test_delete_lifecycle(client, queued_manager):
    job = _submit(client, "sim", {}).get_json()
    # active (queued) jobs cannot be deleted
    assert client.post(f"/api/jobs/{job['id']}/delete").status_code == 409
    client.post(f"/api/jobs/{job['id']}/cancel")
    r = client.post(f"/api/jobs/{job['id']}/delete")
    assert r.status_code == 200
    assert r.get_json()["deleted"] == job["id"]
    assert client.get(f"/api/jobs/{job['id']}").status_code == 404
    assert client.post(f"/api/jobs/{job['id']}/delete").status_code == 404
    # the persisted files are gone too
    assert not list(queued_manager._jobs_dir.glob(f"{job['id']}.*"))


# ---------------------------------------------------------------------------
# JobManager lifecycle with trivial real processes (echo / sh / sleep)
# ---------------------------------------------------------------------------

def test_echo_job_completes(running_manager):
    job = running_manager._submit_argv(["/bin/echo", "voxelsim-echo-ok"],
                                       label="echo")
    done = _wait_status(running_manager, job["id"], "done")
    assert done["units"][0]["returncode"] == 0
    assert done["ended_at"] is not None
    win = running_manager.log_window(job["id"])
    assert "voxelsim-echo-ok" in win["text"]


def test_failing_job_marked_failed(running_manager):
    job = running_manager._submit_argv(["/bin/sh", "-c", "exit 3"],
                                       label="fail")
    done = _wait_status(running_manager, job["id"], "failed")
    assert done["units"][0]["returncode"] == 3


def test_cancel_running_sleep(running_manager):
    job = running_manager._submit_argv(["/bin/sleep", "30"], label="sleep")
    _wait_status(running_manager, job["id"], "running")
    cancelled = running_manager.cancel(job["id"])
    assert cancelled["status"] == "cancelled"
    assert cancelled["units"][0]["status"] == "cancelled"


def test_persistence_reload_keeps_terminal_state(tmp_path, monkeypatch):
    monkeypatch.setattr(jobs_mod, "max_workers", lambda: 2)
    d = tmp_path / "persist"
    mgr1 = jobs_mod.JobManager(jobs_dir=d)
    job = mgr1._submit_argv(["/bin/echo", "hi"], label="echo")
    _wait_status(mgr1, job["id"], "done")
    assert (d / f"{job['id']}.json").is_file()
    assert (d / f"{job['id']}.log").is_file()
    mgr2 = jobs_mod.JobManager(jobs_dir=d)  # reloads persisted state
    assert mgr2.get(job["id"])["status"] == "done"


def test_active_job_reloaded_as_interrupted(tmp_path, monkeypatch):
    monkeypatch.setattr(jobs_mod, "max_workers", lambda: 2)
    d = tmp_path / "interrupt"
    mgr1 = jobs_mod.JobManager(jobs_dir=d)
    job = mgr1._submit_argv(["/bin/sleep", "60"], label="sleep")
    _wait_status(mgr1, job["id"], "running")
    mgr2 = jobs_mod.JobManager(jobs_dir=d)  # simulates a server restart
    try:
        reloaded = mgr2.get(job["id"])
        assert reloaded["status"] == "interrupted"
        assert reloaded["units"][0]["status"] == "cancelled"
    finally:
        try:
            mgr1.cancel(job["id"])  # kill the real sleep process
        except ValueError:
            pass
