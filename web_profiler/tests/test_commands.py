"""commands.py: sim command construction (decode/prefill), expected output
path shape, cartesian spec expansion, hw_config schema, and ensure_hw_config
idempotency — with HW_CONFIG_DIR redirected to tmp_path so the real
hw_config/ directory is never touched."""

import json

import pytest

from web_profiler.server import commands


def _flag_value(argv, flag):
    return argv[argv.index(flag) + 1]


# ---------------------------------------------------------------------------
# build_sim_command
# ---------------------------------------------------------------------------

def test_build_sim_command_decode():
    cmd = commands.build_sim_command({"model": "llama2-13", "mode": "decode"})
    argv = cmd["argv"]
    assert argv[1] == "icbm_launch.py"
    assert "llama2-13" in argv
    assert "--prefill" not in argv
    assert _flag_value(argv, "--split_factor") == \
        str(commands.SPLIT_FACTOR_DECODE)
    assert _flag_value(argv, "--sim_layers") == "0"
    assert _flag_value(argv, "--batch_size") == "32"
    assert _flag_value(argv, "--sequence_length") == "2048"
    assert _flag_value(argv, "--hw_json") == cmd["hw_config_name"]
    assert "--output_dir" in argv
    assert not _flag_value(argv, "--output_dir").endswith("_prefill")
    assert cmd["expected_output"].endswith("output_cg_8_row_8192.log")
    assert cmd["display"] == " ".join(argv)  # no shell-special chars here
    assert cmd["label"]
    assert cmd["params"]["mode"] == "decode"


def test_build_sim_command_prefill():
    cmd = commands.build_sim_command({"mode": "prefill", "batch_size": 1})
    argv = cmd["argv"]
    assert "--prefill" in argv
    assert _flag_value(argv, "--split_factor") == \
        str(commands.SPLIT_FACTOR_PREFILL)
    assert _flag_value(argv, "--sim_layers") == "1"
    assert _flag_value(argv, "--output_dir").endswith("_prefill")


def test_build_sim_command_impl_flag():
    cmd = commands.build_sim_command({"impl": "uniform_dram"})
    assert "--uniform_dram_mapping" in cmd["argv"]
    cmd = commands.build_sim_command({"impl": "best"})
    assert not any(a.startswith("--uniform") for a in cmd["argv"])


# ---------------------------------------------------------------------------
# expected_output_path
# ---------------------------------------------------------------------------

_BASE_PARAMS = {"model": "llama2-13", "mode": "decode", "batch_size": 32,
                "num_cores": 256, "sa_size": 32, "sram_kb": 2048,
                "dram_bw": 12288, "noc_topo": 1, "noc_bw": 16,
                "core_group": 8, "row": 8192, "impl": "best",
                "root": "logs_web", "trp": None, "trcd": 14}


def test_expected_output_path_shape():
    path = commands.expected_output_path(dict(_BASE_PARAMS))
    parts = path.parts
    assert path.name == "output_cg_8_row_8192.log"
    assert parts[-2] == "best"
    assert parts[-3] == "topo_1-nocbw16"
    assert parts[-4] == "sram_2048-drambw_12288_PLACEHOLDER"
    assert parts[-5] == "sa_32-vu_32"
    assert parts[-6] == "decode"
    assert parts[-7] == "core_256"
    assert parts[-8] == "bs_32"
    assert parts[-9] == "llama2-13"
    assert "logs_web" in parts


def test_expected_output_path_trp_suffix():
    p = dict(_BASE_PARAMS, trp=12)
    path = commands.expected_output_path(p)
    assert path.name == "output_cg_8_row_8192_trcd_14_trp_12.log"


# ---------------------------------------------------------------------------
# expand_sim_specs
# ---------------------------------------------------------------------------

def test_expand_sim_specs_cartesian():
    specs = commands.expand_sim_specs(
        {"sa_size": [16, 32], "noc_bw": [4, 8, 16], "model": "llama2-13"})
    assert len(specs) == 6
    assert {(s["sa_size"], s["noc_bw"]) for s in specs} == \
        {(16, 4), (16, 8), (16, 16), (32, 4), (32, 8), (32, 16)}
    for s in specs:
        assert not isinstance(s["sa_size"], list)
        assert s["model"] == "llama2-13"


def test_expand_sim_specs_scalar_passthrough():
    params = {"model": "llama2-13", "sa_size": 32}
    assert commands.expand_sim_specs(params) == [params]


# ---------------------------------------------------------------------------
# make_hw_config_dict
# ---------------------------------------------------------------------------

def test_make_hw_config_dict_schema():
    d = commands.make_hw_config_dict(sa=32, noc_topo=1, noc_bw=16,
                                     dram_bw=12288, row=8192, trp=14)
    assert set(d) == {"compute", "noc", "dram"}
    comp = d["compute"]
    assert comp["mm_pad_shape"] == [32, 32, 32]
    assert comp["ew_pad_len"] == 32
    assert comp["mm_reuse_list"] == [32, 32 * 32, 32]
    assert comp["ew_mm_overlap"] is True
    noc = d["noc"]
    assert noc["topology"] == 1 and noc["bandwidth_bytepc"] == 16
    assert noc["default_noc"] is True
    dram = d["dram"]
    assert dram["bandwidth_GBps"] == 12288
    assert dram["num_access_per_row"] == 8192
    assert dram["tRP"] == 14
    assert dram["CL"] == commands.CL and dram["tRCD"] == commands.TRCD


# ---------------------------------------------------------------------------
# ensure_hw_config (HW_CONFIG_DIR pinned to tmp_path)
# ---------------------------------------------------------------------------

@pytest.fixture()
def tmp_hw_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(commands, "HW_CONFIG_DIR", tmp_path)
    return tmp_path


def test_ensure_hw_config_creates_and_is_idempotent(tmp_hw_dir):
    args = dict(sa=33, noc_topo=2, noc_bw=3, dram_bw=9999, row=1234, trp=11)
    name1 = commands.ensure_hw_config(**args)
    path = tmp_hw_dir / f"{name1}.json"
    assert path.is_file()
    want = commands.make_hw_config_dict(**args)
    assert json.loads(path.read_text()) == want
    content, mtime = path.read_text(), path.stat().st_mtime_ns
    # second call: same name, no rewrite
    name2 = commands.ensure_hw_config(**args)
    assert name2 == name1
    assert path.read_text() == content
    assert path.stat().st_mtime_ns == mtime
    assert len(list(tmp_hw_dir.glob("*.json"))) == 1


def test_ensure_hw_config_name_conflict_gets_web_prefix(tmp_hw_dir):
    base = commands.hw_config_name(44, 1, 16, 5000)
    (tmp_hw_dir / f"{base}.json").write_text(json.dumps({"bogus": True}))
    name = commands.ensure_hw_config(sa=44, noc_topo=1, noc_bw=16,
                                     dram_bw=5000, row=77)
    assert name.startswith("web_") and name != base
    # the conflicting original file is left untouched
    assert json.loads((tmp_hw_dir / f"{base}.json").read_text()) == \
        {"bogus": True}
    written = json.loads((tmp_hw_dir / f"{name}.json").read_text())
    assert written == commands.make_hw_config_dict(44, 1, 16, 5000, 77)
    # idempotent on the web-prefixed variant too
    assert commands.ensure_hw_config(44, 1, 16, 5000, 77) == name


def test_ensure_hw_config_reuses_matching_existing(tmp_hw_dir):
    args = dict(sa=8, noc_topo=3, noc_bw=64, dram_bw=4000, row=512, trp=14)
    name = commands.hw_config_name(8, 3, 64, 4000)
    (tmp_hw_dir / f"{name}.json").write_text(
        json.dumps(commands.make_hw_config_dict(**args)))
    # identical content -> reused as-is, no web_ fallback
    assert commands.ensure_hw_config(**args) == name
