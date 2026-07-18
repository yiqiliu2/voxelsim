"""Results index: scans all result roots and caches summary metrics.

The index maps every ``output_cg*_row_*.log`` under the known result roots
to a config dict parsed from its directory path.  Summary metrics are parsed
lazily (head-read only) and cached on disk keyed by (path, mtime, size).
"""

import json
import threading
import time
from pathlib import Path
from typing import Dict, List, Optional

from . import parsers
from .config import (CACHE_DIR, LOG_NAME_RE, DIR_PATTERNS, PROJECT_ROOT,
                     discover_result_roots)

_SUMMARY_CACHE_FILE = CACHE_DIR / "summary_cache.json"


def _parse_config_path(root: Dict, log_path: Path) -> Optional[Dict]:
    """Build a config dict from a log file's path under a result root."""
    try:
        rel = log_path.relative_to(root["path"])
    except ValueError:
        return None
    parts = rel.parts
    # expect: model/bs_B/core_C/mode/sa..-vu../sram..-drambw.._name/
    #         topo..-nocbw../impl/file.log   (9 parts)
    if len(parts) != 9:
        return None
    name_m = LOG_NAME_RE.match(parts[8])
    if not name_m:
        return None
    bs_m = DIR_PATTERNS["batch_size"].match(parts[1])
    core_m = DIR_PATTERNS["num_cores"].match(parts[2])
    mode = parts[3]
    sa_m = DIR_PATTERNS["sa"].match(parts[4])
    sram_m = DIR_PATTERNS["sram"].match(parts[5])
    noc_m = DIR_PATTERNS["noc"].match(parts[6])
    if not (bs_m and core_m and sa_m and sram_m and noc_m):
        return None
    if mode not in ("decode", "prefill"):
        return None
    cfg = {
        "model": parts[0],
        "batch_size": int(bs_m.group(1)),
        "num_cores": int(core_m.group(1)),
        "mode": mode,
        "sa_size": int(sa_m.group(1)),
        "vu_size": int(sa_m.group(2)),
        "sram_kb": int(sram_m.group(1)),
        "dram_bw": int(sram_m.group(2)),
        "dram_name": sram_m.group(3),
        "noc_topo": int(noc_m.group(1)),
        "noc_bw": int(noc_m.group(2)),
        "impl": parts[7],
        "core_group": int(name_m.group(1)),
        "row": int(name_m.group(2)),
        "trcd": int(name_m.group(3)) if name_m.group(3) else None,
        "trp": int(name_m.group(4)) if name_m.group(4) else None,
        "root": root["name"],
        "seq_length": root.get("seq_length"),
    }
    # id = path relative to project root (stable, URL-safe)
    cfg["id"] = log_path.relative_to(PROJECT_ROOT).as_posix()
    cfg["log_file"] = str(log_path)
    base = log_path.with_suffix("")  # strip .log
    cfg["overlap_file"] = str(log_path.parent /
                              log_path.name.replace("output_", "overlap_"))
    cfg["top_power_file"] = str(log_path.parent /
                                log_path.name.replace("output_", "top_power_"))
    cfg["pickle_file"] = str(log_path.parent /
                             f"output_cg_{cfg['core_group']}.pickle")
    cfg["has_overlap"] = Path(cfg["overlap_file"]).exists()
    cfg["has_pickle"] = Path(cfg["pickle_file"]).exists()
    try:
        st = log_path.stat()
        cfg["mtime"] = st.st_mtime
        cfg["size"] = st.st_size
    except OSError:
        cfg["mtime"] = 0
        cfg["size"] = 0
    _ = base
    return cfg


class ResultsIndex:
    """Thread-safe index of all simulation results."""

    def __init__(self, roots=None):
        self._lock = threading.RLock()
        self._configs: List[Dict] = []
        self._by_id: Dict[str, Dict] = {}
        self._roots = roots if roots is not None else discover_result_roots()
        self._summary_cache: Dict[str, Dict] = {}
        self._built_at: Optional[float] = None
        self._load_summary_cache()
        self.rebuild()

    # ------------------------------------------------------------------
    # Index build
    # ------------------------------------------------------------------
    def rebuild(self):
        # Re-discover roots so newly created ones (e.g. results/logs_web from
        # web-launched runs) are picked up without a server restart.
        self._roots = discover_result_roots()
        configs = []
        for root in self._roots:
            root_path = Path(root["path"])
            if not root_path.is_dir():
                continue
            for log_path in root_path.rglob("output_cg*_row_*.log"):
                cfg = _parse_config_path(root, log_path)
                if cfg:
                    configs.append(cfg)
        with self._lock:
            self._configs = configs
            self._by_id = {c["id"]: c for c in configs}
            self._built_at = time.time()
        return len(configs)

    @property
    def roots(self):
        return [{"name": r["name"], "seq_length": r.get("seq_length")}
                for r in self._roots]

    @property
    def built_at(self):
        return self._built_at

    def all(self) -> List[Dict]:
        with self._lock:
            return list(self._configs)

    def get(self, cfg_id: str) -> Optional[Dict]:
        with self._lock:
            return self._by_id.get(cfg_id)

    def models(self) -> List[str]:
        return sorted({c["model"] for c in self.all()})

    def filter_values(self) -> Dict[str, List]:
        """Distinct values per filterable dimension (for UI dropdowns)."""
        keys = ("root", "model", "mode", "batch_size", "num_cores", "sa_size",
                "sram_kb", "dram_bw", "noc_topo", "noc_bw", "core_group",
                "impl", "row", "seq_length")
        out = {}
        cfgs = self.all()
        for k in keys:
            vals = sorted({c[k] for c in cfgs if c.get(k) is not None})
            out[k] = vals
        return out

    # ------------------------------------------------------------------
    # Summary cache
    # ------------------------------------------------------------------
    def _load_summary_cache(self):
        try:
            with open(_SUMMARY_CACHE_FILE) as f:
                raw = json.load(f)
            if isinstance(raw, dict):
                self._summary_cache = raw
        except (OSError, ValueError):
            self._summary_cache = {}

    def _save_summary_cache(self):
        try:
            tmp = _SUMMARY_CACHE_FILE.with_suffix(".tmp")
            with open(tmp, "w") as f:
                json.dump(self._summary_cache, f)
            tmp.replace(_SUMMARY_CACHE_FILE)
        except OSError:
            pass

    def summary(self, cfg: Dict) -> Optional[Dict]:
        """Summary metrics for a config, with persistent mtime-keyed cache."""
        key = cfg["id"]
        ent = self._summary_cache.get(key)
        if ent and ent.get("mtime") == cfg.get("mtime") \
                and ent.get("size") == cfg.get("size"):
            return ent.get("metrics")
        metrics = parsers.parse_summary_file(cfg["log_file"])
        if metrics is None:
            return None
        self._summary_cache[key] = {
            "mtime": cfg.get("mtime"), "size": cfg.get("size"),
            "metrics": metrics,
        }
        return metrics

    def summaries(self, configs: List[Dict]) -> Dict[str, Dict]:
        """Bulk summaries; returns {id: metrics}. Flushes cache to disk."""
        out = {}
        dirty = False
        for cfg in configs:
            key = cfg["id"]
            before = len(self._summary_cache)
            m = self.summary(cfg)
            if m is not None:
                out[key] = m
            if len(self._summary_cache) != before:
                dirty = True
        if dirty:
            self._save_summary_cache()
        return out

    def prewarm(self):
        """Background: parse summaries for every indexed config once."""
        cfgs = self.all()
        for i in range(0, len(cfgs), 200):
            self.summaries(cfgs[i:i + 200])
        self._save_summary_cache()


_INDEX: Optional[ResultsIndex] = None
_INDEX_LOCK = threading.Lock()


def get_index() -> ResultsIndex:
    global _INDEX
    with _INDEX_LOCK:
        if _INDEX is None:
            _INDEX = ResultsIndex()
        return _INDEX


def reset_index():
    global _INDEX
    with _INDEX_LOCK:
        _INDEX = None
