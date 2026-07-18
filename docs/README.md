# Documentation Archive

Historical documentation of major debugging sessions and architectural decisions.

## Session Archives (2026-02-13)

### Session A: Temporal Estimation & Parallelism Architecture

**Topics**:
- Temporal workload estimation accuracy (pruning-aware recursive estimator)
- Intra-spatial parallelism for mega spatials
- Multiprocessing state hygiene (`__getstate__` / `__setstate__`)
- **Critical fix**: `cold_config_candidates` must NOT be stripped — causes `AssertionError: min_cold_configs empty`

**Key outcomes**:
- Replaced under-estimating formula with tree-pruning-aware recursive counter
- Added memoization and overlapped estimate precompute
- Implemented safe mega-spatial splitting without nested pools
- Fixed cold/hot assertion crash by preserving cross-phase state

---

### Session B: RAM Spike Investigation
**File**: [`SESSION_2026-02-13B_RAM_SPIKES.md`](SESSION_2026-02-13B_RAM_SPIKES.md)

**Topics**:
- Per-batch RAM spikes during `search_optimal_config`
- Workload-based batch sizing (not spatial count)
- LPT scheduling for chunk balancing
- Intermediate deletion patterns

**Key outcomes**:
- Added `del chunks`, `del chunk_results`, `del params` to prevent 5 intermediates living simultaneously
- Changed batch count to `ceil(total_est / TARGET_BATCH_WORKLOAD)` — scales with actual work
- Replaced greedy-fill chunking with LPT — eliminates long tails
- Calibrated `TARGET_BATCH_WORKLOAD = 1.5e7` for ~100-200K configs per batch

---

## Design Principles Extracted

From both sessions, these principles emerged:

1. **Estimation must mirror pruning semantics** — not just raw combinatorics
2. **Single-level process topology** — avoid nested pools
3. **Explicit state reconstruction** — workers call `update_spatial_dim_parts_if_valid(...)` before accessing spatial-derived fields
4. **Preserve cross-phase data** — don't strip fields needed after process round-trips
5. **Delete intermediates at sync points** — prevents peak RAM accumulation
6. **Workload-based scheduling** — use estimates, not counts, when variance is high
7. **LPT for parallel distribution** — greedy fill creates tails

---

## Related Documentation

- **Memory files**: `~/.claude/projects/.../memory/` — persistent lessons for Claude
  - `MEMORY.md` — top-level index
  - `ram_patterns.md` — pickle hygiene, deletion patterns, multiprocessing architecture
  - `balancing_patterns.md` — batch/chunk load balancing strategies
  - `session_2026-02-13b_ram_spikes.md` — detailed Session B narrative

---

## Common Error Signatures

| Error | Root Cause | Fix |
|-------|-----------|-----|
| `AttributeError: ... has no attribute 'spatial_var_replicas'` | Worker accessed stripped field before rebuild | Call `update_spatial_dim_parts_if_valid(...)` first |
| `AssertionError: min_cold_configs is empty` | `cold_config_candidates` stripped in `__getstate__` | Remove from strip list — cold/hot needs it |
| RAM spikes after batches | Intermediates not deleted | Add `del chunks`, `del chunk_results`, `del params` |
| Batch imbalance (long tails) | Greedy fill chunking | Use LPT scheduling |
| Too many/few batches | Count-based batch sizing | Use `ceil(total_est / TARGET_BATCH_WORKLOAD)` |
| Nested pool crashes | Double pickling, state fragility | Flatten to single pool with mode dispatch |

---

## Quick Reference: What NOT to Do

From both sessions' hard-learned lessons:

- ❌ **DO NOT** strip `cold_config_candidates` in `__getstate__`
- ❌ **DO NOT** strip `config_dict` in `__getstate__`
- ❌ **DO NOT** remove `op.variables` / `op.dim_lengths` from `TensorOperator`
- ❌ **DO NOT** use greedy fill for parallel chunk assignment
- ❌ **DO NOT** use count-based batch sizing when workload variance is high
- ❌ **DO NOT** use nested multiprocessing pools
- ❌ **DO NOT** access spatial-derived fields in workers without rebuilding first
- ❌ **DO NOT** assume lru_cache estimates are free — memoize recursive counters
