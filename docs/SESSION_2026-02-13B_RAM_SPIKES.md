# Session 2026-02-13B: RAM Spike Investigation & Workload-Based Batch Sizing

**Timeline**: Follow-on session after `CONVERSATION_RETROSPECTIVE.md` (Session A)
**Focus**: Memory management during batch processing in `search_optimal_config`

---

## Problem Statement

User observed: **"each time after a batch of an op ends, RAM usage flies"**

Example from logs:
```
[Op_7_Dot] batch [1/5] 208024 cfg | gen=95.82s eval=34.22s → 8 hot
```

User feedback: "half of it is good size" → wanted ~100K configs/batch, not 208K.

---

## Root Causes

### 1. Peak RAM = All Intermediates Live Simultaneously

Between Stage 2 (temporal generation) and Stage 3 (evaluation), these all existed at once:
- `chunks` — LPT bucket assignments
- `chunk_results` — pool.map output from `_get_temporal_configs_chunk`
- `batch_configs` — flattened from `chunk_results`
- `params` — built from `batch_configs` for evaluate
- `batch_scores` — pool.map output from `evaluate_config_helper`

Only `batch_configs` and `batch_scores` were deleted. The other 3 stayed live through the entire batch loop.

### 2. Batch Count Ignored Workload Variance

Old formula:
```python
BATCH_SIZE = max(1, int(3e7 / max(1, len(all_spatial_configs))))
num_batches = max(1, (len(all_spatial_configs) + BATCH_SIZE - 1) // BATCH_SIZE)
```

Problem: Based on spatial count, not estimated temporal fan-out.
- Op with 100 spatials × 2K temporals each = 200K total → 5 batches → 40K/batch ✓
- Op with 10 spatials × 20K temporals each = 200K total → 1-2 batches → 100-200K/batch ✗

The sizing was blind to the massive variance in how many temporals each spatial generates.

---

## Fixes Applied

### Fix 1: Delete Intermediates Immediately After Consumption

```python
# Stage 2: Temporal generation
with Pool(num_threads) as pool:
    chunk_results = pool.map(self._get_temporal_configs_chunk, chunks)
del chunks  # ← NEW: worker inputs no longer needed

batch_configs = [
    (spatial_key, tuple(temporal))
    for chunk in chunk_results
    for spatial_key, temporals in chunk
    for temporal in temporals
]
del chunk_results  # ← CRITICAL: batch_configs now owns flattened data

# Stage 3: Evaluation
with Pool(num_threads) as pool:
    params = [(config, i) for i, config in enumerate(batch_configs)]
    batch_scores = pool.map(self.evaluate_config_helper, params)
del params  # ← NEW: batch_scores owns the results
```

**Impact**: Each `del` cuts peak RAM significantly. User confirmed RAM spikes disappeared after this change.

---

### Fix 2: Batch Sizing by Total Estimated Workload

```python
# Pre-compute workload estimates for all spatials
all_estimates = [(self._estimate_temporal_workload(sc), sc) for sc in all_spatial_configs]
total_est = sum(e for e, _ in all_estimates) or 1

# Determine num_batches from total workload budget
TARGET_BATCH_WORKLOAD = 1.5e7  # configs per batch
num_batches = max(1, math.ceil(total_est / TARGET_BATCH_WORKLOAD))
```

**Why this works**:
- `total_est` = sum of estimated temporal configs across all spatials
- Batch count scales with actual work, not spatial count
- Heavy ops with huge temporal fans → more batches automatically
- Light ops → fewer batches, less overhead

**Calibration**: Started at 3e7 (gave 208K/batch), user wanted half → settled at 1.5e7.

---

### Fix 3: LPT Scheduling for Chunk Balancing

User: "find ways to make tails of chunks as mini as possible"

**Problem with greedy fill**:
```python
# OLD: greedy fill creates long tails
cur_chunk, cur_est = [], 0
for est, sc in sorted_spatials:
    cur_chunk.append(sc)
    cur_est += est
    if cur_est >= target:
        chunks.append(cur_chunk)
        cur_chunk, cur_est = [], 0
# Last chunk gets ALL leftovers → can be 10x larger than others
```

**Solution: LPT (Longest Processing Time first)**
```python
n_buckets = min(num_threads, len(batch_est_sc))
buckets = [[] for _ in range(n_buckets)]
bucket_ests = [0] * n_buckets

for est, sc in batch_est_sc:  # already sorted heavy-first
    i = min(range(n_buckets), key=lambda b: bucket_ests[b])
    buckets[i].append(sc)
    bucket_ests[i] += est

chunks = [b for b in buckets if b]
```

**Why LPT wins**: Each spatial assigned to least-loaded bucket → balanced distribution across ALL buckets, no long tail.

---

### Fix 4: Enhanced Logging

```python
if not is_light:
    print(f"[{self.name}] {len(all_spatial_configs)} spatial configs, "
          f"depth={len(self.dim_lengths)-1}, cores={np.prod(self.num_cores)}, "
          f"est={total_est}, batches={num_batches}", flush=True)
```

Now users see estimated workload and resulting batch count upfront.

---

## Files Modified

**`t10_TensorExpression.py`**:
- Lines 1435-1439: Batch count formula → workload-based
- Lines 1470-1480: LPT scheduling for chunks
- Lines 1483, 1490, 1501: Added `del chunks`, `del chunk_results`, `del params`
- Line 1440: Enhanced print with `est` and `batches`

---

## Key Learnings

1. **Count-based heuristics fail when item size varies wildly**
   → Always use workload estimates when available

2. **Profile peak RAM, not just average**
   → Intermediates stack up at synchronization points (between stages)

3. **Delete early, delete often**
   → Especially for large list/dict intermediates after pool.map

4. **LPT > greedy fill for parallel work**
   → Greedy creates tails; LPT balances across all buckets

5. **Batch sizing should match actual RAM pressure**
   → Temporal config count (workload) matters, not spatial count

---

## What This Session Did NOT Cover

Topics handled in prior Session A (see `CONVERSATION_RETROSPECTIVE.md`):
- Temporal estimation accuracy (pruning-aware recursive estimator)
- Intra-spatial splitting for mega spatials
- `cold_config_candidates` pickle stripping bug fix
- Nested pool instability and single-pool architecture

---

## Documentation Updated

- **`CLAUDE.md`**: Added batch sizing strategy, LPT scheduling, and deletion patterns
- **Memory files** (`~/.claude/memory/`):
  - `MEMORY.md` — session timeline, quick reference
  - `ram_patterns.md` — deletion patterns with `del chunks`
  - `balancing_patterns.md` — workload-based batch count formula
  - `session_2026-02-13b_ram_spikes.md` — detailed narrative

---

## Validation

All changes passed `python3 -m py_compile t10_TensorExpression.py`.
User confirmed RAM spikes resolved after applying fixes.
