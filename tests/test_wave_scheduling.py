"""Unit tests for memory-aware wave scheduling helpers in dask_util."""

from cellmap_analyze.util import dask_util


def test_balanced_batches_empty():
    assert dask_util.balanced_batches([], 4) == []


def test_balanced_batches_fewer_items_than_batches():
    # 3 items, 10 batches → returns 3 single-item batches.
    items = [("a", 100), ("b", 200), ("c", 150)]
    batches = dask_util.balanced_batches(items, 10)
    assert len(batches) == 3
    assert sorted(sum(batches, [])) == ["a", "b", "c"]


def test_balanced_batches_load_balance():
    # Heavy items should be split across batches, not stacked.
    weights = [100, 90, 80, 70, 60, 50, 40, 30, 20, 10]
    items = [(f"i{i}", w) for i, w in enumerate(weights)]
    batches = dask_util.balanced_batches(items, 4)
    assert len(batches) == 4

    # Each batch's total weight should be close to the mean (550 / 4 ≈ 137).
    weights_by_id = {item_id: w for item_id, w in items}
    totals = [sum(weights_by_id[i] for i in batch) for batch in batches]
    # No batch is more than 30% above the average — greedy LPT is good
    # enough for this load.
    avg = sum(weights) / 4
    assert max(totals) <= avg * 1.3, totals


def test_plan_memory_waves_no_config_single_wave():
    # Without a jobqueue config, the planner produces one wave covering
    # all items with processes=1.
    items = [(i, 10_000_000 * i) for i in range(1, 6)]
    waves = dask_util.plan_memory_waves(items, requested_workers=4, config=None)
    assert len(waves) == 1
    wave = waves[0]
    assert wave.processes == 1
    assert set(wave.item_ids) == {1, 2, 3, 4, 5}
    assert wave.config is None
    # Heaviest-first ordering within the wave.
    assert wave.item_ids[0] == 5


def test_plan_memory_waves_groups_by_memory_class():
    # 16 GB slot, base_processes=16 → tiny items fit at 16/slot, big items
    # fit fewer per slot. We expect at least two waves.
    config = {
        "jobqueue": {
            "lsf": {
                "processes": 16,
                "cores": 16,
                "memory": "16GB",
            }
        }
    }
    items = [
        ("tiny1", 100_000_000),    # ~100 MB ⇒ many per slot
        ("tiny2", 200_000_000),
        ("medium", 2_000_000_000),  # 2 GB ⇒ fewer per slot
        ("big", 8_000_000_000),     # 8 GB ⇒ 1 per slot
    ]
    waves = dask_util.plan_memory_waves(items, requested_workers=8, config=config)
    # 3+ distinct memory classes → at least 3 waves.
    assert len(waves) >= 3

    # Waves are sorted low → high processes (high-memory first).
    procs = [w.processes for w in waves]
    assert procs == sorted(procs)
    # The biggest item lands in the smallest-processes wave.
    big_wave = waves[0]
    assert "big" in big_wave.item_ids
    # The smallest items land in the largest-processes wave.
    small_wave = waves[-1]
    assert "tiny1" in small_wave.item_ids and "tiny2" in small_wave.item_ids

    # Each wave's config has been tuned: its processes value matches the
    # wave's planned per-slot processes.
    for wave in waves:
        assert wave.config["jobqueue"]["lsf"]["processes"] == wave.processes
        assert wave.config["jobqueue"]["lsf"]["cores"] == wave.processes


def test_plan_memory_waves_buckets_procs_to_powers_of_2():
    # 12 items with peaks that would naively land in 12 distinct integer
    # process counts; bucketed planner should collapse to {1,2,4,8,16}.
    config = {"jobqueue": {"lsf": {"processes": 16, "cores": 16, "memory": "240GB"}}}
    # usable = 144 GB. Pick peaks that fall in many narrow integer slots
    # without rounding (10.5 → 13 procs, 11 → 13, 12 → 12, etc.).
    items = [
        ("a", int(120e9)),   # 144/120 = 1.2 → bucket 1
        ("b", int(60e9)),    # 144/60  = 2.4 → bucket 2
        ("c", int(48e9)),    # 144/48  = 3   → bucket 2
        ("d", int(36e9)),    # 144/36  = 4   → bucket 4
        ("e", int(24e9)),    # 144/24  = 6   → bucket 4
        ("f", int(18e9)),    # 144/18  = 8   → bucket 8
        ("g", int(16e9)),    # 144/16  = 9   → bucket 8
        ("h", int(14e9)),    # 144/14  = 10  → bucket 8
        ("i", int(12e9)),    # 144/12  = 12  → bucket 8
        ("j", int(10e9)),    # 144/10  = 14  → bucket 8
        ("k", int(9e9)),     # 144/9   = 16  → bucket 16
        ("l", int(5e9)),     # well-fit at base → bucket 16
    ]
    waves = dask_util.plan_memory_waves(items, requested_workers=10, config=config)
    procs_seen = sorted(w.processes for w in waves)
    assert procs_seen == [1, 2, 4, 8, 16], procs_seen


def test_plan_memory_waves_all_items_distributed():
    # Pathological case: many small items + one giant. All items must end
    # up in some wave, none dropped or duplicated.
    config = {
        "jobqueue": {
            "lsf": {"processes": 8, "cores": 8, "memory": "16GB"}
        }
    }
    items = [(i, 50_000_000) for i in range(100)] + [("giant", 30_000_000_000)]
    waves = dask_util.plan_memory_waves(items, requested_workers=10, config=config)
    all_ids = [iid for w in waves for iid in w.item_ids]
    assert sorted(str(x) for x in all_ids) == sorted(str(x) for x in [iid for iid, _ in items])
