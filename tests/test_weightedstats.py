import math
import random

import pytest

from boltons.weightedstats import WeightedStats


def _reference_moments(samples):
    """Brute-force, fsum-stabilized reference moments."""
    total_w = math.fsum(w for _, w in samples)
    mean = math.fsum(w * x for x, w in samples) / total_w
    m2 = math.fsum(w * (x - mean) ** 2 for x, w in samples)
    wsq = math.fsum(w * w for _, w in samples)
    return total_w, mean, m2, wsq


def test_matches_unweighted_reference():
    samples = [(float(v), 1.0) for v in range(1, 21)]
    ws = WeightedStats(samples)
    total_w, mean, m2, _ = _reference_moments(samples)

    assert ws.total_weight == total_w
    assert ws.mean == pytest.approx(mean)
    assert ws.m2 == pytest.approx(m2)
    # population denominator W vs sample denominator W - 1
    assert ws.pvariance == pytest.approx(m2 / 20.0)
    assert ws.variance == pytest.approx(m2 / 19.0)
    assert ws.std_dev == pytest.approx(math.sqrt(m2 / 19.0))


def test_weighted_frequency_variance():
    # weight-3 sample == three duplicate observations
    samples = [(2.0, 2.0), (4.0, 3.0), (9.0, 1.0)]
    expanded = [(2.0, 1.0)] * 2 + [(4.0, 1.0)] * 3 + [(9.0, 1.0)]

    ws = WeightedStats(samples)
    exp = WeightedStats(expanded)

    assert ws.mean == pytest.approx(exp.mean)
    assert ws.pvariance == pytest.approx(exp.pvariance)
    assert ws.variance == pytest.approx(exp.variance)


def test_reliability_variance_denominator():
    samples = [(1.0, 0.5), (3.0, 2.0), (8.0, 1.0)]
    ws = WeightedStats(samples)
    total_w, _, m2, wsq = _reference_moments(samples)
    denom = total_w - wsq / total_w
    assert ws.reliability_variance() == pytest.approx(m2 / denom)


def test_zero_weight_is_noop():
    ws = WeightedStats()
    ws.add(5.0, 0.0)
    assert not ws
    assert ws.mean == 0.0
    ws.add(2.0, 1.0).add(9.0, 0.0)
    assert ws.total_weight == 1.0
    assert ws.mean == 2.0
    assert ws.m2 == 0.0


@pytest.mark.parametrize('value, weight', [
    (float('nan'), 1.0),
    (float('inf'), 1.0),
    (1.0, float('nan')),
    (1.0, float('inf')),
    (1.0, -0.1),
])
def test_invalid_inputs_rejected(value, weight):
    ws = WeightedStats()
    with pytest.raises(ValueError):
        ws.add(value, weight)


def test_precision_large_offset_small_fluctuations():
    # Small perturbations around 1e12: a naive E[x^2] - E[x]^2
    # implementation loses the variance entirely in double precision.
    base = 1e12
    deviations = [-3.0, -1.0, 2.0, 2.0, 4.0, -4.0, 0.0]
    ws = WeightedStats((base + d, 1.0) for d in deviations)

    ref_mean = base + sum(deviations) / len(deviations)
    ref_pvar = sum(d ** 2 for d in deviations) / len(deviations)
    ref_svar = sum(d ** 2 for d in deviations) / (len(deviations) - 1)

    assert ws.mean == pytest.approx(ref_mean, abs=1e-6)
    assert ws.pvariance == pytest.approx(ref_pvar, rel=1e-9)
    assert ws.variance == pytest.approx(ref_svar, rel=1e-9)

    # A large-square subtraction drowns the ~7.0 signal entirely:
    # E[x^2] - E[x]^2 produces a nonsensical value on the scale of
    # 2^27 instead of the true variance.
    naive_sq = sum((base + d) ** 2 for d in deviations) / len(deviations)
    naive_mean = sum(base + d for d in deviations) / len(deviations)
    naive = naive_sq - naive_mean ** 2
    assert not math.isclose(naive, ref_pvar, rel_tol=0.5)


def test_merge_independent_of_shard_order():
    rng = random.Random(20261003)
    samples = [(rng.uniform(-100.0, 100.0), rng.uniform(0.1, 5.0))
               for _ in range(500)]
    total_w, mean, m2, wsq = _reference_moments(samples)

    def merge_in_order(order):
        shards = [[samples[i] for i in range(len(samples)) if i % 3 == k]
                  for k in order]
        return WeightedStats.combine(*(WeightedStats(s) for s in shards))

    orders = [(0, 1, 2), (2, 1, 0), (1, 0, 2)]
    merged = [merge_in_order(o) for o in orders]

    for ws in merged:
        assert ws.total_weight == pytest.approx(total_w)
        assert ws.mean == pytest.approx(mean, rel=1e-12)
        assert ws.m2 == pytest.approx(m2, rel=1e-10)
        assert ws._weight_sq_sum == pytest.approx(wsq)

    # identical to sequential feeding
    sequential = WeightedStats(samples)
    for ws in merged:
        assert ws.mean == pytest.approx(sequential.mean)
        assert ws.m2 == pytest.approx(sequential.m2, rel=1e-10)


def test_merge_empty_shards():
    ws = WeightedStats([(1.0, 1.0)])
    ws.merge(WeightedStats())
    assert ws.total_weight == 1.0 and ws.mean == 1.0

    empty = WeightedStats()
    empty.merge(WeightedStats([(2.0, 2.0)]))
    assert empty.total_weight == 2.0 and empty.mean == 2.0


def test_remove_all_returns_to_empty_and_reusable():
    samples = [(3.0, 2.0), (7.0, 1.0), (11.0, 4.0)]
    ws = WeightedStats(samples)

    for value, weight in samples:
        ws.remove(value, weight)

    assert not ws
    assert ws.total_weight == 0.0
    assert ws.m2 == 0.0
    assert ws.mean == 0.0
    assert ws.pvariance == 0.0

    # empty state is usable again
    ws.update([(5.0, 1.0), (9.0, 3.0)])
    assert ws.total_weight == 4.0
    assert ws.mean == pytest.approx(8.0)


def test_remove_reconstructs_intermediate_states():
    samples = [(1.0, 1.0), (6.0, 2.0), (10.0, 1.0), (4.0, 3.0)]

    # undo the tail one sample at a time and compare against the
    # accumulator that never saw those samples
    for cut in range(len(samples), 0, -1):
        ws = WeightedStats(samples)
        for value, weight in samples[cut:]:
            ws.remove(value, weight)
        ref = WeightedStats(samples[:cut])
        assert ws.total_weight == pytest.approx(ref.total_weight)
        assert ws.mean == pytest.approx(ref.mean)
        assert ws.m2 == pytest.approx(ref.m2, abs=1e-9)


def test_remove_weight_exceeding_total_rejected():
    ws = WeightedStats([(1.0, 1.0), (2.0, 1.0)])
    with pytest.raises(ValueError):
        ws.remove(1.0, 3.0)
    # state untouched
    assert ws.total_weight == 2.0


def test_remove_unknown_sample_rejected():
    ws = WeightedStats([(1.0, 1.0), (2.0, 1.0), (3.0, 1.0)])
    # weight is valid, but retracting this "sample" forces m2 negative
    with pytest.raises(ValueError):
        ws.remove(99.0, 1.0)
    assert ws.total_weight == 3.0
    assert ws.m2 >= 0.0


def test_remove_empty_rejected():
    with pytest.raises(ValueError):
        WeightedStats().remove(1.0, 1.0)


def test_large_offset_remove_roundtrip_resets_cleanly():
    base = 1e12
    samples = [(base + d, 1.0)
               for d in (0.5, -1.25, 2.0, -0.25, 3.0)]
    ws = WeightedStats(samples)
    for value, weight in samples:
        ws.remove(value, weight)
    assert not ws
    assert ws.total_weight == 0.0
    assert ws.mean == 0.0 and ws.m2 == 0.0


def test_update_accepts_mixed_items():
    ws = WeightedStats([1.0, (2.0, 2.0), 3.0])
    assert ws.total_weight == 4.0
    assert ws.mean == pytest.approx((1.0 + 4.0 + 3.0) / 4.0)
