import pytest

from boltons.statsutils import OnlineWeightedStats, Stats, mode


def test_stats_basic():
    da = Stats(range(20))
    assert da.mean == 9.5
    assert round(da.std_dev, 2) == 5.77
    assert da.variance == 33.25
    assert da.skewness == 0
    assert round(da.kurtosis, 1) == 1.9
    assert da.median == 9.5


def test_pearson_type_covers_kappa_ge_zero():
    assert Stats([8, 4, 4, 4, 5, 1, 6, 5]).pearson_type == 4
    assert Stats([5, 5, 2, 4, 5, 9, 5, 5]).pearson_type == 5
    assert Stats([0, 0, 0, 0, 0, 0, -11, 17]).pearson_type == 6


def test_mode():
    assert Stats([2, 1, 3, 1]).mode == 1
    # ties resolve to the value seen first in the data
    assert mode([1, 1, 2, 2, 3]) == 1
    # non-numeric, categorical data is supported
    assert mode(['a', 'b', 'b', 'c', 'c', 'c']) == 'c'
    # empty data falls back to the configured default
    assert Stats([], default=None).mode is None


def test_pearson_type_zero_denominator():
    # Supply exact moments to exercise c0 == 0, c2 == 0, and the
    # normal-distribution boundary without sampling noise.
    for skewness, kurtosis, expected in ((2.0, 3.0, 1),
                                         (2.0, 9.0, 3),
                                         (0.0, 3.0, 0)):
        stats = Stats([0])
        stats.skewness = skewness
        stats.kurtosis = kurtosis
        assert stats.pearson_type == expected


def test_histogram_zero_interquartile_range():
    for data in ([5] * 10, [0] * 10 + [100]):
        counts = Stats(data).get_histogram_counts()
        assert counts == [(float(min(data)), len(data))]


def test_online_weighted_stats_variance_conventions():
    stats = OnlineWeightedStats([2.0, 4.0], [0.5, 1.5])

    assert stats.total_weight == 2.0
    assert stats.mean == 3.5
    assert stats.m2 == 1.5
    assert stats.population_variance() == 0.75
    assert stats.sample_variance() == 1.5


def test_online_weighted_stats_zero_weight_is_noop():
    stats = OnlineWeightedStats()
    stats.add(1.0, 0.0)

    assert stats.total_weight == 0.0
    assert stats.mean == 0.0
    assert stats.m2 == 0.0


@pytest.mark.parametrize('value, weight', [
    (float('inf'), 1.0),
    (float('nan'), 1.0),
    (1.0, -1.0),
    (1.0, float('nan')),
])
def test_online_weighted_stats_rejects_invalid_inputs(value, weight):
    with pytest.raises((TypeError, ValueError)):
        OnlineWeightedStats().add(value, weight)


def test_online_weighted_stats_precision_near_large_value():
    base = 1_000_000_000_000.0
    values = [base - 2.0, base - 1.0, base + 1.0, base + 2.0]
    stats = OnlineWeightedStats(values)
    naive_m2 = sum(value * value for value in values) - len(values) * base ** 2

    assert stats.mean == base
    assert stats.m2 == pytest.approx(10.0, rel=1e-4)
    assert naive_m2 == 0.0


def test_online_weighted_stats_merge_order():
    base = 1_000_000_000_000.0
    first = OnlineWeightedStats([base - 1.0, base + 1.0])
    second = OnlineWeightedStats([base - 2.0, base + 2.0])

    merged_first_second = OnlineWeightedStats.from_shards([first, second])
    merged_second_first = OnlineWeightedStats.from_shards([second, first])

    assert merged_first_second.total_weight == 4.0
    assert merged_first_second.mean == base
    assert merged_first_second.m2 == pytest.approx(10.0)
    assert merged_second_first.m2 == pytest.approx(merged_first_second.m2)


def test_online_weighted_stats_remove_all_returns_to_empty_state():
    values = [2.0, 4.0, 7.0]
    weights = [1.0, 2.0, 0.5]
    stats = OnlineWeightedStats(values, weights)

    for value, weight in reversed(list(zip(values, weights))):
        stats.remove(value, weight)

    assert stats.total_weight == 0.0
    assert stats.mean == 0.0
    assert stats.m2 == 0.0

    stats.add(5.0, 2.0)
    assert stats.total_weight == 2.0
    assert stats.mean == 5.0
    assert stats.m2 == 0.0


def test_online_weighted_stats_invalid_removal_is_atomic():
    stats = OnlineWeightedStats([0.0, 0.0])

    with pytest.raises(ValueError):
        stats.remove(10.0)

    assert stats.total_weight == 2.0
    assert stats.mean == 0.0
    assert stats.m2 == 0.0

    with pytest.raises(ValueError):
        stats.remove(1.0, 3.0)

    assert stats.total_weight == 2.0


def test_online_weighted_stats_rejects_wrong_last_sample():
    stats = OnlineWeightedStats([1.0])

    with pytest.raises(ValueError):
        stats.remove(2.0)

    assert stats.total_weight == 1.0
    assert stats.mean == 1.0
    assert stats.m2 == 0.0


def test_online_weighted_stats_merge_rejects_negative_moment():
    stats = OnlineWeightedStats([1.0, 2.0])
    invalid_shard = OnlineWeightedStats([3.0])
    invalid_shard._m2 = -1.0

    with pytest.raises(ValueError):
        stats.merge(invalid_shard)

    assert stats.total_weight == 2.0
    assert stats.mean == 1.5
    assert stats.m2 == 0.5


def test_online_weighted_stats_empty_variance_undefined():
    stats = OnlineWeightedStats()

    with pytest.raises(ValueError):
        stats.population_variance()
    with pytest.raises(ValueError):
        stats.sample_variance()
