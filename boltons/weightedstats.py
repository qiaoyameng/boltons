"""
Weighted online statistics
==========================

:class:`WeightedStats` accumulates the weighted mean and the weighted
second central moment of a (possibly very long) data stream without
ever storing the individual datapoints. It supports three incremental
operations:

1. Adding a sample with :meth:`WeightedStats.add`.
2. Combining two independently collected shards with
   :meth:`WeightedStats.merge`.
3. Retracting a sample that was previously added with
   :meth:`WeightedStats.remove`.

The state consists of the total weight :math:`W = \\\\sum_i w_i`, the
weighted mean :math:`\\\\bar{x}`, and the weighted second central
moment :math:`M_2 = \\\\sum_i w_i (x_i - \\\\bar{x})^2`. Updates use the
weighted Welford/Chan formulation, which keeps each correction as a
product of two small differences instead of subtracting two large
squares, so precision survives values clustered tightly around a large
magnitude. Internally, values are translated by the origin of the
first sample (West's translation trick): the recurrence only ever
touches numbers on the scale of the fluctuations, not the absolute
values, which keeps :meth:`remove` precise at the fluctuation scale as
well.

Variance denominators
---------------------

The moment alone fixes the numerator of the variance but not its
denominator. Three conventions are exposed deliberately:

* :attr:`WeightedStats.pvariance` uses :math:`W`, for *frequency*
  weights where each sample represents :math:`w_i` observations of the
  full population.
* :attr:`WeightedStats.variance` uses :math:`W - 1`, the
  frequency-weighted sample estimator (equivalent to the
  :mod:`statistics`-style sample variance with duplicate observations).
* :meth:`WeightedStats.reliability_variance` uses
  :math:`W - \\\\sum_i w_i^2 / W`, the reliability-weight convention
  (an unbiased estimator of the population variance with normalized
  weights, as used by the GSL/Praat family).

Population and sample variance therefore never share a denominator.
"""

import math


class WeightedStats:
    """An online accumulator for the weighted mean and variance.

    Args:
        samples: Optional iterable of ``(value, weight)`` pairs (or
            bare values, which receive weight 1). Consumed eagerly via
            :meth:`update`.
        default: Value returned by statistics that are not defined for
            an empty accumulator, such as the mean or variance.
        moment_tol: Relative floating-point tolerance used when
            :meth:`remove` produces a slightly negative second central
            moment. A negative moment whose magnitude is within
            ``moment_tol`` times the moment's running scale is treated
            as roundoff and clamped to zero.

    >>> ws = WeightedStats()
    >>> _ = ws.add(1e12 + 1)
    >>> _ = ws.add(1e12 - 1)
    >>> round(ws.pvariance, 6)
    1.0
    """

    def __init__(self, samples=(), default=0.0, moment_tol=1e-12):
        self.default = default
        self.moment_tol = moment_tol
        self._total_weight = 0.0
        # Fixed translation origin (first nonzero-weight sample) and
        # S = sum_i w_i * (x_i - origin). The reported mean is
        # origin + S / W.
        self._origin = None
        self._offset_sum = 0.0
        self._m2 = 0.0
        # Sum of squared weights; backs the reliability-weight sample
        # variance denominator.
        self._weight_sq_sum = 0.0
        if samples:
            self.update(samples)

    @property
    def total_weight(self):
        """Sum of the weights accumulated so far (:math:`W`)."""
        return self._total_weight

    @property
    def mean(self):
        """Weighted mean :math:`M_1 / W`, or ``default`` when empty."""
        if self._total_weight == 0.0:
            return self.default
        return self._origin + self._offset_sum / self._total_weight

    @property
    def m2(self):
        """Weighted sum of squared deviations from the mean
        (:math:`M_2`). Always non-negative for a valid state.
        """
        return self._m2

    @property
    def pvariance(self):
        """Population variance :math:`M_2 / W`.

        Defined for frequency weights, including a single sample
        (returns ``0.0``). Returns ``default`` when empty.
        """
        if self._total_weight <= 0.0:
            return self.default
        return self._m2 / self._total_weight

    @property
    def variance(self):
        """Frequency-weighted *sample* variance :math:`M_2 / (W - 1)`.

        Use this when weights count repeated observations and the data
        is a sample of a larger population. Defined only once the total
        weight exceeds 1; otherwise ``default`` is returned.
        """
        denom = self._total_weight - 1.0
        if denom <= 0.0:
            return self.default
        return self._m2 / denom

    def reliability_variance(self):
        """Reliability-weight sample variance
        :math:`M_2 / (W - \\\\sum_i w_i^2 / W)`.

        Use this when weights express observation reliability rather
        than observation counts. Returns ``default`` when the
        denominator is not positive.
        """
        if self._total_weight <= 0.0:
            return self.default
        denom = (self._total_weight
                 - self._weight_sq_sum / self._total_weight)
        if denom <= 0.0:
            return self.default
        return self._m2 / denom

    @property
    def pstd_dev(self):
        """Square root of :attr:`pvariance`."""
        if self._total_weight <= 0.0:
            return self.default
        return math.sqrt(self._m2 / self._total_weight)

    @property
    def std_dev(self):
        """Square root of :attr:`variance`."""
        denom = self._total_weight - 1.0
        if denom <= 0.0:
            return self.default
        return math.sqrt(self._m2 / denom)

    def __bool__(self):
        return self._total_weight > 0.0

    def __repr__(self):
        return ('{cls}(total_weight={w!r}, mean={m!r}, m2={m2!r})'
                .format(cls=self.__class__.__name__,
                        w=self._total_weight,
                        m=self.mean,
                        m2=self._m2))

    def _reset(self):
        self._total_weight = 0.0
        self._origin = None
        self._offset_sum = 0.0
        self._m2 = 0.0
        self._weight_sq_sum = 0.0

    @staticmethod
    def _check_finite(value, weight):
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TypeError('expected a real numeric value, not %r'
                            % (value,))
        if isinstance(weight, bool) or not isinstance(weight, (int, float)):
            raise TypeError('expected a real numeric weight, not %r'
                            % (weight,))
        value = float(value)
        weight = float(weight)
        if math.isnan(value) or math.isinf(value):
            raise ValueError('value must be finite, not %r' % (value,))
        if math.isnan(weight) or math.isinf(weight) or weight < 0.0:
            raise ValueError('weight must be a finite non-negative number,'
                             ' not %r' % (weight,))
        return value, weight

    def add(self, value, weight=1.0):
        """Add one weighted sample.

        A weight of zero (including weights that round to zero) leaves
        the state untouched. Returns ``self`` for chaining.

        Raises :exc:`ValueError` for non-finite values or negative /
        non-finite weights.
        """
        value, weight = self._check_finite(value, weight)
        if weight == 0.0:
            return self
        if self._total_weight == 0.0:
            self._origin = value
        d = value - self._origin
        w = self._total_weight
        new_w = w + weight
        # m2 += w * weight / new_w * d**2 (in offset coordinates the
        # current mean offset is S / w).
        if w > 0.0:
            mean_d = self._offset_sum / w
            self._m2 += (w * weight / new_w) * (d - mean_d) ** 2
        self._offset_sum += weight * d
        self._total_weight = new_w
        self._weight_sq_sum += weight * weight
        return self

    def update(self, samples):
        """Add many samples from an iterable.

        Each item is either a bare numeric value (weight 1) or a
        ``(value, weight)`` pair. Returns ``self``.
        """
        for sample in samples:
            if isinstance(sample, tuple):
                self.add(sample[0], sample[1])
            else:
                self.add(sample)
        return self

    def merge(self, other):
        """Fold another accumulator's shard into this one, in place.

        Uses the parallel (Chan) combination formulas, so merging
        shards yields the same moments as feeding every sample
        sequentially, regardless of shard order. Returns ``self``.

        >>> a = WeightedStats([(1.0, 1), (2.0, 1)])
        >>> b = WeightedStats([(3.0, 1), (4.0, 1)])
        >>> a.merge(b).mean
        2.5
        """
        if not isinstance(other, WeightedStats):
            raise TypeError('can only merge WeightedStats instances,'
                            ' not %r' % (other,))
        w_b = other._total_weight
        if w_b == 0.0:
            return self
        if self._total_weight == 0.0:
            self._total_weight = w_b
            self._origin = other._origin
            self._offset_sum = other._offset_sum
            self._m2 = other._m2
            self._weight_sq_sum = other._weight_sq_sum
            return self

        w_a = self._total_weight
        total_w = w_a + w_b
        # Re-express the other shard's weighted sums in this origin.
        origin_delta = other._origin - self._origin
        other_s = other._offset_sum + w_b * origin_delta
        other_mean_d = other_s / w_b
        self_mean_d = self._offset_sum / w_a
        delta = other_mean_d - self_mean_d
        self._m2 += other._m2 + (w_a * w_b / total_w) * delta ** 2
        self._offset_sum += other_s
        self._total_weight = total_w
        self._weight_sq_sum += other._weight_sq_sum
        return self

    @classmethod
    def combine(cls, *shards):
        """Return a new accumulator merging all ``shards`` without
        mutating any of them.
        """
        combined = cls()
        for shard in shards:
            combined.merge(shard)
        return combined

    def remove(self, value, weight=1.0):
        """Retract a previously added weighted sample.

        This is the exact inverse of :meth:`add` for a sample still
        contained in the accumulator. Membership is not tracked, so
        removing a sample (or weight) that was never added makes the
        state physically invalid and is rejected.

        The operation is refused, leaving the state unchanged, when:

        * the weight is negative or non-finite;
        * the weight exceeds the accumulated total weight;
        * recomputation yields a second central moment that is
          significantly negative.

        A moment that is negative only within :attr:`moment_tol` of the
        running scale is attributed to floating-point roundoff and
        clamped to zero. Retracting the final sample resets the
        accumulator to a reusable empty state.
        """
        value, weight = self._check_finite(value, weight)
        if weight == 0.0:
            return self

        w = self._total_weight
        if w <= 0.0:
            raise ValueError('cannot remove a sample from an empty'
                             ' accumulator')
        # Allow roundoff on the order of moment_tol when many weighted
        # samples have been summed.
        w_tol = self.moment_tol * max(1.0, abs(w), abs(weight))
        if weight > w + w_tol:
            raise ValueError(
                'cannot remove weight %r greater than total weight %r'
                % (weight, w))
        collapse = weight >= w - w_tol
        if weight > w:
            # The excess is pure roundoff.
            weight = w

        new_w = w - weight
        d = value - self._origin
        new_s = self._offset_sum - weight * d
        if collapse:
            # Removing the last bit of weight. The current moment must
            # be fully explained by this one sample's deviation; an
            # unknown value produces a far larger correction.
            mean_d = self._offset_sum / w
            correction = weight * (d - mean_d) ** 2
            new_m2 = self._m2 - correction
            scale = max(1.0, abs(self._m2), correction)
            if new_m2 < -self.moment_tol * scale:
                raise ValueError(
                    'removing %r with weight %r yields an invalid central'
                    ' moment (sample was not part of the accumulator?)'
                    % (value, weight))
            # Every sample has been retracted; floating-point residue
            # in the sums is meaningless.
            self._reset()
            return self

        new_mean_d = new_s / new_w
        # Forward correction for this sample, seen from the state left
        # behind: m2_before = m2_after + (new_w * weight / w) *
        # (d - new_mean_d)**2, where the pre-removal total was w.
        correction = (new_w * weight / w) * (d - new_mean_d) ** 2
        new_m2 = self._m2 - correction
        new_wsq = self._weight_sq_sum - weight * weight

        scale = max(1.0, abs(self._m2), abs(new_m2), correction)
        tol = self.moment_tol * scale
        if new_m2 < -tol or not math.isfinite(new_m2) \
                or not math.isfinite(new_mean_d) or not math.isfinite(new_s):
            raise ValueError(
                'removing %r with weight %r yields an invalid central'
                ' moment (sample was not part of the accumulator?)'
                % (value, weight))
        if new_m2 < 0.0:
            new_m2 = 0.0
        wsq_scale = max(1.0, self._weight_sq_sum, weight * weight)
        if new_wsq < 0.0 and new_wsq > -self.moment_tol * wsq_scale:
            new_wsq = 0.0

        self._total_weight = new_w
        self._offset_sum = new_s
        self._m2 = new_m2
        self._weight_sq_sum = new_wsq
        return self
