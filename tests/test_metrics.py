from unittest import TestCase

import chex
import jax.numpy as jnp

from flaxfit.train_state import AverageMetric


class TestMetrics(TestCase):
    def test_Avg(self):
        m = AverageMetric.create(dict(a=jnp.array([1]), b=jnp.array([1])))
        chex.assert_trees_all_equal(m.value_sum, dict(a=0, b=0))
        m = m.update(dict(a=10, b=100))
        m = m.update(dict(a=20, b=200))
        chex.assert_trees_all_equal(m.value_sum, dict(a=30, b=300))
        chex.assert_trees_all_equal(m.collect(), dict(a=15, b=150))
