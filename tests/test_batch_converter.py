from unittest import TestCase

import chex
import jax
import jax.numpy as jnp

from flaxfit.converters_and_functions import ModelCallBatchConverter
from flaxfit.dataset import Dataset
from flaxfit.train_state import AverageMetric
from tests.common import ExampleData


class TestBatchConverter(TestCase):
    def test_2_batch_dims(self):
        converter = ModelCallBatchConverter(number_of_batch_dimension=2)

        second_batch_dim_size = 5
        base_data = jnp.arange(second_batch_dim_size)[:, jnp.newaxis]
        data = ExampleData(
            a=jnp.stack([
                base_data,
                base_data*10,
                base_data*100,
            ]),
            b=jnp.stack([
                base_data,
                base_data+1,
                base_data+2,
            ]),
        )

        data_flatted, batch_non_flatted, batch_unflatten_shape = converter._batch_to_model_input(data)
        print(data_flatted)
        chex.assert_tree_shape_prefix(data_flatted, (second_batch_dim_size*3,))
        
        # change b
        data_flatted = data_flatted.replace(b=jnp.stack([data_flatted.b, jnp.ones(data_flatted.b.shape)], axis=-1))
        print(data_flatted)
        
        data_unflatted = converter._model_output_convert(batch_unflatten_shape, data_flatted)

        chex.assert_trees_all_equal(data.a, data_unflatted.a)
        chex.assert_trees_all_equal(
            jnp.stack([data.b, jnp.ones(data.b.shape)], axis=-1),
            data_unflatted.b
        )
        print(jnp.stack([data.b, jnp.ones(data.b.shape)], axis=-1))


