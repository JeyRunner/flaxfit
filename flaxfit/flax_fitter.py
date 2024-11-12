import math
from typing import Protocol, Union, Literal, Any

import chex
import jax
import flax
import jax.numpy as jnp
import jaxtyping
import optax
from batchix.batching import (
    pytree_sub_index_each_leaf,
    pytree_split_in_batches_with_remainder,
    pytree_combine_batches,
)
from batchix.tree_shape import (
    pytree_get_shape_first_axis_equal,
    pytree_get_shape_first_n_equal,
)
from flax import nnx, struct
from flax.nnx import filterlib
from jaxtyping import Integer, Array, Float

from flaxfit.converters_and_functions import LossFunction, MetricsFunction, EpochBatchSplitter, BatchConverter
from flaxfit.dataset import Dataset
from flaxfit.fitter import ModelFitter
from flaxfit.train_state import (
    TrainStateWithMetrics,
    Metric,
    TrainState,
    ModelForwardFn,
)
from flaxfit.train_state_flax import TrainStateFlax


class CallModuleFunction(Protocol):
    def __call__(
        self,
        model: nnx.Module,
        batch_x,
    ) -> Any:
        """
        Custom call to the module.
        This is a stateful function that will change the passed model.
        :param model: the model as merged nnx module, do calls to this module.
        :return: the model predictions for the batch_x, that will be passed to the loss function
        """


class FlaxModelFitter(ModelFitter):
    """
    Abstract Model fitter contains general training loop code.
    """

    def __init__(self, loss_function: LossFunction = None, metrics_function: MetricsFunction = None,
                 call_model_function: CallModuleFunction = None,
                 update_batch_size=256, evaluate_batch_size=None, batch_converter: BatchConverter = None,
                 epoch_batch_splitter: EpochBatchSplitter = None):
        super().__init__(loss_function, metrics_function, update_batch_size, evaluate_batch_size, batch_converter,
                         epoch_batch_splitter)
        self.call_model_function = call_model_function


    @staticmethod
    def create_train_state(model: nnx.Module, optimizer=optax.adam(learning_rate=0.0002)) -> TrainStateFlax:
        """
        Creates an initial train state.
        """
        model.train()  # set deterministic=False
        graphdef, params, model_state = nnx.split(model, nnx.Param, filterlib.Everything())
        return TrainStateFlax.create(
            graphdef=graphdef,
            params=params,
            model_state=model_state,
            tx=optimizer
        )


    def make_model_forward_fn(self, train_state: TrainStateFlax) -> ModelForwardFn:
        def forward(params, model_state, batch):
            # recombine to model, do calls to model which changes the model state, split the model again into states
            model = nnx.merge(train_state.graphdef, params, model_state)
            if self.call_model_function is None:
                prediction = model(batch)
            else:
                prediction = self.call_model_function(model, batch)
            model_graph_def, params, model_state_new = nnx.split(model, nnx.Param, filterlib.Everything())
            return prediction, model_state_new
        return forward

