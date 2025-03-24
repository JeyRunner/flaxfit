import math
from typing import Protocol, Union, Literal, Any, Callable

import chex
import jax
import flax
import jax.numpy as jnp
import jaxtyping
import optax
import rich
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

from flaxfit.converters_and_functions import LossFunction, MetricsFunction, EpochBatchSplitter, ModelCallBatchConverter, \
    DatasetBatchConverter, BatchProcessStepFunction, BatchProcessStepFunctionDefault
from flaxfit.dataset import Dataset
from flaxfit.fitter import ModelFitter
from flaxfit.train_state import (
    TrainStateWithMetrics,
    Metric,
    TrainState,
    ModelForwardFn, ModelFromTrainStateFn,
)
from flaxfit.train_state_flax import TrainStateFlax


class CallModuleFunction(Protocol):
    def __call__(
        self,
        model: nnx.Module,
        batch_x,
        **model_call_kwargs
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
                 update_batch_size=256, evaluate_batch_size=None, model_call_batch_converter: ModelCallBatchConverter = None,
                 dataset_batch_converter: DatasetBatchConverter = None,
                 epoch_batch_splitter: EpochBatchSplitter = None,
                 batch_process_step_function: BatchProcessStepFunction = BatchProcessStepFunctionDefault(),
                 ):
        """
        :param loss_function: returns the loss for a given batch and model output.
        :param metrics_function: returns the metrics for a given batch and model output.
        :param update_batch_size: batch size for passing the train dataset through the model during training.
        :param evaluate_batch_size: batch size for passing the evaluation dataset through the model and for general inference.
        :param model_call_batch_converter: converts batches within the model forward function, also used during inference.
        :param dataset_batch_converter: converts batches of the dataset used just during training (train and eval sets).
        :param epoch_batch_splitter:
        :param batch_process_step_function: Defines how the model update(s) and evals are handled for each batch.
        """
        super().__init__(
            loss_function, metrics_function, update_batch_size, evaluate_batch_size, model_call_batch_converter,
            dataset_batch_converter, epoch_batch_splitter,
            batch_process_step_function=batch_process_step_function
        )
        self.call_model_function = call_model_function


    @staticmethod
    def create_train_state(
        model: nnx.Module,
        optimizer=optax.adam(learning_rate=0.0002),
        optimizer_only_update: filterlib.Filter = nnx.Param
    ) -> TrainStateFlax:
        """
        Creates an initial train state.
        """
        model.train()  # set deterministic=False
        graphdef, params, model_state = nnx.split(model, nnx.Param, filterlib.Everything())
        return TrainStateFlax.create(
            graphdef=graphdef,
            params=params,
            model_state=model_state,
            tx=optimizer,
            wrt=optimizer_only_update
        )


    def make_model_from_train_state_fn(self, train_state: TrainStateFlax) -> ModelFromTrainStateFn[nnx.Module]:
        def model_from_train_state(params, model_state):
            # recombine to model, do calls to model which changes the model state, split the model again into states
            model = nnx.merge(train_state.graphdef, params, model_state)
            return model
        return model_from_train_state

    def make_model_forward_fn(self, train_state: TrainStateFlax) -> ModelForwardFn:
        def forward(params, model_state, batch, model: nnx.Module = None, model_call_kwargs: dict = {}):
            # recombine to model, do calls to model which changes the model state, split the model again into states
            if model is None:
                model = nnx.merge(train_state.graphdef, params, model_state)
            if self.call_model_function is None:
                prediction = model(batch)
            else:
                prediction = self.call_model_function(model, batch, **model_call_kwargs)
            model_graph_def, params, model_state_new = nnx.split(model, nnx.Param, filterlib.Everything())
            return prediction, model_state_new
        return forward



    @staticmethod
    def get_param_count_of_model(model) -> int:
        graphdef, params, rest = nnx.split(model, nnx.Param, filterlib.Everything())
        return FlaxModelFitter.get_param_count_of_model_params(params)


    @staticmethod
    def get_param_count_of_model_params(model_params) -> int:
        # jax.tree_util.tree_map(lambda leaf: print(leaf.shape), params)
        num_params = jax.tree_util.tree_reduce(lambda acc, leaf: acc + math.prod(leaf.shape), model_params, initializer=0)
        return num_params

    @staticmethod
    def get_param_shapes(model_params) -> int:
        return jax.tree_util.tree_map(lambda leaf: leaf.shape, model_params)



    def _model_forward_and_loss(
        self,
        model_params: jaxtyping.PyTree,
        model_state: jaxtyping.PyTree,
        model_forward_fn: ModelForwardFn,
        model_from_train_state_fn: ModelFromTrainStateFn,
        batch: Dataset,
        model: nnx.Module = None,
        model_call_kwargs: dict = {}
    ):
        """
        Pass batch through the model and calculate loss and metrics.
        If model is given all other args except for batch are ignored. This is used for gradient calc with nnx.grad.
        :param model_params:
        :param model_state:
        :param model_forward_fn:
        :param batch:
        :return: loss, (prediction, model_state, loss_dict, metrics)
        """
        batch_non_flatted = batch
        if self.model_call_batch_converter is not None:
            batch, batch_non_flatted, batch_unflatten_shape = (
                self.model_call_batch_converter._batch_to_model_input(batch)
            )

        if model is None:
            # pass through model
            prediction, model_state = model_forward_fn(
                model_params, model_state, self.batch_to_model_input(batch), model_call_kwargs=model_call_kwargs
            )
        else:
            # pass through model nnx style
            prediction, model_state = model_forward_fn(
                # params, model_state, batch, model
                params=None, model_state=None,
                batch=self.batch_to_model_input(batch), model=model,
                model_call_kwargs=model_call_kwargs
            )
            # note that the model_state coming from model_forward_fn will not be used later


        if self.model_call_batch_converter is not None:
            prediction = self.model_call_batch_converter._model_output_convert(
                batch_unflatten_shape, prediction
            )

        if model is None:
            model_after_forward = model_from_train_state_fn(model_params, model_state)
        else:
            model_after_forward = model

        loss = self.loss_function(prediction, batch_non_flatted, model_after_forward)
        loss, loss_dict = self._get_loss_sum_and_dict_from_loss(loss)
        metrics = {}
        if self.metrics_function is not None:
            metrics = self.metrics_function(prediction, batch_non_flatted)
        return loss, (prediction, model_state, loss_dict, metrics)



    def train_update_step__one_model_grad_step(
        self,
        state: TrainStateWithMetrics,
        batch: Dataset,
        update_model_states=True,
        model_call_kwargs: dict = {}
    ):
        """
        Train for a single step given the input data x and labels y.
        :param ignore_last_n_elements_in_batch: exclude these last n elements from calculating the loss.
        """
        # if we don't want to update all model params we need to use nnx.grad
        wrt = state.train_state.wrt #filterlib.All(nnx.Param, filterlib.PathContains('layers_out'))
        use_nnx_grad = wrt != nnx.Param
        print(f'> flaxfit using {"default jax.grad" if (not use_nnx_grad) else "nnx.grad (required since not all params are updated)"} for updating parameters')

        # normal grad with normal jax model call
        if not use_nnx_grad:
            (loss, (prediction, model_state, loss_dict, metrics)), grads = jax.value_and_grad(
                self._model_forward_and_loss, has_aux=True
            )(
                state.train_state.params,
                state.train_state.model_state,
                self.make_model_forward_fn(state.train_state),
                self.make_model_from_train_state_fn(state.train_state),
                batch,
                None,
                model_call_kwargs
            )
        else:
            model = state.train_state.as_model()
            (loss, (prediction, __model_state, loss_dict, metrics)), grads = nnx.value_and_grad(
                self._model_forward_and_loss, has_aux=True, argnums=nnx.DiffState(5, filter=wrt)
            )(
                None,  # params
                None,  # model_state
                self.make_model_forward_fn(None),
                None,  # make_model_from_train_state_fn
                batch,
                model,
                model_call_kwargs
            )
            model_graph_def, params, model_state = nnx.split(model, nnx.Param, filterlib.Everything())

        # rich.print(grads)

        train_state = state.train_state.apply_gradients(grads)
        # just update batchstats when the full batch is valid, otherwise the calculated batch stats are not accurate
        if update_model_states:
            train_state = train_state.update_model_state(model_state)

        # update the metrics
        state = state.replace(
            train_state=train_state, metrics_train=state.metrics_train.update(loss_dict, metrics)
        )
        return state, loss, prediction
