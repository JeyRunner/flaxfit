from typing import Protocol, Any, Dict

import jaxtyping
from flax import nnx
from flax.nnx import filterlib

from flaxfit.converters_and_functions import LossEntry
from flaxfit.logger.util import squeeze_to_scalar
from flaxfit.train_state import (
    TrainStateWithMetrics, AverageMetric,
)
from flaxfit.train_state_flax import TrainStateFlax
from flaxfit.train_state import TrainState, MetricsWithLoss
import jax.numpy as jnp


class ModuleForwardAndLossFunction(Protocol):
    def __call__(
        self,
        model: nnx.Module,
        batch,
        **model_foward_and_loss_kwargs
    ) -> tuple[dict, dict] | tuple[dict, dict, Any]:
        """
        Pass batch through the model and calc the loss terms.
        This is a stateful function that will change the passed model.
        :param model: the model as merged nnx module, do calls to this module.
        :return: (loss terms as dict, metric terms as dict, optional output e.g. model raw output)
        """


def get_loss_sum_and_dict_from_loss_terms(loss_terms: Dict[str, LossEntry | float]):
    loss_dict = {}
    if isinstance(loss_terms, dict):
        loss_sum = 0.0
        for key, value in loss_terms.items():
            if isinstance(value, LossEntry):
                loss_value = squeeze_to_scalar(value.value)
                loss_sum += value.weight * loss_value
                loss_dict[key] = loss_value
            else:
                loss_value = squeeze_to_scalar(value)
                loss_sum += loss_value
                loss_dict[key] = loss_value
    else:
        loss_sum = squeeze_to_scalar(loss_terms)
    # include loss sum in loss dict for metrics
    loss_dict['total'] = loss_sum
    return loss_sum, loss_dict


class TrainUpdateStepFunction(Protocol):
    """
    Get the current state and batch to process, calls the model_forward_and_loss_fn
    and does one update step based on the gradient of the loss.
    :return: new state, loss_additional_out
    """
    def __call__(self, state: TrainStateWithMetrics, batch: jaxtyping.PyTree, model_foward_and_loss_kwargs: dict = {}) -> tuple[TrainStateWithMetrics, jaxtyping.PyTree]:
        ...


def make_train_update_step(model_forward_and_loss_fn: ModuleForwardAndLossFunction, update_params_filter=nnx.Param) -> TrainUpdateStepFunction:
    """
    Make the update step function. This function gets the current state and batch to process, calls the model_forward_and_loss_fn
    and does one update step based on the gradient of the loss. It returns the new state, loss_additional_out.
    :return: train_update_step function
    """
    def loss_fn(model: nnx.Module, batch, model_call_kwargs):
        loss = model_forward_and_loss_fn(model, batch, **model_call_kwargs)
        loss_additional_out = None
        assert isinstance(loss, tuple)
        if len(loss) > 2:
            loss_dict, metrics_dict, loss_additional_out = loss
        else:
            loss_dict, metrics_dict = loss
        loss, loss_dict = get_loss_sum_and_dict_from_loss_terms(loss_dict)
        return loss, (loss_dict, metrics_dict, loss_additional_out)

    def train_update_step(state: TrainStateWithMetrics, batch: jaxtyping.PyTree, model_foward_and_loss_kwargs = {}):
        train_state: TrainStateFlax = state.train_state
        model = train_state.as_model()

        loss_val_and_grad = nnx.value_and_grad(loss_fn, has_aux=True, argnums=nnx.DiffState(0, filter=update_params_filter))
        (loss, (loss_dict, metrics_dict, loss_additional_out)), grads = loss_val_and_grad(model, batch, model_foward_and_loss_kwargs)

        # update params and put back into train_state
        model_graph_def, params, model_state = nnx.split(model, nnx.Param, filterlib.Everything())
        train_state = state.train_state.apply_gradients(grads)
        train_state = train_state.update_model_state(model_state)
        state = state.replace(train_state=train_state)

        # update the metrics
        state = state.replace(
          train_state=train_state, metrics_train=state.metrics_train.update(loss_dict, metrics_dict)
        )

        return state, loss_additional_out

    return train_update_step


def make_initial_train_state_with_metrics(
    train_state: TrainState,
    loss_keys: list[str], metrics_keys: list[str] = {},
    metrics_train_epoch: dict = None,
    info: dict = None,
    use_eval: bool = True
):
    """
    Make the initial train state with metrics.
    """
    def make_metrics(make_empty = False):
        return MetricsWithLoss(
            loss=AverageMetric.create({} if make_empty else {k: jnp.zeros((), dtype=float) for k in loss_keys}),
            metrics=AverageMetric.create({} if make_empty else {k: jnp.zeros((), dtype=float) for k in metrics_keys})
        )
    state = TrainStateWithMetrics(
        train_state=train_state,
        metrics_train=make_metrics(),
        metrics_eval=make_metrics(make_empty=not use_eval),
        metrics_train_epoch=metrics_train_epoch,
        info=info
    )
    return state