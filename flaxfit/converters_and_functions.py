import math
from typing import Protocol, Union, Literal, Any, Optional, Callable, Dict, Tuple

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
from jaxtyping import Integer, Array, Float

from flaxfit.dataset import Dataset
from flaxfit.train_state import (
    TrainStateWithMetrics,
    Metric,
    TrainState,
    ModelForwardFn,
)


class ModelCallBatchConverter:
    """
    Converts a batch directly before it is passed to the model.
    Note that this happens within the __model_forward_and_loss function of which we compute the gradient,
    w.r.t. the model parameters.
    This converter is can be used for generally converting data passed to the model, not just for training it.
    """

    def __init__(self, number_of_batch_dimension: int = 1):
        """
        :param number_of_batch_dimension:
            if this is bigger than 1 the additional dimensions will be flatted before being passed to the model
            (this data may come from convert_batch(...)).
            And flatted in convert_model_output_back_before_loss(...) before being passed to the loss function.
        """
        self.number_of_batch_dimension = number_of_batch_dimension

    def convert_batch(self, batch: Dataset) -> Dataset:
        """
        Is called for each batch in an epoch before passing the batch to the model.
        This allows to convert the batch data before being passed to the model.
        For example when the training set has the shape (num_examples, num_sub_examples, dims),
        this function may convert each batch (batch_size, num_sub_examples, dims) from the dataset to (batch_size*num_sub_examples, dims).
        Keep in mind that in the case the real batch size is batch_size*num_sub_examples.
        :param batch: the batch to convert
        """
        return batch

    def convert_model_output_back_before_loss(self, model_output_y) -> Any:
        """
        Directly converts the model output after calling model forward.
        This transformed output is provided to the loss function and to convert_model_output_back(...)
        """
        return model_output_y

    def _batch_to_model_input(self, batch):
        batch_non_flatted = self.convert_batch(batch)
        batch_unflatten_shape = None
        if self.number_of_batch_dimension > 1:
            flatten_n = self.number_of_batch_dimension
            batch_unflatten_shape = pytree_get_shape_first_n_equal(batch_non_flatted, flatten_n)
            batch = jax.tree_util.tree_map(lambda x: jnp.reshape(x, (-1,) + x.shape[flatten_n:]), batch_non_flatted)
        return batch, batch_non_flatted, batch_unflatten_shape

    def _model_output_convert(self, batch_unflatten_shape, model_output_y):
        if self.number_of_batch_dimension > 1:
            model_output_y = jax.tree_util.tree_map(
                lambda y: jnp.reshape(y, batch_unflatten_shape + y.shape[1:]),
                model_output_y
            )
        return self.convert_model_output_back_before_loss(model_output_y)


class DatasetBatchConverter:
    """
    Converter used during training for the eval and train set.
    This converter is only meant to convert batch data used during training.
    """

    def convert_batch(self, batch: Dataset, rng_key: jax.random.PRNGKey) -> Dataset:
        """
        After splitting the dataset into batches in a training epoch, this function is called for each batch.
        This function converts a batch (e.g. adds noise to each element)
        before the batch is passed to the train_step function (which calculates the loss and updates the gradient).
        :param batch: the batch to convert
        :param rng_key: a random key that changes for each batch, can be used to e.g. add noise to the batch.
        """
        return batch


class EpochBatchSplitter(Protocol):
    def __call__(
        self, key: jax.random.PRNGKey, train_dataset: Dataset, batch_size: int
    ) -> Dataset:
        """
        Split the dataset into batches used for training in the following epoch.
        This also allows to randomize the ordering.
        Note if shuffle_dataset_before_each_epoch is true, the dataset is shuffled before being passed to this function.

        :param key: key to use for randomization
        :param train_dataset:
        :return: train_dataset but with an additional first dimension with the size of batch_size
        """


@flax.struct.dataclass
class DatasetAndModelPredictions[D, D_CONVERT, PRED]:
    dataset: D
    dataset_converted_to_model_input: D
    """
    Dataset after being converted by the batch converter. 
    If not batch converter is used then this is equal to dataset.
    """

    model_predictions: PRED


class EpochCallbackFunction(Protocol):
    def __call__(
        self,
        epoch: int, metrics: dict,
        train_model_predictions: DatasetAndModelPredictions,
        eval_model_predictions: DatasetAndModelPredictions,
        train_state: TrainState
    ) -> None | bool:
        """
        Callback after running some number of epochs and doing the evaluation.
        Here all elements in predictions and labels are valid.
        :param epoch: the current epoch.
        :param metrics: the calculated metrics, averaged over all the previous epochs till the last call to this function.
        :param train_model_predictions: A portion of the train dataset and the model output for it.
                                Set the dataset entries used here by epoch_callback_pass_train_dataset_prediction_idx.
                                Note that when there is randomness in sampling/converting the batches used for training,
                                this dataset will not be equivalent to that used in training (since the random keys are computed separately)
        :param eval_model_predictions: A portion of the eval dataset and the model output for it.
                                Set the dataset entries used here by epoch_callback_pass_train_dataset_prediction_idx.
        :return: Nothing or a boolean that indicates if the training should be continued.
        """


@flax.struct.dataclass
class LossEntry:
    value: Float[Array, ""]
    weight: float = flax.struct.field(pytree_node=False)


class LossFunction(Protocol):
    def __call__(
        self, predictions_y, dataset: Dataset, model=None
    ) -> Union[Float[Array, ""], dict[str, LossEntry | Float[Array, ""]]]:
        """
        Return the loss for given batch of model predictions and dataset (contains labels).
        :param model: The model class where the model parameters can be used in the loss.
                        The model should not be called (since the model state will not be updated!).
        :return: either a single loss or a dict of individual loss values optionally with weights (these will be summed)
        """


class MetricsFunction(Protocol):
    def __call__(
        self, predictions_y, dataset: Dataset
    ) -> dict[str, Float[Array, ""]] | Metric:
        """
        Return a dict of metric scalar values for given batch of model predictions and dataset (contains labels).
        """



class PassBatchThroughModelAndUpdateStateFn(Protocol):
    """Protocol for the function that processes a batch through the model and updates the state."""
    def __call__(self, state: TrainStateWithMetrics, batch: Dataset, model_call_kwargs: Dict) -> Tuple[TrainStateWithMetrics, Dict, Dict, jaxtyping.PyTree]:
        """
        Given state, batch and model args. Return new state and loss and metric values, also the second tuple value returned by the model.
        """

class BatchProcessStepFunction(Protocol):
    def __call__(
        self,
        state: TrainStateWithMetrics,
        batch: Dataset,
        pass_batch_through_model_and_update_state_fn: PassBatchThroughModelAndUpdateStateFn
    ) -> tuple[TrainStateWithMetrics, dict, dict]:
        """
        This function is used for doing one train batch update or one evaluation of the given batch.
        The function inner_batch_process_fn is called to change the batch based on the batch (either update model params or just eval metrics).
        Train/Evaluate for a single step given (or multiple) the input data x and labels y.
        The function should internally call pass_batch_through_model_and_update_state_fn one or multiple times.
        :param update_model_states: enable updates for all model states except the params (params are allays updated).
        :return: (new state, loss_dict, metrics_dict) Note that the value of loss_dict, metrics_dict are not used.
                    Just their pytree structure is used to infer the initial loss and metrics keys/values types (thus the shape also does not matter).
                    The metrics are updated by pass_batch_through_model_and_update_state_fn.
        """
        model_call_kwargs = {}
        state = pass_batch_through_model_and_update_state_fn(
            state, batch, model_call_kwargs
        )
        return state

class BatchProcessStepFunctionDefault(BatchProcessStepFunction):
    pass