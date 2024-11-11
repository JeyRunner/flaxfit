import inspect
import math
from functools import partial
from typing import Protocol, Union, Literal, Callable, Any, Tuple, Dict

import batchix.vmap_scan
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
from batchix.vmap_scan import scan_batched
from flax import nnx, struct
from jaxtyping import Integer, Array, Float

from flaxfit.callbacks.call_sub_args import call_fn_just_with_defined_args
from flaxfit.converters_and_functions import LossFunction, MetricsFunction, BatchConverter, EpochBatchSplitter, \
    LossEntry, EpochCallbackFunction, DatasetAndModelPredictions
from flaxfit.dataset import Dataset, DatasetTyped
from flaxfit.logger.util import squeeze_to_scalar
from flaxfit.train_state import (
    TrainStateWithMetrics,
    Metric,
    TrainState,
    ModelForwardFn, MetricsWithLoss, AverageMetric,
)



class ModelFitter:
    """
    Abstract Model fitter contains general training loop code.
    """

    def __init__(
        self,
        loss_function: LossFunction = None,
        metrics_function: MetricsFunction = None,
        update_batch_size=256,
        evaluate_batch_size=None,
        batch_converter: BatchConverter = None,
        epoch_batch_splitter: EpochBatchSplitter = None,
    ):
        self.update_batch_size = update_batch_size
        self.evaluate_batch_size = (
            evaluate_batch_size
            if evaluate_batch_size is not None
            else update_batch_size
        )
        self.batch_converter = batch_converter
        self.epoch_batch_splitter = epoch_batch_splitter

        self.loss_function = loss_function
        self.metrics_function = metrics_function


    def batch_to_model_input(self, batch: Dataset):
        """
        Get the data that is passed into the model from a batch of the dataset.
        Overwrite this method when using other dataset types.
        """
        return batch.x


    def make_model_forward_fn(self, train_state: TrainState) -> ModelForwardFn:
        raise NotImplementedError()


    def __get_loss_sum_and_dict_from_loss(self, loss_fn_output):
        loss_dict = {}
        if isinstance(loss_fn_output, dict):
            loss_sum = 0.0
            for key, value in loss_fn_output.items():
                if isinstance(value, LossEntry):
                    loss_value = squeeze_to_scalar(value.value)
                    loss_sum += value.weight * loss_value
                    loss_dict[key] = loss_value
                else:
                    loss_value = squeeze_to_scalar(value)
                    loss_sum += loss_value
                    loss_dict[key] = loss_value
        else:
            loss_sum = squeeze_to_scalar(loss_fn_output)
        # include loss sum in loss dict for metrics
        loss_dict['total'] = loss_sum
        return loss_sum, loss_dict


    def __model_forward_and_loss(
        self,
        model_params: jaxtyping.PyTree,
        model_state: jaxtyping.PyTree,
        model_forward_fn: ModelForwardFn,
        batch: Dataset,
    ):
        """
        Pass batch through the model and calculate loss and metrics
        :param model_params:
        :param model_state:
        :param model_forward_fn:
        :param batch:
        :return: loss, (prediction, model_state, loss_dict, metrics)
        """
        batch_non_flatted = batch
        if self.batch_converter is not None:
            batch, batch_non_flatted, batch_unflatten_shape = (
                self.batch_converter._batch_to_model_input(batch)
            )
        # pass through model
        prediction, model_state = model_forward_fn(
            model_params, model_state, self.batch_to_model_input(batch)
        )

        if self.batch_converter is not None:
            prediction = self.batch_converter._model_output_convert(
                batch_unflatten_shape, prediction
            )
        loss = self.loss_function(prediction, batch_non_flatted)
        loss, loss_dict = self.__get_loss_sum_and_dict_from_loss(loss)
        metrics = {}
        if self.metrics_function is not None:
            metrics = self.metrics_function(prediction, batch_non_flatted)
        return loss, (prediction, model_state, loss_dict, metrics)


    def train_update_step(
        self,
        state: TrainStateWithMetrics,
        batch: Dataset,
        update_model_states=True,
    ):
        """
        Train for a single step given the input data x and labels y.
        :param ignore_last_n_elements_in_batch: exclude these last n elements from calculating the loss.
        """
        (loss, (prediction, model_state, loss_dict, metrics)), grads = jax.value_and_grad(
            self.__model_forward_and_loss, has_aux=True
        )(
            state.train_state.params,
            state.train_state.model_state,
            self.make_model_forward_fn(state.train_state),
            batch,
        )
        train_state = state.train_state.apply_gradients(grads)
        # just update batchstats when the full batch is valid, otherwise the calculated batch stats are not accurate
        if update_model_states:
            train_state = train_state.update_model_state(model_state)

        # update the metrics
        state = state.replace(
            train_state=train_state, metrics_train=state.metrics_train.update(loss_dict, metrics)
        )
        return state, loss, prediction


    def train_epoch(
        self,
        state: TrainStateWithMetrics,
        dataset: Dataset,
        shuffle: bool,
        rng: jax.random.PRNGKey,
        sample_each_batch_independent_from_dataset: bool = False,
        batch_remainder_strategy: Literal[
            "None", "PadValidSampled", "ExtraLastBatch"
        ] = "ExtraLastBatch",
    ) -> TrainStateWithMetrics:
        """
        Does training steps over all the batches over the given dataset.
        :param state: The current train state.
        :param dataset: train dataset
        :param shuffle: shuffle dataset with rng key before splitting it into batches.
        :param sample_each_batch_independent_from_dataset:
            instead of scanning over the whole dataset sample each batch independently.
            This may resul in not all samples of the dataset being processed in one epoch.
            This implicitly shuffles the dataset.
        :param batch_remainder_strategy: How to handle the case if the dataset is not dividable by the batch size.
            'None': fail if dataset is not dividable by batch size.
            'PadValidSampled': Pad missing elements in last batch with random samples from dataset.
            'ExtraLastBatch': Do a separate last pass and gradient though the model with a last smaller batch.
                        For the last batch only change the model params via the gradient but not the other model states.
        :return: new training state
        """
        dataset_size = pytree_get_shape_first_axis_equal(dataset)
        # sample independent batches from dataset
        if sample_each_batch_independent_from_dataset:
            assert (
                self.batch_converter is None
            ), "when sample_each_batch_independent_from_dataset batch_converter is not supported"
            rng, sample_key = jax.random.split(rng)
            num_batches = math.ceil(dataset_size / self.update_batch_size)

            # go over all batches
            def train_update_step(carry, x):
                state, sample_key = carry
                sample_key, _ = jax.random.split(sample_key)
                batch_idx = jax.random.choice(
                    sample_key, dataset_size, shape=(self.update_batch_size,)
                )
                batch = pytree_sub_index_each_leaf(dataset, batch_idx)
                # train on batch:
                state, loss, prediction = self.train_update_step(
                    state, batch, update_model_states=True
                )
                return (state, sample_key), None

            (state, _), _ = jax.lax.scan(
                train_update_step, init=(state, sample_key), length=num_batches
            )
            return state

        # split dataset into batches
        if shuffle:
            # shuffle dataset before splitting into batches
            dataset_randomized_order = jax.random.permutation(
                rng, jnp.arange(dataset_size)
            )
            dataset = pytree_sub_index_each_leaf(dataset, dataset_randomized_order)

        last_batch_remainder = None
        rng, key_split = jax.random.split(rng)
        # do normal shuffling and splitting
        if self.epoch_batch_splitter is None:
            # split the dataset into batches
            dataset_batched, last_batch_remainder = (
                pytree_split_in_batches_with_remainder(
                    dataset,
                    batch_size=self.update_batch_size,
                    batch_remainder_strategy=batch_remainder_strategy,
                    rng_key=key_split,
                )
            )
        else:
            # custom splitting
            dataset_batched = self.epoch_batch_splitter(
                key_split, dataset, batch_size=self.update_batch_size
            )
        # check if we have valid batched dataset
        num_batches, batch_size = pytree_get_shape_first_n_equal(dataset_batched, 2)
        chex.assert_tree_shape_prefix(
            dataset_batched,
            (
                num_batches,
                self.update_batch_size,
            ),
        )

        # go over all batches
        def train_update_step(carry, batch, update_model_states=True):
            state = carry
            # train on batch:
            state, loss, prediction = self.train_update_step(
                state, batch, update_model_states=update_model_states
            )
            return state, prediction

        state, predictions = jax.lax.scan(
            train_update_step, init=state, xs=dataset_batched
        )

        predictions_extra_last_batch = None
        if last_batch_remainder is not None:
            print(
                f"> WARN: dataset size {dataset_size} is not dividable by batch size {self.update_batch_size}, "
                "will do separate last call to the model for last batch "
                "(this will just update the model params but not the other model states, since batch size is smaller)"
            )
            # just update model params but not other model states, since here the batch size is different
            state, predictions_extra_last_batch = train_update_step(
                state, last_batch_remainder, update_model_states=False
            )

        # recombine predictions from all batches
        # not needed anymore since we are not returning predictions
        predictions = pytree_combine_batches(
            predictions, batch_remainder=predictions_extra_last_batch
        )

        # reconstruct order before shuffle, currently unused
        if shuffle and self.epoch_batch_splitter is None and batch_remainder_strategy != 'PadValidSampled':
            predictions = jax.tree_util.tree_map(
                lambda x: x.at[dataset_randomized_order].set(x), predictions
            )
        return state


    @partial(jax.jit, static_argnames=[
        'self', 'shuffle', 'num_epochs', 'sample_each_batch_independent_from_dataset', 'batch_remainder_strategy'
    ])
    def train_epochs(
        self,
        state: TrainStateWithMetrics,
        dataset_train: Dataset,
        shuffle: bool,
        rng: jax.random.PRNGKey,
        num_epochs: int,
        sample_each_batch_independent_from_dataset: bool = False,
        batch_remainder_strategy: Literal[
            "None", "PadValidSampled", "ExtraLastBatch"
        ] = "ExtraLastBatch",
    ) -> TrainStateWithMetrics:
        """
        Do one or multiple training epochs.
        :param state: The current train state.
        :param dataset_train: train dataset
        :param shuffle: shuffle dataset with rng key before splitting it into batches.
        :param sample_each_batch_independent_from_dataset:
            instead of scanning over the whole dataset sample each batch independently.
            This may resul in not all samples of the dataset being processed in one epoch.
            This implicitly shuffles the dataset.
        :param batch_remainder_strategy: How to handle the case if the dataset is not dividable by the batch size.
            'None': fail if dataset is not dividable by batch size.
            'PadValidSampled': Pad missing elements in last batch with random samples from dataset.
            'ExtraLastBatch': Do a separate last pass and gradient though the model with a last smaller batch.
                        For the last batch only change the model params via the gradient but not the other model states.
        :param num_epochs: Number of epochs to train.
        :return: new training state
        """
        def train_epoch_step(carry, x):
            nonlocal dataset_train
            state, rng = carry
            rng, rng_shuffle_dataset = jax.random.split(rng, 2)

            state = self.train_epoch(
                state=state, dataset=dataset_train, shuffle=shuffle,
                batch_remainder_strategy=batch_remainder_strategy,
                sample_each_batch_independent_from_dataset=sample_each_batch_independent_from_dataset,
                rng = rng_shuffle_dataset
            )
            return (state, rng), None

        # loop over training epochs
        (state, _), _ = jax.lax.scan(
            train_epoch_step,
            init=(state, rng),
            length=num_epochs
        )
        return state


    @partial(jax.jit, static_argnames=['self'])
    def evaluate(
        self,
        state: TrainStateWithMetrics,
        dataset_eval: Dataset,
    ) -> TrainStateWithMetrics:
        """
        Pass the evaluation dataset through the model and calc loss and metrics.
        :param state: The current train state.
        :param dataset_eval: evaluation dataset
        :return: new training state with just the eval metrics updated (model state is not changed).
        """
        def eval_step(carry, batch):
            state: TrainStateWithMetrics = carry
            loss, (prediction, model_state, loss_dict, metrics) = self.__model_forward_and_loss(
                state.train_state.params,
                state.train_state.model_state,
                model_forward_fn=self.make_model_forward_fn(state.train_state),
                batch=batch
            )
            # do not update the model state
            # just update the eval metrics
            # update the metrics
            state = state.replace(
                metrics_eval=state.metrics_eval.update(loss_dict, metrics)
            )
            return state, None

        state, _ = batchix.vmap_scan.scan_batched(
            eval_step,
            x=dataset_eval,
            fn_carry_init=state,
            batch_size=self.evaluate_batch_size,
            batch_remainder_strategy='None'
        )
        return state


    @partial(jax.jit, static_argnames=['self', 'batch_size'])
    def model_forward_dataset(
        self,
        train_state: TrainState,
        dataset: Dataset,
        batch_size: int = None
    ):
        """
        Pass batch through the model and return model predictions.
        :return: train_state, predictions, dataset_converted_to_model_input
        """
        if batch_size is None:
            batch_size = self.evaluate_batch_size

        def forward_batch(train_state: TrainState, batch):
            model_forward_fn = self.make_model_forward_fn(train_state)
            batch_non_flatted = batch
            if self.batch_converter is not None:
                batch, batch_non_flatted, batch_unflatten_shape = (
                    self.batch_converter._batch_to_model_input(batch)
                )
            # pass through model
            prediction, model_state = model_forward_fn(
                train_state.params, train_state.model_state, self.batch_to_model_input(batch)
            )

            if self.batch_converter is not None:
                prediction = self.batch_converter._model_output_convert(
                    batch_unflatten_shape, prediction
                )
            train_state = train_state.update_model_state(model_state)
            return train_state, (prediction, batch_non_flatted)

        train_state, (predictions, dataset_converted_to_model_input) = scan_batched(
            forward_batch,
            x=dataset,
            fn_carry_init=train_state,
            batch_size=batch_size,
            batch_remainder_strategy='ExtraLastBatch'
        )
        return train_state, predictions, dataset_converted_to_model_input


    def __dummy_forward_get_initial_metrics(self, train_state, dummy_dataset) -> tuple[MetricsWithLoss, MetricsWithLoss]:
        """
        Create the initial train and eval metrics by dummy calling the loss and metrics functions.
        """
        def forward(batch):
            (loss, (prediction, model_state, loss_dict, metrics)) = self.__model_forward_and_loss(
                train_state.params,
                train_state.model_state,
                self.make_model_forward_fn(train_state),
                batch,
            )
            return loss_dict, metrics
        loss_dict, metrics = jax.eval_shape(forward, pytree_sub_index_each_leaf(dummy_dataset, jnp.s_[:1]))
        assert isinstance(metrics, dict), \
            "current just a dictionary is supported as output from the metrics function (will calc avg value)"
        metrics_train = MetricsWithLoss(
            loss=AverageMetric.create(loss_dict),
            metrics=AverageMetric.create(metrics)
        )
        print('> initial metrics train:', metrics_train)
        # make a copy
        metrics_eval = metrics_train.replace()
        return metrics_train, metrics_eval


    def __collect_metrics_with_epoch_index(self, state: TrainStateWithMetrics, epoch_index: int):
        return dict(
            __epoch=epoch_index,
            train=state.metrics_train.collect(),
            eval=state.metrics_eval.collect()
        )



    def train_fit(
        self,
        initial_train_state: TrainState,
        dataset: DatasetTyped,
        num_epochs: int,
        evaluate_each_n_epochs: int,
        shuffle: bool = True,
        sample_each_batch_independent_from_dataset: bool = False,
        train_batch_remainder_strategy: Literal[
            "None", "PadValidSampled", "ExtraLastBatch"
        ] = "ExtraLastBatch",
        dataset_eval: DatasetTyped | None = None,
        epoch_callback_fn: EpochCallbackFunction | list[EpochCallbackFunction] | None = None,
        epoch_callback_pass_train_dataset_prediction_idx: Any | Integer[Array, 'num'] = None,
        epoch_callback_pass_eval_dataset_prediction_idx: Any | Integer[Array, 'num2'] = None,
        rng: jaxtyping.PRNGKeyArray = jax.random.PRNGKey(0),
        jit: bool = True
    ) -> Tuple[TrainState, Dict[str, Float[Array, 'num_epochs']]]:
        """
        Run the training for multiple epochs on the given train dataset.
        :param dataset: training dataset.
        :param dataset_eval: evaluation dataset.
        :param epoch_callback_fn: Callback function executed every 'evaluate_each_n_epochs' train epochs.
                    Can be used for logging.
                    Takes parameters: (epoch_i, a dict with the metrics, train predictions, eval predictions, train_state).
        :param epoch_callback_pass_train_dataset_prediction_idx:
            When given the provided indices of the train data will be passed through the model
            and given to the epoch_callback_fn as parameter
            This can be used to e.g. plot model predictions.
        :param epoch_callback_pass_eval_dataset_prediction_idx:
            When given the provided indices of the evaluation data will be passed through the model
            and given to the epoch_callback_fn as parameter
            This can be used to e.g. plot model predictions.

        :return: (The new train state that contains the new model weights, the metrics per epoch).
        """
        num_eval_steps = int(math.ceil(num_epochs / evaluate_each_n_epochs))

        # make initial state with metrics
        metrics_train, metrics_eval = self.__dummy_forward_get_initial_metrics(initial_train_state, dataset)
        state = TrainStateWithMetrics(
            train_state=initial_train_state,
            metrics_train=metrics_train,
            metrics_eval=metrics_eval
        )

        # init metrics over epochs
        metrics_over_epochs = jax.tree_util.tree_map(
            lambda x: jnp.zeros(num_eval_steps + 1) * jnp.nan, self.__collect_metrics_with_epoch_index(state, epoch_index=0)
        )



        def __make_callback_model_predictions(state, dataset, dataset_idx):
            if (dataset is None) or dataset_idx is None:
                return None
            callback_dataset = pytree_sub_index_each_leaf(dataset, dataset_idx)
            train_state, predictions, dataset_converted_to_model_input = self.model_forward_dataset(
                state.train_state,
                callback_dataset,
                batch_size=self.evaluate_batch_size
            )
            return DatasetAndModelPredictions(
                dataset=callback_dataset,
                dataset_converted_to_model_input=dataset_converted_to_model_input,
                model_predictions=predictions
            )


        # handle epoch callback function
        def handle_callback_and_rest_metrics(
            state: TrainStateWithMetrics, epoch: int, collected_metrics: dict
        ):
            def host_callback(
                epoch: int, metrics: dict,
                train_model_predictions: DatasetAndModelPredictions,
                eval_model_predictions: DatasetAndModelPredictions,
                train_state: TrainState
            ):
                args = dict(
                    epoch=epoch, metrics=metrics,
                    train_model_predictions=train_model_predictions,
                    eval_model_predictions=eval_model_predictions,
                    train_state=train_state
                )
                not_abort = True
                if isinstance(epoch_callback_fn, list):
                    for fn in epoch_callback_fn:
                        # only pass supported args
                        fn_return = call_fn_just_with_defined_args(fn, args)
                        if fn_return is not None:
                            not_abort &= fn_return
                else:
                    fn_return = call_fn_just_with_defined_args(epoch_callback_fn, args)
                    if fn_return is not None:
                        not_abort = fn_return
                assert isinstance(not_abort, bool)
                return not_abort

            not_abort = True
            if epoch_callback_fn is not None:
                train_model_predictions = __make_callback_model_predictions(
                    state, dataset, epoch_callback_pass_train_dataset_prediction_idx
                )
                # @todo
                eval_model_predictions = __make_callback_model_predictions(
                    state, dataset_eval, epoch_callback_pass_eval_dataset_prediction_idx
                )
                not_abort = jax.experimental.io_callback(
                    host_callback,
                    result_shape_dtypes=jnp.array(True),  # result dtype
                    epoch=epoch, metrics=collected_metrics,
                    train_model_predictions=train_model_predictions,
                    eval_model_predictions=eval_model_predictions,
                    train_state=state.train_state
                )

            return state.replace(
                metrics_train=state.metrics_train.reset(),
                metrics_eval=state.metrics_eval.reset()
            ), not_abort

        # initial eval
        state = self.evaluate(state, dataset_eval)
        collected_metrics = self.__collect_metrics_with_epoch_index(state, 0)
        metrics_over_epochs = jax.tree_util.tree_map(
            lambda m_o_e, m: m_o_e.at[0].set(m), metrics_over_epochs, collected_metrics
        )
        state, _ = handle_callback_and_rest_metrics(
            state, epoch=0, collected_metrics=collected_metrics
        )

        # train loop
        # for epoch_i in range(num_eval_steps):
        def train_for_n_epochs(carry):
            state, metrics_over_epochs, rng_train_epochs, main_loop_i, not_abort = carry
            rng_train_epochs, _ = jax.random.split(rng_train_epochs)

            state = self.train_epochs(
                state=state,
                dataset_train=dataset,
                shuffle=shuffle,
                rng=rng_train_epochs,
                num_epochs=evaluate_each_n_epochs,
                sample_each_batch_independent_from_dataset=sample_each_batch_independent_from_dataset,
                batch_remainder_strategy=train_batch_remainder_strategy
            )
            state = self.evaluate(
                state=state,
                dataset_eval=dataset_eval
            )

            # save collected metrics
            epoch = main_loop_i * evaluate_each_n_epochs
            collected_metrics = self.__collect_metrics_with_epoch_index(state, epoch)
            metrics_over_epochs = jax.tree_util.tree_map(
                lambda m_o_e, m: m_o_e.at[main_loop_i].set(m), metrics_over_epochs, collected_metrics
            )

            # does the callback and resets the metrics
            state, not_abort = handle_callback_and_rest_metrics(
                state, epoch=epoch, collected_metrics=collected_metrics
            )

            main_loop_i = main_loop_i + 1
            return state, metrics_over_epochs, rng_train_epochs, main_loop_i, not_abort

        def loop_cond(carry):
            state, metrics_over_epochs, rng_train_epochs, main_loop_i, not_abort = carry
            return jnp.logical_and(not_abort, main_loop_i < num_eval_steps + 1)

        # run the loop
        def run_train_loop(state, metrics_over_epochs, rng):
            state, metrics_over_epochs, _, _, _ = jax.lax.while_loop(
                loop_cond,
                train_for_n_epochs,
                init_val=(state, metrics_over_epochs, rng, 1, True)
            )
            return state, metrics_over_epochs

        if jit:
            run_train_loop = jax.jit(run_train_loop)
        state, metrics_over_epochs = run_train_loop(state, metrics_over_epochs, rng)

        return state.train_state, metrics_over_epochs