import os
import pathlib
from typing import Any
from unittest import TestCase

import chex
import jax
import jax.numpy as jnp
import optax
from flax import nnx
from matplotlib import pyplot as plt
from pyarrow.dataset import dataset

from flaxfit.flax_fitter import FlaxModelFitter
from flaxfit.callbacks.checkpointer import CheckpointerCallback, load_train_state_from_checkpoint, \
    save_model_checkpoint, create_checkpointer
from flaxfit.callbacks.trigger_every_n_steps import CallbackAtLeastEveryNEpochs
from flaxfit.converters_and_functions import LossEntry, DatasetAndModelPredictions, BatchProcessStepFunction, \
    PassBatchThroughModelAndUpdateStateFn
from flaxfit.dataset import Dataset, DatasetXY
from flaxfit.train_state import AverageMetric, TrainState, TrainStateWithMetrics


class TestFlaxTrainComplex(TestCase):
    def test_custom_simple_model(self):
        rngs = nnx.Rngs(0)

        class Model(nnx.Module):
            def __init__(self, rngs):
                self.layers = nnx.Sequential(
                    nnx.Linear(in_features=1, out_features=10, rngs=rngs),
                    nnx.Linear(in_features=10, out_features=1, rngs=rngs),
                )
                self.rng = rngs

            def __call__(self, x):
                return self.layers(x)
        model = Model(rngs)

        def loss(predictions_y, dataset: Dataset):
            #return predictions_y
            return dict(
                mse=jnp.mean((predictions_y - dataset.y)**2)
            )

        def callback(epoch: int, metrics: dict):
            print(f'> epoch {epoch} - {metrics}')


        # custom
        class CustomTrain:
            def model_forward(
                self,
                model: Model,
                batch_x,
                info: dict,
                batch_sub_update: int
            ) -> Any:
                """
                Custom call to the module.
                This is a stateful function that will change the passed model.
                :param model: the model as merged nnx module, do calls to this module.
                :return: the model predictions for the batch_x, that will be passed to the loss function
                """
                jax.debug.print("info {}, batch_sub_update {}", info, batch_sub_update)
                random = jax.random.normal(model.rng(), shape=batch_x.shape)
                jax.debug.print("random {}", random[0, 0])
                batch_x += random*0.05*batch_sub_update
                return model(batch_x)
                #return dict(new_loss=jnp.mean((model(batch_x) - batch_x**2)**2))

            def train_update_step(
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

                # do multiple update steps for the one batch
                def sub_step(carry, x):
                    state = carry
                    model_call_kwargs = dict(
                        info=state.info,
                        batch_sub_update=x
                    )
                    state, loss_dict, metrics_dict = pass_batch_through_model_and_update_state_fn(
                        state, batch, model_call_kwargs
                    )
                    return state, (loss_dict, metrics_dict)

                state, (loss_dict, metrics_dict) = jax.lax.scan(sub_step, init=state, xs=jnp.arange(5))
                return state, loss_dict, metrics_dict

        custom_train = CustomTrain()


        fitter = FlaxModelFitter(
            call_model_function=custom_train.model_forward,
            batch_process_step_function=custom_train.train_update_step,
            loss_function=loss,
            update_batch_size=5
        )


        x = (jnp.arange(17)*0.1 - 0.1)[:, jnp.newaxis]
        gen_data = lambda x: x**2
        dataset = DatasetXY(
            x=x,
            y=gen_data(x)
        )
        x_eval = (jnp.arange(10)*0.1 - 0.5)[:, jnp.newaxis]
        dataset_eval = DatasetXY(
            x=x_eval,
            y=gen_data(x_eval)
        )
        train_state = fitter.create_train_state(
            model,
            optax.adam(learning_rate=optax.linear_schedule(
                init_value=0.0002,#0.0002
                end_value=0.0,
                transition_steps=1000
            ))
        )

        train_state, metrics_over_epochs = fitter.train_fit(
            train_state,
            dataset=dataset,
            dataset_eval=dataset_eval,
            evaluate_each_n_epochs=1,
            epoch_callback_fn=callback,
            num_epochs=50 #200
        )
        print(metrics_over_epochs)


        plt.plot(metrics_over_epochs['train']['loss']['total'])
        plt.show()

        assert metrics_over_epochs['train']['loss']['total'][-1] < 0.1

