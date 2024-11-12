import os
from unittest import TestCase

import chex
import jax
import jax.numpy as jnp
import optax
from flax import nnx
from matplotlib import pyplot as plt
from pyarrow.dataset import dataset

from flaxfit.callbacks.checkpointer import CheckpointerCallback, load_train_state_from_checkpoint
from flaxfit.callbacks.trigger_every_n_steps import CallbackAtLeastEveryNEpochs
from flaxfit.converters_and_functions import LossEntry, DatasetAndModelPredictions
from flaxfit.dataset import Dataset, DatasetXY
from flaxfit.flax_fitter import FlaxModelFitter
from flaxfit.train_state import AverageMetric, TrainState
from flaxfit.train_state_flax import TrainStateFlax


class TestFlaxTrain(TestCase):

    def get_model(self):
        rngs = nnx.Rngs(0)
        activation_fn = nnx.relu
        model = nnx.Sequential(
            nnx.Linear(in_features=1, out_features=1000, rngs=rngs),
            activation_fn,
            nnx.Linear(in_features=1000, out_features=1000, rngs=rngs),
            activation_fn,
            nnx.Linear(in_features=1000, out_features=1000, rngs=rngs),
            activation_fn,
            nnx.Linear(in_features=1000, out_features=1, rngs=rngs)
        )
        return model

    data_x = jnp.arange(1000)[:, jnp.newaxis]
    data_y = jnp.sin(data_x/20)

    epochs = 30
    batch_size = 10
    learning_rate = 0.002



    def fit_with_flax_fit(self):
        def loss(predictions_y, dataset: DatasetXY):
            return dict(
                mse=jnp.mean((predictions_y - dataset.y)**2),
            )

        def callback(
             epoch: int, metrics: dict,
             train_model_predictions: DatasetAndModelPredictions,
             eval_model_predictions: DatasetAndModelPredictions,
             train_state: TrainState
         ):
            print(f'epoch {epoch} - metrics: {metrics}')


        fitter = FlaxModelFitter(
            update_batch_size=self.batch_size,
            loss_function=loss,
        )

        model = self.get_model()
        train_state: TrainStateFlax = fitter.create_train_state(
            model, optimizer=optax.adam(learning_rate=self.learning_rate)
        )

        train_state, metrics_over_epochs = fitter.train_fit(
            initial_train_state=train_state,
            dataset=DatasetXY(x=self.data_x, y=self.data_y),
            num_epochs=self.epochs,
            evaluate_each_n_epochs=1,
            epoch_callback_fn=[
                callback
            ],
            epoch_callback_pass_train_dataset_prediction_idx=jnp.s_[:],
            shuffle=False,
            sample_each_batch_independent_from_dataset=False,
            train_batch_remainder_strategy='None'
        )
        print(metrics_over_epochs)
        return train_state.params, metrics_over_epochs['train']['loss']['total'][1:]




    def fit_with_nnx(self):
        def loss_fn(model: nnx.Module, batch: DatasetXY):
            predictions_y = model(batch.x)
            loss = jnp.mean((predictions_y - batch.y)**2)
            return loss, predictions_y

        @nnx.jit
        def train_step(model: nnx.Module, optimizer: nnx.Optimizer, metrics: nnx.MultiMetric, batch):
            """Train for a single step."""
            grad_fn = nnx.value_and_grad(loss_fn, has_aux=True)
            (loss, predictions_y), grads = grad_fn(model, batch)
            #jax.debug.print("{}", loss)
            metrics.update(loss=loss)  # In-place updates.
            optimizer.update(grads)  # In-place updates.

        metrics_history = {
            'train_loss': [],
        }

        model = self.get_model()

        optimizer = nnx.Optimizer(model, optax.adam(self.learning_rate))
        metrics = nnx.MultiMetric(
            loss=nnx.metrics.Average('loss'),
        )

        for epoch in range(self.epochs):
            for step in range(int(self.data_x.shape[0] / self.batch_size)):
                # Run the optimization for one step and make a stateful update to the following:
                # - The train state's model parameters
                # - The optimizer state
                # - The training loss and accuracy batch metrics
                idx = jnp.s_[self.batch_size*step:self.batch_size*(step + 1)]
                batch = DatasetXY(x=self.data_x[idx], y=self.data_y[idx])
                train_step(model, optimizer, metrics, batch)

            # Log the training metrics.
            for metric, value in metrics.compute().items():  # Compute the metrics.
                metrics_history[f'train_{metric}'].append(value)  # Record the metrics.
            metrics.reset()  # Reset the metrics for the test set.

            print(
                f"[train] epoch: {epoch}, "
                f"loss: {metrics_history['train_loss'][-1]}, "
            )
        graphdef, params, _ = nnx.split(model, nnx.Param, nnx.filterlib.Everything())
        return params, jnp.array(metrics_history['train_loss'])



    def test_fit_compare_with_nnx(self):
        model, metrics_history = self.fit_with_nnx()
        fitter_model, fitter_metrics_history = self.fit_with_flax_fit()

        plt.plot(jnp.arange(metrics_history.shape[0]), metrics_history, label='nnx')
        plt.plot(jnp.arange(fitter_metrics_history.shape[0]), fitter_metrics_history, label='flaxfit')
        plt.legend()

        # plt.show()

        chex.assert_trees_all_equal(fitter_metrics_history, metrics_history)
        chex.assert_trees_all_equal(model, fitter_model)