import os
import pathlib
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
from flaxfit.converters_and_functions import LossEntry, DatasetAndModelPredictions
from flaxfit.dataset import Dataset, DatasetXY
from flaxfit.train_state import AverageMetric, TrainState


class TestFlaxTrain(TestCase):
    def test_simple_model(self):
        rngs = nnx.Rngs(0)
        model = nnx.Sequential(
            nnx.Linear(in_features=1, out_features=10, rngs=rngs),
            nnx.Linear(in_features=10, out_features=1, rngs=rngs),
        )

        def loss(predictions_y, dataset: Dataset):
            return dict(
                mse=jnp.mean((predictions_y - dataset.x)**2)
            )

        def callback(epoch: int, metrics: dict):
            print(f'> epoch {epoch} - {metrics}')

        fitter = FlaxModelFitter(
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
        train_state = fitter.create_train_state(model)

        train_state, metrics_over_epochs = fitter.train_fit(
            train_state,
            dataset=dataset,
            dataset_eval=dataset_eval,
            evaluate_each_n_epochs=1,
            epoch_callback_fn=callback,
            num_epochs=200
        )
        print(metrics_over_epochs)
        assert metrics_over_epochs['train']['loss']['total'][-1] < 0.1


    def test_fit_simple_NN(self):
        plt.switch_backend('agg')
        rng = jax.random.PRNGKey(0)

        t = jnp.arange(105)[:, jnp.newaxis] * 2 / 100 - 1.0
        dataset_orig = jnp.sin(t)
        dataset = DatasetXY(x=t, y=jnp.sin(t * 10) + jax.random.normal(rng, t.shape) * 0.02)
        t_eval = jnp.arange(100)[:, jnp.newaxis] * 2 / 100 - 1.0
        dataset_eval = DatasetXY(x=t_eval, y=jnp.sin(t_eval * 10))

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

        # test model
        # out = model(dataset.x)
        # print(out)

        with_final_layer_reg = False
        def loss(predictions_y, dataset: DatasetXY, model: nnx.Sequential):
            l = dict(
                mse=jnp.mean((predictions_y - dataset.y)**2),
                output_small=LossEntry(value=jnp.mean(jnp.abs(predictions_y)), weight=0.000),
            )
            if with_final_layer_reg:
                l |= dict(
                    regularize_final_bias=5000 * (jnp.mean(model.layers[-1].kernel ** 2) + jnp.mean(model.layers[-1].bias ** 2))
                )
            return l

        fig = plt.figure()
        ax = fig.subplots(1, 1)
        fig.show()

        out_folder = 'out/test_fit_simple_NN'
        if with_final_layer_reg:
            out_folder += '_withmodelReg'
        os.makedirs(out_folder, exist_ok=True)
        def callback(
             epoch: int, metrics: dict,
             train_model_predictions: DatasetAndModelPredictions,
             eval_model_predictions: DatasetAndModelPredictions,
             train_state: TrainState
         ):
            print(f'epoch {epoch} - metrics: {metrics}')
            ax.clear()
            if train_model_predictions is not None:
                ax.plot(t, train_model_predictions.dataset.y)
                ax.plot(t, train_model_predictions.model_predictions)
                print(train_model_predictions.model_predictions[:2])
            fig.suptitle(f"{epoch}")
            fig.canvas.draw()
            fig.canvas.flush_events()
            plt.savefig(f'{out_folder}/epoch_{epoch}.png')

        fitter = FlaxModelFitter(
            update_batch_size=10,
            evaluate_batch_size=10,
            loss_function=loss,
        )

        train_state = fitter.create_train_state(
            model, optimizer=optax.adam(learning_rate=0.00001)
        )

        train_state, metrics_over_epochs = fitter.train_fit(
            initial_train_state=train_state,
            dataset=dataset,
            dataset_eval=dataset_eval,
            num_epochs=4000,
            evaluate_each_n_epochs=200,
            epoch_callback_fn=[
                callback,
                CallbackAtLeastEveryNEpochs(
                    CheckpointerCallback(path='out/checkpoints', max_to_keep=2, delete_existing_checkpoints=True),
                    at_least_every_n_epochs=500
                )
            ],
            epoch_callback_pass_train_dataset_prediction_idx=jnp.s_[:],
            epoch_callback_pass_eval_dataset_prediction_idx=jnp.s_[:],
            shuffle=True,
            sample_each_batch_independent_from_dataset=False,
            train_batch_remainder_strategy='PadValidSampled',
            jit=True
        )

        print('metrics_over_epochs', metrics_over_epochs)

        train_state = load_train_state_from_checkpoint(
            train_state_init=train_state,
            path='out/checkpoints',
            evaluation_mode=True
        )

        plt.figure()
        plt.plot(metrics_over_epochs['__epoch'], metrics_over_epochs['train']['loss']['total'])
        plt.savefig(f'{out_folder}/loss.png')
        plt.show()

        assert metrics_over_epochs['train']['loss']['total'][-1] <= 1e-2



    def test_save_load(self):
        rngs = nnx.Rngs(0)
        model = nnx.Sequential(
            nnx.Linear(in_features=1, out_features=1, rngs=rngs),
        )

        out_folder = 'out/test_save_load'
        os.makedirs(out_folder, exist_ok=True)

        fitter = FlaxModelFitter(
            update_batch_size=10,
            evaluate_batch_size=10,
            loss_function=None,
        )

        train_state = fitter.create_train_state(
            model, optimizer=optax.adam(learning_rate=0.00001)
        )
        checkpointer = create_checkpointer(path=out_folder)
        save_model_checkpoint(checkpointer, train_state, epoch=1, remove_rng_state=True)
        checkpointer.close()

        train_state = load_train_state_from_checkpoint(
            train_state_init=train_state,
            path=out_folder,
            evaluation_mode=True,
            step=1
        )
        print(train_state)

