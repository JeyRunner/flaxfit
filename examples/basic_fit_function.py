import jax.numpy as jnp
from flax import nnx

from flaxfit.callbacks.checkpointer import CheckpointerCallback
from flaxfit.callbacks.trigger_every_n_steps import CallbackAtLeastEveryNEpochs
from flaxfit.converters_and_functions import DatasetAndModelPredictions
from flaxfit.dataset import Dataset, DatasetXY
from flaxfit.flax_fitter import FlaxModelFitter
from flaxfit.train_state import TrainState

# make the model
rngs = nnx.Rngs(0)
model = nnx.Sequential(
    nnx.Linear(in_features=1, out_features=10, rngs=rngs),
    nnx.relu,
    nnx.Linear(in_features=10, out_features=1, rngs=rngs),
)

def loss(predictions_y, dataset: Dataset):
    return dict(
        mse=jnp.mean((predictions_y - dataset.y)**2)
    )

def epoch_callback(
    epoch: int, metrics: dict,
    train_model_predictions: DatasetAndModelPredictions,
    eval_model_predictions: DatasetAndModelPredictions,
    train_state: TrainState
):
    print(f'> epoch {epoch} - {metrics}')



fitter = FlaxModelFitter(
    loss_function=loss,
    update_batch_size=5
)

# dataset
x = jnp.arange(20)[:, jnp.newaxis]/20.0
gen_data = lambda x: x**2
dataset = DatasetXY(
    x=x,
    y=x**2
)
x_eval = jnp.arange(30)[:, jnp.newaxis]/20.0
dataset_eval = DatasetXY(
    x=x_eval,
    y=x_eval**2
)

# fit
train_state = fitter.create_train_state(model)
train_state, history = fitter.train_fit(
    train_state,
    dataset=dataset,
    dataset_eval=dataset_eval,
    evaluate_each_n_epochs=50,
    epoch_callback_fn=[
      epoch_callback,
      CallbackAtLeastEveryNEpochs(
        CheckpointerCallback(path='out/checkpoints', max_to_keep=2, delete_existing_checkpoints=True),
        at_least_every_n_epochs=100
      )
    ],
    num_epochs=200
)
print(history)