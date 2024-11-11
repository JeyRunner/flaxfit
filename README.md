# Flaxfit
Fitting you flax models made simple!
Thereby the whole fitting function can be used in a jit context.

## Install
```bash
pip install git+https://github.com/JeyRunner/flaxfit.git
```

## Usage
### Fit to dataset
For full examples see `examples/` folder.
```python
from flax.nnx import nnx
# ...

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

# dataset
x = jnp.arange(20)[:, jnp.newaxis]
dataset = DatasetXY(
    x=x,
    y=x**2
)

# fit
train_state = fitter.create_train_state(model)
train_state, history = fitter.train_fit(
    train_state,
    dataset=dataset,
    dataset_eval=dataset_eval,
    evaluate_each_n_epochs=1,
    epoch_callback_fn=callback,
    num_epochs=200
)
print(history)
```

### Dev setup
Install deps:
```bash
pip install .[dev, test]
```