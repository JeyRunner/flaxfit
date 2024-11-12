import chex
import flax
import jax
import matplotlib.pyplot as plt
import numpy as np
import optax
import pcax
from einshape.src.jax.jax_ops import einshape
from flax import nnx
import jax.numpy as jnp
from jaxtyping import PyTree, Float, Array

from flaxfit.converters_and_functions import LossEntry, DatasetAndModelPredictions
from flaxfit.dataset import Dataset, DatasetXY
from flaxfit.flax_fitter import FlaxModelFitter
from flaxfit.logger.logger import AbstractLogger, AbstractLoggerConfig
from flaxfit.train_state import TrainState

plt.switch_backend('agg')


@flax.struct.dataclass
class GaussianEncoderOutput[T]:
  mean: Float[Array, '{T}']
  logvar: Float[Array, '{T}']
  """log(sigma**2)"""

  @property
  def var(self):
    return jnp.exp(self.logvar)

  @property
  def std(self):
    return jnp.exp(self.logvar*0.5)



class Encoder(nnx.Module):
  def __init__(self, latent_size, x_shape, rngs):
    self.rngs = rngs
    self.x_shape = x_shape
    self.latent_size = latent_size

  def __call__(self, x) -> GaussianEncoderOutput:
    ...


  def sample(self, out_gaussian: GaussianEncoderOutput):
    """
    Sample in the latent space given mean and variance.
    """
    std = jnp.exp(out_gaussian.logvar * 0.5)  # = (e^logvar)^0.5
    noise_normal = jax.random.normal(self.rngs.vae_z_noise(), out_gaussian.mean.shape)
    z = out_gaussian.mean + std * noise_normal
    return z

  def encode_and_add_noise(self, x):
    """
    Encode x and add noise to resulting latentspace vector z.
    """
    out_gaussian = self.__call__(x)
    z = self.sample(out_gaussian)
    return z, out_gaussian


  def kl_decoder_output_to_unit_gaussian(self, out_gaussian: GaussianEncoderOutput):
    """
    Get the kl of a batch of z latent space vectors (mean and variance) and the unit standard gaussian (per sample).
    :param out_gaussian: Output of the encoder.
    :return: kl per sample, shape is (batch_size,).
    """
    batch_size = out_gaussian.mean.shape[0]
    chex.assert_shape(out_gaussian.mean, (None, self.latent_size))
    # kl to unit gaussian
    var = jnp.exp(out_gaussian.logvar)  # = sigma^2
    std = jnp.exp(out_gaussian.logvar*0.5)
    kl = -0.5*jnp.sum(  # should be sum
      out_gaussian.logvar - var - out_gaussian.mean**2 + 1,
      #jnp.log(std**2) - std**2 - out_gaussian.mean**2 + 1,
      axis=-1
    )
    chex.assert_shape(kl, (batch_size,))
    return kl



class EncoderMLP(Encoder):
  def __init__(self, latent_size, x_shape, rngs):
    super().__init__(latent_size, x_shape, rngs)
    hidden = 256*2
    self.activation = nnx.relu
    self.lin1 = nnx.Linear(x_shape**2, hidden, rngs=rngs)
    self.lin_in_features = hidden
    self.linear_mean = nnx.Linear(in_features=self.lin_in_features, out_features=latent_size, rngs=rngs)
    self.linear_logvar = nnx.Linear(in_features=self.lin_in_features, out_features=latent_size, rngs=rngs)
    #self.activation = nnx.relu
    #self.lin_out = GaussianOutputLinearLayer(in_features=latent_size, out_dims=latent_size, rngs=rngs)


  def __call__(self, x):
    x = einshape("bssc->b(ssc)", x)
    x = self.activation(self.lin1(x))
    out_gaussian = GaussianEncoderOutput(
      mean=self.linear_mean(x),
      logvar=self.linear_logvar(x)
    )
    chex.assert_shape(out_gaussian.mean, (x.shape[0], self.latent_size))
    return out_gaussian



class DecoderMLP(nnx.Module):
  def __init__(self, latent_size, x_shape, rngs):
    self.x_shape = x_shape
    hidden = 256*2
    self.linear1 = nnx.Linear(latent_size, hidden, rngs=rngs)
    self.linear2 = nnx.Linear(hidden, self.x_shape**2, rngs=rngs)

  def __call__(self, z: jax.Array) -> jax.Array:
    z = self.linear1(z)
    z = jax.nn.relu(z)
    logits = self.linear2(z)
    y = einshape("b(ssf)->bssf", logits, s=self.x_shape, f=1)
    return y



################################################################
## VAE:
class VAE(nnx.Module):
  def __init__(self, x_shape, latent_size, rngs):
    self.latent_size = latent_size
    self.encoder = EncoderMLP(latent_size, x_shape, rngs)
    self.decoder = DecoderMLP(latent_size, x_shape, rngs)

  def encode_decode(self, x):
    z, z_gaussian = self.encoder.encode_and_add_noise(x)
    x_reconstructed = self.decoder(z)
    return x_reconstructed, z_gaussian

  def __call__(self, x):
    return self.encode_decode(x)


# create named rng streams (separate one for creating the vae noise)
rngs = nnx.Rngs(0, vae_z_noise=jax.random.key(1))
model = VAE(
  x_shape=28, latent_size=2, rngs=rngs
)



# Load the MNIST dataset
import tensorflow_datasets as tfds
ds_builder = tfds.builder('mnist')
ds_builder.download_and_prepare()
train_ds = tfds.as_numpy(ds_builder.as_dataset(split="train", batch_size=-1))
test_ds = tfds.as_numpy(ds_builder.as_dataset(split="test", batch_size=-1))

# Normalize data
train_images, train_labels = train_ds['image'], train_ds['label']
test_images, test_labels = test_ds['image'], test_ds['label']
def norm_data(d):
  return (d/255.0) * 2.0 - 1.0
train_images, test_images = norm_data(train_images), norm_data(test_images)



# test model
model_test_out_z, model_test_out_gaussian = model.encode_decode(train_images[0:1000])


def plot_grid2(x_sample, y_pred):
  figure = plt.figure(figsize=(3, 3*0.5))
  for i in range(5):
    plt.subplot(1, 2, 1)
    plt.imshow(x_sample, cmap='gray')
    plt.subplot(1, 2, 2)
    plt.imshow(y_pred, cmap='gray')
  return figure


kl_weight = 1.0 #0.5
logger = AbstractLogger(
  output_folder='out/vae', run_name=f'lrhigh_mlp_latent{model.latent_size}_kl{kl_weight}',
  config=AbstractLoggerConfig(use_wandb=False)
)

def loss_fn(model_output, dataset: Dataset):
  target_x = dataset.x
  x_reconstructed, z_gaussian = model_output

  # sum error over dims, but mean over samples
  reconstruct_loss = jnp.mean(
    jnp.sum((x_reconstructed - target_x)**2, axis=(1, 2)),
    axis=0
  )
  kl_to_unit_gaussian = model.encoder.kl_decoder_output_to_unit_gaussian(z_gaussian)
  chex.assert_shape(kl_to_unit_gaussian, (target_x.shape[0],))

  return dict(
    reconstruct=reconstruct_loss,
    kl=LossEntry(value=jnp.mean(kl_to_unit_gaussian, axis=0), weight=kl_weight)
  )




# train
fitter = FlaxModelFitter(
  update_batch_size=250*1,
  loss_function=loss_fn
)
train_state = fitter.create_train_state(
  model,
  optimizer=optax.adam(
    learning_rate=1e-3 #0.0001 #0.00004
  )
)


def plot_grid(x_sample, y_pred):
  figure = plt.figure(figsize=(3 * 5, 3 * 2))
  plt.title('Reconstruction Samples')
  for i in range(5):
    plt.subplot(2, 5, i + 1)
    plt.imshow(x_sample[i], cmap='gray')
    plt.subplot(2, 5, 5 + i + 1)
    plt.imshow(y_pred[i], cmap='gray')
  return figure


def plot_latent_space(model_train_state):
  # use eval set
  _, (z_sampled, gaussian_out), _ = fitter.model_forward_dataset(model_train_state, Dataset(x=train_images), batch_size=250)
  z = gaussian_out.mean
  # use pca to two dimension when latent space has more than two dims
  f = plt.figure()
  if z.shape[1] > 2:
    plt.title('PCA Z project')
    pca_state = pcax.fit(z, n_components=2)
    z_pca = pcax.transform(pca_state, z)
    z = z_pca
  plt.scatter(z[:, 0], z[:, 1], c=train_labels, cmap='tab10', s=2.)
  return f



def epoch_callback_fn(
    epoch: int, metrics: dict,
    train_model_predictions: DatasetAndModelPredictions,
    eval_model_predictions: DatasetAndModelPredictions,
    train_state: TrainState
):
  logger.log_scalars(metrics, step=epoch)
  print('> do plots ...')
  if train_model_predictions is not None:
    train_dataset = train_model_predictions.dataset
    train_predictions_y = train_model_predictions.model_predictions
    for i in range(10):
      logger.log_plot(f'train_img/{i}', plot_grid2(train_dataset.x[i], train_predictions_y[0][i]), step=epoch)
    logger.log_plot(f'train_img_plt', plot_grid(train_dataset.x, train_predictions_y[0]), step=epoch)

    _, gaussian_out = train_predictions_y
    # distribution in latent space
    z_dims = jnp.arange(gaussian_out.mean.shape[1])
    z_dims_std = jnp.mean(gaussian_out.std, axis=0)
    z_dims_std_sorted = jnp.argsort(z_dims_std)
    fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, sharex=True)
    #plt.title("std values per latent space dimension")
    ax1.set_title("z.std encoder output")
    ax1.bar(z_dims, z_dims_std[z_dims_std_sorted], label="z.std encoder output")
    ax2.set_title('z.means std dist')
    ax2.bar(z_dims, jnp.std(gaussian_out.mean, axis=0)[z_dims_std_sorted], label='z.means std dist')
    ax2.set_xlabel("z dims")
    logger.log_plot(f'train_z_dist', fig, step=epoch)

    f = plt.figure()
    z = gaussian_out.mean
    plt.scatter(z[:, z_dims_std_sorted[0]], z[:, z_dims_std_sorted[1]], c=train_labels, cmap='tab10', s=2.)
    logger.log_plot(
      f'train_embedding_top2std',
      f,
      step=epoch
    )


  for i in range(eval_model_predictions.dataset.x.shape[0]):
    logger.log_plot(f'eval_img/{i}', plot_grid2(
      eval_model_predictions.dataset.x[i], eval_model_predictions.model_predictions[0][i]
    ), step=epoch)

  logger.log_plot(f'eval_embedding/', plot_latent_space(train_state), step=epoch)
  print('> plots done')


fitter.train_fit(
    initial_train_state=train_state,
    dataset=DatasetXY(x=train_images, y=train_images),
    dataset_eval=DatasetXY(x=test_images, y=test_images),
    num_epochs=2000,
    evaluate_each_n_epochs=20,
    epoch_callback_fn=epoch_callback_fn,
    epoch_callback_pass_train_dataset_prediction_idx=jnp.s_[:],
    epoch_callback_pass_eval_dataset_prediction_idx=jnp.s_[:10],
    jit=False
  )

model_test_out_z, model_test_out_gaussian = model.encode_decode(train_images[0:1000])
plt.imshow(model_test_out_z[0], cmap='grayscale')
plt.show()