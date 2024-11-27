import chex
import flax
import jax.numpy as jnp
import jax.tree_util
import jaxtyping
import optax
from flax import nnx, struct
from jaxtyping import Integer, Array


class ModelForwardFn:
    def __call__(
        self,
        params: jaxtyping.PyTree,
        model_state: jaxtyping.PyTree,
        batch: jaxtyping.PyTree,
    ) -> tuple[jaxtyping.PyTree, jaxtyping.PyTree]:
        """
        Pass a batch through the model given its parameters and state.
        :param batch: Pass this batch through the model
        :return: The model output for the batch, new change model state
        """


@flax.struct.dataclass
class TrainState(struct.PyTreeNode):
    params: jaxtyping.PyTree
    """Model params"""

    @property
    def model_state(self):
        """Model state that may change from batch to batch, e.g. rng keys, batch stats for normalization, ..."""
        raise NotImplemented()

    opt_state: optax.OptState
    step: Integer[Array, ""]

    tx: optax.GradientTransformation = struct.field(pytree_node=False)
    """Gradient transform"""

    @classmethod
    def create(
        cls,
        params: jaxtyping.PyTree,
        tx: optax.GradientTransformation,
        opt_state: optax.OptState | None = None,
        *,
        step: int = 0,
        **kwargs,
    ):
        """
        Create TrainState.
        Will init opt_state if opt_state is not provided.
        """
        if opt_state is None:
            opt_state = tx.init(params)
        return cls(
            params=params, opt_state=opt_state, tx=tx, step=jnp.asarray(step), **kwargs
        )

    def apply_gradients(self, gradients: jaxtyping.PyTree) -> jaxtyping.PyTree:
        """
        Apply gradients to the model parameters
        :return: The new updated parameters
        """
        raise NotImplemented()


    def update_model_state(self, model_state: jaxtyping.PyTree):
        """Update the model state."""
        return self.replace(model_state=model_state)


@flax.struct.dataclass
class Metric(struct.PyTreeNode):
    """Accumulator for metric values over multiple batches/epochs"""

    def update(self, values: jaxtyping.PyTree) -> "Metric":
        """Append new values"""

    def reset(self) -> "Metric":
        """Rest"""

    def collect(self) -> jaxtyping.PyTree:
        """Summarize the value after multiple update calls."""


@flax.struct.dataclass
class AverageMetric(Metric):
    """Yields average values over multiple batches/episodes"""

    value_sum: jaxtyping.PyTree
    num_summed: Integer[Array, ""]

    @staticmethod
    def create(values: jaxtyping.PyTree):
        """Create initial values which will all be set to 0."""
        chex.assert_tree_shape_suffix(values, tuple())
        return AverageMetric(
            value_sum=jax.tree_util.tree_map(lambda l: jnp.zeros_like(l, dtype=l.dtype), values),
            num_summed=0
        )

    def reset(self):
        return self.replace(
            value_sum=jax.tree_util.tree_map(lambda l: l * 0, self.value_sum),
            num_summed = 0
        )


    def update(self, values: jaxtyping.PyTree) -> "AverageMetric":
        return AverageMetric(
            value_sum=jax.tree_util.tree_map(
                lambda sum, new: sum + new, self.value_sum, values
            ),
            num_summed=self.num_summed + 1,
        )

    def collect(self) -> jaxtyping.PyTree:
        return jax.tree_util.tree_map(lambda sum: sum / self.num_summed, self.value_sum)


@flax.struct.dataclass
class MetricsWithLoss(struct.PyTreeNode):
    loss: AverageMetric
    metrics: Metric

    def update_loss(self, values: jaxtyping.PyTree) -> "MetricsWithLoss":
        return self.replace(loss=self.loss.update(values))

    def update_metrics(self, values: jaxtyping.PyTree) -> "MetricsWithLoss":
        return self.replace(metrics=self.metrics.update(values))

    def update(self, loss: jaxtyping.PyTree, metrics: jaxtyping.PyTree) -> "MetricsWithLoss":
        return self.update_loss(loss).update_metrics(metrics)

    def collect(self) -> dict[str, jaxtyping.PyTree]:
        return dict(loss=self.loss.collect(), metrics=self.metrics.collect())

    def reset(self):
        return self.replace(
            loss=self.loss.reset(),
            metrics=self.metrics.reset(),
        )



@flax.struct.dataclass
class TrainStateWithMetrics(struct.PyTreeNode):
    train_state: TrainState
    metrics_train: MetricsWithLoss
    metrics_eval: MetricsWithLoss
