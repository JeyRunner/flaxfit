import flax
import jax.numpy as jnp
import jax.tree_util
import jaxtyping
import optax
from cfgv import NotIn
from flax import nnx, struct
from flax.nnx import GraphDef
from flax.nnx import filterlib
from jaxtyping import Integer, Array

from flaxfit.train_state import TrainState


@flax.struct.dataclass
class TrainStateFlax(TrainState):
    graphdef: GraphDef
    params: nnx.Param
    model_state_without_rngs: nnx.State
    rng_state: nnx.State

    @classmethod
    def create(
        cls,
        graphdef: GraphDef,
        params: jaxtyping.PyTree,
        model_state: nnx.State,
        tx: optax.GradientTransformation,
        opt_state: optax.OptState | None = None,
        step: int = 0,
        **kwargs,
    ):
        """
        Create TrainState.
        Will init opt_state if opt_state is not provided.
        """
        rng_state, model_state_without_rngs = model_state.split(nnx.RngState, filterlib.Everything())
        return super().create(
            params, tx, opt_state, step=step, graphdef=graphdef,
            rng_state=rng_state,
            model_state_without_rngs=model_state_without_rngs
        )

    def apply_gradients(self, gradients: jaxtyping.PyTree) -> jaxtyping.PyTree:
        """
        Apply gradients to the model parameters
        :return: The new updated parameters
        """
        updates, opt_state = self.tx.update(gradients, self.opt_state, self.params)
        params = optax.apply_updates(self.params, updates)  # type: ignore
        step = self.step + 1
        return self.replace(params=params, opt_state=opt_state, step=step)


    @property
    def model_state(self):
        return nnx.State.merge(self.model_state_without_rngs, self.rng_state)

    def update_model_state(self, model_state: nnx.State):
        rng_state, model_state_without_rngs = model_state.split(nnx.RngState, filterlib.Everything())
        return self.replace(
            rng_state=rng_state,
            model_state_without_rngs=model_state_without_rngs
        )

    def update_model_state_from_module(self, module: nnx.Module):
        """
        Update the model state (without the params) based on a nnx module.
        Example usage:
        model_out = my_module(batch)  # in-place step update
        train_state = train_state.update_model_state_from_module(my_module)
        """
        return self.update_model_state(self.model_state_from_module(module))

    @staticmethod
    def model_state_from_module(module: nnx.Module):
        model_graph_def, params, model_state_new = nnx.split(module, nnx.Param, filterlib.Everything())
        return model_state_new

    @property
    def model_states_with_params(self):
        return nnx.State.merge(self.params, self.model_state)

    def as_model(self) -> nnx.Module:
        return nnx.merge(self.graphdef, self.model_states_with_params)
