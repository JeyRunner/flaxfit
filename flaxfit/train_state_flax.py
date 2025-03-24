import flax
import jax.numpy as jnp
import jax.tree_util
import jaxtyping
import optax
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
    wrt: filterlib.Filter = flax.struct.field(pytree_node=False)

    @classmethod
    def create(
        cls,
        graphdef: GraphDef,
        params: jaxtyping.PyTree,
        model_state: nnx.State,
        tx: optax.GradientTransformation,
        opt_state: optax.OptState | None = None,
        step: int = 0,
        wrt: filterlib.Filter = nnx.Param,
        **kwargs,
    ):
        """
        Create TrainState.
        Will init opt_state if opt_state is not provided.
        """
        # ensure that we just update params
        if wrt != nnx.Param:
            wrt = filterlib.All(nnx.Param, wrt)
        rng_state, model_state_without_rngs = model_state.split(nnx.RngState, filterlib.Everything())
        train_state: TrainStateFlax = super().create(
            params, tx, opt_state, step=step, graphdef=graphdef,
            rng_state=rng_state,
            model_state_without_rngs=model_state_without_rngs,
            wrt=wrt
        )
        if opt_state is None:
            opt_state = tx.init(nnx.state(train_state.as_model(), wrt))
            train_state = train_state.replace(opt_state=opt_state)
        return train_state

    def apply_gradients(self, gradients: jaxtyping.PyTree) -> jaxtyping.PyTree:
        """
        Apply gradients to the model parameters
        :return: The new updated parameters
        """
        model = self.as_model()
        params_to_update = nnx.state(model, self.wrt)
        updates, opt_state = self.tx.update(gradients, self.opt_state, params_to_update)
        params = optax.apply_updates(params_to_update, updates)  # type: ignore
        step = self.step + 1

        # update model params as get full list of params back from model
        nnx.update(model, params)
        model_graph_def, params, model_state_new = nnx.split(model, nnx.Param, filterlib.Everything())
        return self.replace(params=params, opt_state=opt_state, step=step)


    @property
    def model_state(self):
        return nnx.State.merge(self.model_state_without_rngs, self.rng_state)

    def update_model_state(self, model_state: nnx.State, update_non_rng_state: bool = True):
        rng_state, model_state_without_rngs = model_state.split(nnx.RngState, filterlib.Everything())
        replace = dict(
            rng_state=rng_state,
        )
        if update_non_rng_state:
            replace |= dict(
                model_state_without_rngs=model_state_without_rngs
            )
        return self.replace(**replace)

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
