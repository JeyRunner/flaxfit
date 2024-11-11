from pathlib import Path

from flax import nnx
from flax.nnx import filterlib

from flaxfit.converters_and_functions import EpochCallbackFunction, DatasetAndModelPredictions
from flaxfit.train_state import TrainState
import orbax.checkpoint as ocp

from flaxfit.train_state_flax import TrainStateFlax


def create_checkpointer(path: str, max_to_keep: int = None, delete_existing_checkpoints=False):
    checkpointing_path = Path(path)
    checkpointing_path.mkdir(parents=True, exist_ok=True)
    checkpointing_path = checkpointing_path.resolve().absolute().as_posix()
    if delete_existing_checkpoints:
        checkpointing_path = ocp.test_utils.erase_and_create_empty(checkpointing_path)
    checkpointing = ocp.CheckpointManager(
        checkpointing_path,
        options=ocp.CheckpointManagerOptions(max_to_keep=max_to_keep),
    )
    return checkpointing


def save_model_checkpoint(checkpointer: ocp.Checkpointer, train_state: TrainState, epoch: int, remove_rng_state=True):
    """Save the model train state with a checkpointer"""
    # ignore rngState when checkpointing
    if remove_rng_state:
        assert isinstance(train_state, TrainStateFlax)
        train_state = train_state.replace(rng_state=None)
    checkpointer.save(step=epoch, args=ocp.args.StandardSave(
        train_state
    ))


def load_train_state_from_checkpoint(
    path: str,
    train_state_init: TrainStateFlax,
    step: int | None = None,
    evaluation_mode=False,
) -> TrainStateFlax:
    """
    Will return the train state loaded from the checkpoint.
    :param checkpointing_path: path where the checkpoints are stored.
    :param step: step to load. if is not set will load latest step.
    :param train_state_init: train state with random model parameters (params will be overwritten with params from checkpoint.)
    :param evaluation_mode: if set the model to evaluation mode
    :return: the loaded train state
    """
    train_state = train_state_init
    # restore
    train_state_rng_state = train_state.rng_state
    checkpointing = create_checkpointer(path)
    if step is None:
      step = checkpointing.latest_step()
      assert step is not None, f"latest checkpoint of model {path} not found"
      print(f'> use latest checkpoint of model: {step}')
    # not restore the rng state
    train_state: TrainStateFlax = checkpointing.restore(step, args=ocp.args.StandardRestore(
      train_state.replace(rng_state=None)
    ))

    # re add the rng state
    train_state = train_state.replace(rng_state=train_state_rng_state)

    if evaluation_mode:
      model: nnx.Module = nnx.merge(train_state.graphdef, train_state.params, *train_state.model_state)
      # set to evaluation mode: e.g. to effect dropout
      model.eval()
      graph_def, params, model_state = nnx.split(model, nnx.Param, filterlib.Everything())
      train_state = train_state.replace(
        graph_def=graph_def,
        params=params,
        model_state=model_state
      )
    return train_state



class CheckpointerCallback(EpochCallbackFunction):
    """Checkpointer callback function for saving the model."""

    def __init__(
        self,
        path: str, max_to_keep: int = None, delete_existing_checkpoints=False,
        remove_rng_state=True
    ):
        self.checkpointer = create_checkpointer(path, max_to_keep, delete_existing_checkpoints)
        self.remove_rng_state = remove_rng_state


    def __call__(
        self, epoch: int, metrics: dict, train_model_predictions: DatasetAndModelPredictions,
        eval_model_predictions: DatasetAndModelPredictions, train_state: TrainState
    ) -> None:
        save_model_checkpoint(self.checkpointer, train_state, epoch, self.remove_rng_state)

