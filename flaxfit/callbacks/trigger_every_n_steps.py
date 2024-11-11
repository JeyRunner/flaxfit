from flaxfit.callbacks.call_sub_args import call_fn_just_with_defined_args
from flaxfit.converters_and_functions import EpochCallbackFunction, DatasetAndModelPredictions
from flaxfit.train_state import TrainState


class EveryNSteps:
  """Helper to execute something just every n steps."""

  last_step: int

  def __init__(self, every_n: int | None, trigger_at_step_zero: bool = False):
    self.every_n = every_n
    self.trigger_at_step_zero = trigger_at_step_zero
    self.last_step = 0

  def reset(self):
    self.last_step = 0

  def check(self, step: int) -> bool:
    """
    Check if action should be executed (every_n steps are over)
    :param step: the step to check.
    :return: if at least every_n steps are over.
    """
    if step < self.last_step:
      self.last_step = step

    trigger = True
    if self.every_n is not None:
      trigger = self.last_step + self.every_n <= step
    if self.trigger_at_step_zero and step == 0:
      trigger = True
    if trigger:
      self.last_step = step
    return trigger


class CallbackAtLeastEveryNEpochs(EpochCallbackFunction):
  """Debounce for callback functions, inner callback will be not called if n epochs are not over already."""

  def __init__(self, callback: EpochCallbackFunction, at_least_every_n_epochs: int):
    self.__callback = callback
    self.__every_n_steps = EveryNSteps(every_n=at_least_every_n_epochs, trigger_at_step_zero=True)


  def __call__(
      self, epoch: int, metrics: dict,
      train_model_predictions: DatasetAndModelPredictions,
      eval_model_predictions: DatasetAndModelPredictions,
      train_state: TrainState
    ) -> None | bool:
    if self.__every_n_steps.check(epoch):
      return call_fn_just_with_defined_args(self.__callback, dict(
        epoch=epoch, metrics=metrics,
        train_model_predictions=train_model_predictions,
        eval_model_predictions=eval_model_predictions,
        train_state=train_state
      ))



