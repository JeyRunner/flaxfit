import inspect
from typing import Any, Callable


def call_fn_just_with_defined_args(fn: Callable, fn_args: dict[str, Any]):
  """
  Just pass args of fn_args to fn that are supported by fn
  """
  # only pass supported args
  args = inspect.getfullargspec(fn).args
  if 'self' in args:
    args.remove('self')
  # check args
  for a in args:
    if a not in fn_args.keys():
      raise ValueError(f"Defined function arguments {a} is not supported only arguments {fn_args.keys()} can be defined")
  return fn(*[fn_args[arg] for arg in args])
