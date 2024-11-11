from typing import Callable, TypeVar

import flax.struct
from jaxtyping import PyTree

T_X = TypeVar("T_X")
T_Y = TypeVar("T_Y")


@flax.struct.dataclass
class DatasetTyped[T_X]:
    x: T_X
    """Model input"""


@flax.struct.dataclass
class Dataset(DatasetTyped[PyTree]):
    pass


@flax.struct.dataclass
class DatasetXYTyped[T_X, T_Y](DatasetTyped):
    x: T_X
    y: T_Y


@flax.struct.dataclass
class DatasetXY(DatasetXYTyped[PyTree, PyTree]):
    pass
