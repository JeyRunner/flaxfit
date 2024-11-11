from enum import Enum

import chex
import jax
import numpy as np
from jaxtyping import Array, Float
import jax.numpy as jnp



def squeeze_to_1d(x):
	"""Removes all axis with dim 1, checks that output is 1d"""
	o = x.squeeze()
	chex.assert_shape(o, (None,))
	return o

def squeeze_to_scalar(x):
	"""Removes all axis with dim 1, checks that output is 1d"""
	o = x.squeeze()
	assert np.isscalar(o) or (x.shape == (1,) or x.shape == tuple()), \
		f"squeeze_to_1d needs input with shape that has just one element but input shape is {x.shape} (squeezed shape is {o.shape})"
	return o


def is_slice(x):
	return isinstance(x, slice) or (isinstance(x, tuple))


def slice_get_num_dims(s):
	assert is_slice(s)
	if isinstance(s, slice):
		return 1
	elif isinstance(s, tuple):
		return len(s)

def is_array(x):
	return isinstance(x, jnp.ndarray) or isinstance(x, np.ndarray)

def in_jit_context():
	"""Returns true if in jit context or this function call is within any jax transformation (e.g. vmap, grad, ...)."""
	return isinstance(jnp.array(0), jax.core.Tracer)