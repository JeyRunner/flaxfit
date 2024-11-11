from dataclasses import asdict, dataclass
from os import PathLike
from pathlib import Path
from typing import Dict, Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import wandb
from jaxtyping import Float, Array, Integer
from tensorboardX import SummaryWriter

from flaxfit.logger.util import is_array


@dataclass
class AbstractLoggerConfig:
	use_wandb: bool = True
	use_tensorboard: bool = True
	wand_project: str = None

	save_plots: bool = True
	"""Save plots in the output folder"""


class AbstractLogger:
	"""
	A logger that logs to tensorboard and wandb.
	It also saves plots in the given output folder.
	"""

	def __init__(
			self,
			output_folder: PathLike, run_name: str,
			group_name: str = None,
			hyper_params: dict | Any = {},
			metrics: list[str] = [],
			config: AbstractLoggerConfig = AbstractLoggerConfig()
	):
		self.output_folder = Path(output_folder)
		if group_name is not None:
			self.output_folder /= group_name
		if run_name is not None:
			self.output_folder /= run_name
		else:
			self.output_folder /= 'unknown_run'
		self.output_folder.mkdir(parents=True, exist_ok=True)

		self.config = config
		self.tb_writer = None
		self.wandb_run = None

		if not isinstance(hyper_params, dict) and hyper_params is not None:
			hyper_params = asdict(hyper_params)

		if self.config.use_tensorboard:
			self.tb_writer = SummaryWriter(logdir=self.output_folder.as_posix())
			hyper_params_flatten = self.__dict_flatten(hyper_params)
			self.tb_writer_metric_dict = {name: np.nan for name in metrics}
			self.tb_writer.add_hparams(hparam_dict=hyper_params_flatten, metric_dict=self.tb_writer_metric_dict, name='_hparams')

		if self.config.use_wandb:
			wandb_dir = self.output_folder / '.wandb_logs'
			wandb_dir.mkdir(exist_ok=True)
			self.wandb_run = wandb.init(
				project=self.config.wand_project,
				group=group_name,
				name=run_name,
				# track hyperparameters and run metadata
				config=hyper_params,
				dir=wandb_dir,
				allow_val_change=True,
				#sync_tensorboard=True
			)
			for metric_name in metrics:
				wandb.define_metric(metric_name)


	def __dict_flatten(self, d: dict, sep='.'):
		if d is None or len(d.items()) == 0:
			return {}
		flat_dict_list: list[dict] = pd.json_normalize(d, sep=sep).to_dict(orient='records')
		if len(flat_dict_list) > 0:
			return flat_dict_list[0]
		else:
			return {}

	def add_hyper_params(self, hyper_params: dict | Any):
		"""Update or add hyper parameters."""
		if not isinstance(hyper_params, dict):
			hyper_params = asdict(hyper_params)
		if self.config.use_tensorboard:
			hyper_params_flatten = self.__dict_flatten(hyper_params)
			self.tb_writer.add_hparams(hparam_dict=hyper_params_flatten, metric_dict=self.tb_writer_metric_dict, name='_hparams')
		if self.config.use_wandb:
			self.wandb_run.config.update(hyper_params)



	def log_scalars(self, values: dict, step: int, filter_out_prefixes: list[str] = ['__'], print_values: bool = True):
		"""
		Log a dict of scalars.
		"""
		values = {k: v for k, v in values.items() if not k.startswith(tuple(filter_out_prefixes))}
		self.__check_step(step)

		if self.tb_writer is not None:
			# flatten dict
			values_flatten = self.__dict_flatten(values, sep='/')
			for k, v in values_flatten.items():
				if print_values:
					print(f'\t	- {k}: {v}')
				if v is None:
					continue
				# covert to scalar
				if is_array(v) and len(v.shape) >= 1:
					assert np.all(np.array(v.shape) == 1), f"can only log scalars but {k} has shape {v.shape}"
					v = v[(1,)*len(v.shape)]
				self.tb_writer.add_scalar(tag=k, scalar_value=float(v), global_step=step)

		if self.wandb_run is not None:
			self.wandb_run.log(values, step=step) #, commit=False)


	def log_plot(self, tag: str, plot: plt.Figure, step: int, parent_folder: str = '', log_as_image=True):
		"""
		Log a single plt plot.
		"""
		self.__check_step(step)
		if plot is None:
			return
		# log to file
		if self.config.save_plots:
			plot_folder = self.output_folder / 'plots' / parent_folder
			plot_path = plot_folder / (tag.replace('.', '/') + '__' + str(step) + '.png')
			plot_path.parent.mkdir(exist_ok=True, parents=True)
			plot.savefig(plot_path)

		if self.tb_writer is not None:
			self.tb_writer.add_figure(tag=tag, figure=plot, global_step=step)

		if self.wandb_run is not None:
			if log_as_image:
				plot = wandb.Image(plot)
			self.wandb_run.log({tag: plot}, step=step)#, commit=False)


	def log_plots(self, plots: Dict[str, plt.Figure], step: int, parent_folder: str = ''):
		"""
		Log multiple plt plots.
		:param plots: a flat dict where the key is the plot tag and the value is the plot.
		"""
		if plots is None:
			return
		for k, v in plots.items():
			self.log_plot(tag=k, plot=v, step=step, parent_folder=parent_folder)



	def log_image(self, tag: str, image: Float[Array, 'height width channels]'], step: int, parent_folder: str = ''):
		"""
		Log a single plt plot.
		"""
		self.__check_step(step)
		# log to file
		# plot_folder = self.output_folder / 'images' / parent_folder
		# plot_path = plot_folder / (tag.replace('.', '/') + '__' + str(step) + '.png')
		# plot_path.parent.mkdir(exist_ok=True, parents=True)
		# plot.savefig(plot_path)

		if self.tb_writer is not None:
			self.tb_writer.add_image(tag=tag, img_tensor=image.swapaxes(0, 2), global_step=step)

		if self.wandb_run is not None:
			self.wandb_run.log({tag: wandb.Image(image)}, step=step, commit=False)


	def commit(self, step):
		"""
		Commit wandb logs.
		Call this after logging all values for a step.
		"""
		if self.wandb_run is not None:
			self.wandb_run.log({}, step=step, commit=True)


	def __check_step(self, step):
		if np.isnan(step):
			print('WARNING: step in AbstractLogger log_...(...) is nan!')

	@staticmethod
	def array_as_scalar(x, name: str = ''):
		if is_array(x) and len(x.shape) >= 1:
			assert np.all(np.array(x.shape) == 1), f"array_as_scalar expect scalar array type but {name} has shape {x.shape}"
			return x[(1,) * len(x.shape)]
		else:
			return x
