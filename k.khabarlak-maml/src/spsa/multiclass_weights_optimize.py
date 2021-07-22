import abc
from abc import ABC
from typing import Optional

import numpy as np
import torch
from torch import nn


def compute_weighted_loss(weights, losses, device):
    weights = torch.from_numpy(weights).to(device)
    mul = 1. / (weights ** 2)
    add = torch.log(weights ** 2)
    return (losses * mul + add).mean()


def delta(size):
    return np.where(np.random.binomial(1, 0.5, size=size) == 0, -1, 1).astype(np.float32)


class SpsaParamStrategy:
    @abc.abstractmethod
    def __call__(self, iteration):
        pass


class SpsaParamExponentialStrategy(SpsaParamStrategy):
    def __init__(self, initial_value, gamma):
        self._initial_value = initial_value
        self._gamma = gamma

    def __call__(self, iteration):
        return np.float32(self._initial_value / (iteration ** self._gamma))


class SpsaParamConstantStrategy(SpsaParamStrategy):
    def __init__(self, value):
        self._value = value

    def __call__(self, iteration):
        return self._value


class SpsaParamStepStrategy(SpsaParamStrategy):
    def __init__(self, initial_value, step_every, multiplier):
        self._value = initial_value
        self._step_every = step_every
        self._multiplier = multiplier
        self._last_update = 0

    def __call__(self, iteration):
        if self._last_update != iteration and iteration % self._step_every == 0:
            print(f'SpsaParamStepStrategy from: {self._value} to {self._value * self._multiplier}')
            self._value = self._value * self._multiplier
            self._last_update = iteration
        return self._value


def get_param_strategy(strategy_type: str, initial_value: float, exp_gamma: Optional[float],
                       step: Optional[int], step_multiplier: Optional[float]):
    if strategy_type == 'exponential':
        if exp_gamma is None:
            raise ValueError("exp_gamma can't be None for exponential strategy")
        return SpsaParamExponentialStrategy(initial_value, exp_gamma)
    elif strategy_type == 'constant':
        return SpsaParamConstantStrategy(initial_value)
    elif strategy_type == 'step':
        if step is None or step_multiplier is None:
            raise ValueError('StepStrategy requested, but parameters are not set!')
        return SpsaParamStepStrategy(initial_value, step, step_multiplier)
    else:
        raise ValueError(f'Unknown {strategy_type=}')

class TaskWeightingBase:
    def __init__(self, device):
        self.device = device
        self.weights = None

    def before_gradient_step(self, iteration, batch):
        pass

    def compute_weighted_losses_for_each_image(self, task_id, outer_losses_for_each_image):
        return outer_losses_for_each_image.mean()

    @abc.abstractmethod
    def compute_weighted_loss(self, iteration, losses):
        pass

    @abc.abstractmethod
    def update_inner_weights(self, iteration, losses):
        pass



class TaskWeightingNone(TaskWeightingBase):
    def update_inner_weights(self, iteration, losses):
        pass

    def compute_weighted_loss(self, iteration, losses):
        return losses.mean()


class SpsaWeighting(TaskWeightingBase):
    def __init__(self, num_tasks_in_batch,
                 alpha: SpsaParamStrategy,
                 beta: SpsaParamStrategy,
                 device):
        super().__init__(device)

        self.num_tasks_in_batch = num_tasks_in_batch
        self.alpha = alpha
        self.beta = beta
        self.weights = np.array([1. / num_tasks_in_batch] * num_tasks_in_batch, dtype=np.float32)

    def compute_weighted_loss(self, iteration, losses):
        return compute_weighted_loss(self.weights, losses, self.device)

    def update_inner_weights(self, iteration, losses):
        if iteration == 0:
            return

        n = len(self.weights)
        delta_n = delta(n)
        alpha_n = self.alpha(iteration)
        beta_n = self.beta(iteration)

        y_plus = compute_weighted_loss(self.weights + beta_n * delta_n, losses, self.device) \
            .detach().cpu().numpy()
        y_minus = compute_weighted_loss(self.weights - beta_n * delta_n, losses, self.device) \
            .detach().cpu().numpy()

        self.weights -= alpha_n * np.multiply(delta_n, (y_plus - y_minus) / (2 * beta_n))


class SinWeighting(TaskWeightingBase):
    def __init__(self, num_tasks_in_batch, device):
        super().__init__(device)

        self.num_tasks_in_batch = num_tasks_in_batch

    def _get_weight(self, iteration, idx):
        return np.sin(iteration * np.pi / 2.0
                      + idx * np.pi / self.num_tasks_in_batch
                      + np.pi / (2 * self.num_tasks_in_batch)) + 0.9

    def compute_weighted_loss(self, iteration, losses: torch.Tensor):
        self.weights = torch.FloatTensor([self._get_weight(iteration, idx) for idx in range(self.num_tasks_in_batch)]) \
            .to(self.device)
        return (losses * self.weights).mean()

    def update_inner_weights(self, iteration, losses):
        pass
