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
