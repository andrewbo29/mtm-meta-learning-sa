import abc

import numpy as np
import torch

SPSA_GAMMA = 1. / 6


class TaskWeightingBase:
    def __init__(self, device):
        self.device = device
        self.weights = None

    @abc.abstractmethod
    def compute_weighted_loss(self, iteration, losses):
        pass

    @abc.abstractmethod
    def after_gradient_step(self, iteration, losses):
        pass



class TaskWeightingNone(TaskWeightingBase):
    def after_gradient_step(self, iteration, losses):
        pass

    def compute_weighted_loss(self, iteration, losses):
        return losses.mean()


class SpsaWeightingDelta(TaskWeightingBase):
    def __init__(self, num_tasks_in_batch, device):
        super().__init__(device)

        self.num_tasks_in_batch = num_tasks_in_batch
        self.weights = np.array([1. / num_tasks_in_batch] * num_tasks_in_batch, dtype=np.float32)

    @staticmethod
    def _delta(size):
        return np.where(np.random.binomial(1, 0.5, size=size) == 0, -1, 1).astype(np.float32)

    @staticmethod
    def _alpha(iteration):
        return np.float32(0.25 / (iteration ** SPSA_GAMMA))

    @staticmethod
    def _beta(iteration):
        return np.float32(15. / (iteration ** (SPSA_GAMMA / 4)))

    @staticmethod
    def _compute_weighted_loss(weights, losses, device):
        weights = torch.from_numpy(weights).to(device)
        mul = 1. / (weights ** 2)
        add = torch.log(weights ** 2)
        return (losses * mul + add).mean()

    def compute_weighted_loss(self, iteration, losses):
        return self._compute_weighted_loss(self.weights, losses, self.device)

    def after_gradient_step(self, iteration, losses):
        if iteration == 0:
            return

        n = len(self.weights)
        delta_n = self._delta(n)
        alpha_n = self._alpha(iteration)
        beta_n = self._beta(iteration)

        y_plus = self._compute_weighted_loss(self.weights + beta_n * delta_n, losses, self.device) \
            .detach().cpu().numpy()
        y_minus = self._compute_weighted_loss(self.weights - beta_n * delta_n, losses, self.device) \
            .detach().cpu().numpy()

        self.weights -= alpha_n * np.multiply(delta_n, (y_plus - y_minus) / (2 * beta_n))
