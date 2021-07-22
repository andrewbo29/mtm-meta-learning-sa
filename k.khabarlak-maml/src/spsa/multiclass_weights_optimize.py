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


class SpsaTrackWeighting(SpsaWeighting):
    def __init__(self, num_tasks_in_batch, alpha, beta, device):
        super().__init__(num_tasks_in_batch, alpha, beta, device)
        self.old_losses = None

    def update_inner_weights(self, iteration, losses):
        if self.old_losses is None:
            self.old_losses = losses.detach()
            return

        n = len(self.weights)
        delta_n = delta(n)
        alpha_n = self.alpha(iteration)
        beta_n = self.beta(iteration)

        y_plus = compute_weighted_loss(self.weights + beta_n * delta_n, self.old_losses, self.device) \
            .detach().cpu().numpy()
        y_minus = compute_weighted_loss(self.weights - beta_n * delta_n, losses, self.device) \
            .detach().cpu().numpy()

        self.weights -= alpha_n * np.multiply(delta_n, (y_plus - y_minus) / (2 * beta_n))

        self.old_losses = losses.detach()


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


class SpsaWeightingPerClass(TaskWeightingBase):
    def __init__(self, max_classes, class_info_label, skip_for_iterations,
                 alpha: SpsaParamStrategy, beta: SpsaParamStrategy, device):
        super().__init__(device)

        self.class_info_label = class_info_label
        self.skip_for_iterations = skip_for_iterations

        self.alpha = alpha
        self.beta = beta

        self.weights = np.array([1.] * max_classes, dtype=np.float32)
        self.iteration = None
        self.batch = None

        self.weights_update = None
        self.class_ids = None

    @property
    def test_class_info_label(self):
        return 'test' + self.class_info_label

    def before_gradient_step(self, iteration, batch):
        self.iteration = iteration
        self.batch = batch

    def _get_weights_for_batch_items(self, class_ids):
        weights_for_batch = np.ndarray(shape=(len(class_ids),))
        for idx, class_id in enumerate(class_ids.flatten()):
            weights_for_batch[idx] = self.weights[class_id]
        return weights_for_batch

    def _update_weights_for_batch_items(self, weights_update, class_ids):
        num_tasks = len(self.batch[self.test_class_info_label])
        for update, class_id in zip(weights_update, class_ids):
            # / num_task for the update to be of the same scale
            # as ordinary spsa
            self.weights[class_id] -= update / num_tasks

    def _compute_weights_update(self, weights, outer_losses_for_each_image):
        n = len(weights)
        delta_n = delta(n)
        alpha_n = self.alpha(self.iteration)
        beta_n = self.beta(self.iteration)

        y_plus = compute_weighted_loss(weights + beta_n * delta_n, outer_losses_for_each_image, self.device) \
            .detach().cpu().numpy()
        y_minus = compute_weighted_loss(weights - beta_n * delta_n, outer_losses_for_each_image, self.device) \
            .detach().cpu().numpy()

        return alpha_n * np.multiply(delta_n, (y_plus - y_minus) / (2 * beta_n))

    def compute_weighted_losses_for_each_image(self, task_id, outer_losses_for_each_image):
        if self.iteration % 100 < self.skip_for_iterations:
            return outer_losses_for_each_image.mean()

        # 'test_class_ids' because we weight outer losses (computed on Test (Query) subset)
        class_ids = self.batch[self.test_class_info_label][task_id]
        weights = self._get_weights_for_batch_items(class_ids)

        result_loss = compute_weighted_loss(weights, outer_losses_for_each_image, self.device)

        if self.iteration > 0:
            self.weights_update = self._compute_weights_update(weights, outer_losses_for_each_image)
            self.class_ids = class_ids

        return result_loss

    def compute_weighted_loss(self, iteration, losses):
        return losses.mean()

    def update_inner_weights(self, iteration, losses):
        if self.weights_update is not None:
            self._update_weights_for_batch_items(self.weights_update, self.class_ids)


class GradientWeightingBase(TaskWeightingBase, ABC):
    def __init__(self, use_inner_optimizer, num_tasks_in_batch, device):
        super().__init__(device)
        self.weights = nn.Parameter(
            torch.ones(size=(num_tasks_in_batch,),
                       requires_grad=True,
                       dtype=torch.float,
                       device=device))
        self.optimizer = None
        if use_inner_optimizer:
            self.optimizer = torch.optim.Adam([self.weights], lr=1e-3)

    @property
    def outer_optimization_weights(self):
        return [] if self.optimizer is None else [self.weights]

    def update_inner_weights(self, iteration, losses):
        if self.optimizer is None:
            # Update is done in backward pass
            return
        self.optimizer.step()
        self.optimizer.zero_grad()


class GradientWeighting(GradientWeightingBase):
    def __init__(self, use_inner_optimizer, num_tasks_in_batch, device):
        super().__init__(use_inner_optimizer, num_tasks_in_batch, device)

    def compute_weighted_loss(self, iteration, losses):
        return (losses * self.weights).mean()


class GradientNovelLossWeighting(GradientWeightingBase):
    def __init__(self, use_inner_optimizer, num_tasks_in_batch, device):
        super().__init__(use_inner_optimizer, num_tasks_in_batch, device)

    def compute_weighted_loss(self, iteration, losses):
        mul = 1. / (self.weights ** 2)
        add = torch.log(self.weights ** 2)
        return (losses * mul + add).mean()


class WeightNormalizer:
    def __init__(self, normalize_after: Optional[int]):
        self.normalize_after = normalize_after

    def normalize(self, iteration, weights):
        if (self.normalize_after is None) or (iteration == 0) or (iteration % self.normalize_after != 0):
            return weights
        return weights / np.linalg.norm(weights, ord=2)
