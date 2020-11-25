# -*- coding: utf-8 -*-
import numpy as np

spsa_gamma = 1. / 6
spsa_alpha = lambda x: 0.25 / (x ** spsa_gamma)
spsa_beta = lambda x: 15. / (x ** (spsa_gamma / 4))

def delta_fabric(d):
    return np.where(np.random.binomial(1, 0.5, size = d) == 0, -1, 1)

def alpha_fabric(iteration_num):
    return spsa_alpha(iteration_num)

def beta_fabric(iteration_num):
    return spsa_beta(iteration_num)

def y_loss(weights, losses):
    y = np.sum([1 / (float(weights[i]) ** 2) * losses[i].detach().cpu().numpy() +
                np.log(weights[i] ** 2) for i in range(len(losses))])
    return y

def optimize(weights, losses, iteration_num):
    delta_n = delta_fabric(len(losses))
    alpha_n = alpha_fabric(iteration_num)
    beta_n = beta_fabric(iteration_num)
    y_plus = y_loss([weights[i] + beta_n * delta_n[i] for i in range(len(weights))], losses)
    y_minus = y_loss([weights[i] - beta_n * delta_n[i] for i in range(len(weights))], losses)
    weights_update = weights - alpha_n * np.multiply(delta_n, (y_plus - y_minus) / (2 * beta_n))
    return weights_update