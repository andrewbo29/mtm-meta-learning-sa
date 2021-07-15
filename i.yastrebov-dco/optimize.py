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

def optimize(weights, losses, iteration_num, alpha = 0, beta = 0):
    delta_n = delta_fabric(len(losses))
    if alpha == 0:
        alpha_n = alpha_fabric(iteration_num)
    else:
        alpha_n = alpha
    if beta == 0:
        beta_n = beta_fabric(iteration_num)
    else:
        beta_n = beta
    y_plus = y_loss([weights[i] + beta_n * delta_n[i] for i in range(len(weights))], losses)
    y_minus = y_loss([weights[i] - beta_n * delta_n[i] for i in range(len(weights))], losses)
    weights_update = weights - alpha_n * np.multiply(delta_n, (y_plus - y_minus) / (2 * beta_n))
    return weights_update
    
def optimize_weights_track(weights, losses_2, iteration_num, alpha = 0, beta = 0):
    delta_n = delta_fabric(len(weights))
    if alpha == 0:
        alpha_n = alpha_fabric(iteration_num)
    else:
        alpha_n = alpha
    if beta == 0:
        beta_n = beta_fabric(iteration_num)
    else:
        beta_n = beta
    y_plus = y_loss([weights[i] + beta_n * delta_n[i] for i in range(len(weights))], losses_2[0])
    y_minus = y_loss([weights[i] - beta_n * delta_n[i] for i in range(len(weights))], losses_2[1])
    weights_update = weights - alpha_n * np.multiply(delta_n, (y_plus - y_minus) / (2 * beta_n))
    return weights_update