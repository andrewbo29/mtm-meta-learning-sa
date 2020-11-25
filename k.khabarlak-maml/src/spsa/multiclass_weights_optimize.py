# coding=utf-8
import numpy as np
import copy
import torch


# spsa_alpha = lambda x: 0.001
# spsa_beta = lambda x: 0.001

# spsa_alpha = lambda x: 0.0001
# spsa_beta = lambda x: 0.01

spsa_gamma = 1. / 6
spsa_alpha = lambda x: 0.25 / (x ** spsa_gamma)
spsa_beta = lambda x: 15. / (x ** (spsa_gamma / 4))


def delta_fabric(d):
    return np.where(np.random.binomial(1, 0.5, size=d) == 0, -1, 1)
    # return np.random.binomial(1, 0.5, size=d)


def alpha_fabric(iteration_num):
    return spsa_alpha(iteration_num)


def beta_fabric(iteration_num):
    return spsa_beta(iteration_num)


def y_loss(weights, losses):
    y = np.sum([1 / (float(weights[i]) ** 2) * losses[i].detach().numpy() + np.log(weights[i] ** 2) for i in range(len(losses))])

    # y = np.sum([1 / (float(weights[i]) ** 4) * losses[i].detach().numpy() + np.log(weights[i] ** 2) for i in range(len(losses))])

    # y = np.sum([1 / (float(weights[i]) ** 2) * losses[i].detach().numpy() + np.log(weights[i] ** 2) for i in
    #             range(len(losses) - 1)])
    # y += losses[-1].detach().numpy()

    return y


def y_loss_one_loss(losses):
    return losses[0].detach().numpy()


def optimize(weights, losses, iteration_num):
    delta_n = delta_fabric(len(losses))
    # delta_n = delta_fabric(len(losses) - 1)
    alpha_n = alpha_fabric(iteration_num)
    beta_n = beta_fabric(iteration_num)

    y_plus = y_loss([weights[i] + beta_n * delta_n[i] for i in range(len(weights))], losses)
    y_minus = y_loss([weights[i] - beta_n * delta_n[i] for i in range(len(weights))], losses)

    weights_update = weights - alpha_n * np.multiply(delta_n, (y_plus - y_minus) / (2 * beta_n))

    return weights_update


def compute_mean_weighted_loss(weights, losses, device):
    weights = torch.from_numpy(weights).to(device)
    mul = 1. / (weights ** 2)
    add = torch.log(weights ** 2)
    return (losses.dot(mul) + add).mean()


def update_params_all_loss(net, weights, batch, device, opt, iteration_num):
    # lr = 0.001
    lr = 0.01

    optim = torch.optim.SGD(params=net.model.parameters(), lr=lr)

    net.model.to(device)
    net.model.train()

    outer_losses, _ = net.get_outer_losses(batch)
    loss = compute_mean_weighted_loss(weights, outer_losses, device)

    loss.backward()
    optim.step()

    outer_losses, _ = net.get_outer_losses(batch)
    return compute_mean_weighted_loss(weights, outer_losses, device)


def optimize_grad_all_loss(weights, iteration_num, net, batch, device, opt):
    delta_n = delta_fabric(len(weights))
    alpha_n = alpha_fabric(iteration_num)
    beta_n = beta_fabric(iteration_num)

    net_plus = copy.deepcopy(net)
    weights_plus = (weights + beta_n * delta_n).astype(np.float32)
    y_plus = update_params_all_loss(net_plus, weights_plus, batch, device, opt, iteration_num)

    net_minus = copy.deepcopy(net)
    weights_minus = (weights - beta_n * delta_n).astype(np.float32)
    y_minus = update_params_all_loss(net_minus, weights_minus, batch, device, opt, iteration_num)

    difference = (y_plus - y_minus).detach().cpu().numpy()
    weights_update = weights - alpha_n * np.multiply(delta_n, difference / (2 * beta_n))

    return weights_update.astype(np.float32)


def update_params_one_loss(net, weights, batches, loss_fn, device, opt, iteration_num):
    # lr = 0.03
    lr = 0.01

    # epoch_num = iteration_num // opt.iterations
    # step_ep = 10
    # lr = 0.001 * 1 / 10 ** (epoch_num // step_ep)

    # optim = torch.optim.Adam(params=net.parameters(), lr=0.001)
    optim = torch.optim.SGD(params=net.parameters(), lr=lr, weight_decay=opt.weight_decay)

    net.to(device)
    net.train()

    losses = []
    for i in range(1, len(batches)):
        x, y = batches[i]
        x, y = x.to(device), y.to(device)

        net_output = net(x)
        loss, _, __ = loss_fn(net_output, target=y, n_support=opt.num_support_tr, args=opt, is_test=False)
        losses.append(loss)

    loss_all = 0
    for j, loss_val in enumerate(losses):
        loss_all += 1 / (float(weights[j]) ** 2) * loss_val + np.log(weights[j] ** 2)

    loss_all.backward()
    optim.step()

    losses = []
    x, y = batches[0]
    x, y = x.to(device), y.to(device)
    net_output = net(x)
    loss, _, __ = loss_fn(net_output, target=y, n_support=opt.num_support_tr, args=opt, is_test=False)
    losses.append(loss)

    return y_loss_one_loss(losses)


def optimize_grad_one_loss(weights, iteration_num, net, batches, loss_fn, device, opt):
    delta_n = delta_fabric(len(weights))
    alpha_n = alpha_fabric(iteration_num)
    beta_n = beta_fabric(iteration_num)

    net_plus = copy.deepcopy(net)
    weights_plus = [weights[i] + beta_n * delta_n[i] for i in range(len(weights))]
    y_plus = update_params_one_loss(net_plus, weights_plus, batches, loss_fn, device, opt, iteration_num)

    net_minus = copy.deepcopy(net)
    weights_minus = [weights[i] - beta_n * delta_n[i] for i in range(len(weights))]
    y_minus = update_params_one_loss(net_minus, weights_minus, batches, loss_fn, device, opt, iteration_num)

    weights_update = weights - alpha_n * np.multiply(delta_n, (y_plus - y_minus) / (2 * beta_n))

    return weights_update


def update_loss_scale(net, weights, batches, loss_fn, device, opt, alphas):
    losses = []
    for i, batch in enumerate(batches):
        x, y = batch
        x, y = x.to(device), y.to(device)

        net_output = net(x)
        loss, _, __ = loss_fn(net_output, target=y, n_support=opt.num_support_tr, args=opt, is_test=False, alpha=alphas[i])
        losses.append(loss)

    return y_loss(weights, losses)


def optimize_loss_scale(alphas, weights, iteration_num, net, batches, loss_fn, device, opt):
    delta_n = delta_fabric(len(alphas))
    alpha_n = alpha_fabric(iteration_num)
    beta_n = beta_fabric(iteration_num)

    alphas_plus = [alphas[i] + beta_n * delta_n[i] for i in range(len(alphas))]
    y_plus = update_loss_scale(net, weights, batches, loss_fn, device, opt, alphas_plus)

    alphas_minus = [alphas[i] - beta_n * delta_n[i] for i in range(len(alphas))]
    y_minus = update_loss_scale(net, weights, batches, loss_fn, device, opt, alphas_minus)

    alphas_update = alphas - alpha_n * np.multiply(delta_n, (y_plus - y_minus) / (2 * beta_n))

    return alphas_update


def optimize_weights_track(weights, losses_2, iteration_num):
    delta_n = delta_fabric(len(weights))
    alpha_n = alpha_fabric(iteration_num)
    beta_n = beta_fabric(iteration_num)

    y_plus = y_loss([weights[i] + beta_n * delta_n[i] for i in range(len(weights))], losses_2[0])
    y_minus = y_loss([weights[i] - beta_n * delta_n[i] for i in range(len(weights))], losses_2[1])

    weights_update = weights - alpha_n * np.multiply(delta_n, (y_plus - y_minus) / (2 * beta_n))

    return weights_update
