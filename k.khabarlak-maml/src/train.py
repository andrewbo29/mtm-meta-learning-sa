import torch
import math
import os
import time
import json
import logging

import spsa.multiclass_weights_optimize as weighting

from os import path

from torchmeta.utils.data import BatchMetaDataLoader

from maml.datasets import get_benchmark_by_name
from maml.metalearners import ModelAgnosticMetaLearning


def main(args):
    logging.basicConfig(level=logging.INFO if args.silent else logging.DEBUG)
    device = torch.device('cuda' if args.use_cuda and torch.cuda.is_available() else 'cpu')

    if not path.exists(args.output_folder):
        os.makedirs(args.output_folder)
        logging.debug('Creating folder `{0}`'.format(args.output_folder))

    if args.run_name is None:
        args.run_name = time.strftime('%Y-%m-%d_%H%M%S')

    folder = path.join(args.output_folder, args.run_name)
    os.makedirs(folder, exist_ok=False)
    logging.debug('Creating folder `{0}`'.format(folder))

    args.folder = path.abspath(args.folder)
    args.model_path = path.abspath(path.join(folder, 'model.th'))
    # Save the configuration in a config.json file
    with open(path.join(folder, 'config.json'), 'w') as f:
        stored_args = argparse.Namespace(**vars(args))
        stored_args.folder = path.relpath(stored_args.folder, folder)
        stored_args.model_path = path.relpath(stored_args.model_path, folder)
        json.dump(vars(stored_args), f, indent=2)
    logging.info('Saving configuration file in `{0}`'.format(
        path.abspath(path.join(folder, 'config.json'))))

    benchmark = get_benchmark_by_name(args.dataset,
                                      args.folder,
                                      args.num_ways,
                                      args.num_shots,
                                      args.num_shots_test,
                                      args.no_max_pool,
                                      hidden_size=args.hidden_size)

    meta_train_dataloader = BatchMetaDataLoader(benchmark.meta_train_dataset,
                                                batch_size=args.batch_size,
                                                shuffle=True,
                                                num_workers=args.num_workers,
                                                pin_memory=True)
    meta_val_dataloader = BatchMetaDataLoader(benchmark.meta_val_dataset,
                                              batch_size=args.batch_size,
                                              shuffle=True,
                                              num_workers=args.num_workers,
                                              pin_memory=True)
    meta_optimizer = torch.optim.Adam(benchmark.model.parameters(), lr=args.meta_lr)
    metalearner = ModelAgnosticMetaLearning(benchmark.model,
                                            meta_optimizer,
                                            first_order=args.first_order,
                                            num_adaptation_steps=args.num_steps,
                                            step_size=args.step_size,
                                            loss_function=benchmark.loss_function,
                                            device=device)

    if args.load is not None:
        with open(args.load, 'rb') as f:
            benchmark.model.load_state_dict(torch.load(f, map_location=device))

    best_value = None

    alpha = weighting.get_param_strategy(args.spsa_alpha_strategy,
                                         args.spsa_alpha,
                                         args.spsa_alpha_exp_gamma,
                                         args.spsa_alpha_step_step_every,
                                         args.spsa_alpha_step_multiplier)
    beta = weighting.get_param_strategy(args.spsa_beta_strategy,
                                        args.spsa_beta,
                                        args.spsa_beta_exp_gamma,
                                        args.spsa_beta_step_step_every,
                                        args.spsa_beta_step_multiplier)

    if args.task_weighting == 'none':
        task_weighting = weighting.TaskWeightingNone(device)
    elif args.task_weighting == 'spsa-delta':
        task_weighting = weighting.SpsaWeighting(args.batch_size, alpha, beta, device)
    elif args.task_weighting == 'sin':
        task_weighting = weighting.SinWeighting(args.batch_size, device)
    elif args.task_weighting == 'gradient':
        task_weighting = weighting.GradientWeighting(args.use_inner_optimizer, args.batch_size, device=device)
        meta_optimizer = torch.optim.Adam(list(benchmark.model.parameters()) + task_weighting.outer_optimization_weights,
                                          lr=args.meta_lr)
    elif args.task_weighting == 'gradient-novel-loss':
        task_weighting = weighting.GradientNovelLossWeighting(args.use_inner_optimizer, args.batch_size, device=device)
        meta_optimizer = torch.optim.Adam(list(benchmark.model.parameters()) + task_weighting.outer_optimization_weights,
                                          lr=args.meta_lr)
    else:
        raise ValueError(f'Unknown weighting value: {args.task_weighting}')

    # Training loop
    epoch_desc = 'Epoch {{0: <{0}d}}'.format(1 + int(math.log10(args.num_epochs)))

    for epoch in range(args.num_epochs):
        metalearner.train(meta_train_dataloader,
                          task_weighting,
                          epoch,
                          max_batches=args.num_batches,
                          silent=args.silent,
                          desc='Training',
                          leave=False)
        results = metalearner.evaluate(meta_val_dataloader,
                                       max_batches=args.num_batches,
                                       silent=args.silent,
                                       desc=epoch_desc.format(epoch + 1))

        # Save best model
        if 'accuracies_after' in results:
            if (best_value is None) or (best_value < results['accuracies_after']):
                best_value = results['accuracies_after']
                save_model = True
        elif (best_value is None) or (best_value > results['mean_outer_loss']):
            best_value = results['mean_outer_loss']
            save_model = True
        else:
            save_model = False

        if save_model and (args.output_folder is not None):
            with open(args.model_path, 'wb') as f:
                torch.save(benchmark.model.state_dict(), f)

    if hasattr(benchmark.meta_train_dataset, 'close'):
        benchmark.meta_train_dataset.close()
        benchmark.meta_val_dataset.close()


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser('MAML')

    # General
    parser.add_argument('folder', type=str,
                        help='Path to the folder the data is downloaded to.')
    parser.add_argument('--load', type=str, default=None,
                        help='Finetune already trained model')
    parser.add_argument('--dataset', type=str,
                        choices=['sinusoid', 'omniglot', 'miniimagenet', 'tieredimagenet', 'cifarfs', 'fc100'],
                        default='omniglot', help='Name of the dataset (default: omniglot).')
    parser.add_argument('--output-folder', type=str, default=None,
                        help='Path to the output folder to save the model.')
    parser.add_argument('--num-ways', type=int, default=5,
                        help='Number of classes per task (N in "N-way", default: 5).')
    parser.add_argument('--num-shots', type=int, default=5,
                        help='Number of training example per class (k in "k-shot", default: 5).')
    # following Ravi: "15 examples per class were used for evaluating the post-update meta-gradient"
    parser.add_argument('--num-shots-test', type=int, default=15,
                        help='Number of test example per class. If negative, same as the number '
                             'of training examples `--num-shots` (default: 15).')

    # Model
    parser.add_argument('--hidden-size', type=int, default=64,
                        help='Number of channels in each convolution layer of the network '
                             '(default: 64).')
    # Should be False for cifarfs and miniimagenet, True for omniglot
    parser.add_argument('--no-max-pool', action='store_true', default=False,
                        help='True to use strided conv model, False to use MaxPooled model')

    # Optimization
    parser.add_argument('--batch-size', type=int, default=25,
                        help='Number of tasks in a batch of tasks (default: 25).')
    parser.add_argument('--num-steps', type=int, default=1,
                        help='Number of fast adaptation steps, ie. gradient descent '
                             'updates (default: 1).')
    parser.add_argument('--num-epochs', type=int, default=50,
                        help='Number of epochs of meta-training (default: 50).')
    parser.add_argument('--num-batches', type=int, default=100,
                        help='Number of batch of tasks per epoch (default: 100).')
    parser.add_argument('--step-size', type=float, default=0.1,
                        help='Size of the fast adaptation step, ie. learning rate in the '
                             'gradient descent update (default: 0.1).')
    parser.add_argument('--first-order', action='store_true',
                        help='Use the first order approximation, do not use higher-order '
                             'derivatives during meta-optimization.')
    parser.add_argument('--meta-lr', type=float, default=0.001,
                        help='Learning rate for the meta-optimizer (optimization of the outer '
                             'loss). The default optimizer is Adam (default: 1e-3).')

    # SPSA
    parser.add_argument('--task-weighting', type=str,
                        choices=['none', 'spsa-delta', 'sin'], default='none',
                        help='Type of multi-tasking weighting')

    parser.add_argument('--spsa-alpha-strategy', type=str,
                        choices=['exponential', 'constant', 'step'],
                        default='exponential', help='alpha weighting strategy for spsa')
    parser.add_argument('--spsa-alpha', type=float, default=0.25,
                        help='alpha parameter to the spsa method')
    parser.add_argument('--spsa-alpha-exp-gamma', type=float, default=1 / 6.,
                        help='Gamma parameter for exponential alpha weighting')
    parser.add_argument('--spsa-alpha-step-step-every', type=int,
                        help='Step value for SpsaStepParamStrategy')
    parser.add_argument('--spsa-alpha-step-multiplier', type=float,
                        help='Multiplier value SpsaStepParamStrategy. Should be < 1.')

    parser.add_argument('--spsa-beta-strategy', type=str,
                        choices=['exponential', 'constant', 'step'],
                        default='exponential', help='beta weighting strategy for spsa')
    parser.add_argument('--spsa-beta', type=float, default=15.,
                        help='beta parameter to the spsa method')
    parser.add_argument('--spsa-beta-exp-gamma', type=float, default=1 / 24.,
                        help='Gamma parameter for exponential beta weighting')
    parser.add_argument('--spsa-beta-step-step-every', type=int,
                        help='Step value for SpsaStepParamStrategy')
    parser.add_argument('--spsa-beta-step-multiplier', type=float,
                        help='Multiplier value SpsaStepParamStrategy')

    parser.add_argument('--normalize-spsa-weights-after', type=int, default=None,
                        help='normalize spsa weights after specified number of iterations to avoid '
                             'loss explosion')
    parser.add_argument('--min-weight', type=float, default=None,
                        help='Min spsa weight allowed during optimization')

    # Gradient weighting
    parser.add_argument('--use-inner-optimizer', action='store_true', default=False,
                        help='Use inner gradient optimizer for task weight optimization '
                             '(in contrast to optimizing weights together with network weights)')
    
    # Misc
    parser.add_argument('--run-name', type=str, default=None, help='Custom name for run results')
    parser.add_argument('--num-workers', type=int, default=1,
                        help='Number of workers to use for data-loading (default: 1).')
    parser.add_argument('--silent', action='store_true', default=False)
    parser.add_argument('--use-cuda', action='store_true')

    args = parser.parse_args()

    if args.num_shots_test <= 0:
        args.num_shots_test = args.num_shots

    main(args)
