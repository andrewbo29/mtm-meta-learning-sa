# -*- coding: utf-8 -*-
import argparse
import base64
import numpy as np
import os
import torch
import torch.nn.functional as F
from torchvision.models.resnet import resnet18

from dataloaders import BatchMetaDataLoaderWithLabels
from itertools import combinations
from models.classification_heads import ClassificationHead
from models.protonet_embedding import ProtoNetEmbedding
from models.ResNet12_embedding import resnet12
from optimize import optimize
from optimize import optimize_weights_track
from scipy.stats import rankdata
from torchmeta.transforms import Categorical, ClassSplitter
from torchmeta.utils.data import BatchMetaDataLoader
from torchvision.transforms import ColorJitter, Compose, Normalize, RandomCrop, RandomResizedCrop, RandomHorizontalFlip, Resize, ToTensor
from tqdm import tqdm
from utils import check_dir, count_accuracy, log, Timer

def get_model(options):
    # Choose the embedding network
    if options.network == 'ProtoNet':
        network = ProtoNetEmbedding().to(options.device)
    elif options.network == 'ResNet12':
        network = torch.nn.DataParallel(resnet12(options.device, avg_pool = False, drop_rate = .1, dropblock_size = 2).to(options.device))
    elif options.network == 'ResNet18':
       #  network = torch.hub.load('pytorch/vision', 'resnet18', pretrained = False, verbose = False).to(options.device)
       network = torch.nn.DataParallel(resnet18(pretrained=False).to(options.device))
    else:
        print ("Cannot recognize the network type")
        assert(False)

    # Choose the classification head
    if options.head == 'Proto':
        cls_head = ClassificationHead(options.device, base_learner='Proto').to(options.device)
    elif options.head == 'SVM-CS':
        cls_head = ClassificationHead(options.device, base_learner='SVM-CS').to(options.device)
    else:
        print ("Cannot recognize the base learner type")
        assert(False)

    return (network, cls_head)

def get_dataset(options):
    # Choose the learning datdset
    if options.dataset == 'miniImageNet':
        from torchmeta.datasets import MiniImagenet
        mean_pix = [x / 255 for x in [120.39586422,  115.59361427, 104.54012653]]
        std_pix = [x / 255 for x in [70.68188272,  68.27635443,  72.54505529]]
        if options.network == 'ResNet18':
            train_start = RandomResizedCrop(224)
        else:
            train_start = RandomCrop(84, padding = 8)
        dataset_train = MiniImagenet(
            "data",
            num_classes_per_task = options.train_way,
            transform = Compose([train_start,
                                 ColorJitter(brightness = .4,
                                             contrast = .4,
                                             saturation = .4),
                                 RandomHorizontalFlip(),
                                 ToTensor(),
                                 Normalize(mean = mean_pix,
                                           std = std_pix),
                                ]),
            target_transform = Categorical(num_classes = options.train_way),
            meta_train = True,
            download = True)
        dataset_train = ClassSplitter(dataset_train, shuffle = True,
                                      num_train_per_class = options.train_shot,
                                      num_test_per_class = options.train_query)
        dataloader_train = BatchMetaDataLoader(
            dataset_train,
            batch_size = options.episodes_per_batch,
            num_workers = options.num_workers)
        if options.network == 'ResNet18':
          dataset_val = MiniImagenet(
            "data",
            num_classes_per_task = options.test_way,
            transform = Compose([Resize(224),
                                 ToTensor(),
                                 Normalize(mean = mean_pix,
                                           std = std_pix),
                                ]),
            target_transform = Categorical(num_classes = options.test_way),
            meta_val = True,
            download = False)
        else:
          dataset_val = MiniImagenet(
            "data",
            num_classes_per_task = options.test_way,
            transform = Compose([ToTensor(),
                                 Normalize(mean = mean_pix,
                                           std = std_pix),
                                ]),
            target_transform = Categorical(num_classes = options.test_way),
            meta_val = True,
            download = False)
        dataset_val = ClassSplitter(dataset_val, shuffle = True,
                                    num_train_per_class = options.val_shot,
                                    num_test_per_class = options.val_query)
        dataloader_val = BatchMetaDataLoader(
          dataset_val,
          batch_size = 1,
          num_workers = options.num_workers)
    elif options.dataset == 'tieredImageNet':
        from torchmeta.datasets import TieredImagenet
        mean_pix = [x / 255 for x in [120.39586422,  115.59361427, 104.54012653]]
        std_pix = [x / 255 for x in [70.68188272,  68.27635443,  72.54505529]]
        dataset_train = TieredImagenet(
            "data",
            num_classes_per_task = options.train_way,
            transform = Compose([RandomCrop(84, padding = 8),
                                 ColorJitter(brightness = .4,
                                             contrast = .4,
                                             saturation = .4),
                                 RandomHorizontalFlip(),
                                 ToTensor(),
                                 Normalize(mean = mean_pix,
                                           std = std_pix),
                                ]),
            target_transform = Categorical(num_classes = options.train_way),
            meta_train = True,
            download = True)
        dataset_train = ClassSplitter(dataset_train, shuffle = True,
                                      num_train_per_class = options.train_shot,
                                      num_test_per_class = options.train_query)
        dataloader_train = BatchMetaDataLoader(
            dataset_train,
            batch_size = options.episodes_per_batch,
            num_workers = options.num_workers)
        dataset_val = TieredImagenet(
          "data",
          num_classes_per_task = options.test_way,
          transform = Compose([ToTensor(),
                               Normalize(mean = mean_pix,
                                         std = std_pix),
                              ]),
          target_transform = Categorical(num_classes = options.test_way),
          meta_val = True,
          download = False)
        dataset_val = ClassSplitter(dataset_val, shuffle = True,
                                    num_train_per_class = options.val_shot,
                                    num_test_per_class = options.val_query)
        dataloader_val = BatchMetaDataLoader(
          dataset_val,
          batch_size = 1,
          num_workers = options.num_workers)
    elif options.dataset == 'CIFAR_FS':
        from torchmeta.datasets import CIFARFS
        mean_pix = [x / 255 for x in [129.37731888, 124.10583864, 112.47758569]]
        std_pix = [x / 255 for x in [68.20947949, 65.43124043, 70.45866994]]
        if options.coarse:
            dataset_train = CIFARFS(
                "data",
                num_classes_per_task = 1,
                meta_train = True,
                download = True)
            dataset_train = ClassSplitter(dataset_train, shuffle = False,
                num_train_per_class = 1,
                num_test_per_class = 1)
            li = {}
            for i in range(len(dataset_train)):
                li[i] = dataset_train[(i,)]['train'].__getitem__(0)[1][0][0]
            sli = list(li.values())
            dli = {x: ix for ix, x in enumerate(set(sli))}
            if options.super_coarse:
                dli['aquatic_mammals'] = 21
                dli['fish'] = 21
                dli['flowers'] = 22
                dli['fruit_and_vegetables'] = 22
                dli['food_containers'] = 23
                dli['household_electrical_devices'] = 23
                dli['household_furniture'] = 23
                dli['insects'] = 24
                dli['non-insect_invertebrates'] = 24
                dli['large_carnivores'] = 25
                dli['reptiles'] = 25
                dli['large_natural_outdoor_scenes'] = 26
                dli['trees'] = 26
                dli['large_omnivores_and_herbivores'] = 27
                dli['medium_mammals'] = 27
                dli['people'] = 27
                dli['vehicles_1'] = 28
                dli['vehicles_2'] = 28
            nli = rankdata([dli[item] for item in sli], 'dense')
            def new__iter__(self):
                num_coarse = max(nli) + 1
                for ix in range(1, num_coarse):
                    for index in combinations([n for n in range(len(li)) if nli[n] == ix], self.num_classes_per_task):
                        yield self[index]
            def newsample_task(self):
                num = self.np_random.randint(1, max(nli) + 1)
                sample = [n for n in range(len(li)) if nli[n] == num]
                index = self.np_random.choice(sample, size=self.num_classes_per_task, replace=False)
                return self[tuple(index)]
            def new__len__(self):
                total_length = 0
                num_coarse = max(nli) + 1
                for jx in range(1, num_coarse):
                    num_classes, length = len([n for n in range(len(li)) if nli[n] == jx]), 1
                    for ix in range(1, self.num_classes_per_task + 1):
                        length *= (num_classes - ix + 1) / ix
                    total_length += length
                return int(total_length)
            CIFARFS.__iter__ = new__iter__
            CIFARFS.sample_task = newsample_task
            CIFARFS.__len__ = new__len__
        dataset_train = CIFARFS(
            "data",
            num_classes_per_task = options.train_way,
            transform = Compose([RandomCrop(32, padding = 4),
                                 ColorJitter(brightness = .4,
                                             contrast = .4,
                                             saturation = .4),
                                 RandomHorizontalFlip(),
                                 ToTensor(),
                                 Normalize(mean = mean_pix,
                                           std = std_pix),
                                ]),
            target_transform = Categorical(num_classes = options.train_way),
            meta_train = True,
            download = True)
        dataset_train = ClassSplitter(dataset_train, shuffle = True,
                                      num_train_per_class = options.train_shot,
                                      num_test_per_class = options.train_query)
        if options.coarse_weights:
            dataloader_train = BatchMetaDataLoaderWithLabels(
                dataset_train,
                batch_size = options.episodes_per_batch,
                num_workers = options.num_workers)
        else:
            dataloader_train = BatchMetaDataLoader(
                dataset_train,
                batch_size = options.episodes_per_batch,
                num_workers = options.num_workers)
        dataset_val = CIFARFS(
          "data",
          num_classes_per_task = options.test_way,
          transform = Compose([ToTensor(),
                               Normalize(mean = mean_pix,
                                         std = std_pix),
                              ]),
          target_transform = Categorical(num_classes = options.test_way),
          meta_val = True,
          download = False)
        dataset_val = ClassSplitter(dataset_val, shuffle = True,
                                    num_train_per_class = options.val_shot,
                                    num_test_per_class = options.val_query)
        dataloader_val = BatchMetaDataLoader(
          dataset_val,
          batch_size = 1,
          num_workers = options.num_workers)
    elif options.dataset == 'FC100':
        from torchmeta.datasets import FC100
        mean_pix = [x / 255 for x in [129.37731888, 124.10583864, 112.47758569]]
        std_pix = [x / 255 for x in [68.20947949, 65.43124043, 70.45866994]]
        if options.coarse:
            dataset_train = FC100(
                "data",
                num_classes_per_task = 1,
                meta_train = True,
                download = True)
            dataset_train = ClassSplitter(dataset_train, shuffle = False,
                num_train_per_class = 1,
                num_test_per_class = 1)
            li = {}
            for i in range(len(dataset_train)):
                li[i] = dataset_train[(i,)]['train'].__getitem__(0)[1][0][0]
            sli = list(li.values())
            dli = {x: ix for ix, x in enumerate(set(sli))}
            if options.super_coarse:
                dli['aquatic_mammals'] = 21
                dli['fish'] = 21
                dli['flowers'] = 22
                dli['fruit_and_vegetables'] = 22
                dli['food_containers'] = 23
                dli['household_electrical_devices'] = 23
                dli['household_furniture'] = 23
                dli['insects'] = 24
                dli['non-insect_invertebrates'] = 24
                dli['large_carnivores'] = 25
                dli['reptiles'] = 25
                dli['large_natural_outdoor_scenes'] = 26
                dli['trees'] = 26
                dli['large_omnivores_and_herbivores'] = 27
                dli['medium_mammals'] = 27
                dli['people'] = 27
                dli['vehicles_1'] = 28
                dli['vehicles_2'] = 28
            nli = rankdata([dli[item] for item in sli], 'dense')
            def new__iter__(self):
                num_coarse = max(nli) + 1
                for ix in range(1, num_coarse):
                    for index in combinations([n for n in range(len(li)) if nli[n] == ix], self.num_classes_per_task):
                        yield self[index]
            def newsample_task(self):
                num = self.np_random.randint(1, max(nli) + 1)
                sample = [n for n in range(len(li)) if nli[n] == num]
                index = self.np_random.choice(sample, size=self.num_classes_per_task, replace=False)
                return self[tuple(index)]
            def new__len__(self):
                total_length = 0
                num_coarse = max(nli) + 1
                for jx in range(1, num_coarse):
                    num_classes, length = len([n for n in range(len(li)) if nli[n] == jx]), 1
                    for ix in range(1, self.num_classes_per_task + 1):
                        length *= (num_classes - ix + 1) / ix
                    total_length += length
                return int(total_length)
            FC100.__iter__ = new__iter__
            FC100.sample_task = newsample_task
            FC100.__len__ = new__len__
        dataset_train = FC100(
            "data",
            num_classes_per_task = options.train_way,
            transform = Compose([RandomCrop(32, padding = 4),
                                 ColorJitter(brightness = .4,
                                             contrast = .4,
                                             saturation = .4),
                                 RandomHorizontalFlip(),
                                 ToTensor(),
                                 Normalize(mean = mean_pix,
                                           std = std_pix),
                                ]),
            target_transform = Categorical(num_classes = options.train_way),
            meta_train = True,
            download = True)
        dataset_train = ClassSplitter(dataset_train, shuffle = True,
                                      num_train_per_class = options.train_shot,
                                      num_test_per_class = options.train_query)
        if options.coarse_weights:
            dataloader_train = BatchMetaDataLoaderWithLabels(
                dataset_train,
                batch_size = options.episodes_per_batch,
                num_workers = options.num_workers)
        else:
            dataloader_train = BatchMetaDataLoader(
                dataset_train,
                batch_size = options.episodes_per_batch,
                num_workers = options.num_workers)
        dataset_val = FC100(
          "data",
          num_classes_per_task = options.test_way,
          transform = Compose([ToTensor(),
                               Normalize(mean = mean_pix,
                                         std = std_pix),
                              ]),
          target_transform = Categorical(num_classes = options.test_way),
          meta_val = True,
          download = False)
        dataset_val = ClassSplitter(dataset_val, shuffle = True,
                                    num_train_per_class = options.val_shot,
                                    num_test_per_class = options.val_query)
        dataloader_val = BatchMetaDataLoader(
          dataset_val,
          batch_size = 1,
          num_workers = options.num_workers)
    else:
        print ("Cannot recognize the dataset type")
        assert(False)

    return (dataloader_train, dataloader_val)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num-epoch', type=int, default=60,
                            help='number of training epochs')
    parser.add_argument('--save-epoch', type=int, default=10,
                            help='frequency of model saving')
    parser.add_argument('--train-episode', type=int, default=1000,
                            help='number of episodes per validation')
    parser.add_argument('--val-episode', type=int, default=2000,
                            help='number of episodes per validation')
    parser.add_argument('--train-shot', type=int, default=1,
                            help='number of support examples per training class')
    parser.add_argument('--val-shot', type=int, default=1,
                            help='number of support examples per validation class')
    parser.add_argument('--train-query', type=int, default=6,
                            help='number of query examples per training class')
    parser.add_argument('--val-query', type=int, default=15,
                            help='number of query examples per validation class')
    parser.add_argument('--train-way', type=int, default=5,
                            help='number of classes in one training episode')
    parser.add_argument('--test-way', type=int, default=5,
                            help='number of classes in one test (or validation) episode')
    parser.add_argument('--save-path', default='./experiments')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--num-workers', type=int, default=2,
                            help='number of cpu workers for loading data')
    parser.add_argument('--network', type=str, default='ProtoNet',
                            help='choose which embedding network to use: ProtoNet, ResNet12, ResNet18')
    parser.add_argument('--head', type=str, default='Proto',
                            help='choose which classification head to use: Proto, SVM-CS')
    parser.add_argument('--dataset', type=str, default='miniImageNet',
                            help='choose which classification head to use: miniImageNet, tieredImageNet, CIFAR_FS, FC100')
    parser.add_argument('--episodes-per-batch', type=int, default=8,
                            help='number of episodes per batch')
    parser.add_argument('--task-number', type=int, default=4,
                            help='number of task runs before update')
    parser.add_argument('--pretrain', type=int, default=20,
                            help='number of epochs without MTL')
    parser.add_argument('--tracking', action='store_true',
                            help='track loss for SPSA')
    parser.add_argument('--alpha', type=float, default=0.0,
                            help='alpha in SPSA optimization')
    parser.add_argument('--beta', type=float, default=0.0,
                            help='beta in SPSA optimization')
    parser.add_argument('--decrease-alpha', action='store_true',
                            help='decrease alpha in SPSA optimization')
    parser.add_argument('--coarse', action='store_true',
                            help='use coarse classes only for subtasks')
    parser.add_argument('--super-coarse', action='store_true',
                            help='use super coarse classes only for subtasks')
    parser.add_argument('--coarse-weights', action='store_true',
                            help='use separate weight for each coarse class')
    parser.add_argument('--half-spsa', action='store_false',
                            help='do SPSA optimization only for epoch half, do backpropagation only for epoch half')
    parser.add_argument('--epoch-spsa', action='store_true',
                            help='do SPSA after every epoch')
    parser.add_argument('--train-weights', action='store_true',
                            help='train weights like a layer')
    parser.add_argument('--train-weights-layer', action='store_true',
                            help='train weights like a layer with spsa loss')
    parser.add_argument('--train-weights-opt', action='store_true',
                            help='train weights with separate optimizer')
    parser.add_argument('--weight_decay', type=float, default=5e-4)

    opt = parser.parse_args()

    (dataloader_train, dataloader_val) = get_dataset(opt)

    check_dir('./experiments/')
    check_dir(opt.save_path)

    log_file_path = os.path.join(opt.save_path, "train_log.txt")
    log(log_file_path, str(vars(opt)))

    weight_file_path = os.path.join(opt.save_path, "weight_log.txt")

    (embedding_net, cls_head) = get_model(opt)

    if opt.train_weights:
      weights = torch.ones(opt.task_number, device=opt.device, requires_grad = True)
      if opt.train_weights_layer:
        optimizer = torch.optim.SGD([{'params': embedding_net.parameters()},
                                    {'params': cls_head.parameters()},
                                    {'params': weights}], lr=0.1,
                                    momentum=0.9, weight_decay=5e-4, nesterov=True)
      elif opt.train_weights_opt:
        optimizer = torch.optim.SGD([{'params': embedding_net.parameters()},
                                   {'params': cls_head.parameters()}], lr=0.1,
                                   momentum=0.9, weight_decay=5e-4, nesterov=True)
        optimizer1 = torch.optim.Adam([{'params': weights}], lr=1e-3)
    elif opt.coarse_weights:
      weights = np.ones(20)
      loss_hist = torch.zeros(20, device=opt.device)
      optimizer = torch.optim.SGD([{'params': embedding_net.parameters()},
                                   {'params': cls_head.parameters()}], lr=0.1,
                                   momentum=0.9, weight_decay=5e-4, nesterov=True)
    else:
      weights = np.array([1 / opt.task_number for _ in range(opt.task_number)])
      optimizer = torch.optim.SGD([{'params': embedding_net.parameters()},
                                   {'params': cls_head.parameters()}], lr=0.1,
                                   momentum=0.9, weight_decay=opt.weight_decay, nesterov=True)

    lambda_epoch = lambda e: (1.0 if e < 20 else (0.06 if e < 40 else 0.012 if e < 50 else (0.0024))) if e < opt.pretrain or opt.train_weights else (1.0 if e < 20 else (0.06 if e < 40 else 0.012 if e < 50 else (0.0024))) / 10
    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_epoch, last_epoch=-1)

    max_val_acc = 0.0

    timer = Timer()
    x_entropy = torch.nn.CrossEntropyLoss()

    i_cum = 0

    if (opt.task_number > 1) and not opt.train_weights:
        spsa_start = opt.pretrain
    else:
        spsa_start = opt.num_epoch + 1

    for epoch in range(1, opt.num_epoch + 1):
        # Train on the training split
        losses_all = []
        acc_all = []

        losses_2n_1 = []
        losses_2n = []

        # Fetch the current epoch's learning rate
        epoch_learning_rate = 0.1
        for param_group in optimizer.param_groups:
            epoch_learning_rate = param_group['lr']

        log(log_file_path, 'Train Epoch: {}\tLearning Rate: {:.4f}'.format(
                            epoch, epoch_learning_rate))

        _, _ = [x.train() for x in (embedding_net, cls_head)]

        train_accuracies = []
        train_losses = []

        with tqdm(total = opt.train_episode, initial = 1) as pbar:
          i = 1
          s = 0
          if opt.epoch_spsa and epoch > spsa_start:
              opt.alpha = .25 / (((epoch - opt.pretrain) * opt.task_number) ** (1 / 6))
              opt.beta = 15 / (((epoch - opt.pretrain) * opt.task_number) ** (1 / 24))
          while i < opt.train_episode:
           for batch in dataloader_train:
            data_support, labels_support = batch["train"]
            data_query, labels_query = batch["test"]
            if opt.coarse_weights:
                labels_query_class = batch["test_coarse_class_ids"]
            data_support = data_support.to(opt.device)
            labels_support = labels_support.to(opt.device)
            data_query = data_query.to(opt.device)
            labels_query = labels_query.to(opt.device)

            train_n_support = opt.train_way * opt.train_shot
            train_n_query = opt.train_way * opt.train_query

            emb_support = embedding_net(data_support.reshape([-1] + list(data_support.shape[-3:])))
            emb_support = emb_support.reshape(opt.episodes_per_batch, train_n_support, -1)

            emb_query = embedding_net(data_query.reshape([-1] + list(data_query.shape[-3:])))
            emb_query = emb_query.reshape(opt.episodes_per_batch, train_n_query, -1)

            logit_query = cls_head(emb_query, emb_support, labels_support, opt.train_way, opt.train_shot)

            res_one_hot = F.one_hot(labels_query.reshape(-1), opt.train_way)
            if opt.coarse_weights and (epoch > spsa_start):
                loss_weights = []
                for label in labels_query_class.reshape(-1):
                    loss_weights.append(weights[label])
                loss_weights = torch.Tensor(loss_weights).to(opt.device)
                res_one_hot = res_one_hot * loss_weights.reshape(-1, 1)

            log_prb = F.log_softmax(logit_query.reshape(-1, opt.train_way), dim=1)
            loss = -(res_one_hot * log_prb).sum(dim=1)
            if opt.coarse_weights and epoch > spsa_start:
                for pair in zip(labels_query_class.reshape(-1), loss):
                    loss_hist[pair[0]] += pair[1]
            loss = loss.mean()

            acc = count_accuracy(logit_query.reshape(-1, opt.train_way), labels_query.reshape(-1))

            losses_all.append(loss)
            acc_all.append(acc)

            loss_all = 0
            if (opt.task_number == 1) or (epoch <= opt.pretrain):
                loss_all = loss
            elif (i % opt.task_number == 0):
                if not losses_2n_1:
                    losses_2n_1 = losses_all
                elif not losses_2n:
                    losses_2n = losses_all
                else:
                    losses_2n_1 = losses_2n
                    losses_2n = losses_all
                if opt.train_weights & (epoch >= opt.pretrain + 1):
                    loss_all = torch.tensor(0, dtype = torch.float).to(opt.device)
                    for j, loss_val in enumerate(losses_all):
                        loss_all += (1 / (weights[j] ** 2) * loss_val + torch.log(weights[j] ** 2))
                else:
                    for j, loss_val in enumerate(losses_all):
                        loss_all += (1 / (weights[j] ** 2) * loss_val + np.log(weights[j] ** 2))
                if (epoch > spsa_start) and not opt.train_weights:
                  s += 1
                  if opt.coarse_weights:
                    weights = optimize(weights, loss_hist, i_cum + s, opt.alpha, opt.beta)
                    loss_hist = torch.zeros(20, device=opt.device)
                  elif opt.tracking and losses_2n_1 and losses_2n:
                    if opt.epoch_spsa:
                      s = opt.task_number
                    weights = optimize_weights_track(weights, (losses_2n_1, losses_2n), i_cum + s, opt.alpha, opt.beta)
                  elif opt.half_spsa or (i <= opt.train_episode // 2):
                    if opt.epoch_spsa:
                      s = opt.task_number
                    weights = optimize(weights, losses_all, i_cum + s, opt.alpha, opt.beta)
                with open(weight_file_path, 'a+') as f:
                  if opt.train_weights:
                    weights_save = np.float64(weights.detach().cpu().numpy())
                  else:
                    weights_save = weights
                  f.write(base64.binascii.b2a_base64(weights_save).decode("ascii"))
                  f.flush()

            if loss_all != 0:
                train_losses.append(loss_all.item() / len(losses_all))
                train_accuracies.append(np.mean([acc.item() for acc in acc_all]))
                if opt.half_spsa or (i > opt.train_episode // 2) or (epoch <= opt.pretrain):
                    loss_all.backward()
                    optimizer.step()
                    optimizer.zero_grad()
                    if opt.train_weights_opt:
                      optimizer1.step()
                      optimizer1.zero_grad()
                losses_all = []
                acc_all = []
                if opt.epoch_spsa and epoch > spsa_start:
                  for w in range(len(weights)):
                    if weights[w] < .1:
                      weights[w] = .1

            if (i % (opt.train_episode // 10) == 0):
                train_acc_avg = np.mean(np.array(train_accuracies))
                log(log_file_path, 'Train Epoch: {}\tBatch: [{}/{}]\tLoss: {:.4f}\tAccuracy: {:.2f} % ({:.2f} %)'.format(
                            epoch, i, opt.train_episode, train_losses[-1], train_acc_avg, train_accuracies[-1]))

            i += 1

            if i == opt.train_episode + 1:
                i_cum += s
                if opt.decrease_alpha:
                    if epoch == 40:
                        opt.alpha /= 5
                    if epoch == 50:
                        opt.alpha /= 5
                break
            else:
                pbar.update()

          if opt.train_weights == opt.epoch_spsa:
            if not opt.coarse_weights:
                weights = weights / np.sum(weights)

        lr_scheduler.step()
        # Evaluate on the validation split
        _, _ = [x.eval() for x in (embedding_net, cls_head)]

        val_accuracies = []
        val_losses = []

        with tqdm(total = opt.val_episode, initial = 1) as pbar:
          i = 1
          while i < opt.val_episode:
           for batch in dataloader_val:
            data_support, labels_support = batch["train"]
            data_query, labels_query = batch["test"]
            data_support = data_support.to(opt.device)
            labels_support = labels_support.to(opt.device)
            data_query = data_query.to(opt.device)
            labels_query = labels_query.to(opt.device)

            test_n_support = opt.test_way * opt.val_shot
            test_n_query = opt.test_way * opt.val_query

            emb_support = embedding_net(data_support.reshape([-1] + list(data_support.shape[-3:])))
            emb_support = emb_support.reshape(1, test_n_support, -1)
            emb_query = embedding_net(data_query.reshape([-1] + list(data_query.shape[-3:])))
            emb_query = emb_query.reshape(1, test_n_query, -1)

            logit_query = cls_head(emb_query, emb_support, labels_support, opt.test_way, opt.val_shot)

            loss = x_entropy(logit_query.reshape(-1, opt.test_way), labels_query.reshape(-1))
            acc = count_accuracy(logit_query.reshape(-1, opt.test_way), labels_query.reshape(-1))

            val_accuracies.append(acc.item())
            val_losses.append(loss.item())

            i += 1

            if i == opt.val_episode + 1:
                break
            else:
                pbar.update()

        val_acc_avg = np.mean(np.array(val_accuracies))
        val_acc_ci95 = 1.96 * np.std(np.array(val_accuracies)) / np.sqrt(opt.val_episode)

        val_loss_avg = np.mean(np.array(val_losses))

        if val_acc_avg > max_val_acc:
            max_val_acc = val_acc_avg
            torch.save({'embedding': embedding_net.state_dict(), 'head': cls_head.state_dict()},\
                       os.path.join(opt.save_path, 'best_model.pth'))
            log(log_file_path, 'Validation Epoch: {}\t\t\tLoss: {:.4f}\tAccuracy: {:.2f} ± {:.2f} % (Best)'\
                  .format(epoch, val_loss_avg, val_acc_avg, val_acc_ci95))
        else:
            log(log_file_path, 'Validation Epoch: {}\t\t\tLoss: {:.4f}\tAccuracy: {:.2f} ± {:.2f} %'\
                  .format(epoch, val_loss_avg, val_acc_avg, val_acc_ci95))

        torch.save({'embedding': embedding_net.state_dict(), 'head': cls_head.state_dict()}\
                   , os.path.join(opt.save_path, 'last_epoch.pth'))

        if epoch % opt.save_epoch == 0:
            torch.save({'embedding': embedding_net.state_dict(), 'head': cls_head.state_dict()}\
                       , os.path.join(opt.save_path, 'epoch_{}.pth'.format(epoch)))

        log(log_file_path, 'Elapsed Time: {}/{}\n'.format(timer.measure(), timer.measure(epoch / float(opt.num_epoch))))
