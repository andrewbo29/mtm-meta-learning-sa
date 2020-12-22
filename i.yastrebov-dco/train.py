# -*- coding: utf-8 -*-
import argparse
import numpy as np
import os
import torch
import torch.nn.functional as F

from models.classification_heads import ClassificationHead
from models.ResNet12_embedding import resnet12
from optimize import optimize
from torchmeta.transforms import Categorical, ClassSplitter
from torchmeta.utils.data import BatchMetaDataLoader
from torchvision.transforms import ColorJitter, Compose, Normalize, RandomCrop, RandomHorizontalFlip, ToTensor
from tqdm import tqdm
from utils import check_dir, count_accuracy, log, set_gpu, Timer 

def one_hot(indices, depth):
    """
    Returns a one-hot tensor.
    This is a PyTorch equivalent of Tensorflow's tf.one_hot.
        
    Parameters:
      indices:  a (n_batch, m) Tensor or (m) Tensor.
      depth: a scalar. Represents the depth of the one hot dimension.
    Returns: a (n_batch, m, depth) Tensor or (m, depth) Tensor.
    """

    encoded_indicies = torch.zeros(indices.size() + torch.Size([depth])).cuda()
    index = indices.view(indices.size()+torch.Size([1]))
    encoded_indicies = encoded_indicies.scatter_(1,index,1)
    
    return encoded_indicies

def get_model(options):
    # Choose the embedding network
    if options.network == 'ResNet':
        if options.dataset == 'miniImageNet' or options.dataset == 'tieredImageNet':
            network = resnet12(avg_pool = False, drop_rate = .1, dropblock_size = 5).cuda()
            if opt.gpu != '0':
                network = torch.nn.DataParallel(network, device_ids=[0, 1, 2, 3])
        else:
            network = resnet12(avg_pool = False, drop_rate = .1, dropblock_size = 2).cuda()
    else:
        print ("Cannot recognize the network type")
        assert(False)
        
    # Choose the classification head
    if options.head == 'Ridge':
        cls_head = ClassificationHead(base_learner='Ridge').cuda()
    elif options.head == 'Proto':
        cls_head = ClassificationHead(base_learner='Proto').cuda()
    elif options.head == 'SVM-CS':
        cls_head = ClassificationHead(base_learner='SVM-CS').cuda()
    elif options.head == 'SVM-He':
        cls_head = ClassificationHead(base_learner='SVM-He').cuda()
    elif options.head == 'SVM-WW':
        cls_head = ClassificationHead(base_learner='SVM-WW').cuda()
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
        dataset_train = MiniImagenet(
            "data",
            num_classes_per_task = opt.train_way,
            transform = Compose([RandomCrop(84, padding = 8),
                                 ColorJitter(brightness = .4,
                                             contrast = .4,
                                             saturation = .4),
                                 RandomHorizontalFlip(),
                                 ToTensor(),
                                 Normalize(mean = mean_pix,
                                           std = std_pix),
                                ]),
            target_transform = Categorical(num_classes = opt.train_way),
            meta_train = True,
            download = True)
        dataset_train = ClassSplitter(dataset_train, shuffle = True,
                                      num_train_per_class = opt.train_shot,
                                      num_test_per_class = opt.train_query)
        dataloader_train = BatchMetaDataLoader(
            dataset_train,
            batch_size = opt.episodes_per_batch,
            num_workers = opt.num_workers)
        dataset_val = MiniImagenet(
          "data",
          num_classes_per_task = opt.test_way,
          transform = Compose([ToTensor(),
                               Normalize(mean = mean_pix,
                                         std = std_pix),
                              ]),
          target_transform = Categorical(num_classes = opt.test_way),
          meta_val = True,
          download = False)
        dataset_val = ClassSplitter(dataset_val, shuffle = True,
                                    num_train_per_class = opt.val_shot,
                                    num_test_per_class = opt.val_query)
        dataloader_val = BatchMetaDataLoader(
          dataset_val,
          batch_size = 1,
          num_workers = opt.num_workers)
    elif options.dataset == 'tieredImageNet':
        from torchmeta.datasets import TieredImagenet
        mean_pix = [x / 255 for x in [120.39586422,  115.59361427, 104.54012653]]
        std_pix = [x / 255 for x in [70.68188272,  68.27635443,  72.54505529]]
        dataset_train = TieredImagenet(
            "data",
            num_classes_per_task = opt.train_way,
            transform = Compose([RandomCrop(84, padding = 8),
                                 ColorJitter(brightness = .4,
                                             contrast = .4,
                                             saturation = .4),
                                 RandomHorizontalFlip(),
                                 ToTensor(),
                                 Normalize(mean = mean_pix,
                                           std = std_pix),
                                ]),
            target_transform = Categorical(num_classes = opt.train_way),
            meta_train = True,
            download = True)
        dataset_train = ClassSplitter(dataset_train, shuffle = True,
                                      num_train_per_class = opt.train_shot,
                                      num_test_per_class = opt.train_query)
        dataloader_train = BatchMetaDataLoader(
            dataset_train,
            batch_size = opt.episodes_per_batch,
            num_workers = opt.num_workers)
        dataset_val = TieredImagenet(
          "data",
          num_classes_per_task = opt.test_way,
          transform = Compose([ToTensor(),
                               Normalize(mean = mean_pix,
                                         std = std_pix),
                              ]),
          target_transform = Categorical(num_classes = opt.test_way),
          meta_val = True,
          download = False)
        dataset_val = ClassSplitter(dataset_val, shuffle = True,
                                    num_train_per_class = opt.val_shot,
                                    num_test_per_class = opt.val_query)
        dataloader_val = BatchMetaDataLoader(
          dataset_val,
          batch_size = 1,
          num_workers = opt.num_workers)
    elif options.dataset == 'CIFAR_FS':
        from torchmeta.datasets import CIFARFS
        mean_pix = [x / 255 for x in [129.37731888, 124.10583864, 112.47758569]]
        std_pix = [x / 255 for x in [68.20947949, 65.43124043, 70.45866994]]
        dataset_train = CIFARFS(
            "data",
            num_classes_per_task = opt.train_way,
            transform = Compose([RandomCrop(32, padding = 4),
                                 ColorJitter(brightness = .4,
                                             contrast = .4,
                                             saturation = .4),
                                 RandomHorizontalFlip(),
                                 ToTensor(),
                                 Normalize(mean = mean_pix,
                                           std = std_pix),
                                ]),
            target_transform = Categorical(num_classes = opt.train_way),
            meta_train = True,
            download = True)
        dataset_train = ClassSplitter(dataset_train, shuffle = True,
                                      num_train_per_class = opt.train_shot,
                                      num_test_per_class = opt.train_query)
        dataloader_train = BatchMetaDataLoader(
            dataset_train,
            batch_size = opt.episodes_per_batch,
            num_workers = opt.num_workers)
        dataset_val = CIFARFS(
          "data",
          num_classes_per_task = opt.test_way,
          transform = Compose([ToTensor(),
                               Normalize(mean = mean_pix,
                                         std = std_pix),
                              ]),
          target_transform = Categorical(num_classes = opt.test_way),
          meta_val = True,
          download = False)
        dataset_val = ClassSplitter(dataset_val, shuffle = True,
                                    num_train_per_class = opt.val_shot,
                                    num_test_per_class = opt.val_query)
        dataloader_val = BatchMetaDataLoader(
          dataset_val,
          batch_size = 1,
          num_workers = opt.num_workers)
    elif options.dataset == 'FC100':
        from torchmeta.datasets import FC100
        mean_pix = [x / 255 for x in [129.37731888, 124.10583864, 112.47758569]]
        std_pix = [x / 255 for x in [68.20947949, 65.43124043, 70.45866994]]
        dataset_train = FC100(
            "data",
            num_classes_per_task = opt.train_way,
            transform = Compose([RandomCrop(32, padding = 4),
                                 ColorJitter(brightness = .4,
                                             contrast = .4,
                                             saturation = .4),
                                 RandomHorizontalFlip(),
                                 ToTensor(),
                                 Normalize(mean = mean_pix,
                                           std = std_pix),
                                ]),
            target_transform = Categorical(num_classes = opt.train_way),
            meta_train = True,
            download = True)
        dataset_train = ClassSplitter(dataset_train, shuffle = True,
                                      num_train_per_class = opt.train_shot,
                                      num_test_per_class = opt.train_query)
        dataloader_train = BatchMetaDataLoader(
            dataset_train,
            batch_size = opt.episodes_per_batch,
            num_workers = opt.num_workers)
        dataset_val = FC100(
          "data",
          num_classes_per_task = opt.test_way,
          transform = Compose([ToTensor(),
                               Normalize(mean = mean_pix,
                                         std = std_pix),
                              ]),
          target_transform = Categorical(num_classes = opt.test_way),
          meta_val = True,
          download = False)
        dataset_val = ClassSplitter(dataset_val, shuffle = True,
                                    num_train_per_class = opt.val_shot,
                                    num_test_per_class = opt.val_query)
        dataloader_val = BatchMetaDataLoader(
          dataset_val,
          batch_size = 1,
          num_workers = opt.num_workers)
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
    parser.add_argument('--train-shot', type=int, default=15,
                            help='number of support examples per training class')
    parser.add_argument('--val-shot', type=int, default=5,
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
    parser.add_argument('--gpu', default='0')
    parser.add_argument('--num-workers', type=int, default=2,
                            help='number of cpu workers for loading data')
    parser.add_argument('--network', type=str, default='ResNet',
                            help='choose which embedding network to use. ResNet')
    parser.add_argument('--head', type=str, default='SVM-CS',
                            help='choose which classification head to use. Ridge, Proto, SVM')
    parser.add_argument('--dataset', type=str, default='miniImageNet',
                            help='choose which classification head to use. miniImageNet, tieredImageNet, CIFAR_FS, FC100')
    parser.add_argument('--episodes-per-batch', type=int, default=8,
                            help='number of episodes per batch')
    parser.add_argument('--eps', type=float, default=0.0,
                            help='epsilon of label smoothing')
    parser.add_argument('--task-number', type=int, default=5,
                            help='number of task runs before update')
    parser.add_argument('--pretrain', type=int, default=20,
                            help='number of epochs without MTL')
    parser.add_argument('--mid-stage', type=int, default=10,
                            help='number of epochs for stage after pretrain')
    parser.add_argument('--sin-first', action='store_true',
                            help='using sin weights before spsa weights')

    opt = parser.parse_args()
    
    (dataloader_train, dataloader_val) = get_dataset(opt)

    set_gpu(opt.gpu)
    check_dir('./experiments/')
    check_dir(opt.save_path)
    
    log_file_path = os.path.join(opt.save_path, "train_log.txt")
    log(log_file_path, str(vars(opt)))

    (embedding_net, cls_head) = get_model(opt)
    
    optimizer = torch.optim.SGD([{'params': embedding_net.parameters()}, 
                                 {'params': cls_head.parameters()}], lr=0.1, momentum=0.9, \
                                          weight_decay=5e-4, nesterov=True)
    
    lambda_epoch = lambda e: 1.0 if e < 20 else (0.06 if e < 40 else 0.012 if e < 50 else (0.0024))
    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_epoch, last_epoch=-1)

    max_val_acc = 0.0

    timer = Timer()
    x_entropy = torch.nn.CrossEntropyLoss()
    
    weights0 = np.array([1 / opt.task_number for _ in range(opt.task_number)])
    weights1 = np.array([np.sin(i * np.pi / opt.task_number + np.pi / (2 * opt.task_number)) + 0.9 for i in range(1, opt.task_number + 1)])
    weights1 = weights1 / weights1.sum()
    weights = np.ones_like(weights0)
    i_cum = 0

    
    if opt.sin_first:
      sin_start = opt.pretrain
      sin_finish = opt.pretrain + opt.mid_stage
      spsa_start = opt.pretrain + opt.mid_stage
      spsa_finish = opt.num_epoch
    else:
      spsa_start = opt.pretrain
      spsa_finish = opt.pretrain + opt.mid_stage
      sin_start = opt.pretrain + opt.mid_stage
      sin_finish = opt.num_epoch
    
    for epoch in range(1, opt.num_epoch + 1):
        # Train on the training split
        losses_all = []
        acc_all = []
        
        # Fetch the current epoch's learning rate
        epoch_learning_rate = 0.1
        for param_group in optimizer.param_groups:
            epoch_learning_rate = param_group['lr']
            
        log(log_file_path, 'Train Epoch: {}\tLearning Rate: {:.4f}'.format(
                            epoch, epoch_learning_rate))
        
        _, _ = [x.train() for x in (embedding_net, cls_head)]
        
        train_accuracies = []
        train_losses = []

        with tqdm(dataloader_train, total = opt.train_episode, initial = 1) as pbar:
          if epoch == spsa_start + 1:
            if opt.sin_first:
              weights = weights1
            else:
              weights = weights0
          for i, batch in enumerate(pbar, 1):
            if (epoch > sin_start) & (epoch <= sin_finish):
              weight = np.sin(i * np.pi / opt.task_number + np.pi / (2 * opt.task_number)) + 0.9
              weights[(i - 1) % opt.task_number] = weight
            data_support, labels_support = batch["train"]
            data_query, labels_query = batch["test"]
            data_support = data_support.to(device='cuda')
            labels_support = labels_support.to(device='cuda')
            data_query = data_query.to(device='cuda')
            labels_query = labels_query.to(device='cuda')

            train_n_support = opt.train_way * opt.train_shot
            train_n_query = opt.train_way * opt.train_query

            emb_support = embedding_net(data_support.reshape([-1] + list(data_support.shape[-3:])))
            emb_support = emb_support.reshape(opt.episodes_per_batch, train_n_support, -1)
            
            emb_query = embedding_net(data_query.reshape([-1] + list(data_query.shape[-3:])))
            emb_query = emb_query.reshape(opt.episodes_per_batch, train_n_query, -1)
            
            logit_query = cls_head(emb_query, emb_support, labels_support, opt.train_way, opt.train_shot)

            smoothed_one_hot = one_hot(labels_query.reshape(-1), opt.train_way)
            smoothed_one_hot = smoothed_one_hot * (1 - opt.eps) + (1 - smoothed_one_hot) * opt.eps / (opt.train_way - 1)

            log_prb = F.log_softmax(logit_query.reshape(-1, opt.train_way), dim=1)
            loss = -(smoothed_one_hot * log_prb).sum(dim=1)
            loss = loss.mean()
            
            acc = count_accuracy(logit_query.reshape(-1, opt.train_way), labels_query.reshape(-1))
            
            losses_all.append(loss)
            acc_all.append(acc)
            
            loss_all = 0
            if (opt.task_number == 1) or (epoch <= opt.pretrain):
                loss_all = loss
            elif (i % opt.task_number == 0):
                if (epoch > sin_start) & (epoch <= sin_finish):
                  weights = weights / np.sum(weights)             
                for j, loss_val in enumerate(losses_all):
                  if (epoch > sin_start) & (epoch <= sin_finish):
                    loss_all += weights[j] * loss_val
                  else:
                    loss_all += (1 / (weights[j] ** 2) * loss_val + np.log(weights[j] ** 2))
                if (epoch > spsa_start) & (epoch <= spsa_finish):
                  weights = optimize(weights, losses_all, i_cum + i / opt.task_number)
                  weights = weights / np.sum(weights)
                
            if loss_all != 0:
                train_losses.append(loss_all.item() / len(losses_all))
                train_accuracies.append(np.mean([acc.item() for acc in acc_all]))                
                loss_all.backward()
                optimizer.step()
                optimizer.zero_grad()
                losses_all = []
                acc_all = []

            if (i % (opt.train_episode // 10) == 0):
                train_acc_avg = np.mean(np.array(train_accuracies))
                log(log_file_path, 'Train Epoch: {}\tBatch: [{}/{}]\tLoss: {:.4f}\tAccuracy: {:.2f} % ({:.2f} %)'.format(
                            epoch, i, opt.train_episode, train_losses[-1], train_acc_avg, train_accuracies[-1]))            
            
            if i == opt.train_episode:
                if epoch > spsa_start:
                    i_cum += i // opt.task_number
                break

        lr_scheduler.step()
        # Evaluate on the validation split
        _, _ = [x.eval() for x in (embedding_net, cls_head)]

        val_accuracies = []
        val_losses = []
        
        with tqdm(dataloader_val, total = opt.val_episode, initial = 1) as pbar:
          for i, batch in enumerate(pbar, 1):
            data_support, labels_support = batch["train"]
            data_query, labels_query = batch["test"]
            data_support = data_support.to(device='cuda')
            labels_support = labels_support.to(device='cuda')
            data_query = data_query.to(device='cuda')
            labels_query = labels_query.to(device='cuda')

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
            if i == opt.val_episode:
                break
            
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
# -*- coding: utf-8 -*-
import argparse
import numpy as np
import os
import torch
import torch.nn.functional as F

from models.classification_heads import ClassificationHead
from models.ResNet12_embedding import resnet12
from optimize import optimize
from torchmeta.transforms import Categorical, ClassSplitter
from torchmeta.utils.data import BatchMetaDataLoader
from torchvision.transforms import ColorJitter, Compose, Normalize, RandomCrop, RandomHorizontalFlip, ToTensor
from tqdm import tqdm
from utils import check_dir, count_accuracy, log, set_gpu, Timer 

def one_hot(indices, depth):
    """
    Returns a one-hot tensor.
    This is a PyTorch equivalent of Tensorflow's tf.one_hot.
        
    Parameters:
      indices:  a (n_batch, m) Tensor or (m) Tensor.
      depth: a scalar. Represents the depth of the one hot dimension.
    Returns: a (n_batch, m, depth) Tensor or (m, depth) Tensor.
    """

    encoded_indicies = torch.zeros(indices.size() + torch.Size([depth])).cuda()
    index = indices.view(indices.size()+torch.Size([1]))
    encoded_indicies = encoded_indicies.scatter_(1,index,1)
    
    return encoded_indicies

def get_model(options):
    # Choose the embedding network
    if options.network == 'ResNet':
        if options.dataset == 'miniImageNet' or options.dataset == 'tieredImageNet':
            network = resnet12(avg_pool = False, drop_rate = .1, dropblock_size = 5).cuda()
            if opt.gpu != '0':
                network = torch.nn.DataParallel(network, device_ids=[0, 1, 2, 3])
        else:
            network = resnet12(avg_pool = False, drop_rate = .1, dropblock_size = 2).cuda()
    else:
        print ("Cannot recognize the network type")
        assert(False)
        
    # Choose the classification head
    if options.head == 'Ridge':
        cls_head = ClassificationHead(base_learner='Ridge').cuda()
    elif options.head == 'SVM-CS':
        cls_head = ClassificationHead(base_learner='SVM-CS').cuda()
    elif options.head == 'SVM-He':
        cls_head = ClassificationHead(base_learner='SVM-CS').cuda()
    elif options.head == 'SVM-WW':
        cls_head = ClassificationHead(base_learner='SVM-CS').cuda()
    else:
        print ("Cannot recognize the dataset type")
        assert(False)
        
    return (network, cls_head)

def get_dataset(options):
    # Choose the learning datdset
    if options.dataset == 'miniImageNet':
        from torchmeta.datasets import MiniImagenet
        mean_pix = [x / 255 for x in [120.39586422,  115.59361427, 104.54012653]]
        std_pix = [x / 255 for x in [70.68188272,  68.27635443,  72.54505529]]
        dataset_train = MiniImagenet(
            "data",
            num_classes_per_task = opt.train_way,
            transform = Compose([RandomCrop(84, padding = 8),
                                 ColorJitter(brightness = .4,
                                             contrast = .4,
                                             saturation = .4),
                                 RandomHorizontalFlip(),
                                 ToTensor(),
                                 Normalize(mean = mean_pix,
                                           std = std_pix),
                                ]),
            target_transform = Categorical(num_classes = opt.train_way),
            meta_train = True,
            download = True)
        dataset_train = ClassSplitter(dataset_train, shuffle = True,
                                      num_train_per_class = opt.train_shot,
                                      num_test_per_class = opt.train_query)
        dataloader_train = BatchMetaDataLoader(
            dataset_train,
            batch_size = opt.episodes_per_batch,
            num_workers = opt.num_workers)
        dataset_val = MiniImagenet(
          "data",
          num_classes_per_task = opt.test_way,
          transform = Compose([ToTensor(),
                               Normalize(mean = mean_pix,
                                         std = std_pix),
                              ]),
          target_transform = Categorical(num_classes = opt.test_way),
          meta_val = True,
          download = False)
        dataset_val = ClassSplitter(dataset_val, shuffle = True,
                                    num_train_per_class = opt.val_shot,
                                    num_test_per_class = opt.val_query)
        dataloader_val = BatchMetaDataLoader(
          dataset_val,
          batch_size = 1,
          num_workers = opt.num_workers)
    elif options.dataset == 'tieredImageNet':
        from torchmeta.datasets import TieredImagenet
        mean_pix = [x / 255 for x in [120.39586422,  115.59361427, 104.54012653]]
        std_pix = [x / 255 for x in [70.68188272,  68.27635443,  72.54505529]]
        dataset_train = TieredImagenet(
            "data",
            num_classes_per_task = opt.train_way,
            transform = Compose([RandomCrop(84, padding = 8),
                                 ColorJitter(brightness = .4,
                                             contrast = .4,
                                             saturation = .4),
                                 RandomHorizontalFlip(),
                                 ToTensor(),
                                 Normalize(mean = mean_pix,
                                           std = std_pix),
                                ]),
            target_transform = Categorical(num_classes = opt.train_way),
            meta_train = True,
            download = True)
        dataset_train = ClassSplitter(dataset_train, shuffle = True,
                                      num_train_per_class = opt.train_shot,
                                      num_test_per_class = opt.train_query)
        dataloader_train = BatchMetaDataLoader(
            dataset_train,
            batch_size = opt.episodes_per_batch,
            num_workers = opt.num_workers)
        dataset_val = TieredImagenet(
          "data",
          num_classes_per_task = opt.test_way,
          transform = Compose([ToTensor(),
                               Normalize(mean = mean_pix,
                                         std = std_pix),
                              ]),
          target_transform = Categorical(num_classes = opt.test_way),
          meta_val = True,
          download = False)
        dataset_val = ClassSplitter(dataset_val, shuffle = True,
                                    num_train_per_class = opt.val_shot,
                                    num_test_per_class = opt.val_query)
        dataloader_val = BatchMetaDataLoader(
          dataset_val,
          batch_size = 1,
          num_workers = opt.num_workers)
    elif options.dataset == 'CIFAR_FS':
        from torchmeta.datasets import CIFARFS
        mean_pix = [x / 255 for x in [129.37731888, 124.10583864, 112.47758569]]
        std_pix = [x / 255 for x in [68.20947949, 65.43124043, 70.45866994]]
        dataset_train = CIFARFS(
            "data",
            num_classes_per_task = opt.train_way,
            transform = Compose([RandomCrop(32, padding = 4),
                                 ColorJitter(brightness = .4,
                                             contrast = .4,
                                             saturation = .4),
                                 RandomHorizontalFlip(),
                                 ToTensor(),
                                 Normalize(mean = mean_pix,
                                           std = std_pix),
                                ]),
            target_transform = Categorical(num_classes = opt.train_way),
            meta_train = True,
            download = True)
        dataset_train = ClassSplitter(dataset_train, shuffle = True,
                                      num_train_per_class = opt.train_shot,
                                      num_test_per_class = opt.train_query)
        dataloader_train = BatchMetaDataLoader(
            dataset_train,
            batch_size = opt.episodes_per_batch,
            num_workers = opt.num_workers)
        dataset_val = CIFARFS(
          "data",
          num_classes_per_task = opt.test_way,
          transform = Compose([ToTensor(),
                               Normalize(mean = mean_pix,
                                         std = std_pix),
                              ]),
          target_transform = Categorical(num_classes = opt.test_way),
          meta_val = True,
          download = False)
        dataset_val = ClassSplitter(dataset_val, shuffle = True,
                                    num_train_per_class = opt.val_shot,
                                    num_test_per_class = opt.val_query)
        dataloader_val = BatchMetaDataLoader(
          dataset_val,
          batch_size = 1,
          num_workers = opt.num_workers)
    elif options.dataset == 'FC100':
        from torchmeta.datasets import FC100
        mean_pix = [x / 255 for x in [129.37731888, 124.10583864, 112.47758569]]
        std_pix = [x / 255 for x in [68.20947949, 65.43124043, 70.45866994]]
        dataset_train = FC100(
            "data",
            num_classes_per_task = opt.train_way,
            transform = Compose([RandomCrop(32, padding = 4),
                                 ColorJitter(brightness = .4,
                                             contrast = .4,
                                             saturation = .4),
                                 RandomHorizontalFlip(),
                                 ToTensor(),
                                 Normalize(mean = mean_pix,
                                           std = std_pix),
                                ]),
            target_transform = Categorical(num_classes = opt.train_way),
            meta_train = True,
            download = True)
        dataset_train = ClassSplitter(dataset_train, shuffle = True,
                                      num_train_per_class = opt.train_shot,
                                      num_test_per_class = opt.train_query)
        dataloader_train = BatchMetaDataLoader(
            dataset_train,
            batch_size = opt.episodes_per_batch,
            num_workers = opt.num_workers)
        dataset_val = FC100(
          "data",
          num_classes_per_task = opt.test_way,
          transform = Compose([ToTensor(),
                               Normalize(mean = mean_pix,
                                         std = std_pix),
                              ]),
          target_transform = Categorical(num_classes = opt.test_way),
          meta_val = True,
          download = False)
        dataset_val = ClassSplitter(dataset_val, shuffle = True,
                                    num_train_per_class = opt.val_shot,
                                    num_test_per_class = opt.val_query)
        dataloader_val = BatchMetaDataLoader(
          dataset_val,
          batch_size = 1,
          num_workers = opt.num_workers)
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
    parser.add_argument('--train-shot', type=int, default=15,
                            help='number of support examples per training class')
    parser.add_argument('--val-shot', type=int, default=5,
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
    parser.add_argument('--gpu', default='0')
    parser.add_argument('--num-workers', type=int, default=2,
                            help='number of cpu workers for loading data')
    parser.add_argument('--network', type=str, default='ResNet',
                            help='choose which embedding network to use. ResNet')
    parser.add_argument('--head', type=str, default='SVM-CS',
                            help='choose which classification head to use. Ridge, SVM')
    parser.add_argument('--dataset', type=str, default='miniImageNet',
                            help='choose which classification head to use. miniImageNet, tieredImageNet, CIFAR_FS, FC100')
    parser.add_argument('--episodes-per-batch', type=int, default=8,
                            help='number of episodes per batch')
    parser.add_argument('--eps', type=float, default=0.0,
                            help='epsilon of label smoothing')
    parser.add_argument('--task-number', type=int, default=5,
                            help='number of task runs before update')
    parser.add_argument('--pretrain', type=int, default=20,
                            help='number of task runs before update')
    parser.add_argument('--sin-stage', type=int, default=10,
                            help='number of task runs before update')

    opt = parser.parse_args()
    
    (dataloader_train, dataloader_val) = get_dataset(opt)

    set_gpu(opt.gpu)
    check_dir('./experiments/')
    check_dir(opt.save_path)
    
    log_file_path = os.path.join(opt.save_path, "train_log.txt")
    log(log_file_path, str(vars(opt)))

    (embedding_net, cls_head) = get_model(opt)
    
    optimizer = torch.optim.SGD([{'params': embedding_net.parameters()}, 
                                 {'params': cls_head.parameters()}], lr=0.1, momentum=0.9, \
                                          weight_decay=5e-4, nesterov=True)
    
    lambda_epoch = lambda e: 1.0 if e < 20 else (0.06 if e < 40 else 0.012 if e < 50 else (0.0024))
    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_epoch, last_epoch=-1)

    max_val_acc = 0.0

    timer = Timer()
    x_entropy = torch.nn.CrossEntropyLoss()
    
    weights0 = np.array([1 / opt.task_number for _ in range(opt.task_number)])
    weights1 = np.array([np.sin(i * np.pi / opt.task_number + np.pi / (2 * opt.task_number)) + 0.9 for i in range(1, opt.task_number + 1)])
    weights1 = weights1 / weights1.sum()
    assert weights0.shape == weights1.shape
    weights = np.ones_like(weights0)
    i_cum = 0
    
    for epoch in range(1, opt.num_epoch + 1):
        # Train on the training split
        losses_all = []
        acc_all = []
        
        # Fetch the current epoch's learning rate
        epoch_learning_rate = 0.1
        for param_group in optimizer.param_groups:
            epoch_learning_rate = param_group['lr']
            
        log(log_file_path, 'Train Epoch: {}\tLearning Rate: {:.4f}'.format(
                            epoch, epoch_learning_rate))
        
        _, _ = [x.train() for x in (embedding_net, cls_head)]
        
        train_accuracies = []
        train_losses = []        

        with tqdm(dataloader_train, total = opt.train_episode, initial = 1) as pbar:
          if epoch > opt.pretrain + opt.sin_stage:
            if opt.sin_stage > 0:
              weights = weights1
            else:
              weights = weights0
          for i, batch in enumerate(pbar, 1):
            if (epoch > opt.pretrain) & (epoch <+ opt.pretrain + opt.sin_stage):
              weight = np.sin(i * np.pi / opt.task_number + np.pi / (2 * opt.task_number)) + 0.9
              weights[(i - 1) % opt.task_number] = weight
            data_support, labels_support = batch["train"]
            data_query, labels_query = batch["test"]
            data_support = data_support.to(device='cuda')
            labels_support = labels_support.to(device='cuda')
            data_query = data_query.to(device='cuda')
            labels_query = labels_query.to(device='cuda')

            train_n_support = opt.train_way * opt.train_shot
            train_n_query = opt.train_way * opt.train_query

            emb_support = embedding_net(data_support.reshape([-1] + list(data_support.shape[-3:])))
            emb_support = emb_support.reshape(opt.episodes_per_batch, train_n_support, -1)
            
            emb_query = embedding_net(data_query.reshape([-1] + list(data_query.shape[-3:])))
            emb_query = emb_query.reshape(opt.episodes_per_batch, train_n_query, -1)
            
            logit_query = cls_head(emb_query, emb_support, labels_support, opt.train_way, opt.train_shot)

            smoothed_one_hot = one_hot(labels_query.reshape(-1), opt.train_way)
            smoothed_one_hot = smoothed_one_hot * (1 - opt.eps) + (1 - smoothed_one_hot) * opt.eps / (opt.train_way - 1)

            log_prb = F.log_softmax(logit_query.reshape(-1, opt.train_way), dim=1)
            loss = -(smoothed_one_hot * log_prb).sum(dim=1)
            loss = loss.mean()
            
            acc = count_accuracy(logit_query.reshape(-1, opt.train_way), labels_query.reshape(-1))
            
            losses_all.append(loss)
            acc_all.append(acc)
            
            loss_all = 0
            if (opt.task_number == 1) or (epoch <= opt.pretrain):
                loss_all = loss
            elif (i % opt.task_number == 0):
                if epoch <= opt.pretrain + opt.sin_stage:
                  weights = weights / np.sum(weights)             
                for j, loss_val in enumerate(losses_all):
                  if epoch <= opt.pretrain + opt.sin_stage:
                    loss_all += weights[j] * loss_val
                  else:
                    loss_all += (1 / (weights[j] ** 2) * loss_val + np.log(weights[j] ** 2))
                if epoch > opt.pretrain + opt.sin_stage:
                  weights = optimize(weights, losses_all, i_cum + i / opt.task_number)
                  weights = weights / np.sum(weights)
                
            if loss_all != 0:
                train_losses.append(loss_all.item() / len(losses_all))
                train_accuracies.append(np.mean([acc.item() for acc in acc_all]))                
                loss_all.backward()
                optimizer.step()
                optimizer.zero_grad()
                losses_all = []
                acc_all = []

            if (i % (opt.train_episode // 10) == 0):
                train_acc_avg = np.mean(np.array(train_accuracies))
                log(log_file_path, 'Train Epoch: {}\tBatch: [{}/{}]\tLoss: {:.4f}\tAccuracy: {:.2f} % ({:.2f} %)'.format(
                            epoch, i, opt.train_episode, train_losses[-1], train_acc_avg, train_accuracies[-1]))            
            
            if i == opt.train_episode:
                if epoch > opt.pretrain + opt.sin_stage:
                    i_cum += i // opt.task_number
                break

        lr_scheduler.step()
        # Evaluate on the validation split
        _, _ = [x.eval() for x in (embedding_net, cls_head)]

        val_accuracies = []
        val_losses = []
        
        with tqdm(dataloader_val, total = opt.val_episode, initial = 1) as pbar:
          for i, batch in enumerate(pbar, 1):
            data_support, labels_support = batch["train"]
            data_query, labels_query = batch["test"]
            data_support = data_support.to(device='cuda')
            labels_support = labels_support.to(device='cuda')
            data_query = data_query.to(device='cuda')
            labels_query = labels_query.to(device='cuda')

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
            if i == opt.val_episode:
                break
            
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
