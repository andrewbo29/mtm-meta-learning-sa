import torch.nn.functional as F

from collections import namedtuple
from torchmeta.datasets import Omniglot, MiniImagenet, CIFARFS, FC100, TieredImagenet
from torchmeta.toy import Sinusoid
from torchmeta.transforms import ClassSplitter, Categorical, Rotation
from torchvision.transforms import ToTensor, Resize, Compose

from maml.model import ModelConvOmniglot, ModelConvMiniImagenet, ModelConvCifar, ModelMLPSinusoid
from maml.utils import ToTensor1D

Benchmark = namedtuple('Benchmark', 'meta_train_dataset meta_val_dataset '
                                    'meta_test_dataset model loss_function')


def get_benchmark_by_name(name,
                          folder,
                          num_ways,
                          num_shots,
                          num_shots_test,
                          no_max_pool,
                          hidden_size=None):
    dataset_transform = ClassSplitter(shuffle=True,
                                      num_train_per_class=num_shots,
                                      num_test_per_class=num_shots_test)
    if name == 'sinusoid':
        transform = ToTensor1D()

        meta_train_dataset = Sinusoid(num_shots + num_shots_test,
                                      num_tasks=1000000,
                                      transform=transform,
                                      target_transform=transform,
                                      dataset_transform=dataset_transform)
        meta_val_dataset = Sinusoid(num_shots + num_shots_test,
                                    num_tasks=1000000,
                                    transform=transform,
                                    target_transform=transform,
                                    dataset_transform=dataset_transform)
        meta_test_dataset = Sinusoid(num_shots + num_shots_test,
                                     num_tasks=1000000,
                                     transform=transform,
                                     target_transform=transform,
                                     dataset_transform=dataset_transform)

        model = ModelMLPSinusoid(hidden_sizes=[40, 40])
        loss_function = F.mse_loss

    elif name == 'omniglot':
        class_augmentations = [Rotation([90, 180, 270])]
        transform = Compose([Resize(28), ToTensor()])

        meta_train_dataset = Omniglot(folder,
                                      transform=transform,
                                      target_transform=Categorical(num_ways),
                                      num_classes_per_task=num_ways,
                                      meta_train=True,
                                      class_augmentations=class_augmentations,
                                      dataset_transform=dataset_transform,
                                      download=True)
        meta_val_dataset = Omniglot(folder,
                                    transform=transform,
                                    target_transform=Categorical(num_ways),
                                    num_classes_per_task=num_ways,
                                    meta_val=True,
                                    class_augmentations=class_augmentations,
                                    dataset_transform=dataset_transform)
        meta_test_dataset = Omniglot(folder,
                                     transform=transform,
                                     target_transform=Categorical(num_ways),
                                     num_classes_per_task=num_ways,
                                     meta_test=True,
                                     dataset_transform=dataset_transform)

        model = ModelConvOmniglot(num_ways, no_max_pool, hidden_size=hidden_size)
        loss_function = F.cross_entropy

    elif name == 'miniimagenet' or name == 'tieredimagenet':
        transform = Compose([Resize(84), ToTensor()])
        ctor = MiniImagenet if name == 'miniimagenet' else TieredImagenet

        meta_train_dataset = ctor(folder,
                                  transform=transform,
                                  target_transform=Categorical(num_ways),
                                  num_classes_per_task=num_ways,
                                  meta_train=True,
                                  dataset_transform=dataset_transform,
                                  download=True)
        meta_val_dataset = ctor(folder,
                                transform=transform,
                                target_transform=Categorical(num_ways),
                                num_classes_per_task=num_ways,
                                meta_val=True,
                                dataset_transform=dataset_transform)
        meta_test_dataset = ctor(folder,
                                 transform=transform,
                                 target_transform=Categorical(num_ways),
                                 num_classes_per_task=num_ways,
                                 meta_test=True,
                                 dataset_transform=dataset_transform)

        model = ModelConvMiniImagenet(num_ways, no_max_pool, hidden_size=hidden_size)
        loss_function = F.cross_entropy
    elif name == 'cifarfs' or name == 'fc100':
        transform = Compose([Resize(32), ToTensor()])
        ctor = CIFARFS if name == 'cifarfs' else FC100

        meta_train_dataset = ctor(folder,
                                  transform=transform,
                                  target_transform=Categorical(num_ways),
                                  num_classes_per_task=num_ways,
                                  meta_train=True,
                                  dataset_transform=dataset_transform,
                                  download=True)
        meta_val_dataset = ctor(folder,
                                transform=transform,
                                target_transform=Categorical(num_ways),
                                num_classes_per_task=num_ways,
                                meta_val=True,
                                dataset_transform=dataset_transform)
        meta_test_dataset = ctor(folder,
                                 transform=transform,
                                 target_transform=Categorical(num_ways),
                                 num_classes_per_task=num_ways,
                                 meta_test=True,
                                 dataset_transform=dataset_transform)

        model = ModelConvCifar(num_ways, no_max_pool, hidden_size=hidden_size)
        loss_function = F.cross_entropy

    else:
        raise NotImplementedError('Unknown dataset `{0}`.'.format(name))

    return Benchmark(meta_train_dataset=meta_train_dataset,
                     meta_val_dataset=meta_val_dataset,
                     meta_test_dataset=meta_test_dataset,
                     model=model,
                     loss_function=loss_function)
