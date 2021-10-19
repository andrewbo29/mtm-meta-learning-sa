# -*- coding: utf-8 -*-
import bisect
import torch

from collections import OrderedDict
from torch.utils.data.dataloader import default_collate
from torch.utils.data.dataset import Dataset as TorchDataset
from torchmeta.utils.data import MetaDataLoader


class BatchMetaCollateWithLabels:
    def __init__(self, collate_fn):
        super().__init__()
        self.collate_fn = collate_fn
        self._coarse_class_mapping = self.get_coarse_class_mapping()

    @staticmethod
    def get_coarse_class_mapping():
        return {
            'aquatic_mammals': 0,
            'fish': 1,
            'flowers': 2,
            'food_containers': 3,
            'fruit_and_vegetables': 4,
            'household_electrical_devices': 5,
            'household_furniture': 6,
            'insects': 7,
            'large_carnivores': 8,
            'large_man-made_outdoor_things': 9,
            'large_natural_outdoor_scenes': 10,
            'large_omnivores_and_herbivores': 11,
            'medium_mammals': 12,
            'non-insect_invertebrates': 13,
            'people': 14,
            'reptiles': 15,
            'small_mammals': 16,
            'trees': 17,
            'vehicles_1': 18,
            'vehicles_2': 19
        }

    @staticmethod
    def get_dataset_idx(dataset, global_idx):
        return bisect.bisect_right(dataset.cumulative_sizes, global_idx)

    @staticmethod
    def get_class_id(dataset, global_idx):
        return dataset.index[BatchMetaCollateWithLabels.get_dataset_idx(dataset, global_idx)]

    def get_coarse_class_id(self, dataset, global_idx):
        cur_dataset = dataset.datasets[self.get_dataset_idx(dataset, global_idx)]
        coarse_label_name = getattr(cur_dataset, 'coarse_label_name', None)
        return self._coarse_class_mapping[coarse_label_name] if coarse_label_name is not None else None

    def collate_task(self, task):
        if isinstance(task, TorchDataset):
            data = self.collate_fn([task[idx] for idx in range(len(task))])
            class_indices = [self.get_class_id(task.dataset, item_idx) for item_idx in task.indices]
            coarse_class_indices = [self.get_coarse_class_id(task.dataset, item_idx) for item_idx in task.indices]
            return data, class_indices, coarse_class_indices
        elif isinstance(task, OrderedDict):
            result = OrderedDict()
            for (key, subtask) in task.items():
                data, class_indices, coarse_class_indices = self.collate_task(subtask)
                result[key] = data
                result[key + '_class_ids'] = torch.IntTensor(class_indices)
                if None not in coarse_class_indices:
                    result[key + '_coarse_class_ids'] = torch.IntTensor(coarse_class_indices)
            return result
        else:
            raise NotImplementedError()

    def __call__(self, batch):
        result = self.collate_fn([self.collate_task(task) for task in batch])
        return result


class BatchMetaDataLoaderWithLabels(MetaDataLoader):
    def __init__(self, dataset, batch_size=1, shuffle=True, sampler=None, num_workers=0,
                 pin_memory=False, drop_last=False, timeout=0, worker_init_fn=None):
        collate_fn = BatchMetaCollateWithLabels(default_collate)

        super().__init__(dataset,
                         batch_size=batch_size, shuffle=shuffle, sampler=sampler,
                         batch_sampler=None, num_workers=num_workers,
                         collate_fn=collate_fn, pin_memory=pin_memory, drop_last=drop_last,
                         timeout=timeout, worker_init_fn=worker_init_fn)
