from __future__ import absolute_import

import copy
import random
from collections import defaultdict

import numpy as np
import torch.distributed as dist
from torch.utils.data.sampler import Sampler


class DistributedRandomIdentitySampler(Sampler):
    """
    Randomly sample N identities, then for each identity,
    randomly sample K instances, therefore batch size is N*K.
    Args:
    - data_source (list): list of (img_path, pid, camid).
    - num_instances (int): number of instances per identity in a batch.
    - batch_size (int): number of examples in a batch.
    """
    def __init__(self, data_source, batch_size, num_instances, start_iter=0, max_iter=30000):
        self.task_rank = dist.get_rank()
        self.task_size = dist.get_world_size()
        self.data_source = data_source
        self.batch_size = batch_size
        self.num_instances = num_instances
        self.num_pids_per_batch = self.batch_size // self.num_instances
        self.index_dic = defaultdict(list)
        for index, (_, _, pid, _, _) in enumerate(self.data_source):
            self.index_dic[pid].append(index)
        self.pids = list(self.index_dic.keys())

        # estimate number of examples in an epoch
        self.max_iter = max_iter
        self.start_iter = start_iter
        self.ret = []
        self.target_length = self.max_iter * self.batch_size
        self.init_selected_list()

    def __iter__(self):
        return iter(self.ret[self.start_iter*self.batch_size:self.target_length])

    def __len__(self):
        return self.target_length - self.start_iter*self.batch_size

    def init_selected_list(self):
        seed = 1
        while len(self.ret) < self.target_length:
            batch_idxs_dict = defaultdict(list)
            for pid in self.pids:
                idxs = copy.deepcopy(self.index_dic[pid])
                if len(idxs) < self.num_instances:
                    np.random.seed(seed)
                    idxs = np.random.choice(idxs, size=self.num_instances, replace=True)
                random.shuffle(idxs)
                batch_idxs = []
                for idx in idxs:
                    batch_idxs.append(idx)
                    if len(batch_idxs) == self.num_instances:
                        batch_idxs_dict[pid].append(batch_idxs)
                        batch_idxs = []

            avai_pids = copy.deepcopy(self.pids)
            final_idxs = []

            while len(avai_pids) >= self.num_pids_per_batch:
                random.seed(seed)
                selected_pids = random.sample(avai_pids, self.num_pids_per_batch)
                for pid in selected_pids:
                    batch_idxs = batch_idxs_dict[pid].pop(0)
                    final_idxs.extend(batch_idxs)
                    if len(batch_idxs_dict[pid]) == 0:
                        avai_pids.remove(pid)

            total_length = len(final_idxs)
            per_length = total_length // self.task_size
            beg = self.task_rank * per_length
            end = beg+per_length
            assert end <= total_length
            self.ret.extend(final_idxs[beg:end])
            seed += 1


class MultiTaskDistributedRandomIdentitySampler(Sampler):
    """
    Randomly sample N identities, then for each identity,
    randomly sample K instances, therefore batch size is N*K.
    Args:
    - data_source (list): list of (img_path, pid, camid).
    - num_instances (int): number of instances per identity in a batch.
    - batch_size (int): number of examples in a batch.
    """
    def __init__(self, data_source, batch_size, num_instances, this_task_info, start_iter=0, max_iter=30000):
        self.task_rank = this_task_info.task_rank
        self.task_size = this_task_info.task_size
        self.data_source = data_source
        self.batch_size = batch_size
        self.num_instances = num_instances
        self.num_pids_per_batch = self.batch_size // self.num_instances
        self.index_dic = defaultdict(list)
        for index, (_, _, pid, _, _) in enumerate(self.data_source):
            self.index_dic[pid].append(index)
        self.pids = list(self.index_dic.keys())

        # estimate number of examples in an epoch
        self.max_iter = max_iter
        self.start_iter = start_iter
        self.ret = []
        self.target_length = self.max_iter * self.batch_size
        self.init_selected_list()

    def __iter__(self):
        return iter(self.ret[self.start_iter*self.batch_size:self.target_length])

    def __len__(self):
        return self.target_length - self.start_iter*self.batch_size

    def init_selected_list(self):
        seed = 1
        while len(self.ret) < self.target_length:
            batch_idxs_dict = defaultdict(list)
            for pid in self.pids:
                idxs = copy.deepcopy(self.index_dic[pid])
                if len(idxs) < self.num_instances:
                    np.random.seed(seed)
                    idxs = np.random.choice(idxs, size=self.num_instances, replace=True)
                random.shuffle(idxs)
                batch_idxs = []
                for idx in idxs:
                    batch_idxs.append(idx)
                    if len(batch_idxs) == self.num_instances:
                        batch_idxs_dict[pid].append(batch_idxs)
                        batch_idxs = []

            avai_pids = copy.deepcopy(self.pids)
            final_idxs = []

            while len(avai_pids) >= self.num_pids_per_batch:
                random.seed(seed)
                selected_pids = random.sample(avai_pids, self.num_pids_per_batch)
                for pid in selected_pids:
                    batch_idxs = batch_idxs_dict[pid].pop(0)
                    final_idxs.extend(batch_idxs)
                    if len(batch_idxs_dict[pid]) == 0:
                        avai_pids.remove(pid)

            total_length = len(final_idxs)
            per_length = total_length // self.task_size
            beg = self.task_rank * per_length
            end = beg+per_length
            assert end <= total_length
            self.ret.extend(final_idxs[beg:end])
            seed += 1
