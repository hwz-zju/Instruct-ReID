from torch.utils.data import DataLoader

from .data import transforms as T
from .data.preprocessor_t2i import PreProcessor
from .data.sampler import DistributedRandomIdentitySampler, MultiTaskDistributedRandomIdentitySampler
from .image_layer_multi import Image_Layer
from .image_layer import Image_Layer as Image_Layer_test
import os

def merge_sub_datasets(train_list, train_root_list):
    if not isinstance(train_list, list):
        task_list = [train_list]
    else:
        task_list = train_list
    if not isinstance(train_root_list, list):
        task_pref = [train_root_list]
    else:
        task_pref = train_root_list
        
    num_sub_datasets = len(task_list)
    assert num_sub_datasets == len(task_pref)

    label_base = 0
    list_lines_all = []
    for list_file, prefix in zip(task_list, task_pref):
        prefix = None
        list_lines = []
        sub_dataset_pid = 0
        sub_dataset_pid_list = dict()
        
        with open(list_file) as f:
            for line in f.readlines():
                info = line.strip('\n').split(" ")
                imgs = info[0]
                clothes = info[1]
                if info[2] not in sub_dataset_pid_list.keys():
                    pids = label_base + sub_dataset_pid
                    sub_dataset_pid_list[info[2]] = pids
                    sub_dataset_pid += 1
                else:
                    pids = sub_dataset_pid_list[info[2]]
                cids = info[3]
                if len(info) > 4:
                    cams = info[4]
                    list_lines.append('{} {} {} {} {}'.format(imgs, clothes, pids, cids, cams))
                else:
                    list_lines.append('{} {} {} {}'.format(imgs, clothes, pids, cids))
        label_base = label_base + sub_dataset_pid # update label_base

        list_lines_all.extend(list_lines)

    return list_lines_all


class DataBuilder_t2i(object):
    def __init__(self, args, task_info=None):
        super(DataBuilder_t2i, self).__init__()
        self.args = args
        self.this_task_info = task_info
        self.normalizer = T.Normalize(mean=[0.485, 0.456, 0.406],
                                      std=[0.229, 0.224, 0.225])

    def _build_train_loader(self, dataset):
        if self.args.colorjitter =='all':
            train_transformer = T.Compose([
                #T.ImageNetPolicy(self.args.iters),
                T.ColorJitter(brightness=.5, contrast=0.5, saturation=0.5, hue=.4),
                T.Resize((self.args.height, self.args.width)),
                T.RandomHorizontalFlip(p=0.5),
                T.Pad(10),
                T.RandomCrop((self.args.height, self.args.width)),
                T.RandomSizedEarserImage(),
                T.ToTensor(),
                self.normalizer
            ])
            
            train_clothes_transformer = T.Compose([
                T.Resize((self.args.width, self.args.width)),
                T.RandomCrop((self.args.width, self.args.width)),
                T.ToTensor(),
                T.Normalize(mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711])
                    ])
        else:
            train_transformer = T.Compose([
                T.Resize((self.args.height, self.args.width)),
                T.RandomHorizontalFlip(p=0.5),
                T.Pad(10),
                T.RandomCrop((self.args.height, self.args.width)),
                T.RandomSizedEarserImage(),
                T.ToTensor(),
                self.normalizer
            ])
            train_clothes_transformer = T.Compose([
                T.Resize((self.args.width, self.args.width)),
                T.RandomCrop((self.args.width, self.args.width)),
                T.ToTensor(),
                T.Normalize(mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711])
            ])
        if self.this_task_info:
            sampler = MultiTaskDistributedRandomIdentitySampler(dataset,
                                                    24,
                                                    # self.args.num_instances,
                                                    1,
                                                    self.this_task_info,
                                                    max_iter=self.args.iters
                                                    )
        else:
            sampler = DistributedRandomIdentitySampler(dataset,
                                                    24,
                                                    # self.args.num_instances,
                                                    1,
                                                    max_iter=self.args.iters
                                                    )
        json_list = self.args.data_config['tasks'][self.this_task_info['task_id']]['attt_file']
        data_root = self.this_task_info.root_path if self.this_task_info else self.args.root
        train_loader = DataLoader(PreProcessor(dataset, root=data_root, json_list=json_list,
                                               transform=train_transformer,
                                               clothes_transform=train_clothes_transformer, blur_clo=self.args.pool_clo),
                                  batch_size=24,
                                  num_workers=self.args.workers,
                                  sampler=sampler,
                                  shuffle=False,
                                  pin_memory=False
                                  )

        return train_loader

    def _build_test_loader(self, query_dataset, gallery_dataset):
        test_transformer = T.Compose([
            T.Resize((self.args.height, self.args.width)),
            T.ToTensor(),
            self.normalizer
        ])
        
            
        
        test_clothes_transformer = T.Compose([
            T.Resize((self.args.width, self.args.width)),
            T.ToTensor(),
            self.normalizer
        ])
        
        data_root = self.this_task_info.root_path if self.this_task_info else self.args.root
        test_set = list(set(query_dataset) | set(gallery_dataset))
        json_list = ' <your project root> + Instruct-ReID/data/cuhk_pedes/caption_t2i_v2.json'
        test_loader = DataLoader(PreProcessor(test_set, root=data_root, json_list=json_list, root_additional=None, 
                                              transform=test_transformer,
                                              clothes_transform=test_clothes_transformer,blur_clo=self.args.pool_clo),
                                 batch_size=self.args.batch_size,
                                 num_workers=self.args.workers,
                                 shuffle=False,
                                 pin_memory=False)

        return test_loader

    def build_data(self, is_train, image_list=None):
        if image_list is not None:
            dataset = Image_Layer(image_list, is_train=True)
            data_loader = self._build_train_loader(dataset.data)
            return data_loader, dataset

        # default
        if is_train:
            train_list = self.this_task_info.train_file_path if self.this_task_info is not None else self.args.train_list
            train_root_list = self.this_task_info.root_path if self.this_task_info else self.args.root
            train_list_all = merge_sub_datasets(train_list, train_root_list)
            train_dataset = Image_Layer(train_list_all, train_list, is_train=True)
            train_loader = self._build_train_loader(train_dataset.data)
            # train_dataset.num_classes = 1
            return train_loader, train_dataset
        query_dataset = Image_Layer_test(self.args.query_list, is_query=True)
        gallery_dataset = Image_Layer_test(self.args.gallery_list, self.args.gallery_list_add, is_gallery=True)
        test_loader = self._build_test_loader(query_dataset.data, gallery_dataset.data)
        return test_loader, query_dataset, gallery_dataset
