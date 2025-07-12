import torch
from torchvision import datasets, transforms
import random
import os
from datasets.cifar import cifar_noniid, cifar_dirichlet_balanced,cifar_dirichlet_unbalanced, cifar_iid, cifar_overlap, cifar_toyset
import torch.nn as nn
import csv
from typing import List, Dict
import copy
import json
from collections import OrderedDict

import numpy as np

__all__ = ['DatasetSplit', 'DatasetSplitSubset', 'DatasetSplitMultiView', 'get_dataset', 'MultiViewDataInjector', 'GaussianBlur', 'TransformTwice'
                                                                                                            ]

create_dataset_log = False

class TransformTwice:
    def __init__(self, transform):
        self.transform = transform

    def __call__(self, inp):
        out1 = self.transform(inp)
        out2 = self.transform(inp)
        return out1, out2


class DatasetSplit(torch.utils.data.Dataset):
    """An abstract Dataset class wrapped around Pytorch Dataset class.
    """

    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = [int(i) for i in idxs]
        self.class_dict = {}
        for idx in self.idxs:
            _, label = self.dataset[idx]
            if torch.is_tensor(label):
                label = str(label.item())
            else:
                label = str(label)
            if label in self.class_dict:
                self.class_dict[str(label)] += 1
            else:
                self.class_dict[str(label)] = 1


    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        image, label = self.dataset[self.idxs[item]]
        return image, label
    
    @property
    def num_classes(self):
        return len(self.class_dict.keys())
    
    @property
    def class_ids(self):
        return self.class_dict.keys()
    
    def importance_weights(self, labels, pow=1):
        class_counts = np.array([self.class_dict[str(label.item())] for label in labels])
        weights = (1/class_counts)**pow
        weights /= weights.mean()
        return weights



class DatasetSplitSubset(DatasetSplit):
    """An abstract Dataset class wrapped around Pytorch Dataset class.
    """

    def __init__(self, dataset, idxs, subset_classes=None):
        self.dataset = dataset

        self.subset_classes = subset_classes

        self.class_dict = {}
        self.indices = []

        for idx in idxs:
            _, label = self.dataset[int(idx)]
            if torch.is_tensor(label):
                label = str(label.item())
            else:
                label = str(label)

            if subset_classes is not None and int(label) not in subset_classes:
                continue

            self.indices.append(idx)

            if label in self.class_dict:
                self.class_dict[str(label)] += 1
            else:
                self.class_dict[str(label)] = 1


    def __len__(self):
        return len(self.indices)

    def __getitem__(self, item):
        image, label = self.dataset[self.indices[item]]
        return image, label
    
    @property
    def num_classes(self):
        return len(self.class_dict.keys())
    
    @property
    def class_ids(self):
        return self.class_dict.keys()
    
    def importance_weights(self, labels, pow=1):
        class_counts = np.array([self.class_dict[str(label.item())] for label in labels])
        weights = (1/class_counts)**pow
        weights /= weights.mean()
        return weights


class DatasetSplitMultiView(torch.utils.data.Dataset):
    """An abstract Dataset class wrapped around Pytorch Dataset class.
    """

    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = [int(i) for i in idxs]

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        (view1, view2), label = self.dataset[self.idxs[item]]
        return torch.tensor(view1), torch.tensor(view2), torch.tensor(label)


def get_dataset(args, trainset, mode='iid'):
    set = args.dataset.name
    if 'leaf' not in set:
        directory = args.dataset.client_path + '/' + set + '/' + ('un' if args.split.unbalanced==True else '') + 'balanced'
        filepath = directory+'/' + mode + (str(args.split.class_per_client) if mode == 'skew' else '') + (str(args.split.alpha) if mode == 'dirichlet' else '') + (str(args.split.overlap_ratio) if mode == 'overlap' else '') + '_clients' +str(args.trainer.num_clients) +  (("_toyinform_" + str(args.split.toy_noniid_rate) + "_" + str(args.split.limit_total_classes)+ "_" +  str(args.split.limit_number_per_class)) if 'toy' in mode else "")   + '.txt'

        check_already_exist = os.path.isfile(filepath) and (os.stat(filepath).st_size != 0)
        create_new_client_data = not check_already_exist or args.split.create_client_dataset

        if create_new_client_data == False:
            try:
                dataset = {}
                with open(filepath) as f:
                    for idx, line in enumerate(f):
                        dataset = eval(line)
            except:
                print("Have problem to read client data")

        if create_new_client_data == True:

            if mode == 'iid':
                dataset = cifar_iid(trainset, args.trainer.num_clients)
            elif mode == 'overlap':
                dataset = cifar_overlap(trainset, args.trainer.num_clients, args.split.overlap_ratio)
            # elif mode[:4] == 'skew' and mode[-5:] == 'class':
            elif mode == 'skew':
                class_per_client = args.split.class_per_client
                # assert class_per_client * args.trainer.num_clients == trainset.dataset.num_classes
                # class_per_user = int(mode[4:-5])
                dataset = cifar_noniid(trainset, args.trainer.num_clients, class_per_client)
            elif mode == 'dirichlet':
                if args.split.unbalanced==True:
                    dataset = cifar_dirichlet_unbalanced(trainset, args.trainer.num_clients, alpha=args.split.alpha)
                else:
                    dataset = cifar_dirichlet_balanced(trainset, args.trainer.num_clients, alpha=args.split.alpha)
            elif mode == 'toy_noniid':
                dataset = cifar_toyset(trainset, args.trainer.num_clients, num_valid_classes=args.split.limit_total_classes, limit_number_per_class = args.split.limit_number_per_class, toy_noniid_rate = args.split.toy_noniid_rate, non_iid = True)
            elif mode == 'toy_iid':
                dataset = cifar_toyset(trainset, args.trainer.num_clients, num_valid_classes=args.split.limit_total_classes, limit_number_per_class = args.split.limit_number_per_class, toy_noniid_rate = args.split.toy_noniid_rate, non_iid = False)                

            
            else:
                print("Invalid mode ==> please select in iid, skewNclass, dirichlet")
                return

            try:
                os.makedirs(directory, exist_ok=True)
                with open(filepath, 'w') as f:
                    print(dataset, file=f)

            except:
                print("Fail to write client data at " + directory)

        return dataset
    elif 'leaf' in set:
        return trainset.get_train_idxs()
    elif set == 'shakespeare':
        return trainset.get_client_dic()



class MultiViewDataInjector(object):
    def __init__(self, *args):
        self.transforms = args[0]
        self.random_flip = transforms.RandomHorizontalFlip()

    def __call__(self, sample, *with_consistent_flipping):
        if with_consistent_flipping:
            sample = self.random_flip(sample)
        output = [transform(sample) for transform in self.transforms]
        return output

class GaussianBlur(object):
    """blur a single image on CPU"""

    def __init__(self, kernel_size):
        radias = kernel_size // 2
        kernel_size = radias * 2 + 1
        self.blur_h = nn.Conv2d(3, 3, kernel_size=(kernel_size, 1),
                                stride=1, padding=0, bias=False, groups=3)
        self.blur_v = nn.Conv2d(3, 3, kernel_size=(1, kernel_size),
                                stride=1, padding=0, bias=False, groups=3)
        self.k = kernel_size
        self.r = radias

        self.blur = nn.Sequential(
            nn.ReflectionPad2d(radias),
            self.blur_h,
            self.blur_v
        )

        self.pil_to_tensor = transforms.ToTensor()
        self.tensor_to_pil = transforms.ToPILImage()


class CutMix(DatasetSplitSubset):
    def __init__(self, dataset, num_classes, num_mix=2, beta=1., prob=1.0, use_cutmix_reg=False):
        self.dataset = dataset.dataset
        self.subset_classes = dataset.subset_classes

        self.class_dict = dataset.class_dict
        self.indices = dataset.indices

        self.total_classes = num_classes
        self.num_mix = num_mix
        self.beta = beta
        self.prob = prob
        
        self.use_cutmix_reg = use_cutmix_reg
        if use_cutmix_reg == True:
            self.probs = self.compute_sampling_probs()
        # updates
        self.noise_prob = -1.
        if self.total_classes > len(self.class_dict.keys()):
            self.noise_prob  = self.compute_noise_prob()
        
    def compute_noise_prob(self):
        counts = np.array(list(self.class_dict.values()))
        p_max = 1. - float(len(counts)) / float(self.total_classes)
        probs = counts / counts.sum()
        entropy = -np.sum(probs * np.log(probs + 1e-8)) / np.log(len(probs))  # 정규화 엔트로피
        return p_max * (1.0 - entropy)
        
    def compute_sampling_probs(self):
        label_list = [self.dataset[idx][-1] for idx in self.indices]
        class_counts = torch.tensor([self.class_dict[str(label)] for label in label_list])
        weights = 1. / (class_counts + 1e-6)
        probs = weights / weights.sum()
        return probs
    
    def __getitem__(self, item):
        img, label = self.dataset[self.indices[item]]
        label_onehot = self.onehot(label)

        for _ in range(self.num_mix):
            r = np.random.rand(1)
            if self.beta <= 0 or r > self.prob:
                continue
            # generate mixed sample
            lamda = np.random.beta(self.beta, self.beta)
            bbx1, bby1, bbx2, bby2 = self.rand_bbox(img.size(), lamda)
            lamda = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (img.size()[-1] * img.size()[-2]))
            
            if self.noise_prob > 0. and np.random.rand() < self.noise_prob:
                # generate noise
                img2 = torch.randn_like(img)
                label_onehot = label_onehot * lamda
            else:
                if self.use_cutmix_reg == True:
                    rand_item = torch.multinomial(self.probs, 1).item()
                else:
                    rand_item = random.choice(range(len(self.indices)))

                img2, label2 = self.dataset[self.indices[rand_item]]
                label2_onehot = self.onehot(label2)
                label_onehot = label_onehot * lamda + label2_onehot * (1. - lamda)

            img[:, bbx1:bbx2, bby1:bby2] = img2[:, bbx1:bbx2, bby1:bby2]
            
        return img, label_onehot

    def onehot(self, target):
        vec = torch.zeros(self.total_classes, dtype=torch.float32)
        vec[target] = 1.
        return vec
    
    @staticmethod
    def rand_bbox(size, lam):
        if len(size) == 4:
            W = size[2]
            H = size[3]
        elif len(size) == 3:
            W = size[1]
            H = size[2]
        else:
            raise Exception

        cut_rat = np.sqrt(1. - lam)
        cut_w = int(W * cut_rat)
        cut_h = int(H * cut_rat)

        # uniform
        cx = np.random.randint(W)
        cy = np.random.randint(H)

        bbx1 = np.clip(cx - cut_w // 2, 0, W)
        bby1 = np.clip(cy - cut_h // 2, 0, H)
        bbx2 = np.clip(cx + cut_w // 2, 0, W)
        bby2 = np.clip(cy + cut_h // 2, 0, H)

        return bbx1, bby1, bbx2, bby2
    
    
class Mixup(DatasetSplitSubset):
    def __init__(self, dataset, num_classes, num_mix=2, beta=1., use_cutmix_reg=False):
        self.dataset = dataset.dataset
        self.subset_classes = dataset.subset_classes

        self.class_dict = dataset.class_dict
        self.indices = dataset.indices

        self.total_classes = num_classes
        self.num_mix = num_mix
        self.beta = beta
        
        self.use_cutmix_reg = use_cutmix_reg
        if use_cutmix_reg == True:
            self.probs = self.compute_sampling_probs()
        
    def compute_sampling_probs(self):
        label_list = [self.dataset[idx][-1] for idx in self.indices]
        class_counts = torch.tensor([self.class_dict[str(label)] for label in label_list])
        weights = 1. / (class_counts + 1e-6)
        probs = weights / weights.sum()
        return probs
    
    def __getitem__(self, item):
        img, label = self.dataset[self.indices[item]]
        label_onehot = self.onehot(label)

        for _ in range(self.num_mix):
            r = np.random.rand(1)
            if self.beta <= 0 or r > self.prob:
                continue

            # generate mixed sample
            lamda = np.random.beta(self.beta, self.beta)
            if self.use_cutmix_reg == True:
                rand_item = torch.multinomial(self.probs, 1).item()
            else:
                rand_item = random.choice(range(len(self.indices)))

            img2, label2 = self.dataset[self.indices[rand_item]]
            label2_onehot = self.onehot(label2)

            img = img * lamda + img2 * (1. - lamda)
            label_onehot = label_onehot * lamda + label2_onehot * (1. - lamda)

        return img, label_onehot

    def onehot(self, target):
        vec = torch.zeros(self.total_classes, dtype=torch.float32)
        vec[target] = 1.
        return vec
