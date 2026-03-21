import os
from typing import Tuple

import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import Dataset

from backbone.NoiseFLCNN import NoiseFLCNN
from backbone.SimpleCNN import SimpleCNN
from backbone.ResNet import resnet18

from datasets.utils.federated_dataset import FederatedDataset, partition_label_skew_loaders
from utils.conf import data_path


class MyTinyImageNet(Dataset):
    """
    Tiny-ImageNet 数据集读取类：
    - train: 读取 train/<class>/images/*.JPEG
    - val  : 读取 val/images/*.JPEG，并根据 val_annotations.txt 找标签

    同时提供：
    - targets
    - clean_targets
    以兼容当前项目的噪声注入和 pure ratio 统计
    """

    def __init__(self, root, split='train', transform=None):
        self.root = root
        self.split = split
        self.transform = transform
        
        self.samples = []
        self.targets = []

        wnids_path = os.path.join(root, 'wnids.txt')
        if not os.path.exists(wnids_path):
            raise FileNotFoundError(f"Cannot find wnids.txt at: {wnids_path}")

        with open(wnids_path, 'r') as f:
            self.classes = [line.strip() for line in f.readlines()]

        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}

        if split == 'train':
            train_dir = os.path.join(root, 'train')
            if not os.path.isdir(train_dir):
                raise FileNotFoundError(f"Cannot find train dir at: {train_dir}")

            for cls_name in self.classes:
                img_dir = os.path.join(train_dir, cls_name, 'images')
                if not os.path.isdir(img_dir):
                    continue

                for fname in os.listdir(img_dir):
                    if fname.lower().endswith(('.jpeg', '.jpg', '.png')):
                        path = os.path.join(img_dir, fname)
                        target = self.class_to_idx[cls_name]
                        self.samples.append((path, target))
                        self.targets.append(target)

        elif split == 'val':
            val_dir = os.path.join(root, 'val')
            img_dir = os.path.join(val_dir, 'images')
            anno_path = os.path.join(val_dir, 'val_annotations.txt')

            if not os.path.isdir(img_dir):
                raise FileNotFoundError(f"Cannot find val/images dir at: {img_dir}")
            if not os.path.exists(anno_path):
                raise FileNotFoundError(f"Cannot find val_annotations.txt at: {anno_path}")

            img_to_class = {}
            with open(anno_path, 'r') as f:
                for line in f.readlines():
                    parts = line.strip().split('\t')
                    img_name = parts[0]
                    cls_name = parts[1]
                    img_to_class[img_name] = cls_name

            for fname in os.listdir(img_dir):
                if fname in img_to_class:
                    cls_name = img_to_class[fname]
                    target = self.class_to_idx[cls_name]
                    path = os.path.join(img_dir, fname)
                    self.samples.append((path, target))
                    self.targets.append(target)

        else:
            raise ValueError(f"Unknown split: {split}")

        self.clean_targets = self.targets.copy()

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index: int) -> Tuple[type(Image), int]:
        path, _ = self.samples[index]

        # 后续会改 self.targets 来注入噪声，所以这里必须从 self.targets[index] 取
        target = self.targets[index]

        img = Image.open(path).convert('RGB')

        if self.transform is not None:
            img = self.transform(img)

        return img, target


class FedLeaTinyImageNet(FederatedDataset):
    NAME = 'fl_tinyimagenet'
    SETTING = 'label_skew'
    N_SAMPLES_PER_Class = None
    N_CLASS = 200

    Nor_TRANSFORM = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor()
    ])

    def get_data_loaders(self, train_transform=None):
        root = os.path.join(data_path(), 'tiny-imagenet-200')

        # 先把 64x64 resize 成 32x32，最容易兼容你当前的 backbone
        train_transform = transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.ToTensor()
        ])

        test_transform = transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.ToTensor()
        ])

        train_dataset = MyTinyImageNet(
            root=root,
            split='train',
            transform=train_transform
        )

        test_dataset = MyTinyImageNet(
            root=root,
            split='val',
            transform=test_transform
        )

        traindls, testdl, net_cls_counts = partition_label_skew_loaders(
            train_dataset, test_dataset, self
        )
        return traindls, testdl, net_cls_counts

    @staticmethod
    def get_transform():
        return transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.ToTensor()
        ])

    @staticmethod
    def get_backbone(parti_num, names_list, model_name=''):
        nets_list = []

        if model_name == 'feddenoise':
            for _ in range(parti_num):
                nets_list.append(
                    resnet18(num_classes=FedLeaTinyImageNet.N_CLASS, name='resnet18')
                )
        else:
            for _ in range(parti_num):
                nets_list.append(SimpleCNN(FedLeaTinyImageNet.N_CLASS))

        return nets_list

    @staticmethod
    def get_normalization_transform():
        return transforms.Compose([])

    @staticmethod
    def get_denormalization_transform():
        return transforms.Compose([])