from typing import Tuple

import torchvision.transforms as transforms
from PIL import Image
from torchvision.datasets import SVHN

from backbone.NoiseFLCNN import NoiseFLCNN
from backbone.SimpleCNN import SimpleCNN
from datasets.utils.federated_dataset import FederatedDataset, partition_label_skew_loaders
from utils.conf import data_path


class MySVHN(SVHN):
    """
    适配当前项目的 SVHN 数据集：
    1. 保留 clean_targets
    2. 对齐出 targets 字段，兼容现有噪声注入逻辑
    3. __getitem__ 返回 (img, target)
    """

    def __init__(self, root, split='train', transform=None,
                 target_transform=None, download=False) -> None:
        super(MySVHN, self).__init__(
            root=root,
            split=split,
            transform=transform,
            target_transform=target_transform,
            download=download
        )

        # torchvision 的 SVHN 默认是 labels，不是 targets
        if hasattr(self, 'labels'):
            if isinstance(self.labels, list):
                self.targets = self.labels.copy()
            else:
                self.targets = self.labels.tolist()
        else:
            raise ValueError("SVHN dataset has no attribute 'labels'.")

        self.clean_targets = self.targets.copy()

    def __getitem__(self, index: int) -> Tuple[type(Image), int]:
        img = self.data[index]
        target = self.targets[index]

        # SVHN 原始格式是 CHW，要转成 HWC 再转 PIL
        img = img.transpose(1, 2, 0)
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target


class FedLeaSVHN(FederatedDataset):
    NAME = 'fl_svhn'
    SETTING = 'label_skew'
    N_SAMPLES_PER_Class = None
    N_CLASS = 10

    Nor_TRANSFORM = transforms.Compose([
        transforms.ToTensor()
    ])

    def get_data_loaders(self, train_transform=None):
        train_transform = transforms.ToTensor()
        test_transform = transforms.ToTensor()

        train_dataset = MySVHN(
            root=data_path(),
            split='train',
            download=True,
            transform=train_transform
        )

        test_dataset = MySVHN(
            root=data_path(),
            split='test',
            download=True,
            transform=test_transform
        )

        traindls, testdl, net_cls_counts = partition_label_skew_loaders(
            train_dataset, test_dataset, self
        )
        return traindls, testdl, net_cls_counts

    @staticmethod
    def get_transform():
        return transforms.Compose([transforms.ToTensor()])

    @staticmethod
    def get_backbone(parti_num, names_list, model_name=''):
        nets_list = []
        for _ in range(parti_num):
            nets_list.append(
                NoiseFLCNN(input_channel=3, n_outputs=FedLeaSVHN.N_CLASS)
            )
        return nets_list

    @staticmethod
    def get_normalization_transform():
        return transforms.Normalize(
            (0.4377, 0.4438, 0.4728),
            (0.1980, 0.2010, 0.1970)
        )

    @staticmethod
    def get_denormalization_transform():
        return transforms.Compose([])