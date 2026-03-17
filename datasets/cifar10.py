from typing import Tuple

import torchvision.transforms as T
import torchvision.transforms as transforms
from PIL import Image
from torchvision.datasets import CIFAR10

from backbone.NoiseFLCNN import NoiseFLCNN
from datasets.transforms.denormalization import DeNormalize
from datasets.utils.federated_dataset import FederatedDataset, partition_label_skew_loaders
from utils.conf import data_path


class MyCIFAR10(CIFAR10):
    """
    与 noise-fl 对齐：
    1. 保留原始 clean_targets
    2. __getitem__ 返回 (img, target)
    """
    def __init__(self, root, train=True, transform=None,
                 target_transform=None, download=False) -> None:
        self.not_aug_transform = transforms.Compose([transforms.ToTensor()])
        super(MyCIFAR10, self).__init__(root, train, transform, target_transform, download)

        if hasattr(self, 'targets'):
            self.clean_targets = self.targets.copy()

    def __getitem__(self, index: int) -> Tuple[type(Image), int]:
        img, target = self.data[index], self.targets[index]
        img = Image.fromarray(img, mode='RGB')

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target


class FedLeaCIFAR10(FederatedDataset):
    NAME = 'fl_cifar10'
    SETTING = 'label_skew'
    N_SAMPLES_PER_Class = None
    N_CLASS = 10

    CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
    CIFAR10_STD = (0.2470, 0.2435, 0.2615)

    torchvision_normalization = T.Normalize(CIFAR10_MEAN, CIFAR10_STD)
    torchvision_denormalization = DeNormalize(CIFAR10_MEAN, CIFAR10_STD)

    Nor_TRANSFORM = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD)
    ])

    CON_TRANSFORMS = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD)
    ])

    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD)
    ])

    def get_data_loaders(self, train_transform=None):
        train_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(self.CIFAR10_MEAN, self.CIFAR10_STD)
        ])

        test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(self.CIFAR10_MEAN, self.CIFAR10_STD)
        ])

        train_dataset = MyCIFAR10(
            root=data_path(),
            train=True,
            download=True,
            transform=train_transform
        )

        test_dataset = MyCIFAR10(
            root=data_path(),
            train=False,
            download=True,
            transform=test_transform
        )

        traindls, testdl, net_cls_counts = partition_label_skew_loaders(train_dataset, test_dataset, self)
        return traindls, testdl, net_cls_counts

    @staticmethod
    def get_transform():
        return transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                FedLeaCIFAR10.CIFAR10_MEAN,
                FedLeaCIFAR10.CIFAR10_STD
            )
        ])

    @staticmethod
    def get_backbone(parti_num, names_list, model_name=''):
        nets_list = []
        for _ in range(parti_num):
            nets_list.append(NoiseFLCNN(input_channel=3, n_outputs=FedLeaCIFAR10.N_CLASS))
        return nets_list

    @staticmethod
    def get_normalization_transform():
        return FedLeaCIFAR10.torchvision_normalization

    @staticmethod
    def get_denormalization_transform():
        return FedLeaCIFAR10.torchvision_denormalization