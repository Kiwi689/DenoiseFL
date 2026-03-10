from typing import Tuple

import torchvision.transforms as transforms
from PIL import Image
from torchvision.datasets import CIFAR100

from backbone.ResNet import resnet50
from backbone.SimpleCNN import SimpleCNN
from backbone.resnet_fedalign import resnet50_fedalign
from backbone.NoiseFLCNN import NoiseFLCNN
from datasets.transforms.denormalization import DeNormalize
from datasets.utils.federated_dataset import FederatedDataset, partition_label_skew_loaders
from utils.conf import data_path


class MyCifar100(CIFAR100):
    def __init__(self, root, train=True, transform=None,
                 target_transform=None, download=False) -> None:
        self.not_aug_transform = transforms.Compose([transforms.ToTensor()])
        super(MyCifar100, self).__init__(
            root, train, transform, target_transform, download
        )

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


class FedLeaCIFAR100(FederatedDataset):
    NAME = 'fl_cifar100'
    SETTING = 'label_skew'
    N_SAMPLES_PER_Class = None
    N_CLASS = 100

    Nor_TRANSFORM = transforms.Compose([
        transforms.ToTensor()
    ])

    def get_data_loaders(self, train_transform=None):
        train_transform = transforms.ToTensor()
        test_transform = transforms.ToTensor()

        train_dataset = MyCifar100(
            root=data_path(),
            train=True,
            download=True,
            transform=train_transform
        )

        test_dataset = MyCifar100(
            root=data_path(),
            train=False,
            download=True,
            transform=test_transform
        )

        traindls, testdl, net_cls_counts = partition_label_skew_loaders(train_dataset, test_dataset, self)
        return traindls, testdl, net_cls_counts

    @staticmethod
    def get_transform():
        return transforms.Compose([transforms.ToTensor()])

    @staticmethod
    def get_backbone(parti_num, names_list, model_name=''):
        nets_list = []
        if model_name == 'moon':
            for _ in range(parti_num):
                nets_list.append(resnet50(num_classes=FedLeaCIFAR100.N_CLASS, name='resnet50'))
        elif model_name == 'fedalign':
            for _ in range(parti_num):
                nets_list.append(resnet50_fedalign(class_num=FedLeaCIFAR100.N_CLASS, name='resnet50'))
        elif model_name == 'feddenoise':
            for _ in range(parti_num):
                nets_list.append(NoiseFLCNN(input_channel=3, n_outputs=FedLeaCIFAR100.N_CLASS))
        else:
            for _ in range(parti_num):
                nets_list.append(SimpleCNN(FedLeaCIFAR100.N_CLASS))
        return nets_list

    @staticmethod
    def get_normalization_transform():
        return transforms.Normalize(
            (0.5070751592371323, 0.48654887331495095, 0.4409178433670343),
            (0.2673342858792401, 0.2564384629170883, 0.27615047132568404)
        )

    @staticmethod
    def get_denormalization_transform():
        return DeNormalize(
            (0.5070751592371323, 0.48654887331495095, 0.4409178433670343),
            (0.2673342858792401, 0.2564384629170883, 0.27615047132568404)
        )