from typing import Tuple

import torchvision.transforms as T
import torchvision.transforms as transforms
from PIL import Image
from torchvision.datasets import CIFAR10

from backbone.SimpleCNN import SimpleCNN
from backbone.SimpleCNNAlign import SimpleCNNAilgn
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

    torchvision_normalization = T.Normalize((0.4914, 0.4822, 0.4465),
                                            (0.2470, 0.2435, 0.2615))
    torchvision_denormalization = DeNormalize((0.4914, 0.4822, 0.4465),
                                              (0.2470, 0.2435, 0.2615))

    Nor_TRANSFORM = transforms.Compose([
        transforms.ToTensor()
    ])

    CON_TRANSFORMS = transforms.Compose([
        transforms.ToTensor()
    ])

    transform_train = transforms.Compose([
        transforms.ToTensor()
    ])

    def get_data_loaders(self, train_transform=None):
        train_transform = transforms.ToTensor()
        test_transform = transforms.ToTensor()

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
        return transforms.Compose([transforms.ToTensor()])

    @staticmethod
    def get_backbone(parti_num, names_list, model_name=''):
        nets_list = []

        if model_name == 'fedalign':
            for _ in range(parti_num):
                nets_list.append(SimpleCNNAilgn(FedLeaCIFAR10.N_CLASS))
        elif model_name == 'feddenoise':
            for _ in range(parti_num):
                nets_list.append(NoiseFLCNN(input_channel=3, n_outputs=FedLeaCIFAR10.N_CLASS))
        else:
            for _ in range(parti_num):
                nets_list.append(SimpleCNN(FedLeaCIFAR10.N_CLASS))

        return nets_list

    @staticmethod
    def get_normalization_transform():
        return FedLeaCIFAR10.torchvision_normalization

    @staticmethod
    def get_denormalization_transform():
        return FedLeaCIFAR10.torchvision_denormalization