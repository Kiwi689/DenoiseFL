import torch
import torchvision.transforms as transforms
from torchvision.datasets import MNIST

from backbone.NoiseFLCNN import NoiseFLCNN
from backbone.SimpleCNN import SimpleCNN
from utils.conf import data_path
from datasets.utils.federated_dataset import FederatedDataset, partition_label_skew_loaders
from datasets.transforms.denormalization import DeNormalize


class MyMNIST(torch.utils.data.Dataset):
    def __init__(self, root, train=True, transform=None,
                 target_transform=None, download=False, data_name=None) -> None:
        self.not_aug_transform = transforms.Compose([transforms.ToTensor()])
        self.data_name = data_name
        self.root = root
        self.train = train
        self.transform = transform
        self.target_transform = target_transform
        self.download = download

        self.dataset = self.__build_truncated_dataset__()

        # 与当前噪声注入逻辑对齐：暴露 targets / clean_targets
        if hasattr(self.dataset, 'targets'):
            if isinstance(self.dataset.targets, list):
                self.targets = self.dataset.targets.copy()
            else:
                self.targets = self.dataset.targets.tolist()
        else:
            raise ValueError("MNIST dataset has no attribute 'targets'.")

        self.clean_targets = self.targets.copy()

    def __build_truncated_dataset__(self):
        dataobj = MNIST(
            self.root,
            self.train,
            self.transform,
            self.target_transform,
            self.download
        )
        return dataobj

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index: int):
        # 直接从底层 dataset 取图像
        img, _ = self.dataset[index]

        # 标签统一从外层 self.targets 读，这样噪声注入写回后能生效
        target = self.targets[index]

        return img, target


class FedMNIST(FederatedDataset):
    NAME = 'fl_mnist'
    SETTING = 'label_skew'
    N_SAMPLES_PER_Class = None
    N_CLASS = 10

    # 注意：去掉 HorizontalFlip，数字翻转通常有害
    Single_Channel_Nor_TRANSFORM = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.repeat(3, 1, 1)),
        transforms.Normalize(
            (0.1307, 0.1307, 0.1307),
            (0.3081, 0.3081, 0.3081)
        )
    ])

    def get_data_loaders(self, train_transform=None):
        pri_aug = self.args.pri_aug

        if pri_aug == 'weak':
            train_transform = self.Single_Channel_Nor_TRANSFORM
        else:
            train_transform = self.Single_Channel_Nor_TRANSFORM

        test_transform = transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x.repeat(3, 1, 1)),
            self.get_normalization_transform()
        ])

        train_dataset = MyMNIST(
            root=data_path(),
            train=True,
            download=True,
            transform=train_transform
        )

        test_dataset = MyMNIST(
            root=data_path(),
            train=False,
            download=True,
            transform=test_transform
        )

        traindls, testdl, net_cls_counts = partition_label_skew_loaders(
            train_dataset, test_dataset, self
        )
        return traindls, testdl, net_cls_counts

    @staticmethod
    def get_transform():
        transform = transforms.Compose([
            transforms.ToPILImage(),
            FedMNIST.Single_Channel_Nor_TRANSFORM
        ])
        return transform

    @staticmethod
    def get_backbone(parti_num, names_list, model_name=''):
        nets_list = []
        for _ in range(parti_num):
            nets_list.append(
                NoiseFLCNN(input_channel=3, n_outputs=FedMNIST.N_CLASS)
            )
        return nets_list

    @staticmethod
    def get_normalization_transform():
        transform = transforms.Normalize(
            mean=(0.1307, 0.1307, 0.1307),
            std=(0.3081, 0.3081, 0.3081)
        )
        return transform

    @staticmethod
    def get_denormalization_transform():
        transform = DeNormalize(
            mean=(0.1307, 0.1307, 0.1307),
            std=(0.3081, 0.3081, 0.3081)
        )
        return transform