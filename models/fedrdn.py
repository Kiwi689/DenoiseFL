import copy
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from models.utils.federated_model import FederatedModel


class FedRDN(FederatedModel):
    NAME = 'fedrdn'
    COMPATIBILITY = ['homogeneity']

    def __init__(self, nets_list, args, transform):
        super(FedRDN, self).__init__(nets_list, args, transform)

        self.rdn_eps = getattr(args, "rdn_eps", 1e-6)

        # 每个客户端一个 (mu, std)，shape: [C]
        self.client_stats = None
        self.stats_ready = False

    def ini(self):
        self.global_net = copy.deepcopy(self.nets_list[0])
        global_w = self.global_net.state_dict()

        for net in self.nets_list:
            net.load_state_dict(global_w)

    def loc_update(self, priloader_list):
        if self.trainloaders is None:
            self.trainloaders = priloader_list

        # 论文要求训练开始前先计算并共享所有客户端统计量
        if not self.stats_ready:
            self._compute_all_client_stats(priloader_list)
            self.stats_ready = True

        total_clients = list(range(self.args.parti_num))
        online_clients = self.random_state.choice(
            total_clients,
            self.online_num,
            replace=False
        ).tolist()
        self.online_clients = online_clients

        for client_id in online_clients:
            self._train_net(client_id, self.nets_list[client_id], priloader_list[client_id])

        # FedRDN 不改服务器聚合，直接沿用 FedAvg-style aggregation
        self.aggregate_nets(None)
        return None

    def _compute_all_client_stats(self, priloader_list):
        """
        预计算每个客户端本地数据的 channel-wise mean/std。
        论文 Algorithm 1 / Eq.(5)(6) 的工程实现版。
        """
        self.client_stats = []

        for client_id in range(self.args.parti_num):
            mu, std = self._compute_dataset_channel_stats(priloader_list[client_id])
            self.client_stats.append({
                "mu": mu.to(self.device),
                "std": std.to(self.device)
            })

    def _compute_dataset_channel_stats(self, train_loader):
        """
        计算一个客户端数据集的 channel-wise mean/std。
        支持图像张量形状 [B,C,H,W]。
        """
        sum_mu = None
        sum_std = None
        num_samples = 0

        for batch in train_loader:
            if len(batch) == 3:
                images, _, _ = batch
            else:
                images, _ = batch

            # images: [B, C, H, W]
            images = images.float()

            # 每张图像的 sample-level channel mean/std
            sample_mu = images.mean(dim=(2, 3))                     # [B, C]
            sample_std = images.std(dim=(2, 3), unbiased=False)    # [B, C]

            if sum_mu is None:
                sum_mu = sample_mu.sum(dim=0)
                sum_std = sample_std.sum(dim=0)
            else:
                sum_mu += sample_mu.sum(dim=0)
                sum_std += sample_std.sum(dim=0)

            num_samples += images.size(0)

        # 这里取 dataset-level 平均统计量，工程上更稳定
        mu = sum_mu / max(num_samples, 1)
        std = sum_std / max(num_samples, 1)
        std = torch.clamp(std, min=self.rdn_eps)

        return mu, std

    def _apply_random_rdn_train(self, images):
        """
        训练时：对 batch 中每张图像，随机抽取一个客户端的 (mu, std) 做归一化。
        对应论文 Eq.(7)。
        """
        # images: [B, C, H, W]
        bsz = images.size(0)
        device = images.device

        sampled_client_ids = self.random_state.choice(
            list(range(len(self.client_stats))),
            size=bsz,
            replace=True
        )

        mus = []
        stds = []
        for cid in sampled_client_ids:
            mus.append(self.client_stats[cid]["mu"])
            stds.append(self.client_stats[cid]["std"])

        mu = torch.stack(mus, dim=0).to(device)     # [B, C]
        std = torch.stack(stds, dim=0).to(device)   # [B, C]

        mu = mu.view(bsz, -1, 1, 1)
        std = std.view(bsz, -1, 1, 1)

        return (images - mu) / std

    def normalize_test_images(self, client_id, images):
        """
        测试时：只使用该客户端自己的统计量。
        对应论文 Eq.(8)。

        用法：
            images = model.normalize_test_images(client_id, images)
            outputs = model.global_net(images)
        """
        stats = self.client_stats[client_id]
        mu = stats["mu"].to(images.device).view(1, -1, 1, 1)
        std = stats["std"].to(images.device).view(1, -1, 1, 1)
        return (images - mu) / std

    def _train_net(self, index, net, train_loader):
        net = net.to(self.device)
        net.train()

        optimizer = optim.SGD(
            net.parameters(),
            lr=self.local_lr,
            momentum=0.9,
            weight_decay=self.args.reg
        )
        criterion = nn.CrossEntropyLoss().to(self.device)

        for _ in range(self.local_epoch):
            for batch in train_loader:
                if len(batch) == 3:
                    images, labels, _ = batch
                else:
                    images, labels = batch

                images = images.to(self.device).float()
                labels = labels.to(self.device)

                # FedRDN 核心：训练时随机注入 federation 内其他客户端的统计量
                images = self._apply_random_rdn_train(images)

                outputs = net(images)
                loss = criterion(outputs, labels)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()