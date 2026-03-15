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

        # 数值稳定项
        self.rdn_eps = getattr(args, "rdn_eps", 1e-6)

        # 每个客户端一个统计量字典：{"mu": [C], "std": [C]}
        self.client_stats = None
        self.stats_ready = False

        # 全局平均统计量，测试时使用，避免 client_id 越界问题
        self.global_mu = None
        self.global_std = None

    def ini(self):
        self.global_net = copy.deepcopy(self.nets_list[0])
        global_w = self.global_net.state_dict()

        for net in self.nets_list:
            net.load_state_dict(global_w)

    def loc_update(self, priloader_list):
        if self.trainloaders is None:
            self.trainloaders = priloader_list

        # 只在第一次训练前预计算客户端统计量
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

        # 服务端仍然使用你工程原有聚合
        self.aggregate_nets(None)
        return None

    def _compute_all_client_stats(self, priloader_list):
        """
        为每个客户端计算 channel-wise mean/std。
        注意这里统计的是“图像输入分布”，不是样本数、不是标签、更不是 batch 数。
        """
        self.client_stats = []

        all_mu = []
        all_std = []

        for client_id in range(self.args.parti_num):
            mu, std = self._compute_dataset_channel_stats(priloader_list[client_id])

            mu = mu.to(self.device)
            std = std.to(self.device)

            self.client_stats.append({
                "mu": mu,
                "std": std
            })

            all_mu.append(mu)
            all_std.append(std)

        # 测试时使用全体客户端平均统计量，避免 evaluate 时 client_id 不可用
        self.global_mu = torch.stack(all_mu, dim=0).mean(dim=0).to(self.device)
        self.global_std = torch.stack(all_std, dim=0).mean(dim=0).to(self.device)
        self.global_std = torch.clamp(self.global_std, min=self.rdn_eps)

    def _compute_dataset_channel_stats(self, train_loader):
        """
        计算一个客户端数据的 channel-wise mean/std。
        输入 images 形状假设为 [B, C, H, W]。
        返回:
            mu:  [C]
            std: [C]
        """
        channel_sum = None
        channel_sq_sum = None
        total_pixels = 0

        for batch in train_loader:
            if len(batch) == 3:
                images, _, _ = batch
            else:
                images, _ = batch

            images = images.float()  # CPU 上统计即可
            # [B, C, H, W]
            bsz, ch, h, w = images.shape

            # 每个通道求和 / 平方和
            cur_sum = images.sum(dim=(0, 2, 3))          # [C]
            cur_sq_sum = (images ** 2).sum(dim=(0, 2, 3))  # [C]

            if channel_sum is None:
                channel_sum = cur_sum
                channel_sq_sum = cur_sq_sum
            else:
                channel_sum += cur_sum
                channel_sq_sum += cur_sq_sum

            total_pixels += bsz * h * w

        if total_pixels == 0:
            # 极端保护，理论上不会发生
            mu = torch.zeros(3)
            std = torch.ones(3)
            return mu, std

        mu = channel_sum / total_pixels
        var = channel_sq_sum / total_pixels - mu ** 2
        var = torch.clamp(var, min=self.rdn_eps)
        std = torch.sqrt(var)

        return mu, std

    def _apply_random_rdn_train(self, images):
        """
        训练时：
        对 batch 中每张图像随机采样一个客户端统计量做归一化。
        """
        if self.client_stats is None or len(self.client_stats) == 0:
            return images

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

        mu = torch.stack(mus, dim=0).to(device)    # [B, C]
        std = torch.stack(stds, dim=0).to(device)  # [B, C]

        mu = mu.view(bsz, -1, 1, 1)
        std = std.view(bsz, -1, 1, 1)

        return (images - mu) / (std + self.rdn_eps)

    def normalize_test_images(self, client_id, images):
        """
        测试阶段：
        不再依赖 client_id，统一使用全局平均统计量。
        这样可以避免:
            IndexError: list index out of range
        也避免 evaluate 时 batch_idx 被误传成 client_id 的问题。
        """
        if self.global_mu is None or self.global_std is None:
            return images

        mu = self.global_mu.to(images.device).view(1, -1, 1, 1)
        std = self.global_std.to(images.device).view(1, -1, 1, 1)
        return (images - mu) / (std + self.rdn_eps)

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

                # FedRDN 核心：训练时随机注入 federation 内其他客户端统计量
                images = self._apply_random_rdn_train(images)

                outputs = net(images)
                loss = criterion(outputs, labels)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()