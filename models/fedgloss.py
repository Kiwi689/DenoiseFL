import copy
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.optim as optim

from models.utils.federated_model import FederatedModel


class FedGLoSS(FederatedModel):
    NAME = 'fedgloss'
    COMPATIBILITY = ['homogeneity']

    def __init__(self, nets_list, args, transform):
        super(FedGLoSS, self).__init__(nets_list, args, transform)

        self.beta = getattr(args, 'beta', 0.0)
        self.sigma = None

    def ini(self):
        self.global_net = copy.deepcopy(self.nets_list[0])
        global_w = self.global_net.state_dict()

        for net in self.nets_list:
            net.load_state_dict(global_w)

        self.sigma = OrderedDict()
        for k, v in global_w.items():
            self.sigma[k] = torch.zeros_like(v)

    def loc_update(self, priloader_list):
        total_clients = list(range(self.args.parti_num))
        online_clients = self.random_state.choice(
            total_clients,
            self.online_num,
            replace=False
        ).tolist()
        self.online_clients = online_clients

        for client_id in online_clients:
            self._train_net(client_id, self.nets_list[client_id], priloader_list[client_id])

        self.aggregate_nets()
        return None

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

                images = images.to(self.device)
                labels = labels.to(self.device)

                outputs = net(images)
                loss = criterion(outputs, labels)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

    def aggregate_nets(self):
        global_w = copy.deepcopy(self.global_net.state_dict())
        online_clients = self.online_clients

        # 1) 聚合权重
        if self.args.averaing == 'weight':
            online_clients_dl = [self.trainloaders[idx] for idx in online_clients]
            online_clients_len = [len(dl.sampler.indices) for dl in online_clients_dl]
            total_num = sum(online_clients_len)
            freq = [x / total_num for x in online_clients_len]
        else:
            freq = [1.0 / len(online_clients) for _ in online_clients]

        # 2) 计算 avg_delta / avg_bn
        avg_delta = OrderedDict()

        for k, v in global_w.items():
            if "num_batches_tracked" in k:
                # int buffer，不做加权平均，后面直接拷贝
                avg_delta[k] = None
            elif "running" in k:
                # BN running stats 可以做 float 平均
                avg_delta[k] = torch.zeros_like(v, dtype=torch.float32)
            else:
                avg_delta[k] = torch.zeros_like(v)

        for idx, client_id in enumerate(online_clients):
            local_w = self.nets_list[client_id].state_dict()

            for k in global_w:
                if "num_batches_tracked" in k:
                    continue
                elif "running" in k:
                    avg_delta[k] += freq[idx] * local_w[k].float()
                else:
                    avg_delta[k] += freq[idx] * (local_w[k] - global_w[k])

        # 3) 更新 global model
        new_global_w = OrderedDict()

        for k in global_w:
            if "num_batches_tracked" in k:
                # 直接拷贝一个在线客户端的值
                ref_client = online_clients[0]
                new_global_w[k] = copy.deepcopy(self.nets_list[ref_client].state_dict()[k])

            elif "running" in k:
                # BN running_mean / running_var
                new_global_w[k] = avg_delta[k].to(global_w[k].dtype)

            else:
                # 关键修正：FedAvg-style update
                new_global_w[k] = global_w[k] + avg_delta[k]

        # 4) sigma correction（只对真正参数，不碰 BN buffers）
        if self.beta > 0:
            for k in new_global_w:
                if "running" in k or "num_batches_tracked" in k:
                    continue
                new_global_w[k] = new_global_w[k] - self.beta * self.sigma[k]

            for k in self.sigma:
                if "running" in k or "num_batches_tracked" in k:
                    continue

                deviation = torch.zeros_like(self.sigma[k])
                for client_id in online_clients:
                    local_w = self.nets_list[client_id].state_dict()
                    deviation += (local_w[k] - global_w[k])

                self.sigma[k] = self.sigma[k] - deviation / (
                    self.beta * self.args.parti_num * len(online_clients)
                )

        # 5) load back
        self.global_net.load_state_dict(new_global_w)

        for net in self.nets_list:
            net.load_state_dict(new_global_w)