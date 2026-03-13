import copy
from collections import OrderedDict, deque

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from models.utils.federated_model import FederatedModel


class FedCDA(FederatedModel):
    NAME = 'fedcda'
    COMPATIBILITY = ['homogeneity']

    def __init__(self, nets_list, args, transform):
        super(FedCDA, self).__init__(nets_list, args, transform)

        # 历史窗口长度：每个客户端保留最近 K 轮本地模型
        self.history_size = getattr(args, 'cda_history_size', 5)

        # 每个客户端一个 deque，保存最近若干轮 local model state_dict
        self.client_model_history = {
            i: deque(maxlen=self.history_size) for i in range(self.args.parti_num)
        }

    def ini(self):
        self.global_net = copy.deepcopy(self.nets_list[0])
        global_w = self.global_net.state_dict()

        for net in self.nets_list:
            net.load_state_dict(global_w)

    def loc_update(self, priloader_list):
        total_clients = list(range(self.args.parti_num))
        online_clients = self.random_state.choice(
            total_clients, self.online_num, replace=False
        ).tolist()
        self.online_clients = online_clients

        # 1) 本地训练
        for client_id in online_clients:
            self._train_net(client_id, self.nets_list[client_id], priloader_list[client_id])

            # 保存当前客户端本轮训练后的 local model 到历史池
            self.client_model_history[client_id].append(
                copy.deepcopy(self.nets_list[client_id].state_dict())
            )

        # 2) 用 cross-round divergence-aware aggregation 聚合
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

    def _model_divergence(self, model_a, model_b):
        """
        计算两个 state_dict 的 L2 divergence
        只比较非 BN running stats / num_batches_tracked 参数
        """
        div = 0.0
        for k in model_a.keys():
            if "running" in k or "num_batches_tracked" in k:
                continue
            diff = model_a[k].float() - model_b[k].float()
            div += torch.sum(diff * diff).item()
        return div

    def aggregate_nets(self, freq=None):
        global_w = copy.deepcopy(self.global_net.state_dict())

        # 1) 对每个在线客户端，从其历史模型中选择一个与当前 global model divergence 最小的模型
        selected_models = []
        selected_client_ids = []

        for client_id in self.online_clients:
            history = self.client_model_history[client_id]

            if len(history) == 0:
                # 理论上不会发生；兜底
                chosen_model = copy.deepcopy(self.nets_list[client_id].state_dict())
            else:
                best_div = None
                best_model = None
                for hist_model in history:
                    div = self._model_divergence(hist_model, global_w)
                    if best_div is None or div < best_div:
                        best_div = div
                        best_model = hist_model
                chosen_model = copy.deepcopy(best_model)

            selected_models.append(chosen_model)
            selected_client_ids.append(client_id)

        # 2) 计算聚合权重
        if self.args.averaing == 'weight':
            online_clients_dl = [self.trainloaders[cid] for cid in selected_client_ids]
            online_clients_len = [len(dl.sampler.indices) for dl in online_clients_dl]
            total = np.sum(online_clients_len)
            freq = [x / total for x in online_clients_len]
        else:
            freq = [1.0 / len(selected_models) for _ in selected_models]

        # 3) 聚合选中的 cross-round models
        first = True
        new_global_w = OrderedDict()

        for idx, local_w in enumerate(selected_models):
            if first:
                first = False
                for k in local_w:
                    if "running" in k or "num_batches_tracked" in k:
                        new_global_w[k] = copy.deepcopy(local_w[k])
                    else:
                        new_global_w[k] = local_w[k] * freq[idx]
            else:
                for k in local_w:
                    if "running" in k or "num_batches_tracked" in k:
                        # BN 统计量先简单采用最近一个被选模型的值
                        new_global_w[k] = copy.deepcopy(local_w[k])
                    else:
                        new_global_w[k] += local_w[k] * freq[idx]

        # 4) 更新 global model，并同步回所有 clients
        self.global_net.load_state_dict(new_global_w)

        for net in self.nets_list:
            net.load_state_dict(self.global_net.state_dict())