import copy
from collections import OrderedDict

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from models.utils.federated_model import FederatedModel


class FedGLoSS(FederatedModel):
    NAME = 'fedgloss'
    COMPATIBILITY = ['homogeneity']

    def __init__(self, nets_list, args, transform):
        super(FedGLoSS, self).__init__(nets_list, args, transform)

        # FedGLoSS hyperparameters
        self.rho = getattr(args, 'rho', 0.05)
        self.beta = getattr(args, 'beta', 1.0)
        self.server_lr = getattr(args, 'server_lr', 1.0)

        self.client_sigma = None
        self.prev_perturbed_delta = None
        self.perturbed_global_w = None

    def ini(self):
        self.global_net = copy.deepcopy(self.nets_list[0]).to(self.device)
        global_w = copy.deepcopy(self.global_net.state_dict())

        for net in self.nets_list:
            net.load_state_dict(global_w)

        self.client_sigma = []
        for _ in range(self.args.parti_num):
            sigma_k = OrderedDict()
            for k, v in global_w.items():
                sigma_k[k] = torch.zeros_like(v, device=v.device)
            self.client_sigma.append(sigma_k)

        self.prev_perturbed_delta = OrderedDict()
        for k, v in global_w.items():
            self.prev_perturbed_delta[k] = torch.zeros_like(v, device=v.device)

        self.perturbed_global_w = copy.deepcopy(global_w)

    def loc_update(self, priloader_list):
        if self.trainloaders is None:
            self.trainloaders = priloader_list

        total_clients = list(range(self.args.parti_num))
        online_clients = self.random_state.choice(
            total_clients,
            self.online_num,
            replace=False
        ).tolist()
        self.online_clients = online_clients

        # 1) build perturbed global model
        self.perturbed_global_w = self._build_perturbed_global_model()

        # 2) local training
        for client_id in online_clients:
            self._train_net(client_id, self.nets_list[client_id], priloader_list[client_id])

        # 3) aggregate
        self.aggregate_nets()
        return None

    def _is_buffer_key(self, key):
        return (
            "running_mean" in key
            or "running_var" in key
            or "num_batches_tracked" in key
        )

    def _get_client_weight(self, client_id):
        dl = self.trainloaders[client_id]

        if hasattr(dl, 'sampler') and hasattr(dl.sampler, 'indices'):
            return len(dl.sampler.indices)

        if hasattr(dl, 'dataset') and dl.dataset is not None:
            try:
                return len(dl.dataset)
            except Exception:
                pass

        return len(dl)

    def _normalize_state_dict(self, state_dict):
        norm_sq = 0.0
        for k, v in state_dict.items():
            if self._is_buffer_key(k):
                continue
            vv = v.detach().float()
            norm_sq += torch.sum(vv * vv).item()
        return float(np.sqrt(norm_sq)) if norm_sq > 0 else 0.0

    def _count_trainable_numel(self, net):
        return sum(p.numel() for p in net.parameters() if p.requires_grad)

    def _build_perturbed_global_model(self):
        """
        epsilon_t = rho * Delta_{t-1} / ||Delta_{t-1}||
        w_t_tilde = w_t + epsilon_t
        """
        global_w = copy.deepcopy(self.global_net.state_dict())
        perturbed_w = OrderedDict()

        delta_norm = self._normalize_state_dict(self.prev_perturbed_delta)

        for k in global_w.keys():
            if self._is_buffer_key(k):
                perturbed_w[k] = copy.deepcopy(global_w[k])
            else:
                if delta_norm == 0.0:
                    eps_k = torch.zeros_like(global_w[k], device=global_w[k].device)
                else:
                    eps_k = self.rho * self.prev_perturbed_delta[k] / delta_norm
                perturbed_w[k] = global_w[k] + eps_k

        return perturbed_w

    def _train_net(self, index, net, train_loader):
        """
        Start from perturbed global model, but regularize against the
        unperturbed global model.
        """
        net = net.to(self.device)
        net.load_state_dict(copy.deepcopy(self.perturbed_global_w))
        net.train()

        optimizer = optim.SGD(
            net.parameters(),
            lr=self.local_lr,
            momentum=0.9,
            weight_decay=self.args.reg
        )
        criterion = nn.CrossEntropyLoss().to(self.device)

        # IMPORTANT:
        # local optimization anchor should be the unperturbed global model
        global_anchor_w = OrderedDict()
        for k, v in self.global_net.state_dict().items():
            global_anchor_w[k] = v.detach().clone().to(v.device)

        sigma_k = self.client_sigma[index]
        param_numel = float(self._count_trainable_numel(net))

        for _ in range(self.local_epoch):
            for batch in train_loader:
                if len(batch) == 3:
                    images, labels, _ = batch
                else:
                    images, labels = batch

                images = images.to(self.device)
                labels = labels.to(self.device).long()

                outputs = net(images)
                ce_loss = criterion(outputs, labels)

                linear_term = torch.tensor(0.0, device=self.device)
                quad_term = torch.tensor(0.0, device=self.device)

                for name, param in net.named_parameters():
                    anchor = global_anchor_w[name].to(param.device)
                    sigma = sigma_k[name].to(param.device)

                    diff = param - anchor

                    # sigma^T (w - w_t)
                    linear_term = linear_term + torch.sum(sigma * diff)

                    # ||w - w_t||^2
                    quad_term = quad_term + torch.sum(diff * diff)

                # IMPORTANT:
                # normalize ADMM terms, otherwise they dominate CE massively
                linear_term = linear_term / param_numel
                quad_term = quad_term / param_numel

                loss = ce_loss + linear_term + (1.0 / (2.0 * self.beta)) * quad_term

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        # dual update
        local_w = net.state_dict()
        for k in sigma_k.keys():
            if self._is_buffer_key(k):
                continue

            sigma_k[k] = sigma_k[k].to(local_w[k].device)
            sigma_k[k] = sigma_k[k] - (1.0 / self.beta) * (
                local_w[k].detach() - global_anchor_w[k].detach().to(local_w[k].device)
            )

    def aggregate_nets(self):
        """
        Delta_t = sum_k p_k * (w_t_tilde - w_k)
        w_{t+1} = w_t - eta_s * Delta_t
        """
        online_clients = self.online_clients
        global_w = copy.deepcopy(self.global_net.state_dict())
        perturbed_w = self.perturbed_global_w

        averaging = getattr(self.args, 'averaging', getattr(self.args, 'averaing', 'weight'))

        if averaging == 'weight':
            sizes = np.array(
                [self._get_client_weight(cid) for cid in online_clients],
                dtype=np.float64
            )
            freq = sizes / np.sum(sizes)
        else:
            freq = np.array(
                [1.0 / len(online_clients) for _ in online_clients],
                dtype=np.float64
            )

        perturbed_delta = OrderedDict()
        for k in global_w.keys():
            perturbed_delta[k] = torch.zeros_like(perturbed_w[k], device=perturbed_w[k].device)

        # parameter delta
        for idx, client_id in enumerate(online_clients):
            local_w = self.nets_list[client_id].state_dict()
            for k in global_w.keys():
                if self._is_buffer_key(k):
                    continue
                perturbed_delta[k] += freq[idx] * (perturbed_w[k] - local_w[k])

        new_global_w = OrderedDict()
        for k in global_w.keys():
            if self._is_buffer_key(k):
                # weighted average buffers instead of copying first client
                buf = torch.zeros_like(global_w[k], device=global_w[k].device, dtype=global_w[k].dtype)
                for idx, client_id in enumerate(online_clients):
                    local_buf = self.nets_list[client_id].state_dict()[k].to(buf.device)
                    if buf.dtype.is_floating_point:
                        buf = buf + float(freq[idx]) * local_buf
                    else:
                        # integer buffers like num_batches_tracked
                        buf = local_buf
                new_global_w[k] = buf
            else:
                new_global_w[k] = global_w[k] - self.server_lr * perturbed_delta[k]

        self.prev_perturbed_delta = OrderedDict()
        for k in perturbed_delta.keys():
            self.prev_perturbed_delta[k] = perturbed_delta[k].detach().clone()

        self.global_net.load_state_dict(new_global_w)

        for net in self.nets_list:
            net.load_state_dict(copy.deepcopy(new_global_w))