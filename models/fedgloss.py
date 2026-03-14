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

        # Paper hyperparameters
        self.rho = getattr(args, 'rho', 0.05)          # SAM neighborhood size
        self.beta = getattr(args, 'beta', 1.0)         # ADMM penalty
        self.server_lr = getattr(args, 'server_lr', 1.0)

        # Per-client dual variables sigma_k
        self.client_sigma = None

        # Previous perturbed pseudo-gradient \tilde{\Delta}^{t-1}_w
        self.prev_perturbed_delta = None

        # Current perturbed global model \tilde{w}^t
        self.perturbed_global_w = None

    def ini(self):
        self.global_net = copy.deepcopy(self.nets_list[0])
        global_w = copy.deepcopy(self.global_net.state_dict())

        for net in self.nets_list:
            net.load_state_dict(global_w)

        # Initialize per-client dual variables sigma_k = 0
        self.client_sigma = []
        for _ in range(self.args.parti_num):
            sigma_k = OrderedDict()
            for k, v in global_w.items():
                sigma_k[k] = torch.zeros_like(v, device=self.device)
            self.client_sigma.append(sigma_k)

        # No previous perturbed pseudo-gradient at round 0
        self.prev_perturbed_delta = OrderedDict()
        for k, v in global_w.items():
            self.prev_perturbed_delta[k] = torch.zeros_like(v, device=self.device)

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

        # 1) Build perturbed global model \tilde{w}^t using previous perturbed pseudo-gradient
        self.perturbed_global_w = self._build_perturbed_global_model()

        # 2) Send perturbed model to selected clients and perform local training
        for client_id in online_clients:
            self._train_net(client_id, self.nets_list[client_id], priloader_list[client_id])

        # 3) Aggregate local models into current perturbed pseudo-gradient \tilde{\Delta}^t_w
        self.aggregate_nets()
        return None

    def _is_buffer_key(self, key):
        return ("running_mean" in key) or ("running_var" in key) or ("num_batches_tracked" in key)

    def _get_client_weight(self, client_id):
        if self.args.averaing == 'weight':
            dl = self.trainloaders[client_id]
            if hasattr(dl, 'dataset') and dl.dataset is not None:
                try:
                    return len(dl.dataset)
                except Exception:
                    pass
            if hasattr(dl, 'sampler') and hasattr(dl.sampler, 'indices'):
                return len(dl.sampler.indices)
            return len(dl)
        return 1.0

    def _normalize_state_dict(self, state_dict):
        norm_sq = 0.0
        for k, v in state_dict.items():
            if self._is_buffer_key(k):
                continue
            vv = v.detach().float()
            norm_sq += torch.sum(vv * vv).item()
        norm = float(np.sqrt(norm_sq)) if norm_sq > 0 else 0.0
        return norm

    def _build_perturbed_global_model(self):
        """
        Paper Sec. 5.2:
            \tilde{\epsilon}^t = rho * \tilde{\Delta}^{t-1} / ||\tilde{\Delta}^{t-1}||
            \tilde{w}^t = w^t + \tilde{\epsilon}^t
        At round 0, no previous perturbed pseudo-gradient exists, so use zero perturbation.
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
        Paper Eq. (7) local subproblem:
            wk,E = argmin_wk { fk
                               - sigma_k^T(w_t - wk)
                               + 1/(2 beta) ||w_t - wk||^2 }
        Here the server sends perturbed model \tilde{w}^t, so we use it as the local anchor.
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

        anchor_w = OrderedDict()
        for k, v in self.perturbed_global_w.items():
            anchor_w[k] = v.detach().clone().to(self.device)

        sigma_k = self.client_sigma[index]

        for _ in range(self.local_epoch):
            for batch in train_loader:
                if len(batch) == 3:
                    images, labels, _ = batch
                else:
                    images, labels = batch

                images = images.to(self.device)
                labels = labels.to(self.device)

                outputs = net(images)
                ce_loss = criterion(outputs, labels)

                linear_term = 0.0
                quad_term = 0.0
                current_state = net.state_dict()

                for name, param in net.named_parameters():
                    diff = param - anchor_w[name]
                    linear_term += torch.sum(sigma_k[name] * param)
                    quad_term += torch.sum(diff * diff)

                # Equivalent up to constants independent of current parameters:
                # fk - sigma_k^T(w_t - w) + 1/(2beta)||w_t - w||^2
                loss = ce_loss + linear_term + (1.0 / (2.0 * self.beta)) * quad_term

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        # Local dual update:
        # sigma_k <- sigma_k - (1/beta) * (w_{k,E} - w_{k,0})
        local_w = net.state_dict()
        for k in sigma_k.keys():
            if self._is_buffer_key(k):
                continue
            sigma_k[k] = sigma_k[k] - (1.0 / self.beta) * (local_w[k].detach() - anchor_w[k].detach())

    def aggregate_nets(self):
        """
        Paper Sec. 5.2:
            \tilde{\Delta}^t_w = sum_k Nk/N * (\tilde{w}^t - \tilde{w}^t_k)
            w^{t+1} = w^t - eta_s * \tilde{\Delta}^t_w
        Also update the global dual variable by averaging (wk - wt).
        """
        online_clients = self.online_clients
        global_w = copy.deepcopy(self.global_net.state_dict())
        perturbed_w = self.perturbed_global_w

        # Client weights Nk / N
        if self.args.averaing == 'weight':
            sizes = np.array([self._get_client_weight(cid) for cid in online_clients], dtype=np.float64)
            freq = sizes / np.sum(sizes)
        else:
            freq = np.array([1.0 / len(online_clients) for _ in online_clients], dtype=np.float64)

        # Current perturbed pseudo-gradient \tilde{\Delta}^t_w
        perturbed_delta = OrderedDict()
        avg_client_minus_anchor = OrderedDict()

        for k, v in global_w.items():
            if self._is_buffer_key(k):
                perturbed_delta[k] = torch.zeros_like(perturbed_w[k], device=perturbed_w[k].device)
                avg_client_minus_anchor[k] = torch.zeros_like(perturbed_w[k], device=perturbed_w[k].device)
            else:
                perturbed_delta[k] = torch.zeros_like(perturbed_w[k], device=perturbed_w[k].device)
                avg_client_minus_anchor[k] = torch.zeros_like(perturbed_w[k], device=perturbed_w[k].device)

        for idx, client_id in enumerate(online_clients):
            local_w = self.nets_list[client_id].state_dict()

            for k in global_w.keys():
                if self._is_buffer_key(k):
                    continue

                perturbed_delta[k] += freq[idx] * (perturbed_w[k] - local_w[k])
                avg_client_minus_anchor[k] += freq[idx] * (local_w[k] - perturbed_w[k])

        # Server update on ORIGINAL global model:
        # w^{t+1} = w^t - eta_s * \tilde{\Delta}^t_w
        new_global_w = OrderedDict()
        for k in global_w.keys():
            if self._is_buffer_key(k):
                # For buffers, copy from first online client for validity
                ref_client = online_clients[0]
                new_global_w[k] = copy.deepcopy(self.nets_list[ref_client].state_dict()[k])
            else:
                new_global_w[k] = global_w[k] - self.server_lr * perturbed_delta[k]

        # Save current perturbed pseudo-gradient for next round perturbation
        self.prev_perturbed_delta = OrderedDict()
        for k in perturbed_delta.keys():
            self.prev_perturbed_delta[k] = perturbed_delta[k].detach().clone()

        # Optional global dual averaging, matching paper description:
        # "The global one sigma is updated by adding the averaged wk - wt"
        # Here we distribute the averaged term to online client duals for stability.
        for client_id in online_clients:
            sigma_k = self.client_sigma[client_id]
            for k in sigma_k.keys():
                if self._is_buffer_key(k):
                    continue
                sigma_k[k] = sigma_k[k] + avg_client_minus_anchor[k].detach()

        self.global_net.load_state_dict(new_global_w)

        # Broadcast the NEW unperturbed global model to all clients.
        for net in self.nets_list:
            net.load_state_dict(copy.deepcopy(new_global_w))