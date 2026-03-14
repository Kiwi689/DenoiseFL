import copy
from collections import deque, OrderedDict

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

        # Paper hyperparameters
        self.history_size = getattr(args, 'cda_history_size', 3)   # K
        self.batch_num = getattr(args, 'cda_batch_num', 3)         # B
        self.warmup_round = getattr(args, 'cda_warmup_round', 50)  # warmup rounds
        self.cda_L = getattr(args, 'cda_L', 1.0)                   # paper uses L; often set to 1

        # Cache recent K local models for each client: W_t^n
        self.client_model_history = {
            i: deque(maxlen=self.history_size) for i in range(self.args.parti_num)
        }

        # Cache recent K local losses F_n(w) for each client
        # Paper line 7 says accumulate local loss in the last local epoch
        self.client_loss_history = {
            i: deque(maxlen=self.history_size) for i in range(self.args.parti_num)
        }

        # Fixed selected model for every client, used when client is offline
        # This is the key "fixed local models" in the approximate objective.
        self.fixed_selected_models = {}
        self.fixed_selected_losses = {}

    def ini(self):
        self.global_net = copy.deepcopy(self.nets_list[0])
        global_w = self.global_net.state_dict()

        for net in self.nets_list:
            net.load_state_dict(global_w)

        # Initialize fixed models with the initial global model,
        # so all N clients always have a valid model in aggregation.
        init_state = copy.deepcopy(global_w)
        for client_id in range(self.args.parti_num):
            self.fixed_selected_models[client_id] = copy.deepcopy(init_state)
            self.fixed_selected_losses[client_id] = 0.0

    def loc_update(self, priloader_list):
        # Keep trainloaders for weighted averaging if the outer pipeline has not set it.
        if self.trainloaders is None:
            self.trainloaders = priloader_list

        total_clients = list(range(self.args.parti_num))
        online_clients = self.random_state.choice(
            total_clients, self.online_num, replace=False
        ).tolist()
        self.online_clients = online_clients

        # 1) Local training on selected clients
        current_round_local_states = {}
        current_round_local_losses = {}

        for client_id in online_clients:
            local_state, local_loss = self._train_net(
                client_id, self.nets_list[client_id], priloader_list[client_id]
            )

            # Save current local result
            current_round_local_states[client_id] = copy.deepcopy(local_state)
            current_round_local_losses[client_id] = float(local_loss)

            # 2) Update client cache W_t^n by replacing the oldest one
            self.client_model_history[client_id].append(copy.deepcopy(local_state))
            self.client_loss_history[client_id].append(float(local_loss))

        # 3) Warmup stage: behave like FedAvg, but still keep caches updated
        if self.epoch_index < self.warmup_round:
            self._fedavg_like_aggregate_current_online(current_round_local_states)
        else:
            # 4) FedCDA aggregation after warmup
            self.aggregate_nets()

        self.epoch_index += 1
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

        last_epoch_loss_sum = 0.0
        last_epoch_loss_count = 0

        for local_ep in range(self.local_epoch):
            epoch_loss_sum = 0.0
            epoch_loss_count = 0

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

                epoch_loss_sum += loss.item()
                epoch_loss_count += 1

            # Paper says accumulate local loss in the last local epoch
            if local_ep == self.local_epoch - 1:
                last_epoch_loss_sum = epoch_loss_sum
                last_epoch_loss_count = epoch_loss_count

        avg_last_epoch_loss = (
            last_epoch_loss_sum / max(last_epoch_loss_count, 1)
        )

        return copy.deepcopy(net.state_dict()), avg_last_epoch_loss

    def _get_client_data_size(self, client_id):
        """
        Get client sample count for weighted aggregation.
        Prefer dataset length; fall back to sampler.indices if needed.
        """
        dl = self.trainloaders[client_id]

        if hasattr(dl, 'dataset') and dl.dataset is not None:
            try:
                return len(dl.dataset)
            except Exception:
                pass

        if hasattr(dl, 'sampler') and hasattr(dl.sampler, 'indices'):
            return len(dl.sampler.indices)

        # Conservative fallback: number of batches
        return len(dl)

    def _is_buffer_key(self, key):
        return ('running_mean' in key) or ('running_var' in key) or ('num_batches_tracked' in key)

    def _squared_l2_of_state(self, state_dict):
        val = 0.0
        for k, v in state_dict.items():
            if self._is_buffer_key(k):
                continue
            vv = v.detach().float()
            val += torch.sum(vv * vv).item()
        return val

    def _average_state_dict(self, state_dict_list, weight_list):
        """
        Weighted average over all clients' selected/fixed models.
        This implements w = sum_n alpha_n * w_n.
        """
        new_global_w = OrderedDict()
        first = True

        for idx, local_w in enumerate(state_dict_list):
            weight = float(weight_list[idx])
            if first:
                first = False
                for k in local_w.keys():
                    if self._is_buffer_key(k):
                        new_global_w[k] = copy.deepcopy(local_w[k])
                    else:
                        new_global_w[k] = local_w[k].detach().clone() * weight
            else:
                for k in local_w.keys():
                    if self._is_buffer_key(k):
                        # Keep a valid buffer value; simple overwrite is common in engineering code.
                        new_global_w[k] = copy.deepcopy(local_w[k])
                    else:
                        new_global_w[k] += local_w[k].detach().clone() * weight

        return new_global_w

    def _objective_value_for_batch_choice(self, tentative_models):
        """
        Approximate objective based on paper Eq. (7)/(8):
            sum_n alpha_n * L_n(w_n) - (L/2) * ||w||^2,
        where w = sum_n alpha_n * w_n.

        Here:
        - for online clients in current batch: use candidate chosen models
        - for online clients not in current batch but already processed: use updated fixed_selected_models
        - for offline clients: use previous fixed_selected_models
        """
        client_ids = list(range(self.args.parti_num))

        if self.args.averaing == 'weight':
            sizes = np.array([self._get_client_data_size(cid) for cid in client_ids], dtype=np.float64)
            weights = sizes / np.sum(sizes)
        else:
            weights = np.array([1.0 / self.args.parti_num for _ in client_ids], dtype=np.float64)

        assembled_models = []
        assembled_losses = []

        for cid in client_ids:
            if cid in tentative_models:
                state_i, loss_i = tentative_models[cid]
            else:
                state_i = self.fixed_selected_models[cid]
                loss_i = self.fixed_selected_losses[cid]

            assembled_models.append(state_i)
            assembled_losses.append(loss_i)

        global_state = self._average_state_dict(assembled_models, weights)

        loss_term = 0.0
        for i in range(self.args.parti_num):
            loss_term += float(weights[i]) * float(assembled_losses[i])

        norm_term = self._squared_l2_of_state(global_state)

        # Paper sets L in the objective; many experiments simply assume L=1.
        obj = loss_term - 0.5 * float(self.cda_L) * norm_term
        return obj, global_state

    def _split_into_batches(self, client_ids, batch_num):
        """
        Split online clients into B batches approximately evenly.
        """
        if len(client_ids) == 0:
            return []

        batch_num = max(1, min(batch_num, len(client_ids)))
        shuffled = list(client_ids)
        self.random_state.shuffle(shuffled)

        batches = np.array_split(np.array(shuffled, dtype=np.int64), batch_num)
        return [batch.tolist() for batch in batches if len(batch) > 0]

    def _select_models_for_batch(self, batch_client_ids):
        """
        Greedy approximate selection for one batch.
        Each client chooses one model from its cached set W_t^n,
        while other clients are fixed by current fixed_selected_models.
        """
        tentative_models = {}

        for cid in batch_client_ids:
            history_states = list(self.client_model_history[cid])
            history_losses = list(self.client_loss_history[cid])

            # Safety fallback
            if len(history_states) == 0:
                history_states = [copy.deepcopy(self.fixed_selected_models[cid])]
                history_losses = [float(self.fixed_selected_losses[cid])]

            best_obj = None
            best_pair = None

            for cand_state, cand_loss in zip(history_states, history_losses):
                tentative_models[cid] = (cand_state, cand_loss)
                obj, _ = self._objective_value_for_batch_choice(tentative_models)

                if (best_obj is None) or (obj < best_obj):
                    best_obj = obj
                    best_pair = (copy.deepcopy(cand_state), float(cand_loss))

            tentative_models[cid] = best_pair

        return tentative_models

    def _fedavg_like_aggregate_current_online(self, current_round_local_states):
        """
        Warmup stage: use standard current-round FedAvg aggregation over online clients,
        then broadcast to all clients. Meanwhile caches are already updated outside.
        """
        if self.args.averaing == 'weight':
            online_sizes = np.array(
                [self._get_client_data_size(cid) for cid in self.online_clients],
                dtype=np.float64
            )
            freq = online_sizes / np.sum(online_sizes)
        else:
            freq = np.array(
                [1.0 / len(self.online_clients) for _ in self.online_clients],
                dtype=np.float64
            )

        local_models = [current_round_local_states[cid] for cid in self.online_clients]
        new_global_w = self._average_state_dict(local_models, freq)

        self.global_net.load_state_dict(new_global_w)

        for cid in range(self.args.parti_num):
            self.nets_list[cid].load_state_dict(self.global_net.state_dict())
            # During warmup, fixed model follows newest available global model
            self.fixed_selected_models[cid] = copy.deepcopy(self.global_net.state_dict())
            self.fixed_selected_losses[cid] = 0.0

    def aggregate_nets(self, freq=None):
        """
        Paper-faithful approximate FedCDA aggregation:
        1) Only select models for current online clients P_t
        2) Offline clients keep previous fixed selected models
        3) Process online clients batch by batch
        4) Aggregate BOTH selected online models and fixed offline models
        """
        if len(self.online_clients) == 0:
            return

        # Batch-based approximate selection
        batches = self._split_into_batches(self.online_clients, self.batch_num)

        for batch_client_ids in batches:
            selected_for_batch = self._select_models_for_batch(batch_client_ids)

            # Update fixed selected models immediately after each batch,
            # matching the batch-wise approximation idea.
            for cid, (state_i, loss_i) in selected_for_batch.items():
                self.fixed_selected_models[cid] = copy.deepcopy(state_i)
                self.fixed_selected_losses[cid] = float(loss_i)

        # Aggregate all clients' selected/fixed models
        client_ids = list(range(self.args.parti_num))
        all_states = [self.fixed_selected_models[cid] for cid in client_ids]

        if self.args.averaing == 'weight':
            sizes = np.array([self._get_client_data_size(cid) for cid in client_ids], dtype=np.float64)
            weights = sizes / np.sum(sizes)
        else:
            weights = np.array([1.0 / self.args.parti_num for _ in client_ids], dtype=np.float64)

        new_global_w = self._average_state_dict(all_states, weights)

        self.global_net.load_state_dict(new_global_w)

        # Broadcast new global model to all clients
        for net in self.nets_list:
            net.load_state_dict(self.global_net.state_dict())