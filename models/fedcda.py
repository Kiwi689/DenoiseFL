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

        # FedCDA hyperparameters
        self.history_size = getattr(args, 'cda_history_size', 3)    # K
        self.batch_num = getattr(args, 'cda_batch_num', 3)          # B
        self.warmup_round = getattr(args, 'cda_warmup_round', 50)
        self.cda_L = getattr(args, 'cda_L', 1.0)

        # recent K local models / losses for each client
        self.client_model_history = {
            i: deque(maxlen=self.history_size) for i in range(self.args.parti_num)
        }
        self.client_loss_history = {
            i: deque(maxlen=self.history_size) for i in range(self.args.parti_num)
        }

        # fixed selected models / losses for all clients
        self.fixed_selected_models = {}
        self.fixed_selected_losses = {}

    def ini(self):
        self.global_net = copy.deepcopy(self.nets_list[0])
        global_w = copy.deepcopy(self.global_net.state_dict())

        for net in self.nets_list:
            net.load_state_dict(global_w)

        # initialize all fixed models with initial global model
        for client_id in range(self.args.parti_num):
            self.fixed_selected_models[client_id] = copy.deepcopy(global_w)
            self.fixed_selected_losses[client_id] = 0.0

    def loc_update(self, priloader_list):
        if self.trainloaders is None:
            self.trainloaders = priloader_list

        total_clients = list(range(self.args.parti_num))
        online_clients = self.random_state.choice(
            total_clients, self.online_num, replace=False
        ).tolist()
        self.online_clients = online_clients

        current_round_local_states = {}
        current_round_local_losses = {}

        # local training on online clients
        for client_id in online_clients:
            local_state, local_loss = self._train_net(
                client_id, self.nets_list[client_id], priloader_list[client_id]
            )

            current_round_local_states[client_id] = copy.deepcopy(local_state)
            current_round_local_losses[client_id] = float(local_loss)

            self.client_model_history[client_id].append(copy.deepcopy(local_state))
            self.client_loss_history[client_id].append(float(local_loss))

        # warmup: standard FedAvg on current online clients
        if self.epoch_index < self.warmup_round:
            self._fedavg_like_aggregate_current_online(current_round_local_states)
        else:
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

        last_epoch_loss_sum = 0.0
        last_epoch_sample_count = 0

        for local_ep in range(self.local_epoch):
            epoch_loss_sum = 0.0
            epoch_sample_count = 0

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

                bs = labels.size(0)
                epoch_loss_sum += loss.item() * bs
                epoch_sample_count += bs

            # only keep the last local epoch loss, as in the paper description
            if local_ep == self.local_epoch - 1:
                last_epoch_loss_sum = epoch_loss_sum
                last_epoch_sample_count = epoch_sample_count

        avg_last_epoch_loss = last_epoch_loss_sum / max(last_epoch_sample_count, 1)
        return copy.deepcopy(net.state_dict()), float(avg_last_epoch_loss)

    def _get_client_data_size(self, client_id):
        dl = self.trainloaders[client_id]

        # be careful: dl.dataset may be the full dataset if sampler/subset is used
        if hasattr(dl, 'sampler') and hasattr(dl.sampler, 'indices'):
            return len(dl.sampler.indices)

        if hasattr(dl, 'dataset') and dl.dataset is not None:
            try:
                return len(dl.dataset)
            except Exception:
                pass

        return len(dl)

    def _is_buffer_key(self, key):
        return (
            ('running_mean' in key) or
            ('running_var' in key) or
            ('num_batches_tracked' in key)
        )

    def _squared_l2_of_state(self, state_dict):
        val = 0.0
        for k, v in state_dict.items():
            if self._is_buffer_key(k):
                continue
            vv = v.detach().float()
            val += torch.sum(vv * vv).item()
        return val

    def _get_equal_weights(self, num_clients):
        return np.array([1.0 / num_clients for _ in range(num_clients)], dtype=np.float64)

    def _get_weighted_weights(self, client_ids):
        sizes = np.array([self._get_client_data_size(cid) for cid in client_ids], dtype=np.float64)
        total = np.sum(sizes)
        if total <= 0:
            return self._get_equal_weights(len(client_ids))
        return sizes / total

    def _get_aggregation_weights(self, client_ids, use_equal_default=True):
        # paper-faithful default: equal averaging across clients
        if use_equal_default:
            return self._get_equal_weights(len(client_ids))

        if getattr(self.args, 'averaing', 'weight') == 'weight':
            return self._get_weighted_weights(client_ids)
        return self._get_equal_weights(len(client_ids))

    def _average_state_dict(self, state_dict_list, weight_list):
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
                        # keep a valid buffer value
                        new_global_w[k] = copy.deepcopy(local_w[k])
                    else:
                        new_global_w[k] += local_w[k].detach().clone() * weight

        return new_global_w

    def _split_into_batches(self, client_ids, batch_num):
        if len(client_ids) == 0:
            return []

        batch_num = max(1, min(batch_num, len(client_ids)))
        shuffled = list(client_ids)
        self.random_state.shuffle(shuffled)

        batches = np.array_split(np.array(shuffled, dtype=np.int64), batch_num)
        return [batch.tolist() for batch in batches if len(batch) > 0]

    def _objective_value_for_selection(self, assembled_client_ids, assembled_model_loss_pairs):
        """
        Approximate FedCDA objective for the currently considered client set:
            sum_i alpha_i * F_i(w_i)
          + (L/2) * sum_i alpha_i * ||w_i||^2
          - (L/2) * || sum_i alpha_i w_i ||^2

        This is closer to the paper's Eq.(6)/(8).
        """
        if len(assembled_client_ids) == 0:
            return 0.0, None

        # paper-faithful default: equal averaging
        weights = self._get_aggregation_weights(assembled_client_ids, use_equal_default=True)

        state_list = [assembled_model_loss_pairs[cid][0] for cid in assembled_client_ids]
        loss_list = [assembled_model_loss_pairs[cid][1] for cid in assembled_client_ids]

        global_state = self._average_state_dict(state_list, weights)

        loss_term = 0.0
        local_norm_term = 0.0
        for i, cid in enumerate(assembled_client_ids):
            alpha_i = float(weights[i])
            state_i = assembled_model_loss_pairs[cid][0]
            loss_i = float(loss_list[i])

            loss_term += alpha_i * loss_i
            local_norm_term += alpha_i * self._squared_l2_of_state(state_i)

        global_norm_term = self._squared_l2_of_state(global_state)

        obj = (
            loss_term
            + 0.5 * float(self.cda_L) * local_norm_term
            - 0.5 * float(self.cda_L) * global_norm_term
        )

        return obj, global_state

    def _build_context_for_current_batch(self, processed_online_clients, current_batch_choices):
        """
        Build the client set used in the current batch-wise approximation:
        - offline clients: fixed selected models
        - already processed online clients: updated fixed selected models
        - current batch clients: tentative choices
        - future online clients: excluded (paper batch-wise approximation spirit)
        """
        all_clients = list(range(self.args.parti_num))
        online_set = set(self.online_clients)
        processed_set = set(processed_online_clients)
        current_batch_set = set(current_batch_choices.keys())

        assembled_client_ids = []
        assembled_pairs = {}

        for cid in all_clients:
            if cid in current_batch_set:
                assembled_client_ids.append(cid)
                assembled_pairs[cid] = current_batch_choices[cid]
            elif cid in processed_set:
                assembled_client_ids.append(cid)
                assembled_pairs[cid] = (
                    copy.deepcopy(self.fixed_selected_models[cid]),
                    float(self.fixed_selected_losses[cid])
                )
            elif cid not in online_set:
                assembled_client_ids.append(cid)
                assembled_pairs[cid] = (
                    copy.deepcopy(self.fixed_selected_models[cid]),
                    float(self.fixed_selected_losses[cid])
                )
            else:
                # future online clients are excluded in the current batch approximation
                continue

        return assembled_client_ids, assembled_pairs

    def _select_models_for_batch(self, batch_client_ids, processed_online_clients):
        """
        Greedy selection for one batch.
        For each client in current batch, select one cached model that minimizes
        the approximate objective conditioned on:
        - offline clients fixed
        - already processed online clients fixed
        - future online clients excluded
        """
        selected_for_batch = {}

        for cid in batch_client_ids:
            history_states = list(self.client_model_history[cid])
            history_losses = list(self.client_loss_history[cid])

            if len(history_states) == 0:
                history_states = [copy.deepcopy(self.fixed_selected_models[cid])]
                history_losses = [float(self.fixed_selected_losses[cid])]

            best_obj = None
            best_pair = None

            for cand_state, cand_loss in zip(history_states, history_losses):
                current_batch_choices = copy.deepcopy(selected_for_batch)
                current_batch_choices[cid] = (cand_state, cand_loss)

                assembled_client_ids, assembled_pairs = self._build_context_for_current_batch(
                    processed_online_clients=processed_online_clients,
                    current_batch_choices=current_batch_choices
                )

                obj, _ = self._objective_value_for_selection(
                    assembled_client_ids, assembled_pairs
                )

                if (best_obj is None) or (obj < best_obj):
                    best_obj = obj
                    best_pair = (copy.deepcopy(cand_state), float(cand_loss))

            selected_for_batch[cid] = best_pair

        return selected_for_batch

    def _fedavg_like_aggregate_current_online(self, current_round_local_states):
        if len(self.online_clients) == 0:
            return

        # paper-faithful default: equal averaging among online clients during warmup
        freq = self._get_aggregation_weights(self.online_clients, use_equal_default=True)

        local_models = [current_round_local_states[cid] for cid in self.online_clients]
        new_global_w = self._average_state_dict(local_models, freq)

        self.global_net.load_state_dict(new_global_w)

        for cid in range(self.args.parti_num):
            self.nets_list[cid].load_state_dict(self.global_net.state_dict())
            # during warmup, fixed models follow global model
            self.fixed_selected_models[cid] = copy.deepcopy(self.global_net.state_dict())
            self.fixed_selected_losses[cid] = 0.0

    def aggregate_nets(self, freq=None):
        """
        FedCDA aggregation:
        1) Split current online clients into B batches
        2) Select one cached model for each online client batch by batch
        3) Offline clients keep previous fixed selected models
        4) Aggregate all fixed/selected client models into a new global model
        """
        if len(self.online_clients) == 0:
            return

        batches = self._split_into_batches(self.online_clients, self.batch_num)
        processed_online_clients = []

        for batch_client_ids in batches:
            selected_for_batch = self._select_models_for_batch(
                batch_client_ids=batch_client_ids,
                processed_online_clients=processed_online_clients
            )

            # update fixed selections immediately after each batch
            for cid, (state_i, loss_i) in selected_for_batch.items():
                self.fixed_selected_models[cid] = copy.deepcopy(state_i)
                self.fixed_selected_losses[cid] = float(loss_i)

            processed_online_clients.extend(batch_client_ids)

        # final aggregation over all clients' fixed selected models
        client_ids = list(range(self.args.parti_num))
        all_states = [self.fixed_selected_models[cid] for cid in client_ids]

        # paper-faithful default: equal averaging across all clients
        weights = self._get_aggregation_weights(client_ids, use_equal_default=True)

        new_global_w = self._average_state_dict(all_states, weights)
        self.global_net.load_state_dict(new_global_w)

        # broadcast global model to all clients
        for net in self.nets_list:
            net.load_state_dict(self.global_net.state_dict())