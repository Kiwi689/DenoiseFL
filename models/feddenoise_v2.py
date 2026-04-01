import copy
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from utils.args import *
from models.utils.federated_model import FederatedModel


def get_parser() -> ArgumentParser:
    parser = ArgumentParser(description='Federated learning via FedDenoise V2.')
    add_management_args(parser)
    add_experiment_args(parser)
    return parser


class FedDenoiseV2(FederatedModel):
    """
    V2 changes compared with the original FedDenoise:
    1. Start evaluator refresh at round 10 (same effective timing as original).
    2. Use multiple evaluator models instead of a fixed pair.
    3. Re-select evaluator models at every refresh point.
    4. Each evaluator scores samples independently.
    5. Final sample score is the weighted average of evaluator losses.
       The weight is proportional to the evaluator client's available local
       training sample count before local training in that round.
    6. Keep the evaluator schedule configurable, e.g. 8,6,4,2 or 8,6,5,3.
    """

    NAME = 'feddenoise_v2'
    COMPATIBILITY = ['homogeneity']

    def __init__(self, nets_list, args, transform):
        super(FedDenoiseV2, self).__init__(nets_list, args, transform)

        self.drop_rate = args.drop_rate
        self.denoise_strategy = getattr(args, 'denoise_strategy', 'least_sim')
        self.refresh_gap = int(getattr(args, 'refresh_gap', 10))

        # Example:
        #   --evaluator_schedule 8,6,4,2
        # Means:
        #   refresh#1 -> 8 evaluators
        #   refresh#2 -> 6 evaluators
        #   refresh#3 -> 4 evaluators
        #   refresh#4 and later -> 2 evaluators
        self.evaluator_schedule = self._parse_evaluator_schedule(
            getattr(args, 'evaluator_schedule', '8,6,4,2')
        )

        # Aggregation of evaluator scores:
        #   weighted_mean: weighted by selected clients' local sample counts
        #   mean: equal-weight average
        self.score_agg = getattr(args, 'score_agg', 'weighted_mean').lower()

        # evaluator state
        # mode: 'global' or 'multi'
        self.eval_mode = None
        self.eval_global_model = None
        self.eval_peer_models = []
        self.eval_peer_weights = []
        self.eval_peer_client_ids = []
        self.eval_refresh_count = 0

        # pure ratio related
        # True=clean, False=noisy
        self.noise_or_not = None
        self.round_pure_ratio = []
        self.client_pure_ratio = {}

    # ----------------------------
    # basic helpers
    # ----------------------------
    def ini(self):
        print(
            "Backbone:",
            self.nets_list[0].__class__.__name__,
            getattr(self.nets_list[0], 'name', 'NA')
        )
        self.global_net = copy.deepcopy(self.nets_list[0])
        global_w = self.nets_list[0].state_dict()
        for net in self.nets_list:
            net.load_state_dict(global_w)

    def _parse_evaluator_schedule(self, schedule_value):
        if isinstance(schedule_value, (list, tuple)):
            schedule = [int(x) for x in schedule_value]
        else:
            schedule = [int(x.strip()) for x in str(schedule_value).split(',') if x.strip()]

        if len(schedule) == 0:
            schedule = [2]

        schedule = [max(1, x) for x in schedule]
        return schedule

    def _current_target_evaluator_num(self):
        """
        After each refresh event, move one step forward in the schedule.
        Example schedule: [8, 6, 4, 2]
        refresh #1 -> 8
        refresh #2 -> 6
        refresh #3 -> 4
        refresh #4+ -> 2
        """
        idx = min(self.eval_refresh_count, len(self.evaluator_schedule) - 1)
        return self.evaluator_schedule[idx]

    def _get_loader_len(self, dl):
        if hasattr(dl, 'sampler') and hasattr(dl.sampler, 'indices'):
            return len(dl.sampler.indices)
        if hasattr(dl, 'dataset'):
            return len(dl.dataset)
        return 0

    def _get_client_weights(self, client_ids):
        lengths = [self._get_loader_len(self.trainloaders[cid]) for cid in client_ids]
        total = float(sum(lengths))
        if total <= 0:
            return [1.0 / len(client_ids) for _ in client_ids]
        return [l / total for l in lengths]

    def _has_ready_evaluator(self):
        if self.eval_mode == 'global':
            return self.eval_global_model is not None
        if self.eval_mode == 'multi':
            return len(self.eval_peer_models) > 0 and len(self.eval_peer_weights) == len(self.eval_peer_models)
        return False

    # ----------------------------
    # FL outer loop
    # ----------------------------
    def loc_update(self, priloader_list):
        total_clients = list(range(self.args.parti_num))
        online_clients = self.random_state.choice(
            total_clients, self.online_num, replace=False
        ).tolist()
        self.online_clients = online_clients

        round_pure_ratios = []
        for i in online_clients:
            client_pr = self._train_net(i, self.nets_list[i], priloader_list[i])
            if client_pr is not None:
                round_pure_ratios.append(client_pr)
            self.client_pure_ratio[i] = client_pr

        self.aggregate_nets(None)

        if len(round_pure_ratios) > 0:
            avg_pr = float(np.mean(round_pure_ratios))
            self.round_pure_ratio.append(avg_pr)
            print(f"[Round {self.epoch_index}] Average Pure Ratio: {avg_pr:.2f}%")
        else:
            avg_pr = None
        return avg_pr

    def aggregate_nets(self, freq=None):
        global_net = self.global_net
        nets_list = self.nets_list
        online_clients = self.online_clients
        global_w = self.global_net.state_dict()

        if self.args.averaing == 'weight':
            online_clients_dl = [self.trainloaders[idx] for idx in online_clients]
            online_clients_len = [self._get_loader_len(dl) for dl in online_clients_dl]
            total_len = np.sum(online_clients_len)
            if total_len > 0:
                freq = np.array(online_clients_len) / total_len
            else:
                freq = np.array([1.0 / len(online_clients) for _ in online_clients])
        else:
            freq = [1.0 / len(online_clients) for _ in online_clients]

        first = True
        for index, net_id in enumerate(online_clients):
            net_para = nets_list[net_id].state_dict()
            if first:
                first = False
                for key in net_para:
                    global_w[key] = net_para[key] * freq[index]
            else:
                for key in net_para:
                    global_w[key] += net_para[key] * freq[index]

        global_net.load_state_dict(global_w)

        # Refresh evaluator every refresh_gap rounds.
        # Effective timing:
        # after round 9 aggregation -> choose evaluator for round 10 local training
        if (self.epoch_index + 1) % self.refresh_gap == 0:
            self._refresh_evaluators(online_clients)

        # Broadcast global model for next round local initialization
        for net in nets_list:
            net.load_state_dict(global_net.state_dict())

    # ----------------------------
    # evaluator selection
    # ----------------------------
    def _refresh_evaluators(self, online_clients):
        strategy = self.denoise_strategy.lower()

        if strategy == 'global':
            self.eval_mode = 'global'
            self.eval_global_model = copy.deepcopy(self.global_net)
            self.eval_peer_models = []
            self.eval_peer_weights = []
            self.eval_peer_client_ids = []
            self.eval_refresh_count += 1

            print(
                f"\n---> [Round {self.epoch_index}] Refresh evaluator: GLOBAL "
                f"(strategy={self.denoise_strategy}) <---"
            )
            return

        sims = []
        global_vec = torch.cat(
            [p.detach().flatten().cpu() for p in self.global_net.parameters()]
        )

        for idx in online_clients:
            local_vec = torch.cat(
                [p.detach().flatten().cpu() for p in self.nets_list[idx].parameters()]
            )
            sim = torch.nn.functional.cosine_similarity(
                global_vec.unsqueeze(0),
                local_vec.unsqueeze(0)
            ).item()
            sims.append((idx, sim))

        if len(sims) == 0:
            print(f"[Round {self.epoch_index}] No online clients available for evaluator refresh.")
            return

        target_k = self._current_target_evaluator_num()
        target_k = min(target_k, len(sims))

        if strategy == 'most_sim':
            sims.sort(key=lambda x: x[1], reverse=True)
            selected_idx = [cid for cid, _ in sims[:target_k]]
        elif strategy == 'least_sim':
            sims.sort(key=lambda x: x[1], reverse=True)
            selected_idx = [cid for cid, _ in sims[-target_k:]]
            # Keep logging order from least to less-least for readability
            selected_idx = selected_idx[::-1]
        elif strategy == 'random':
            selected_idx = self.random_state.choice(
                [s[0] for s in sims], target_k, replace=False
            ).tolist()
        else:
            sims.sort(key=lambda x: x[1], reverse=True)
            selected_idx = [cid for cid, _ in sims[:target_k]]

        selected_weights = self._get_client_weights(selected_idx)

        self.eval_mode = 'multi'
        self.eval_global_model = None
        self.eval_peer_models = [copy.deepcopy(self.nets_list[cid]) for cid in selected_idx]
        self.eval_peer_weights = selected_weights
        self.eval_peer_client_ids = selected_idx
        self.eval_refresh_count += 1

        sim_dict = {cid: sim for cid, sim in sims}
        sample_counts = [self._get_loader_len(self.trainloaders[cid]) for cid in selected_idx]

        print(
            f"\n---> [Round {self.epoch_index}] Refresh evaluator clients: {selected_idx} "
            f"(strategy={self.denoise_strategy}, "
            f"schedule={self.evaluator_schedule}, "
            f"current_k={len(selected_idx)}, "
            f"weights={[round(w, 4) for w in selected_weights]}, "
            f"sample_counts={sample_counts}, "
            f"sims={[round(sim_dict[cid], 4) for cid in selected_idx]}) <---"
        )

    # ----------------------------
    # sample scoring
    # ----------------------------
    def _build_sample_scores(self, images, labels, criterion):
        """
        Returns
            scores: shape [batch_size]
        """
        if self.eval_mode == 'global':
            judge_loss = criterion(self.eval_global_model(images), labels)
            return judge_loss.detach()

        losses = []
        for judge_net in self.eval_peer_models:
            judge_loss = criterion(judge_net(images), labels)  # [B]
            losses.append(judge_loss.detach())

        # [k, B]
        loss_matrix = torch.stack(losses, dim=0)

        if self.score_agg == 'mean':
            scores = loss_matrix.mean(dim=0)
        else:
            # default: weighted_mean
            weights = torch.tensor(
                self.eval_peer_weights,
                dtype=loss_matrix.dtype,
                device=loss_matrix.device
            ).view(-1, 1)  # [k, 1]
            scores = (loss_matrix * weights).sum(dim=0)

        return scores

    # ----------------------------
    # local update
    # ----------------------------
    def _train_net(self, index, net, train_loader):
        net = net.to(self.device)
        net.train()

        optimizer = optim.SGD(
            net.parameters(),
            lr=self.local_lr,
            momentum=0.9,
            weight_decay=self.args.reg
        )
        criterion = nn.CrossEntropyLoss(reduction='none').to(self.device)

        is_denoise_phase = self._has_ready_evaluator()

        if is_denoise_phase:
            if self.eval_mode == 'global':
                self.eval_global_model = self.eval_global_model.to(self.device)
                self.eval_global_model.eval()
            elif self.eval_mode == 'multi':
                moved_models = []
                for judge_model in self.eval_peer_models:
                    judge_model = judge_model.to(self.device)
                    judge_model.eval()
                    moved_models.append(judge_model)
                self.eval_peer_models = moved_models

        pure_ratio_list = []
        iterator = tqdm(range(self.local_epoch))

        for _ in iterator:
            for batch_idx, batch in enumerate(train_loader):
                if len(batch) == 3:
                    images, labels, indexes = batch
                    indexes = indexes.cpu().numpy()
                else:
                    images, labels = batch
                    indexes = None

                images, labels = images.to(self.device), labels.to(self.device)

                optimizer.zero_grad()
                outputs = net(images)
                loss_local = criterion(outputs, labels)

                if not is_denoise_phase:
                    final_loss = loss_local.mean()
                    final_loss.backward()
                    optimizer.step()

                    if indexes is not None and self.noise_or_not is not None:
                        pure_ratio = float(np.mean(self.noise_or_not[indexes])) * 100.0
                        pure_ratio_list.append(pure_ratio)
                    else:
                        pure_ratio = -1.0

                    iterator.desc = (
                        "Local Client %d (Warmup) loss = %0.3f, pure = %0.2f"
                        % (index, final_loss.item(), pure_ratio)
                    )
                else:
                    with torch.no_grad():
                        scores = self._build_sample_scores(images, labels, criterion)

                    batch_size = images.size(0)
                    drop_count = int(batch_size * self.drop_rate)
                    keep_count = max(1, batch_size - drop_count)

                    _, sorted_indices = torch.sort(scores, descending=False)
                    keep_indices = sorted_indices[:keep_count]

                    final_loss = loss_local[keep_indices].mean()
                    final_loss.backward()
                    optimizer.step()

                    if indexes is not None and self.noise_or_not is not None:
                        keep_indices_np = keep_indices.detach().cpu().numpy()
                        kept_global_indices = indexes[keep_indices_np]
                        pure_ratio = float(np.mean(self.noise_or_not[kept_global_indices])) * 100.0
                        pure_ratio_list.append(pure_ratio)
                    else:
                        pure_ratio = -1.0

                    if self.eval_mode == 'global':
                        phase_name = "Denoise-Global"
                    else:
                        phase_name = (
                            f"Denoise-{self.denoise_strategy}-"
                            f"k{len(self.eval_peer_models)}-"
                            f"{self.score_agg}"
                        )

                    iterator.desc = (
                        "Local Client %d (%s) loss = %0.3f, dropped = %d, pure = %0.2f"
                        % (index, phase_name, final_loss.item(), drop_count, pure_ratio)
                    )

        if len(pure_ratio_list) > 0:
            return float(np.mean(pure_ratio_list))
        return None
