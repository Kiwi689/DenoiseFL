import copy
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from utils.args import *
from models.utils.federated_model import FederatedModel


def get_parser() -> ArgumentParser:
    parser = ArgumentParser(description='Federated learning via FedDenoise.')
    add_management_args(parser)
    add_experiment_args(parser)
    return parser


class FedDenoise(FederatedModel):
    NAME = 'feddenoise'
    COMPATIBILITY = ['homogeneity']

    def __init__(self, nets_list, args, transform):
        super(FedDenoise, self).__init__(nets_list, args, transform)

        self.drop_rate = args.drop_rate
        self.denoise_strategy = getattr(args, 'denoise_strategy', 'most_sim')
        self.refresh_gap = 10

        # evaluator state for next refresh window
        # mode: 'global' or 'pair'
        self.eval_mode = None
        self.eval_global_model = None
        self.eval_peer_models = []
        self.eval_peer_weights = []
        self.eval_peer_client_ids = []

        # pure ratio related
        # 与 noise-fl 对齐：True=clean, False=noisy
        self.noise_or_not = None
        self.round_pure_ratio = []
        self.client_pure_ratio = {}

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
        if self.eval_mode == 'pair':
            return len(self.eval_peer_models) == 2 and len(self.eval_peer_weights) == 2
        return False

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

        # 每 refresh_gap 轮刷新一次“评判模型”
        # 刷新时机：本轮在线客户端训练完成并聚合出当前 global 之后
        if (self.epoch_index + 1) % self.refresh_gap == 0:
            strategy = self.denoise_strategy.lower()

            if strategy == 'global':
                self.eval_mode = 'global'
                self.eval_global_model = copy.deepcopy(self.global_net)
                self.eval_peer_models = []
                self.eval_peer_weights = []
                self.eval_peer_client_ids = []

                print(
                    f"\n---> [Round {self.epoch_index}] 刷新评判模型: GLOBAL "
                    f"(策略: {self.denoise_strategy}) <---"
                )
            else:
                sims = []
                global_vec = torch.cat(
                    [p.detach().flatten().cpu() for p in self.global_net.parameters()]
                )

                for idx in online_clients:
                    local_vec = torch.cat(
                        [p.detach().flatten().cpu() for p in nets_list[idx].parameters()]
                    )
                    sim = torch.nn.functional.cosine_similarity(
                        global_vec.unsqueeze(0),
                        local_vec.unsqueeze(0)
                    ).item()
                    sims.append((idx, sim))

                if len(sims) < 2:
                    print(
                        f"[Round {self.epoch_index}] Not enough online clients to select evaluator models."
                    )
                else:
                    sims.sort(key=lambda x: x[1], reverse=True)

                    if strategy == 'most_sim':
                        selected_idx = [sims[0][0], sims[1][0]]
                    elif strategy == 'least_sim':
                        selected_idx = [sims[-1][0], sims[-2][0]]
                    elif strategy == 'random':
                        selected_idx = self.random_state.choice(
                            [s[0] for s in sims], 2, replace=False
                        ).tolist()
                    else:
                        # 默认回退到 most_sim
                        selected_idx = [sims[0][0], sims[1][0]]

                    selected_weights = self._get_client_weights(selected_idx)

                    self.eval_mode = 'pair'
                    self.eval_global_model = None
                    self.eval_peer_models = [
                        copy.deepcopy(nets_list[selected_idx[0]]),
                        copy.deepcopy(nets_list[selected_idx[1]])
                    ]
                    self.eval_peer_weights = selected_weights
                    self.eval_peer_client_ids = selected_idx

                    sim_dict = {cid: sim for cid, sim in sims}
                    print(
                        f"\n---> [Round {self.epoch_index}] 刷新评判模型: 客户端 {selected_idx} "
                        f"(策略: {self.denoise_strategy}, "
                        f"weights={[round(w, 4) for w in selected_weights]}, "
                        f"sims={[round(sim_dict[cid], 4) for cid in selected_idx]}) <---"
                    )

        # 广播 global model 给所有客户端，供下一轮本地训练初始化
        for net in nets_list:
            net.load_state_dict(global_net.state_dict())

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
                judge_global = self.eval_global_model.to(self.device)
                judge_global.eval()
            elif self.eval_mode == 'pair':
                judge_net1 = self.eval_peer_models[0].to(self.device)
                judge_net2 = self.eval_peer_models[1].to(self.device)
                judge_net1.eval()
                judge_net2.eval()
                w1, w2 = self.eval_peer_weights[0], self.eval_peer_weights[1]

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
                        if self.eval_mode == 'global':
                            judge_loss = criterion(judge_global(images), labels)
                            score = judge_loss.detach()
                        else:
                            loss_1 = criterion(judge_net1(images), labels)
                            loss_2 = criterion(judge_net2(images), labels)
                            score = (w1 * loss_1 + w2 * loss_2).detach()

                    batch_size = images.size(0)
                    drop_count = int(batch_size * self.drop_rate)
                    keep_count = max(1, batch_size - drop_count)

                    _, sorted_indices = torch.sort(score, descending=False)
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
                        iterator.desc = (
                            "Local Client %d (Denoise-Global) loss = %0.3f, dropped = %d, pure = %0.2f"
                            % (index, final_loss.item(), drop_count, pure_ratio)
                        )
                    else:
                        iterator.desc = (
                            "Local Client %d (Denoise-%s) loss = %0.3f, dropped = %d, pure = %0.2f"
                            % (index, self.denoise_strategy, final_loss.item(), drop_count, pure_ratio)
                        )

        if len(pure_ratio_list) > 0:
            return float(np.mean(pure_ratio_list))
        return None