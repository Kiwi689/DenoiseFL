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
        self.alpha = args.alpha
        self.drop_rate = args.drop_rate
        self.denoise_models = []

        # pure ratio related
        # 与 noise-fl 对齐：True=clean, False=noisy
        self.noise_or_not = None
        self.round_pure_ratio = []
        self.client_pure_ratio = {}

    def ini(self):
        print("Backbone:", self.nets_list[0].__class__.__name__)
        self.global_net = copy.deepcopy(self.nets_list[0])
        global_w = self.nets_list[0].state_dict()
        for _, net in enumerate(self.nets_list):
            net.load_state_dict(global_w)

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
            online_clients_len = [len(dl.sampler.indices) for dl in online_clients_dl]
            freq = np.array(online_clients_len) / np.sum(online_clients_len)
        else:
            freq = [1 / len(online_clients) for _ in online_clients]

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

        # 每 10 轮选拔一次去噪参考模型
        if (self.epoch_index + 1) % 10 == 0:
            sims = []
            global_vec = torch.cat([p.detach().flatten() for p in self.global_net.parameters()])

            for idx in online_clients:
                local_vec = torch.cat([p.detach().flatten() for p in nets_list[idx].parameters()])
                sim = torch.nn.functional.cosine_similarity(
                    global_vec.unsqueeze(0),
                    local_vec.unsqueeze(0)
                ).item()
                sims.append((idx, sim))

            sims.sort(key=lambda x: x[1], reverse=True)

            if len(sims) < 2:
                print(f"[Round {self.epoch_index}] Not enough online clients to select denoise models.")
            else:
                if self.args.denoise_strategy == 'most_sim':
                    selected_idx = [sims[0][0], sims[1][0]]
                elif self.args.denoise_strategy == 'least_sim':
                    selected_idx = [sims[-1][0], sims[-2][0]]
                elif self.args.denoise_strategy == 'median':
                    mid = len(sims) // 2
                    if len(sims) == 2:
                        selected_idx = [sims[0][0], sims[1][0]]
                    else:
                        if mid - 1 < 0:
                            selected_idx = [sims[0][0], sims[1][0]]
                        else:
                            selected_idx = [sims[mid - 1][0], sims[mid][0]]
                elif self.args.denoise_strategy == 'random':
                    selected_idx = self.random_state.choice(
                        [s[0] for s in sims], 2, replace=False
                    ).tolist()
                elif self.args.denoise_strategy == 'mix':
                    selected_idx = [sims[0][0], sims[-1][0]]
                else:
                    selected_idx = [sims[0][0], sims[1][0]]

                self.denoise_models = [
                    copy.deepcopy(nets_list[selected_idx[0]]),
                    copy.deepcopy(nets_list[selected_idx[1]])
                ]
                print(
                    f"\n---> [Round {self.epoch_index}] 选拔去噪模型: 客户端 {selected_idx} "
                    f"(策略: {self.args.denoise_strategy}) <---"
                )

        for _, net in enumerate(nets_list):
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

        is_denoise_phase = (self.epoch_index >= 10) and (len(self.denoise_models) == 2)

        if is_denoise_phase:
            far_net1 = self.denoise_models[0].to(self.device)
            far_net2 = self.denoise_models[1].to(self.device)
            far_net1.eval()
            far_net2.eval()

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

                if not is_denoise_phase:
                    outputs = net(images)
                    loss = criterion(outputs, labels).mean()
                    loss.backward()
                    optimizer.step()

                    if indexes is not None and self.noise_or_not is not None:
                        pure_ratio = float(np.mean(self.noise_or_not[indexes])) * 100.0
                        pure_ratio_list.append(pure_ratio)
                    else:
                        pure_ratio = -1.0

                    iterator.desc = (
                        "Local Client %d (Warmup) loss = %0.3f, pure = %0.2f"
                        % (index, loss.item(), pure_ratio)
                    )
                else:
                    outputs = net(images)
                    loss_local = criterion(outputs, labels)

                    with torch.no_grad():
                        loss_far1 = criterion(far_net1(images), labels)
                        loss_far2 = criterion(far_net2(images), labels)

                    stacked_remote = torch.stack([loss_far1, loss_far2], dim=0)
                    peer_median = stacked_remote.median(dim=0).values
                    mad = (stacked_remote - peer_median).abs().median(dim=0).values

                    score = (
                        (1 - self.alpha) * loss_local.detach()
                        + self.alpha * (peer_median.detach() + mad.detach())
                    )

                    batch_size = images.size(0)
                    drop_count = int(batch_size * self.drop_rate)
                    keep_count = batch_size - drop_count

                    if keep_count > 0:
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

                        iterator.desc = (
                            "Local Client %d (Denoise) loss = %0.3f, dropped = %d, pure = %0.2f"
                            % (index, final_loss.item(), drop_count, pure_ratio)
                        )

        if len(pure_ratio_list) > 0:
            return float(np.mean(pure_ratio_list))
        else:
            return None