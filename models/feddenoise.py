import torch.optim as optim
import torch.nn as nn
from tqdm import tqdm
import copy
import torch
import numpy as np
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
        self.drop_rate = args.drop_rate  # 固定的丢弃比例
        self.denoise_models = []

    def ini(self):
        self.global_net = copy.deepcopy(self.nets_list[0])
        global_w = self.nets_list[0].state_dict()
        for _, net in enumerate(self.nets_list):
            net.load_state_dict(global_w)

    def loc_update(self, priloader_list):
        total_clients = list(range(self.args.parti_num))
        online_clients = self.random_state.choice(total_clients, self.online_num, replace=False).tolist()
        self.online_clients = online_clients

        for i in online_clients:
            self._train_net(i, self.nets_list[i], priloader_list[i])

        self.aggregate_nets(None)
        return None

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

        # 【核心逻辑】每 10 轮结束时，进行相似度排名和模型选拔
        if (self.epoch_index + 1) % 10 == 0:
            sims = []
            global_vec = torch.cat([p.flatten() for p in self.global_net.parameters()])
            for idx in online_clients:
                local_vec = torch.cat([p.flatten() for p in nets_list[idx].parameters()])
                sim = torch.nn.functional.cosine_similarity(global_vec.unsqueeze(0), local_vec.unsqueeze(0)).item()
                sims.append((idx, sim))

            sims.sort(key=lambda x: x[1], reverse=True)
            
            if self.args.denoise_strategy == 'most_sim':
                selected_idx = [sims[0][0], sims[1][0]]
            elif self.args.denoise_strategy == 'least_sim':
                selected_idx = [sims[-1][0], sims[-2][0]]
            elif self.args.denoise_strategy == 'median':
                mid = len(sims) // 2
                selected_idx = [sims[mid-1][0], sims[mid][0]]
            elif self.args.denoise_strategy == 'random':
                selected_idx = np.random.choice([s[0] for s in sims], 2, replace=False).tolist()
            elif self.args.denoise_strategy == 'mix':
                selected_idx = [sims[0][0], sims[-1][0]]

            self.denoise_models = [copy.deepcopy(nets_list[selected_idx[0]]), copy.deepcopy(nets_list[selected_idx[1]])]
            print(f"\n---> [Round {self.epoch_index}] 选拔去噪模型: 客户端 {selected_idx} (策略: {self.args.denoise_strategy}) <---")

        for _, net in enumerate(nets_list):
            net.load_state_dict(global_net.state_dict())

    def _train_net(self, index, net, train_loader):
        net = net.to(self.device)
        net.train()
        optimizer = optim.SGD(net.parameters(), lr=self.local_lr, momentum=0.9, weight_decay=self.args.reg)
        
        # 使用 reduction='none' 获取每个样本的 loss
        criterion = nn.CrossEntropyLoss(reduction='none') 
        criterion.to(self.device)

        is_denoise_phase = (self.epoch_index >= 10) and (len(self.denoise_models) == 2)

        if is_denoise_phase:
            far_net1 = self.denoise_models[0].to(self.device)
            far_net2 = self.denoise_models[1].to(self.device)
            far_net1.eval()
            far_net2.eval()

        iterator = tqdm(range(self.local_epoch))
        for _ in iterator:
            for batch_idx, (images, labels) in enumerate(train_loader):
                images, labels = images.to(self.device), labels.to(self.device)
                optimizer.zero_grad()

                if not is_denoise_phase:
                    # 预热阶段
                    outputs = net(images)
                    loss = criterion(outputs, labels).mean()
                    loss.backward()
                    iterator.desc = "Local Client %d (Warmup) loss = %0.3f" % (index, loss.item())
                    optimizer.step()
                else:
                    # 去噪阶段：打分 + 固定比例截断
                    outputs = net(images)
                    loss_local = criterion(outputs, labels)

                    with torch.no_grad():
                        loss_far1 = criterion(far_net1(images), labels)
                        loss_far2 = criterion(far_net2(images), labels)

                    # 融合打分 (Score 越高越像噪声)
                    score = self.alpha * loss_local + (1 - self.alpha) * (loss_far1 + loss_far2) / 2.0

                    batch_size = images.size(0)
                    drop_count = int(batch_size * self.drop_rate)
                    keep_count = batch_size - drop_count

                    if keep_count > 0:
                        # 升序排序，保留前面分数小（Loss小）的样本
                        _, sorted_indices = torch.sort(score, descending=False)
                        keep_indices = sorted_indices[:keep_count]

                        # 仅在干净子集上算最终 loss 并反传
                        final_loss = loss_local[keep_indices].mean()
                        final_loss.backward()
                        
                        iterator.desc = "Local Client %d (Denoise) loss = %0.3f, dropped = %d" % (index, final_loss.item(), drop_count)
                        optimizer.step()