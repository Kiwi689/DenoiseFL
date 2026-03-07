import torch
import torch.nn.functional as F
from models.utils.federated_model import FederatedModel
import numpy as np
import random
import copy

class FedDenoiseV2(FederatedModel):
    def __init__(self, nets_list, args, transform):
        super(FedDenoiseV2, self).__init__(nets_list, args, transform)
        # 存储所有客户端与全局模型的相似import torch
import torch.nn.functional as F
from models.utils.federated_model import FederatedModel
import numpy as np
import random
import copy

class FedDenoiseV2(FederatedModel):
    def __init__(self, nets_list, args, transform):
        super(FedDenoiseV2, self).__init__(nets_list, args, transform)
        # 存储所有客户端与全局模型的相似度记录 [parti_num]
        self.similarities = torch.zeros(args.parti_num).to(self.device)
        self.peer_indices = {} 
        self.alpha = args.alpha
        self.beta_penalty = 1.0 # 分歧惩罚权重

    # === 最关键的修复点：初始化全局模型 ===
    def ini(self):
        self.global_net = copy.deepcopy(self.nets_list[0])
        global_w = self.nets_list[0].state_dict()
        for _, net in enumerate(self.nets_list):
            net.load_state_dict(global_w)

    def loc_update(self, priloader_list):
        total_loss = {}
        
        # 随机抽取本轮在线的 10 个客户端
        total_clients = list(range(self.args.parti_num))
        online_clients = self.random_state.choice(total_clients, self.online_num, replace=False).tolist()
        self.online_clients = online_clients  # 保存到 self，给后方的 aggregate_nets 使用
        
        # 1. 构建候选池：当前 10 个 + 全局随机抽 20 个 (对齐 Noise-FL)
        all_indices = list(range(self.args.parti_num))
        remaining_indices = [i for i in all_indices if i not in online_clients]
        extra_candidates = random.sample(remaining_indices, min(20, len(remaining_indices)))
        candidate_pool = online_clients + extra_candidates

        # 2. 评委筛选 (10轮预热后开启)
        if self.epoch_index >= 10:
            self._update_peer_selection(online_clients, candidate_pool)

        # 3. 执行本地训练
        for i in online_clients:
            self._train_net(i, priloader_list[i])
        
        # 4. 训练结束后，更新这 10 个模型的最新相似度，供下一轮参考
        self._update_historical_similarities(online_clients)
        
        # 5. 执行联邦全局聚合
        self.aggregate_nets(None)
        
        return total_loss

    def _update_historical_similarities(self, online_clients):
        self.global_net.eval()
        global_params = torch.cat([p.view(-1) for p in self.global_net.parameters()]).detach()
        for i in online_clients:
            self.nets_list[i].eval()
            local_params = torch.cat([p.view(-1) for p in self.nets_list[i].parameters()]).detach()
            sim = F.cosine_similarity(global_params.unsqueeze(0), local_params.unsqueeze(0))
            self.similarities[i] = sim.item()

    def _update_peer_selection(self, online_clients, candidate_pool):
        for i in online_clients:
            # 备选池排除自己
            current_pool = [idx for idx in candidate_pool if idx != i]
            pool_sims = self.similarities[torch.tensor(current_pool).to(self.device)]
            sorted_indices = torch.argsort(pool_sims) 

            if self.args.denoise_strategy == 'most_sim':
                p1, p2 = sorted_indices[-1], sorted_indices[-2]
            elif self.args.denoise_strategy == 'least_sim':
                p1, p2 = sorted_indices[0], sorted_indices[1]
            elif self.args.denoise_strategy == 'median':
                mid = len(sorted_indices) // 2
                p1, p2 = sorted_indices[mid], sorted_indices[mid-1]
            elif self.args.denoise_strategy == 'mix':
                p1, p2 = sorted_indices[0], sorted_indices[-1]
            else: # random
                idx_choice = np.random.choice(len(current_pool), 2, replace=False)
                p1, p2 = idx_choice[0], idx_choice[1]
                self.peer_indices[i] = (current_pool[p1], current_pool[p2])
                continue
            
            self.peer_indices[i] = (current_pool[p1.item()], current_pool[p2.item()])

    def _train_net(self, index, dataloader):
        net = self.nets_list[index]
        net = net.to(self.device)
        net.train()
        
        optimizer = torch.optim.SGD(net.parameters(), lr=self.local_lr, momentum=0.9, weight_decay=self.args.reg)
        
        criterion = torch.nn.CrossEntropyLoss(reduction='none')
        criterion.to(self.device)

        # 获取评委
        far_net1, far_net2 = None, None
        if self.epoch_index >= 10 and index in self.peer_indices:
            p1, p2 = self.peer_indices[index]
            far_net1 = self.nets_list[p1].to(self.device)
            far_net2 = self.nets_list[p2].to(self.device)
            far_net1.eval()
            far_net2.eval()

        for epoch in range(self.local_epoch):
            for batch_idx, (images, labels) in enumerate(dataloader):
                images, labels = images.to(self.device), labels.to(self.device)
                optimizer.zero_grad()
                outputs = net(images)
                
                loss_vec = criterion(outputs, labels)

                if far_net1 is not None:
                    with torch.no_grad():
                        loss_far1 = criterion(far_net1(images), labels)
                        loss_far2 = criterion(far_net2(images), labels)
                    
                    # === 核心：Disagreement Penalty 打分公式 ===
                    stacked = torch.stack([loss_far1, loss_far2], dim=0)
                    mu_remote = torch.mean(stacked, dim=0)
                    std_remote = torch.std(stacked, dim=0, unbiased=False) + 1e-6
                    
                    score = self.alpha * loss_vec.detach() + (1 - self.alpha) * (mu_remote + self.beta_penalty * std_remote)
                    
                    # 按照 drop_rate 剔除高分样本
                    keep_num = int(len(score) * (1 - self.args.drop_rate))
                    _, keep_idx = torch.topk(score, keep_num, largest=False)
                    loss = loss_vec[keep_idx].mean()
                else:
                    loss = loss_vec.mean()

                loss.backward()
                optimizer.step()
        self.similarities = torch.zeros(args.parti_num).to(self.device)
        self.peer_indices = {} 
        self.alpha = args.alpha
        self.beta_penalty = 1.0 # 分歧惩罚权重

    # === 修复点：必须包含 ini() 以初始化全局模型 ===
    def ini(self):
        self.global_net = copy.deepcopy(self.nets_list[0])
        global_w = self.nets_list[0].state_dict()
        for _, net in enumerate(self.nets_list):
            net.load_state_dict(global_w)

    def loc_update(self, priloader_list):
        total_loss = {}
        
        # 随机抽取本轮在线的 10 个客户端
        total_clients = list(range(self.args.parti_num))
        online_clients = self.random_state.choice(total_clients, self.online_num, replace=False).tolist()
        self.online_clients = online_clients  # 保存到 self，给后方的 aggregate_nets 使用
        
        # 1. 构建候选池：当前 10 个 + 全局随机抽 20 个
        all_indices = list(range(self.args.parti_num))
        remaining_indices = [i for i in all_indices if i not in online_clients]
        extra_candidates = random.sample(remaining_indices, min(20, len(remaining_indices)))
        candidate_pool = online_clients + extra_candidates

        # 2. 评委筛选 (10轮预热后开启)
        if self.epoch_index >= 10:
            self._update_peer_selection(online_clients, candidate_pool)

        # 3. 执行本地训练
        for i in online_clients:
            self._train_net(i, priloader_list[i])
        
        # 4. 训练结束后，更新这 10 个模型的最新相似度
        self._update_historical_similarities(online_clients)
        
        # 5. 执行联邦全局聚合
        self.aggregate_nets(None)
        
        return total_loss

    def _update_historical_similarities(self, online_clients):
        self.global_net.eval()
        global_params = torch.cat([p.view(-1) for p in self.global_net.parameters()]).detach()
        for i in online_clients:
            self.nets_list[i].eval()
            local_params = torch.cat([p.view(-1) for p in self.nets_list[i].parameters()]).detach()
            sim = F.cosine_similarity(global_params.unsqueeze(0), local_params.unsqueeze(0))
            self.similarities[i] = sim.item()

    def _update_peer_selection(self, online_clients, candidate_pool):
        for i in online_clients:
            # 备选池排除自己
            current_pool = [idx for idx in candidate_pool if idx != i]
            pool_sims = self.similarities[torch.tensor(current_pool).to(self.device)]
            sorted_indices = torch.argsort(pool_sims) 

            if self.args.denoise_strategy == 'most_sim':
                p1, p2 = sorted_indices[-1], sorted_indices[-2]
            elif self.args.denoise_strategy == 'least_sim':
                p1, p2 = sorted_indices[0], sorted_indices[1]
            elif self.args.denoise_strategy == 'median':
                mid = len(sorted_indices) // 2
                p1, p2 = sorted_indices[mid], sorted_indices[mid-1]
            elif self.args.denoise_strategy == 'mix':
                p1, p2 = sorted_indices[0], sorted_indices[-1]
            else: # random
                idx_choice = np.random.choice(len(current_pool), 2, replace=False)
                p1, p2 = idx_choice[0], idx_choice[1]
                self.peer_indices[i] = (current_pool[p1], current_pool[p2])
                continue
            
            self.peer_indices[i] = (current_pool[p1.item()], current_pool[p2.item()])

    def _train_net(self, index, dataloader):
        net = self.nets_list[index]
        net = net.to(self.device)
        net.train()
        
        optimizer = torch.optim.SGD(net.parameters(), lr=self.local_lr, momentum=0.9, weight_decay=self.args.reg)
        
        criterion = torch.nn.CrossEntropyLoss(reduction='none')
        criterion.to(self.device)

        # 获取评委
        far_net1, far_net2 = None, None
        if self.epoch_index >= 10 and index in self.peer_indices:
            p1, p2 = self.peer_indices[index]
            far_net1 = self.nets_list[p1].to(self.device)
            far_net2 = self.nets_list[p2].to(self.device)
            far_net1.eval()
            far_net2.eval()

        for epoch in range(self.local_epoch):
            for batch_idx, (images, labels) in enumerate(dataloader):
                images, labels = images.to(self.device), labels.to(self.device)
                optimizer.zero_grad()
                outputs = net(images)
                
                loss_vec = criterion(outputs, labels)

                if far_net1 is not None:
                    with torch.no_grad():
                        loss_far1 = criterion(far_net1(images), labels)
                        loss_far2 = criterion(far_net2(images), labels)
                    
                    # === 核心：Disagreement Penalty 打分公式 ===
                    stacked = torch.stack([loss_far1, loss_far2], dim=0)
                    mu_remote = torch.mean(stacked, dim=0)
                    std_remote = torch.std(stacked, dim=0, unbiased=False) + 1e-6
                    
                    score = self.alpha * loss_vec.detach() + (1 - self.alpha) * (mu_remote + self.beta_penalty * std_remote)
                    
                    # 按照 drop_rate 剔除高分样本
                    keep_num = int(len(score) * (1 - self.args.drop_rate))
                    _, keep_idx = torch.topk(score, keep_num, largest=False)
                    loss = loss_vec[keep_idx].mean()
                else:
                    loss = loss_vec.mean()

                loss.backward()
                optimizer.step()