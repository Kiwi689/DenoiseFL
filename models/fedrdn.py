import torch
import torch.nn as nn
import torch.optim as optim
import copy
import numpy as np
from tqdm import tqdm

from models.utils.federated_model import FederatedModel


class FedRDN(FederatedModel):

    NAME = 'fedrdn'
    COMPATIBILITY = ['homogeneity']

    def __init__(self, nets_list, args, transform):
        super(FedRDN, self).__init__(nets_list, args, transform)

        # 噪声强度
        self.noise_std = getattr(args, "rdn_std", 0.01)

    def ini(self):

        self.global_net = copy.deepcopy(self.nets_list[0])

        global_w = self.global_net.state_dict()

        for net in self.nets_list:
            net.load_state_dict(global_w)

    def loc_update(self, priloader_list):

        total_clients = list(range(self.args.parti_num))

        online_clients = self.random_state.choice(
            total_clients,
            self.online_num,
            replace=False
        ).tolist()

        self.online_clients = online_clients

        for i in online_clients:

            self._train_net(i, self.nets_list[i], priloader_list[i])

        self.aggregate_nets(None)

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

        criterion = nn.CrossEntropyLoss()

        iterator = tqdm(range(self.local_epoch))

        for _ in iterator:

            for batch_idx, (images, labels, *_) in enumerate(train_loader):

                images = images.to(self.device)
                labels = labels.to(self.device)

                outputs = net(images)

                loss = criterion(outputs, labels)

                optimizer.zero_grad()

                loss.backward()

                # RDN 核心：给梯度加随机噪声
                for p in net.parameters():
                    if p.grad is not None:
                        noise = torch.randn_like(p.grad) * self.noise_std
                        p.grad += noise

                optimizer.step()

                iterator.desc = "Client %d loss=%0.3f" % (index, loss)