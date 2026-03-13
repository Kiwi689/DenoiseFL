import copy
import torch
import torch.nn as nn
import torch.optim as optim

from utils.args import *
from models.utils.federated_model import FederatedModel


def get_parser() -> ArgumentParser:
    parser = ArgumentParser(description='Federated learning via FedProx.')
    add_management_args(parser)
    add_experiment_args(parser)
    return parser


class FedProx(FederatedModel):
    NAME = 'fedprox'
    COMPATIBILITY = ['homogeneity']

    def __init__(self, nets_list, args, transform):
        super(FedProx, self).__init__(nets_list, args, transform)
        self.mu = args.mu

    def ini(self):
        self.global_net = copy.deepcopy(self.nets_list[0])
        global_w = self.global_net.state_dict()
        for net in self.nets_list:
            net.load_state_dict(global_w)

    def loc_update(self, priloader_list):
        total_clients = list(range(self.args.parti_num))
        online_clients = self.random_state.choice(
            total_clients, self.online_num, replace=False
        ).tolist()
        self.online_clients = online_clients

        for client_id in online_clients:
            self._train_net(client_id, self.nets_list[client_id], priloader_list[client_id])

        self.aggregate_nets(None)
        return None

    def _train_net(self, index, net, train_loader):
        net = net.to(self.device)
        net.train()

        optimizer = optim.SGD(
            filter(lambda p: p.requires_grad, net.parameters()),
            lr=self.local_lr,
            momentum=0.9,
            weight_decay=self.args.reg
        )
        criterion = nn.CrossEntropyLoss().to(self.device)

        self.global_net = self.global_net.to(self.device)
        global_weight_collector = [param.detach().clone() for param in self.global_net.parameters()]

        for _ in range(self.local_epoch):
            for batch in train_loader:
                if len(batch) == 3:
                    images, labels, _ = batch
                else:
                    images, labels = batch

                images = images.to(self.device)
                labels = labels.to(self.device)

                outputs = net(images)
                loss = criterion(outputs, labels)

                fed_prox_reg = 0.0
                for param_index, param in enumerate(net.parameters()):
                    fed_prox_reg += torch.norm(
                        param - global_weight_collector[param_index]
                    ) ** 2

                loss = loss + (self.mu / 2.0) * fed_prox_reg

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()