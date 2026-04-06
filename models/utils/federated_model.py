import copy
import numpy as np
import torch.nn as nn
import torch
import torchvision
from argparse import Namespace
from utils.conf import get_device
from utils.conf import checkpoint_path
from utils.util import create_if_not_exists
import os


class FederatedModel(nn.Module):
    """
    Federated learning base model.
    """
    NAME = None

    def __init__(self, nets_list: list,
                 args: Namespace, transform: torchvision.transforms) -> None:
        super(FederatedModel, self).__init__()
        self.nets_list = nets_list
        self.args = args
        self.transform = transform

        # online clients
        self.random_state = np.random.RandomState(self.args.seed)
        self.online_num = np.ceil(self.args.parti_num * self.args.online_ratio).item()
        self.online_num = int(self.online_num)

        self.global_net = None
        self.device = get_device(device_id=self.args.device_id)

        self.local_epoch = args.local_epoch
        self.local_batch_size = args.local_batch_size
        self.local_lr = args.local_lr
        self.trainloaders = None
        self.testlodaers = None
        self.net_cls_counts = None

        self.epoch_index = 0  # communication round index

        self.checkpoint_path = checkpoint_path() + self.args.dataset + '/' + self.args.structure + '/'
        create_if_not_exists(self.checkpoint_path)
        self.net_to_device()

    # =========================================================
    # basic utilities
    # =========================================================
    def net_to_device(self):
        for net in self.nets_list:
            net.to(self.device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.global_net(x)

    def get_scheduler(self):
        return

    def ini(self):
        pass

    def col_update(self, communication_idx, publoader):
        pass

    def loc_update(self, priloader_list):
        pass

    # =========================================================
    # model state key helpers
    # =========================================================
    def _reference_state_dict(self):
        if self.global_net is not None:
            return self.global_net.state_dict()
        return self.nets_list[0].state_dict()

    def _is_head_key(self, key: str) -> bool:
        """
        Heuristic classifier-head key matcher.
        Current main experiments are based on NoiseFLCNN, whose head is mainly:
        - l_c1.*
        - bn_c1.*
        Also keep a few common prefixes for compatibility.
        """
        head_prefixes = (
            'l_c1',
            'bn_c1',
            'fc',
            'classifier',
            'linear',
            'head'
        )
        return key.startswith(head_prefixes)

    def get_all_state_keys(self):
        return list(self._reference_state_dict().keys())

    def get_head_keys(self):
        return [k for k in self.get_all_state_keys() if self._is_head_key(k)]

    def get_backbone_keys(self):
        return [k for k in self.get_all_state_keys() if not self._is_head_key(k)]

    # =========================================================
    # dataloader / aggregation helpers
    # =========================================================
    def _get_loader_len(self, dl):
        if hasattr(dl, 'sampler') and hasattr(dl.sampler, 'indices'):
            return len(dl.sampler.indices)
        if hasattr(dl, 'dataset'):
            return len(dl.dataset)
        return 0

    def _get_online_client_freq(self, online_clients):
        if len(online_clients) == 0:
            raise RuntimeError('online_clients is empty.')

        if self.args.averaing == 'weight':
            if self.trainloaders is None:
                return [1.0 / len(online_clients) for _ in online_clients]

            online_clients_dl = [self.trainloaders[idx] for idx in online_clients]
            online_clients_len = [self._get_loader_len(dl) for dl in online_clients_dl]
            total_len = float(np.sum(online_clients_len))

            if total_len > 0:
                return [float(x / total_len) for x in online_clients_len]
            else:
                return [1.0 / len(online_clients) for _ in online_clients]
        else:
            return [1.0 / len(online_clients) for _ in online_clients]

    # =========================================================
    # selective aggregation / broadcast
    # =========================================================
    def aggregate_nets_by_keys(self, keys, freq=None):
        """
        Aggregate only the specified state_dict keys from online clients into global_net.
        Does NOT broadcast automatically.
        """
        if self.global_net is None:
            raise RuntimeError('global_net is None. Please call ini() before aggregation.')

        online_clients = self.online_clients
        if len(online_clients) == 0:
            raise RuntimeError('No online clients available for aggregation.')

        if freq is None:
            freq = self._get_online_client_freq(online_clients)

        global_w = self.global_net.state_dict()

        for key in keys:
            ref_tensor = self.nets_list[online_clients[0]].state_dict()[key]

            if torch.is_floating_point(ref_tensor):
                agg_tensor = None
                for idx, client_id in enumerate(online_clients):
                    local_tensor = self.nets_list[client_id].state_dict()[key].detach().clone()
                    weighted_tensor = local_tensor * freq[idx]
                    if agg_tensor is None:
                        agg_tensor = weighted_tensor
                    else:
                        agg_tensor += weighted_tensor
                global_w[key] = agg_tensor
            else:
                # For integer buffers such as num_batches_tracked, just copy from the first client.
                global_w[key] = ref_tensor.detach().clone()

        self.global_net.load_state_dict(global_w)

    def broadcast_global_by_keys(self, keys):
        """
        Broadcast only the specified keys from global_net to all local client models.
        """
        if self.global_net is None:
            raise RuntimeError('global_net is None. Please call ini() before broadcast.')

        global_w = self.global_net.state_dict()
        for net in self.nets_list:
            local_w = net.state_dict()
            for key in keys:
                local_w[key] = global_w[key].detach().clone()
            net.load_state_dict(local_w)

    # =========================================================
    # default full-model aggregation
    # =========================================================
    def aggregate_nets(self, freq=None):
        """
        Default behavior: aggregate the full model, then broadcast the full model.
        This keeps backward compatibility for existing methods like FedAvg / old FedDenoise.
        """
        all_keys = self.get_all_state_keys()
        self.aggregate_nets_by_keys(all_keys, freq=freq)
        self.broadcast_global_by_keys(all_keys)

    # =========================================================
    # optional pretrained helpers
    # =========================================================
    def load_pretrained_nets(self):
        if self.load:
            for j in range(self.args.parti_num):
                pretrain_path = os.path.join(self.checkpoint_path, 'pretrain')
                save_path = os.path.join(pretrain_path, str(j) + '.ckpt')
                self.nets_list[j].load_state_dict(torch.load(save_path, self.device))
        else:
            pass

    def copy_nets2_prevnets(self):
        nets_list = self.nets_list
        prev_nets_list = self.prev_nets_list
        for net_id, net in enumerate(nets_list):
            net_para = net.state_dict()
            prev_net = prev_nets_list[net_id]
            prev_net.load_state_dict(net_para)