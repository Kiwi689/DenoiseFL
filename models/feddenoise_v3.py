import copy
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from utils.args import *
from models.utils.federated_model import FederatedModel


def get_parser() -> ArgumentParser:
    parser = ArgumentParser(description='Federated learning via FedDenoiseV3.')
    add_management_args(parser)
    add_experiment_args(parser)
    return parser


class FedDenoiseV3(FederatedModel):
    NAME = 'feddenoise_v3'
    COMPATIBILITY = ['homogeneity']

    def __init__(self, nets_list, args, transform):
        super(FedDenoiseV3, self).__init__(nets_list, args, transform)

        # ===== core config =====
        self.warmup_round = args.warmup_round
        self.stage_round = args.stage_round
        self.teacher_schedule = args.teacher_schedule_list
        self.num_stages = len(self.teacher_schedule)

        self.teacher_select_strategy = args.teacher_select_strategy
        self.teacher_similarity = args.teacher_similarity
        self.teacher_score_mode = args.teacher_score_mode
        self.warmup_mode = args.warmup_mode

        self.drop_rate = args.drop_rate
        self.drop_rate_schedule = args.drop_rate_schedule_list
        if self.drop_rate_schedule is None:
            self.stage_drop_rates = [self.drop_rate for _ in range(self.num_stages)]
        else:
            self.stage_drop_rates = self.drop_rate_schedule

        self.exclude_self_teacher = args.exclude_self_teacher

        # ===== runtime state =====
        self.phase_name = 'warmup'
        self.current_stage = -1   # warmup期间记为-1，正式阶段从0开始
        self.teacher_pool = None  # list[model snapshot], post-local & pre-aggregation
        self.current_stage_clean_sets = None   # dict: client_id -> set(global_sample_idx)
        self.current_stage_metrics = {}

        # ===== dataset-side stats =====
        self.noise_or_not = None
        self.client_pure_ratio = {}
        self.round_pure_ratio = []

    # =========================================================
    # init / dispatch
    # =========================================================
    def ini(self):
        print(
            "Backbone:",
            self.nets_list[0].__class__.__name__,
            getattr(self.nets_list[0], 'name', 'NA')
        )
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

        if self.epoch_index < self.warmup_round:
            self.phase_name = 'warmup'
            return self._warmup_round(priloader_list)
        else:
            self.phase_name = f'stage_{self.current_stage + 1}'
            return self._stage_round_train(priloader_list)

    # =========================================================
    # phase / stage helpers
    # =========================================================
    def _is_last_warmup_round(self):
        return self.epoch_index == self.warmup_round - 1

    def _formal_round_index(self):
        return self.epoch_index - self.warmup_round

    def _is_stage_end_round(self):
        if self.epoch_index < self.warmup_round:
            return False
        formal_round_idx = self._formal_round_index() + 1  # 1-based in formal stage training
        return (formal_round_idx % self.stage_round) == 0

    def _get_current_stage_drop_rate(self):
        if self.current_stage < 0:
            return self.drop_rate
        return self.stage_drop_rates[self.current_stage]

    # =========================================================
    # snapshot / similarity / teacher select
    # =========================================================
    def _snapshot_local_nets(self):
        return [copy.deepcopy(net).cpu() for net in self.nets_list]

    def _vectorize_model_by_keys(self, net, keys):
        state = net.state_dict()
        vec_list = []
        for key in keys:
            tensor = state[key]
            if torch.is_floating_point(tensor):
                vec_list.append(tensor.detach().float().view(-1).cpu())
        if len(vec_list) == 0:
            return torch.zeros(1)
        return torch.cat(vec_list, dim=0)

    def _compute_similarity(self, client_net, teacher_net):
        if self.teacher_similarity == 'backbone_cosine':
            keys = self.get_backbone_keys()
        elif self.teacher_similarity == 'full_model_cosine':
            keys = self.get_all_state_keys()
        else:
            raise ValueError(f'Unsupported teacher_similarity: {self.teacher_similarity}')

        v1 = self._vectorize_model_by_keys(client_net, keys)
        v2 = self._vectorize_model_by_keys(teacher_net, keys)

        sim = torch.nn.functional.cosine_similarity(
            v1.unsqueeze(0), v2.unsqueeze(0), dim=1
        ).item()
        return float(sim)

    def _select_teachers_for_client(self, client_id, stage_idx):
        if self.teacher_pool is None:
            raise RuntimeError('teacher_pool is None when selecting teachers.')

        target_teacher_num = self.teacher_schedule[stage_idx]
        current_client_model = self.nets_list[client_id]

        sims = []
        for teacher_id, teacher_model in enumerate(self.teacher_pool):
            if self.exclude_self_teacher and teacher_id == client_id:
                continue
            sim = self._compute_similarity(current_client_model, teacher_model)
            sims.append((teacher_id, sim))

        if len(sims) == 0:
            raise RuntimeError(f'No available teacher candidates for client {client_id}.')

        if self.teacher_select_strategy == 'all':
            selected_teacher_ids = [tid for tid, _ in sims]

        elif self.teacher_select_strategy == 'random':
            candidate_ids = [tid for tid, _ in sims]
            k = min(target_teacher_num, len(candidate_ids))
            selected_teacher_ids = self.random_state.choice(
                candidate_ids, k, replace=False
            ).tolist()

        elif self.teacher_select_strategy == 'least_sim':
            sims.sort(key=lambda x: x[1], reverse=False)
            k = min(target_teacher_num, len(sims))
            selected_teacher_ids = [tid for tid, _ in sims[:k]]

        elif self.teacher_select_strategy == 'most_sim':
            sims.sort(key=lambda x: x[1], reverse=True)
            k = min(target_teacher_num, len(sims))
            selected_teacher_ids = [tid for tid, _ in sims[:k]]

        else:
            raise ValueError(f'Unsupported teacher_select_strategy: {self.teacher_select_strategy}')

        sim_dict = {tid: sim for tid, sim in sims}
        selected_sims = [sim_dict[tid] for tid in selected_teacher_ids]

        return selected_teacher_ids, selected_sims

    # =========================================================
    # sample scoring / clean subset selection
    # =========================================================
    def _score_and_filter_client_samples(self, client_id, teacher_ids, train_loader, stage_idx):
        if len(teacher_ids) == 0:
            raise RuntimeError(f'Client {client_id} has empty teacher_ids.')

        criterion = nn.CrossEntropyLoss(reduction='none').to(self.device)
        teacher_models = []

        for tid in teacher_ids:
            teacher = copy.deepcopy(self.teacher_pool[tid]).to(self.device)
            teacher.eval()
            teacher_models.append(teacher)

        sample_scores = []

        with torch.no_grad():
            for batch in train_loader:
                if len(batch) != 3:
                    raise RuntimeError(
                        'FedDenoiseV3 requires train loader batch format: (images, labels, indexes).'
                    )

                images, labels, indexes = batch
                images = images.to(self.device)
                labels = labels.to(self.device)
                indexes_np = indexes.cpu().numpy()

                teacher_losses = []
                for teacher in teacher_models:
                    logits = teacher(images)
                    per_sample_loss = criterion(logits, labels)
                    teacher_losses.append(per_sample_loss)

                if self.teacher_score_mode == 'teacher_mean':
                    score_tensor = torch.stack(teacher_losses, dim=0).mean(dim=0)
                else:
                    raise ValueError(f'Unsupported teacher_score_mode: {self.teacher_score_mode}')

                score_np = score_tensor.detach().cpu().numpy()
                for gidx, score in zip(indexes_np, score_np):
                    sample_scores.append((int(gidx), float(score)))

        for teacher in teacher_models:
            teacher.cpu()
        del teacher_models

        sample_scores.sort(key=lambda x: x[1], reverse=False)

        total_num = len(sample_scores)
        drop_rate = self.stage_drop_rates[stage_idx]
        keep_num = max(1, int(total_num * (1.0 - drop_rate)))

        selected = sample_scores[:keep_num]
        selected_clean_indices = set([gidx for gidx, _ in selected])

        return selected_clean_indices

    def _compute_client_selection_metrics(self, client_id, selected_clean_indices):
        client_indices = np.array(self.net_dataidx_map[client_id], dtype=int)
        if len(client_indices) == 0:
            return {
                'clean_precision': 0.0,
                'clean_recall': 0.0,
                'noisy_precision': 0.0,
                'noisy_recall': 0.0,
                'keep_ratio': 0.0
            }

        selected_mask = np.isin(client_indices, np.array(list(selected_clean_indices), dtype=int))
        true_clean_mask = self.noise_or_not[client_indices]
        true_noisy_mask = ~true_clean_mask

        # predicted clean = selected
        tp_clean = np.sum(selected_mask & true_clean_mask)
        fp_clean = np.sum(selected_mask & true_noisy_mask)
        fn_clean = np.sum((~selected_mask) & true_clean_mask)

        clean_precision = tp_clean / max(tp_clean + fp_clean, 1)
        clean_recall = tp_clean / max(tp_clean + fn_clean, 1)

        # predicted noisy = not selected
        pred_noisy_mask = ~selected_mask
        tp_noisy = np.sum(pred_noisy_mask & true_noisy_mask)
        fp_noisy = np.sum(pred_noisy_mask & true_clean_mask)
        fn_noisy = np.sum((~pred_noisy_mask) & true_noisy_mask)

        noisy_precision = tp_noisy / max(tp_noisy + fp_noisy, 1)
        noisy_recall = tp_noisy / max(tp_noisy + fn_noisy, 1)

        keep_ratio = np.mean(selected_mask.astype(np.float32))

        return {
            'clean_precision': float(clean_precision * 100.0),
            'clean_recall': float(clean_recall * 100.0),
            'noisy_precision': float(noisy_precision * 100.0),
            'noisy_recall': float(noisy_recall * 100.0),
            'keep_ratio': float(keep_ratio * 100.0)
        }

    def _build_stage_metrics(self, clean_sets, teacher_info):
        metrics_per_client = {}
        clean_precision_list = []
        clean_recall_list = []
        noisy_precision_list = []
        noisy_recall_list = []
        keep_ratio_list = []
        avg_teacher_sim_list = []

        for client_id in range(self.args.parti_num):
            client_metrics = self._compute_client_selection_metrics(
                client_id, clean_sets[client_id]
            )

            teacher_sims = teacher_info[client_id]['teacher_sims']
            avg_teacher_sim = float(np.mean(teacher_sims)) if len(teacher_sims) > 0 else 0.0

            metrics_per_client[client_id] = {
                **client_metrics,
                'teacher_ids': teacher_info[client_id]['teacher_ids'],
                'avg_teacher_sim': avg_teacher_sim
            }

            clean_precision_list.append(client_metrics['clean_precision'])
            clean_recall_list.append(client_metrics['clean_recall'])
            noisy_precision_list.append(client_metrics['noisy_precision'])
            noisy_recall_list.append(client_metrics['noisy_recall'])
            keep_ratio_list.append(client_metrics['keep_ratio'])
            avg_teacher_sim_list.append(avg_teacher_sim)

        return {
            'avg_clean_precision': float(np.mean(clean_precision_list)),
            'avg_clean_recall': float(np.mean(clean_recall_list)),
            'avg_noisy_precision': float(np.mean(noisy_precision_list)),
            'avg_noisy_recall': float(np.mean(noisy_recall_list)),
            'avg_keep_ratio': float(np.mean(keep_ratio_list)),
            'avg_teacher_similarity': float(np.mean(avg_teacher_sim_list)),
            'per_client': metrics_per_client
        }

    def _select_teachers_and_filter_samples(self, priloader_list, stage_idx):
        clean_sets = {}
        teacher_info = {}

        print(f'\n[Teacher Selection] Building clean subsets for stage {stage_idx + 1} ...')

        for client_id in range(self.args.parti_num):
            teacher_ids, teacher_sims = self._select_teachers_for_client(client_id, stage_idx)
            clean_set = self._score_and_filter_client_samples(
                client_id=client_id,
                teacher_ids=teacher_ids,
                train_loader=priloader_list[client_id],
                stage_idx=stage_idx
            )

            clean_sets[client_id] = clean_set
            teacher_info[client_id] = {
                'teacher_ids': teacher_ids,
                'teacher_sims': teacher_sims
            }

            print(
                f'[Stage {stage_idx + 1}] Client {client_id}: '
                f'teachers={teacher_ids}, '
                f'avg_teacher_sim={np.mean(teacher_sims) if len(teacher_sims) > 0 else 0.0:.4f}, '
                f'selected={len(clean_set)}'
            )

        stage_metrics = self._build_stage_metrics(clean_sets, teacher_info)

        print(
            f'[Stage {stage_idx + 1}] Selection summary | '
            f'clean_precision={stage_metrics["avg_clean_precision"]:.2f} | '
            f'clean_recall={stage_metrics["avg_clean_recall"]:.2f} | '
            f'noisy_precision={stage_metrics["avg_noisy_precision"]:.2f} | '
            f'noisy_recall={stage_metrics["avg_noisy_recall"]:.2f} | '
            f'keep_ratio={stage_metrics["avg_keep_ratio"]:.2f} | '
            f'avg_teacher_similarity={stage_metrics["avg_teacher_similarity"]:.4f}'
        )

        return clean_sets, stage_metrics

    # =========================================================
    # warmup round
    # =========================================================
    def _warmup_round(self, priloader_list):
        for client_id in self.online_clients:
            self._train_net_warmup(
                client_id,
                self.nets_list[client_id],
                priloader_list[client_id]
            )

        # warmup最后一轮：先 snapshot + teacher matching，再做聚合/广播
        if self._is_last_warmup_round():
            self.teacher_pool = self._snapshot_local_nets()

            next_clean_sets, next_stage_metrics = self._select_teachers_and_filter_samples(
                priloader_list=priloader_list,
                stage_idx=0
            )
            self.current_stage = 0
            self.current_stage_clean_sets = next_clean_sets
            self.current_stage_metrics = next_stage_metrics

        if self.warmup_mode == 'backbone_only':
            backbone_keys = self.get_backbone_keys()
            self.aggregate_nets_by_keys(backbone_keys)
            self.broadcast_global_by_keys(backbone_keys)

        elif self.warmup_mode == 'full_model':
            all_keys = self.get_all_state_keys()
            self.aggregate_nets_by_keys(all_keys)
            self.broadcast_global_by_keys(all_keys)

        elif self.warmup_mode == 'no_comm':
            # 前几轮完全不聚合；但warmup结束后为了进入正式联邦学习，
            # 仍做一次全模型聚合/广播，作为第1阶段初始化
            if self._is_last_warmup_round():
                all_keys = self.get_all_state_keys()
                self.aggregate_nets_by_keys(all_keys)
                self.broadcast_global_by_keys(all_keys)
        else:
            raise ValueError(f'Unsupported warmup_mode: {self.warmup_mode}')

        return {
            'phase': 'warmup',
            'stage_id': -1,
            'round_pure': None,
            'clean_precision': None,
            'clean_recall': None,
            'noisy_precision': None,
            'noisy_recall': None,
            'keep_ratio': None,
            'avg_teacher_similarity': None
        }

    # =========================================================
    # formal stage training round
    # =========================================================
    def _stage_round_train(self, priloader_list):
        active_stage_id = self.current_stage
        active_metrics = copy.deepcopy(self.current_stage_metrics)

        for client_id in self.online_clients:
            self._train_net_clean_subset(
                client_id,
                self.nets_list[client_id],
                priloader_list[client_id]
            )

        # 阶段最后一轮：先保存本轮本地模型，再为下一阶段构建teacher pool / clean subset，然后再聚合
        if self._is_stage_end_round():
            self.teacher_pool = self._snapshot_local_nets()

            has_next_stage = (self.current_stage + 1) < self.num_stages
            if has_next_stage:
                next_stage_id = self.current_stage + 1
                next_clean_sets, next_stage_metrics = self._select_teachers_and_filter_samples(
                    priloader_list=priloader_list,
                    stage_idx=next_stage_id
                )
            else:
                next_stage_id = None
                next_clean_sets = None
                next_stage_metrics = None

            all_keys = self.get_all_state_keys()
            self.aggregate_nets_by_keys(all_keys)
            self.broadcast_global_by_keys(all_keys)

            if has_next_stage:
                self.current_stage = next_stage_id
                self.current_stage_clean_sets = next_clean_sets
                self.current_stage_metrics = next_stage_metrics

        else:
            all_keys = self.get_all_state_keys()
            self.aggregate_nets_by_keys(all_keys)
            self.broadcast_global_by_keys(all_keys)

        round_pure = active_metrics.get('avg_clean_precision', None)
        if round_pure is not None:
            self.round_pure_ratio.append(float(round_pure))

        return {
            'phase': f'stage_{active_stage_id + 1}',
            'stage_id': active_stage_id,
            'round_pure': round_pure,
            'clean_precision': active_metrics.get('avg_clean_precision', None),
            'clean_recall': active_metrics.get('avg_clean_recall', None),
            'noisy_precision': active_metrics.get('avg_noisy_precision', None),
            'noisy_recall': active_metrics.get('avg_noisy_recall', None),
            'keep_ratio': active_metrics.get('avg_keep_ratio', None),
            'avg_teacher_similarity': active_metrics.get('avg_teacher_similarity', None)
        }

    # =========================================================
    # local training
    # =========================================================
    def _train_net_warmup(self, index, net, train_loader):
        net = net.to(self.device)
        net.train()

        optimizer = optim.SGD(
            net.parameters(),
            lr=self.local_lr,
            momentum=0.9,
            weight_decay=self.args.reg
        )
        criterion = nn.CrossEntropyLoss().to(self.device)

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

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

    def _train_net_clean_subset(self, index, net, train_loader):
        net = net.to(self.device)
        net.train()

        optimizer = optim.SGD(
            net.parameters(),
            lr=self.local_lr,
            momentum=0.9,
            weight_decay=self.args.reg
        )
        criterion = nn.CrossEntropyLoss().to(self.device)

        if self.current_stage_clean_sets is None or index not in self.current_stage_clean_sets:
            selected_clean_set = None
        else:
            selected_clean_set = self.current_stage_clean_sets[index]

        for _ in range(self.local_epoch):
            for batch in train_loader:
                if len(batch) != 3:
                    raise RuntimeError(
                        'FedDenoiseV3 requires train loader batch format: (images, labels, indexes).'
                    )

                images, labels, indexes = batch

                if selected_clean_set is not None:
                    keep_mask_np = np.array(
                        [int(idx) in selected_clean_set for idx in indexes.cpu().numpy()],
                        dtype=bool
                    )
                    if keep_mask_np.sum() == 0:
                        continue
                    keep_mask = torch.from_numpy(keep_mask_np)
                    images = images[keep_mask]
                    labels = labels[keep_mask]

                images = images.to(self.device)
                labels = labels.to(self.device)

                outputs = net(images)
                loss = criterion(outputs, labels)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()