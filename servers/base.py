#!/usr/bin/env python
# coding: utf-8
import copy
import time

import matplotlib.pyplot as plt
import torch.multiprocessing as mp
from sklearn.manifold import TSNE

from utils import *
from utils.metrics import evaluate
from utils.qjl import LayerWiseQJL
from models import build_encoder
from typing import Callable, Dict, Tuple, Union, List

from collections import OrderedDict
import wandb

from servers.build import SERVER_REGISTRY

@SERVER_REGISTRY.register()
class Server():

    def __init__(self, args):
        self.args = args
        return
    
    def aggregate(self, local_weights, local_deltas, client_ids, model_dict, current_lr, epoch=None):
        C = len(client_ids)
        
        if local_deltas is not None and len(local_deltas) > 0:
            grad_var = None
            delta_vecs = []
            for i in range(C):
                flat = []
                for k, ds in local_deltas.items():
                    flat.append(ds[i].reshape(-1))
                delta_vecs.append(torch.cat(flat))
            delta_mat = torch.stack(delta_vecs, dim=0)
            mean_delta = delta_mat.mean(dim=0)
            grad_var = ((delta_mat - mean_delta) ** 2).sum(dim=1).mean()

            # print(f"[Server] epoch={epoch} update_divergence(var)={grad_var.item():.6e}")
                
        for param_key in local_weights:
            local_weights[param_key] = sum(local_weights[param_key])/C
            
        return local_weights, grad_var

@SERVER_REGISTRY.register()
class ServerQJL():

    def __init__(self, args):
        self.args = args
        
        self.qjl_helper = LayerWiseQJL(
            qjl_ratio=getattr(args.server, "qjl_ratio", 1.0),
            use_orthogonal=getattr(args.server, "use_orthogonal", True),
            seed=getattr(args, "seed", 0),
            skip_small_tensors=getattr(args.server, "skip_small_tensors", True),
            small_tensor_threshold=getattr(args.server, "small_tensor_threshold", 256),
            block_size=getattr(args.server, "block_size", 2048),              
            min_m=getattr(args.server, "min_m", 32),                     
            max_m=getattr(args.server, "max_m", 256),        
        )
        return
    
    def aggregate(self, local_weights, local_deltas, client_ids, model_dict, current_lr, epoch=None):
        C = len(client_ids)

        # global delta accumulator
        agg_delta = OrderedDict()
        for param_key in model_dict:
            agg_delta[param_key] = torch.zeros_like(model_dict[param_key])

        # local_deltas[param_key] 안에는 각 client가 보낸 packed update가 들어있다고 가정
        for param_key in local_deltas:
            for packed in local_deltas[param_key]:
                delta_hat_flat = self.qjl_helper.decompress(param_key, packed)

                # raw tensor가 바로 나올 수도 있고, flat tensor가 나올 수도 있으니 shape 맞춤
                if delta_hat_flat.shape != model_dict[param_key].shape:
                    delta_hat = delta_hat_flat.view_as(model_dict[param_key])
                else:
                    delta_hat = delta_hat_flat

                delta_hat = delta_hat.to(
                    device=model_dict[param_key].device,
                    dtype=model_dict[param_key].dtype
                    )

                agg_delta[param_key] += delta_hat / C

        # global model update
        for param_key in model_dict:
            model_dict[param_key] = model_dict[param_key] + agg_delta[param_key]

        return model_dict


@SERVER_REGISTRY.register()
class ServerLOO():

    def __init__(self, args):
        self.args = args
        self.loo_temp = getattr(args.server, "loo_temp", 1.0)
        self.loo_lambda = getattr(args.server, "loo_lambda", 0.2)
        self.loo_w_min = getattr(args.server, "loo_w_min", None)
        self.loo_w_max = getattr(args.server, "loo_w_max", None)
        return
    
    def _average_subset(self, local_weights, selected_idx):

        avg_weights = {}
        n = len(selected_idx)

        if n == 0:
            raise ValueError("selected_idx must contain at least one client.")

        for param_key, tensors in local_weights.items():

            if n == len(tensors):
                avg_weights[param_key] = torch.stack(tensors, dim=0).mean(dim=0)
            else:
                subset = torch.stack([tensors[i] for i in selected_idx], dim=0)
                avg_weights[param_key] = subset.mean(dim=0)

        return avg_weights

    def _weighted_average(self, local_weights, weights):

        weighted_avg = {}
        C = len(weights)

        for param_key, tensors in local_weights.items():
            s = None
            for i in range(C):
                if s is None:
                    s = weights[i] * tensors[i]
                else:
                    s = s + weights[i] * tensors[i]
            weighted_avg[param_key] = s

        return weighted_avg
    
    def _compute_grad_variance(self, local_deltas, C):

        delta_vecs = []

        for i in range(C):
            flat = []
            for _, ds in local_deltas.items():
                flat.append(ds[i].reshape(-1))
            delta_vecs.append(torch.cat(flat, dim=0))

        delta_mat = torch.stack(delta_vecs, dim=0)   # [C, D]
        mean_delta = delta_mat.mean(dim=0)           # [D]
        grad_var = ((delta_mat - mean_delta) ** 2).sum(dim=1).mean()

        return grad_var
    
    def aggregate(self, local_weights, local_deltas, client_ids, model_dict, current_lr, epoch=None):
        C = len(client_ids)

        grad_var = None
        if local_deltas is not None and len(local_deltas) > 0:
            grad_var = self._compute_grad_variance(local_deltas, C)
            # print(f"[Server] epoch={epoch} grad_var={grad_var.item():.6e}")

        fedavg_weights = self._average_subset(local_weights, list(range(C)))

        loo_candidates = []
        for i in range(C):
            subset_idx = [j for j in range(C) if j != i]
            loo_weights = self._average_subset(local_weights, subset_idx)

            loo_candidates.append({
                "removed_client_local_idx": i,
                "removed_client_id": client_ids[i],
                "weights": loo_weights
            })

        return {
            "fedavg_weights": fedavg_weights,
            "loo_candidates": loo_candidates,
            "grad_var": grad_var
        }

    def aggregate_with_contributions(self, local_weights, contributions, epoch=None):

        C = len(contributions)
        if C == 0:
            raise ValueError("contributions must not be empty.")

        contrib = torch.tensor(contributions, dtype=torch.float32)

        # 1) softmax over contributions
        loo_weights = F.softmax(contrib / self.loo_temp, dim=0)

        # 2) uniform(FedAvg)와 interpolation
        uniform_weights = torch.ones(C, dtype=torch.float32) / C
        final_weights = (1.0 - self.loo_lambda) * uniform_weights + self.loo_lambda * loo_weights

        # 3) optional clipping
        if self.loo_w_min is not None:
            final_weights = torch.clamp(final_weights, min=self.loo_w_min)
        if self.loo_w_max is not None:
            final_weights = torch.clamp(final_weights, max=self.loo_w_max)

        # 4) normalize again
        final_weights = final_weights / final_weights.sum()

        weighted_weights = self._weighted_average(local_weights, final_weights)

        # if epoch is not None:
        #     print(f"[Server] epoch={epoch} contribution_weights={final_weights.tolist()}")

        return {
            "weighted_weights": weighted_weights,
            "client_weights": final_weights
        }
        
@SERVER_REGISTRY.register()
class AnalizeServer():

    def __init__(self, args):
        self.args = args
        return
    
    def aggregate(self, local_weights, local_deltas, client_ids, model_dict, current_lr, epoch):
        C = len(client_ids)
        temp = {'scale/mean/conv1': [],
                'scale/mean/layer1': [],
                'scale/mean/layer2': [],
                'scale/mean/layer3': [],
                'scale/mean/layer4': [],
                'scale/std/conv1': [],
                'scale/std/layer1': [],
                'scale/std/layer2': [],
                'scale/std/layer3': [],
                'scale/std/layer4': [],
                'cos/mean/conv1': [],
                'cos/mean/layer1': [],
                'cos/mean/layer2': [],
                'cos/mean/layer3': [],
                'cos/mean/layer4': [],
                'cos/std/conv1': [],
                'cos/std/layer1': [],
                'cos/std/layer2': [],
                'cos/std/layer3': [],
                'cos/std/layer4': [],}
        from torch.nn import CosineSimilarity
        cos = CosineSimilarity(dim=1, eps=1e-10)
        for param_key in local_weights:
            if 'conv' in param_key:
                weight_list = local_weights[param_key]
                delta_list = local_deltas[param_key]
                for (w, d) in zip(weight_list, delta_list):
                    o, i, h_, w_ = w.size()
                    w_prev = w - d
                    # mean/std of cos(prev, curr)
                    # mean/std of std of curr
                    sim = cos(w.view(o, i * h_ * w_), w_prev.view(o, i *  h_ * w_)).abs()
                    sim_mean = sim.mean()
                    sim_var = sim.var()
                    
                    std = w.view(o, -1).std(dim=1)
                    std_mean = std.mean()
                    std_var = std.var()
                    
                    w_mean = w.mean(dim=1, keepdim=True).mean(dim=2, keepdim=True).mean(dim=3, keepdim=True)        
                    w = (w - w_mean).view(o, -1)
                    
                    if param_key == 'conv1.weight':
                        temp['scale/mean/conv1'].append(std_mean.item())
                        temp['scale/std/conv1'].append(std_var.item())
                        temp['cos/mean/conv1'].append(sim_mean.item())
                        temp['cos/std/conv1'].append(sim_var.item())
                    elif 'layer1' in param_key:
                        temp['scale/mean/layer1'].append(std_mean.item())
                        temp['scale/std/layer1'].append(std_var.item())
                        temp['cos/mean/layer1'].append(sim_mean.item())
                        temp['cos/std/layer1'].append(sim_var.item())
                    elif 'layer2' in param_key:
                        temp['scale/mean/layer2'].append(std_mean.item())
                        temp['scale/std/layer2'].append(std_var.item())
                        temp['cos/mean/layer2'].append(sim_mean.item())
                        temp['cos/std/layer2'].append(sim_var.item())
                    elif 'layer3' in param_key:
                        temp['scale/mean/layer3'].append(std_mean.item())
                        temp['scale/std/layer3'].append(std_var.item())
                        temp['cos/mean/layer3'].append(sim_mean.item())
                        temp['cos/std/layer3'].append(sim_var.item())
                    elif 'layer4' in param_key:
                        temp['scale/mean/layer4'].append(std_mean.item())
                        temp['scale/std/layer4'].append(std_var.item())
                        temp['cos/mean/layer4'].append(sim_mean.item())
                        temp['cos/std/layer4'].append(sim_var.item())
                             
        for param_key in local_weights:
            local_weights[param_key] = sum(local_weights[param_key])/C
        
        for name in temp:
            temp[name] = np.mean(temp[name])
            if 'std' in name:
                temp[name] = np.sqrt(temp[name])
        
        # print(temp)
            
        wandb.log(temp, step=epoch)
        # print(temp_total)

        return local_weights
    

@SERVER_REGISTRY.register()
class ServerM(Server):    
    
    def set_momentum(self, model):

        global_delta = copy.deepcopy(model.state_dict())
        for key in global_delta.keys():
            global_delta[key] = torch.zeros_like(global_delta[key])

        global_momentum = copy.deepcopy(model.state_dict())
        for key in global_momentum.keys():
            global_momentum[key] = torch.zeros_like(global_momentum[key])

        self.global_delta = global_delta
        self.global_momentum = global_momentum

    @torch.no_grad()
    def FedACG_lookahead(self, model):
        sending_model_dict = copy.deepcopy(model.state_dict())
        for key in self.global_momentum.keys():
            if 'num_batches_tracked' in key:
                sending_model_dict[key] = self.global_momentum[key]
            else:
                sending_model_dict[key] += self.args.server.momentum * self.global_momentum[key]

        model.load_state_dict(sending_model_dict)
        return copy.deepcopy(model)
    

    def aggregate(self, local_weights, local_deltas, client_ids, model_dict, current_lr, epoch=None):
        C = len(client_ids)
        for param_key in local_weights:
            local_weights[param_key] = sum(local_weights[param_key])/C
            
        if self.args.server.momentum>0:

            if not self.args.server.get('FedACG'): 
                for param_key in local_weights:               
                    local_weights[param_key] += self.args.server.momentum * self.global_momentum[param_key]
                    
            for param_key in local_deltas:
                self.global_delta[param_key] = sum(local_deltas[param_key])/C
                self.global_momentum[param_key] = self.args.server.momentum * self.global_momentum[param_key] + self.global_delta[param_key]

        return local_weights


@SERVER_REGISTRY.register()
class ServerAdam(Server):    
    
    def set_momentum(self, model):

        global_delta = copy.deepcopy(model.state_dict())
        for key in global_delta.keys():
            global_delta[key] = torch.zeros_like(global_delta[key])

        global_momentum = copy.deepcopy(model.state_dict())
        for key in global_momentum.keys():
            global_momentum[key] = torch.zeros_like(global_momentum[key])

        global_v = copy.deepcopy(model.state_dict())
        for key in global_v.keys():
            global_v[key] = torch.zeros_like(global_v[key]) + (self.args.server.tau * self.args.server.tau)

        self.global_delta = global_delta
        self.global_momentum = global_momentum
        self.global_v = global_v
    
    def aggregate(self, local_weights, local_deltas, client_ids, model_dict, current_lr, epoch=None):
        C = len(client_ids)
        server_lr = self.args.trainer.global_lr
        
        for param_key in local_deltas:
            self.global_delta[param_key] = sum(local_deltas[param_key])/C
            self.global_momentum[param_key] = self.args.server.momentum * self.global_momentum[param_key] + (1-self.args.server.momentum) * self.global_delta[param_key]
            self.global_v[param_key] = self.args.server.beta * self.global_v[param_key] + (1-self.args.server.beta) * (self.global_delta[param_key] * self.global_delta[param_key])

        for param_key in model_dict.keys():
            model_dict[param_key] += server_lr *  self.global_momentum[param_key] / ( (self.global_v[param_key]**0.5) + self.args.server.tau)
            
        return model_dict


@SERVER_REGISTRY.register()
class ServerDyn(Server):    
    
    def set_momentum(self, model):
        #global_momentum is h^t in FedDyn paper
        global_delta = copy.deepcopy(model.state_dict())
        for key in global_delta.keys():
            global_delta[key] = torch.zeros_like(global_delta[key])

        global_momentum = copy.deepcopy(model.state_dict())
        for key in global_momentum.keys():
            global_momentum[key] = torch.zeros_like(global_momentum[key])

        self.global_delta = global_delta
        self.global_momentum = global_momentum

    def aggregate(self, local_weights, local_deltas, client_ids, model_dict, current_lr, epoch=None):
        C = len(client_ids)
        for param_key in self.global_momentum:
            self.global_momentum[param_key] -= self.args.client.Dyn.alpha / self.args.trainer.num_clients * sum(local_deltas[param_key])
            local_weights[param_key] = sum(local_weights[param_key])/C - 1/self.args.client.Dyn.alpha * self.global_momentum[param_key]
        return local_weights
    

@SERVER_REGISTRY.register()
class ServerMix(Server):
    def __init__(self, args):
        super().__init__(args)
        self.global_mashed_data = []

    def collect_mashed_data(self, client_mashed_data):
        self.global_mashed_data.extend(client_mashed_data)
        return self.global_mashed_data

    # def broadcast_mashed_data(self):
        # max_samples = getattr(self.args.client.FedMix, "max_mashed", 128) # 샘플수 제한시
        # return random.sample(self.global_mashed_data, min(len(self.global_mashed_data), max_samples))
        
@SERVER_REGISTRY.register()
class ServerFA(Server):

    def __init__(self, args):
        self.args = args
        return
    
    def aggregate(self, local_weights, local_deltas, client_ids, model_dict, current_lr, epoch=None):
        C = len(client_ids)
        eps = 1e-12

        agg = {}
        for k, lst in local_weights.items():
            t = torch.stack(lst, dim=0)  # [M, ...]
            if t.is_floating_point():
                agg[k] = t.mean(dim=0)
            else:
                agg[k] = model_dict[k]

        for key in list(model_dict.keys()):
            if ('running_var_mean_bmic' in key) or ('running_var_std_bmic' in key):
                base_key = key.replace('running_var_', 'running_')  # running_mean/std_bmic 로 대응
                if base_key not in local_weights:
                    agg[key] = model_dict[key]
                    continue

                stack = torch.stack(local_weights[base_key], dim=0)
                stack = stack.to(model_dict[key].device).to(model_dict[key].dtype)
                var_ac = stack.var(dim=0, unbiased=False) + eps
                agg[key] = var_ac

        return agg