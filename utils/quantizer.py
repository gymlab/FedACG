# dff
import torch
import torch.nn as nn
import math
# from typing import Literal
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
import numpy as np
import time
from scipy.stats import shapiro, skew, kurtosis, norm, entropy, wasserstein_distance
from torch.distributions.normal import Normal


def plot_input_distribution(inp: torch.Tensor, title='Input Distribution'):
    inp_flat = inp.detach().cpu().numpy().flatten()
    plt.figure(figsize=(6, 4))
    plt.hist(inp_flat, bins=100, color='skyblue', edgecolor='black')
    plt.title(title)
    plt.xlabel('Value')
    plt.ylabel('Frequency')
    plt.grid(True)
    plt.show()
    
def plot_tensor_distribution(x, title='Weight Distribution'):
    import matplotlib.pyplot as plt

    x_flat = x.detach().cpu().numpy().flatten()
    plt.figure(figsize=(6, 4))
    plt.hist(x_flat, bins=100, color='skyblue', edgecolor='black')
    plt.title(title)
    plt.xlabel('Value')
    plt.ylabel('Frequency')
    plt.grid(True)
    plt.show()
    
def save_grad_output_distribution(grad_output, step=None, save_dir='grad_output_logs', title = None):
    os.makedirs(save_dir, exist_ok=True)

    grad_np = grad_output.detach().cpu().numpy()

    timestamp = time.time()
    prefix = f'step_{step}' if step is not None else f'time_{timestamp:.6f}'

    # np.save(os.path.join(save_dir, f'{prefix}_values.npy'), grad_np)

    plt.figure()
    plt.hist(grad_np.flatten(), bins=100)
    plt.title('Gradient')
    plt.xlabel('Value')
    plt.ylabel('Frequency')
    plt.grid(True)
    plt.savefig(os.path.join(save_dir, f'{prefix}_hist.png'))
    plt.close()


Q_VALUES_TABLE = {
    # 2: torch.tensor([-0.7979, 0.7979]),
    # 3: torch.tensor([-1.224, 0, 0.7646, 1.7242]   ),
    4: torch.tensor([-2.6536, -1.9735, -1.508, -1.149, -0.8337, -0.5439, -0.2686, 0.,
            0.2686, 0.5439, 0.8337, 1.149, 1.508, 1.9735, 2.6536]),
    6: torch.tensor([ -2.099, -1.836, -1.659, -1.524, -1.411, -1.314, -1.228, -1.15, -1.079, -1.013, -0.95, -0.892, -0.836, -0.783, -0.732, -0.682, -0.635, -0.589, -0.544, -0.5,
    -0.457, -0.414, -0.373, -0.332, -0.292, -0.252, -0.213, -0.174, -0.135, -0.096,-0.058, -0.019, 0.019, 0.058, 0.096, 0.135, 0.174, 0.213, 0.252, 0.292,
    0.332, 0.373, 0.414, 0.457, 0.5, 0.544, 0.589, 0.635, 0.682, 0.732, 0.783, 0.836, 0.892, 0.95, 1.013, 1.079, 1.15, 1.228, 1.314, 1.411, 1.524, 
    1.659, 1.836, 2.099]),
    8: torch.tensor([-2.418, -2.154, -1.987, -1.863, -1.762, -1.676, -1.601, -1.534, -1.473, -1.418, -1.366, -1.318, -1.273, -1.23, -1.189, -1.15, -1.113, -1.078, -1.043, -1.01,
                     -0.978, -0.947, -0.917, -0.887, -0.858, -0.831, -0.803, -0.776, -0.75, -0.725, -0.699, -0.674, -0.65, -0.626, -0.602, -0.579, -0.556, -0.533, -0.511, -0.489,
                     -0.467, -0.445, -0.424, -0.402, -0.381, -0.36, -0.339, -0.319, -0.298, -0.278, -0.257, -0.237, -0.217, -0.197, -0.177, -0.157, -0.138, -0.118, -0.098, -0.078,
                     -0.059, -0.039, -0.02, 0, 0.019, 0.039, 0.058, 0.077, 0.097, 0.116, 0.135, 0.155, 0.174, 0.194, 0.214, 0.233, 0.253, 0.273, 0.293, 0.314, 0.334, 0.354, 0.375,
                     0.396, 0.417, 0.438, 0.459, 0.481, 0.502, 0.524, 0.547, 0.569, 0.592, 0.615, 0.639, 0.662, 0.687, 0.711, 0.736, 0.762, 0.788, 0.814, 0.842, 0.869, 0.898, 0.927,
                     0.957, 0.988, 1.02, 1.053, 1.087, 1.123, 1.16, 1.198, 1.239, 1.282, 1.327, 1.375, 1.426, 1.482, 1.542, 1.609, 1.683, 1.769, 1.87, 1.994, 2.16, 2.423])
}
class BlockRounding(torch.autograd.Function):
    @staticmethod
    def forward(self, x, forward_bits, backward_bits, mode, small_block="None", block_dim="B", quant_flag = "DANUQ"):
        self.backward_bits = backward_bits
        self.mode = mode
        self.quant_flag = quant_flag
        self.small_block = small_block
        self.block_dim = block_dim
        if forward_bits == -1: return x
        
        
        
        if self.quant_flag == "DANUQ":
            return DANUQ_quantize(x, forward_bits, self.mode, small_block=self.small_block, block_dim=self.block_dim)
        elif self.quant_flag == "BFP":
            return block_quantize(x, forward_bits, self.mode, small_block=self.small_block, block_dim=self.block_dim)
        
    
    @staticmethod
    def backward(self, grad_output):
        
        if self.needs_input_grad[0]:
            if self.backward_bits != -1:
                # layer_name = self.__class__.__name__
                # save_grad_output_distribution(grad_output, title= layer_name)
                
                # grad_input = block_quantize(grad_output, self.backward_bits, self.mode,
                #                              small_block=self.small_block, block_dim=self.block_dim)
                if self.quant_flag == "BFP":
                    grad_input = block_quantize(grad_output, self.backward_bits, self.mode,
                                             small_block=self.small_block, block_dim=self.block_dim)
    
                else:
                    grad_input = bucket_quantize_blockwise_mask_zero(grad_output, self.backward_bits, self.mode, small_block=self.small_block, block_dim=self.block_dim)
                
                # if self.quant_flag == "BFP":
                #     grad_input = block_quantize(grad_output, self.backward_bits, self.mode,
                #                              small_block=self.small_block, block_dim=self.block_dim)
                    
                # elif self.quant_flag == "DANUQ":
                #     grad_input = DANUQ_quantize(grad_output, self.backward_bits, self.mode,
                #                             small_block=self.small_block, block_dim=self.block_dim)
                
            else:
                grad_input = grad_output
        return grad_input, None, None, None, None, None, None
    

class BlockRounding_ReLU(torch.autograd.Function):
    @staticmethod
    def forward(self, x, forward_bits, backward_bits, mode, mean, std, small_block="None", block_dim="B", quant_flag = "DANUQ"):
        self.backward_bits = backward_bits
        self.mode = mode
        self.mean = mean
        self.std = std
        self.small_block = small_block
        self.block_dim = block_dim
        if forward_bits == -1: return x
        
        self.quant_flag = quant_flag
        
        return DANUQ_ReLU_quantize(x, forward_bits, self.mode, self.mean, self.std, small_block=self.small_block, block_dim=self.block_dim)
        
    
    @staticmethod
    def backward(self, grad_output):
        if self.needs_input_grad[0]:
            if self.backward_bits != -1:
                # layer_name = self.__class__.__name__
                # save_grad_output_distribution(grad_output, title= layer_name)
                # grad_input = DANUQ_quantize(grad_output, self.backward_bits, self.mode, small_block=self.small_block, block_dim=self.block_dim)
                # grad_input = block_quantize(grad_output, self.backward_bits, self.mode,
                #                              small_block=self.small_block, block_dim=self.block_dim)
                grad_input = bucket_quantize_blockwise_mask_zero(grad_output, self.backward_bits, self.mode, small_block=self.small_block, block_dim=self.block_dim)
                
            else:
                grad_input = grad_output
                
        return grad_input, None, None, None, None, None, None, None, None
    
class BlockQuantizer(nn.Module):
    def __init__(self, wl_activate, wl_error, mode,
            small_block="None", block_dim="B", quant_flag = "BFP"):
        super(BlockQuantizer, self).__init__()
        self.wl_activate = wl_activate
        self.wl_error = wl_error
        self.mode = mode
        self.small_block="None"
        self.block_dim="B"
        self.quant_flag= quant_flag
    def forward(self, x):
        return quantize_block(x, self.wl_activate,
                              self.wl_error, self.mode,
                              self.small_block, self.block_dim, self.quant_flag)
        
class BlockQuantizer_ReLU(nn.Module):
    def __init__(self, wl_activate, wl_error, mode,
            small_block="None", block_dim="B", quant_flag = "DANUQ"):
        super(BlockQuantizer_ReLU, self).__init__()
        self.wl_activate = wl_activate
        self.wl_error = wl_error
        self.mode = mode
        self.small_block="None"
        self.block_dim="B"
        self.quant_flag= quant_flag
    def forward(self, x, mean, std):
        return quantize_block_ReLU(x, self.wl_activate,
                              self.wl_error, self.mode, mean, std,
                              self.small_block, self.block_dim, self.quant_flag)
        
def block_quantize(data, bits, mode, ebit=8, small_block="FC", block_dim="B"):
    with torch.no_grad():
        data_h = data.clone().clamp(min=1e-10)
        data_l = data.clone().clamp(max=-1e-10)
        data = torch.where(data>=0, data_h, data_l)
    assert data.dim() <= 4
    if small_block == "Conv":
        dim_threshold = 2
    elif small_block == "FC":
        dim_threshold = 1
    elif small_block == "None":
        dim_threshold = 4
    else:
        raise ValueError("Invalid small block option {}".format(small_block))
    if data.dim() <= dim_threshold:
        max_entry = torch.max(torch.abs(data)).item()
        if max_entry == 0: return data
        try:
            max_exponent = math.floor(math.log2(max_entry))
            max_exponent = float(min(max(max_exponent, -2 ** (ebit - 1)), 2 ** (ebit - 1) - 1))
        except (OverflowError, ValueError):
            print(f"max_entru: {max_entry:e}")
            print(f'there is inf in data: {torch.isinf(data).any()}')
            print(f'there is nan in data: {torch.isnan(data).any()}')
            raise
    else:
        if block_dim == "B":    # Better
            max_entry = torch.max(torch.abs(data.view(data.size(0), -1)), 1)[0]
            max_exponent = torch.floor(torch.log2(max_entry))
            max_exponent = torch.clamp(max_exponent, -2 ** (ebit - 1), 2 ** (ebit - 1) - 1)
            max_exponent = max_exponent.view([data.size(0)] + [1 for _ in range(data.dim() - 1)])
            
        elif block_dim == "BC":     # (8+4*9)/72
            max_entry = torch.max(torch.abs(data.view(data.size(0) * data.size(1), -1)), 1)[0]
            max_exponent = torch.floor(torch.log2(max_entry))
            max_exponent = torch.clamp(max_exponent, -2 ** (ebit - 1), 2 ** (ebit - 1) - 1)
            max_exponent = max_exponent.view([data.size(0), data.size(1)] + [1 for _ in range(data.dim() - 2)])
        else:
            raise ValueError("invalid block dim option {}".format(block_dim))
        
    i_temp = 2 ** (-max_exponent + (bits - 2))
    i = data * i_temp
    if mode == "stochastic":
        add_r_(i)
        i.floor_()
    elif mode == "nearest":
        i.round_()
    i.clamp_(-2 ** (bits - 1), 2 ** (bits - 1) - 1)
    try:
        temp = i * 2 ** (max_exponent - (bits - 2))
    except RuntimeError:
        print(type(i))
        print(block_dim)
        print(type(max_exponent))
        print(data.size())
        print(small_block)
        raise
    return temp

def DANUQ_ReLU_quantize(
        x: torch.Tensor,
        bits: int,
        mode: str,
        mean: torch.Tensor,
        std: torch.Tensor,
        ebit: int = 8,
        small_block: str = "FC",
        block_dim: str = "B",
        sigma_clip: float = torch.tensor(2.1)) -> torch.Tensor:

    n_levels = 2 ** bits                      
    norm = Normal(0.0, 1.0)

    z0       = - mean / (std + 1e-10)   
    cdf_0    = norm.cdf(z0)                    
    cdf_max  = norm.cdf(sigma_clip)
    # cdf_max = 1           
    pos_mass = cdf_max - cdf_0

    target = cdf_0 + pos_mass * torch.linspace(
        1/(n_levels-1), 1.0, steps=n_levels-1, device=x.device) # Can be fixed

    z_vals        = norm.icdf(target)        
    q_values_data = torch.cat(
        [torch.zeros(1, device=x.device),   
         z_vals * std + mean])

    edges_data = 0.5 * (q_values_data[1:] + q_values_data[:-1])

    indices = torch.bucketize(x, edges_data, right=False)
    x_q = q_values_data[indices]

    x_q = torch.clamp(x_q, min=q_values_data[0].item(),
                            max=q_values_data[-1].item())
    return x_q

def DANUQ_quantize(x: torch.Tensor,
                   bits: int,
                   mode: str,
                   ebit: int = 8,
                   small_block: str = "FC",
                   block_dim: str = "B") -> torch.Tensor:
    q_values = Q_VALUES_TABLE[bits].to(x.device)
    q_values_sorted, _ = torch.sort(q_values)
    edges = 0.5 * (q_values_sorted[1:] + q_values_sorted[:-1])
    edges = edges.to(x.device)
    
    if small_block == "Conv":
        dim_threshold = 2
    elif small_block == "FC":
        dim_threshold = 1
    elif small_block == "None":
        dim_threshold = 4
    else:
        raise ValueError(f"Invalid small_block option: {small_block}")
    
    def bucket_quantize_blockwise(inp) -> torch.Tensor:

        x_mean = inp.mean()
        x_std = torch.clamp(inp.std(unbiased = False), min=1e-10)  # row_std = row_std + 1e-6
        q_values_data = q_values * x_std + x_mean
        edges_data = 0.5 * (q_values_data[1:] + q_values_data[:-1])
        indices = torch.bucketize(inp, edges_data, right=False)
        x_q = q_values_data[indices]
        x_min2 = x.min()
        x_max2 = x.max()
        q_min = q_values_data[q_values_data > x_min2].min()
        q_max = q_values_data[q_values_data < x_max2].max()
        
        x_q = torch.clamp(x_q, min=q_min.item(), max=q_max.item())

        return x_q
    
    if x.dim() <= dim_threshold:
        return bucket_quantize_blockwise(x)
    
    else:
        if block_dim == "B":
            # B = x.size(0)
            # x_2d = x.view(B, -1)
            # row_mean = x_2d.mean(dim=1, keepdim=True)
            # row_std = x_2d.std(dim=1, keepdim=True, unbiased = True) + 1e-10   # unbiased = True, row_std = row_std + 1e-6
            # x_normed = (x_2d - row_mean) / row_std
            # indices = torch.bucketize(x_normed, edges, right=False)
            # quant_normed = q_values_sorted[indices]
            # x_deq_2d = quant_normed * row_std + row_mean
            # return x_deq_2d.view_as(x)
        
            B = x.size(0)
            x_2d = x.view(B, -1)
            
            ### new_q_values
            row = q_values_sorted.numel()
            
            row_mean = x_2d.mean(dim=1, keepdim=True)
            row_std = torch.clamp(x_2d.std(dim=1, keepdim=True, unbiased = False), min=1e-10)   # unbiased = True, row_std = row_std + 1e-6
            # x_normed = (x_2d - row_mean) / row_std
                                    
            new_q_values = q_values_sorted.view(1, row) * row_std + row_mean
            new_edges = 0.5 * (new_q_values[:, 1:] + new_q_values[:, :-1])
            
            x_q = torch.empty_like(x_2d)
            for b in range(B):
                idx = torch.bucketize(x_2d[b], new_edges[b], right=False)
                x_q[b] = new_q_values[b][idx]
                q_min = new_q_values[b][new_q_values[b] >= x_2d[b].min()].min()
                q_max = new_q_values[b][new_q_values[b] <= x_2d[b].max()].max()
                x_q[b] = torch.clamp(x_q[b], min=q_min.item(), max=q_max.item())
                
            return x_q.view_as(x)

        elif block_dim == "BC":
            # plot_tensor_distribution(x, title='Conv Weight Distribution')
            # analyze_normality(x)
            B, C = x.size(0), x.size(1)
            x_2d = x.view(B*C, -1)
            row_mean = x_2d.mean(dim=1, keepdim=True)
            row_std = x_2d.std(dim=1, keepdim=True, unbiased = True) + 1e-10   # unbiased = True
            # print(min(row_std), x.shape)
            zero     = row_std == 0
            row_std_safe = torch.where(zero, torch.ones_like(row_std), row_std)
            # row_std = torch.clamp(row_std, min=1e-10)
            x_normed = torch.where(zero, torch.zeros_like(x_2d), (x_2d - row_mean)/row_std_safe)
            # x_normed = (x_2d - row_mean) / row_std
            indices = torch.bucketize(x_normed, edges, right=False)
            quant_normed = q_values_sorted[indices]
            x_deq_2d = quant_normed * row_std_safe + row_mean
            # plot_tensor_distribution(x_deq_2d, title='Conv Weight Distribution (64x64x3x3)')
            return x_deq_2d.view_as(x)
        
        else:
            raise ValueError("Invalid block_dim option: {}".format(block_dim))
        

def StochNormQuant_parallel(x: torch.Tensor,
                            bits: int,
                            small_block: str = "FC",
                            block_dim: str = "B") -> torch.Tensor:
    levels = 2 ** (bits - 1)       
    eps = 1e-12

    if small_block == "Conv":
        dim_threshold = 2
    elif small_block == "FC":
        dim_threshold = 1
    elif small_block == "None":
        dim_threshold = 4
    else:
        raise ValueError

    if x.dim() <= dim_threshold:      
        flat = x.view(1, -1)
        reshape_fn = lambda t: t.view_as(x)
    else:
        if block_dim == "B":
            B = x.size(0)
            flat = x.view(B, -1)                
            reshape_fn = lambda t: t.view_as(x)
        elif block_dim == "BC":
            B, C = x.size(0), x.size(1)
            flat = x.view(B * C, -1)            
            reshape_fn = lambda t: t.view_as(x)
        else:
            raise ValueError(f"invalid block_dim: {block_dim}")


    norm = torch.linalg.norm(flat, dim=1, keepdim=True)         
    zero_mask = norm < eps
    safe_norm = torch.where(zero_mask, torch.ones_like(norm), norm)

    abs_ratio = (flat.abs() / safe_norm).clamp_(0, 1)           
    scaled = abs_ratio * levels
    lower = torch.floor(scaled)
    upper = torch.clamp(lower + 1, max=levels)
    p_upper = (scaled - lower)
    rand_u = torch.rand_like(p_upper)
    sel = torch.where(rand_u < p_upper, upper, lower)         
    sel_level = sel / levels

    quant = safe_norm * flat.sign() * sel_level                
    quant[zero_mask.expand_as(quant)] = 0.                        

    return reshape_fn(quant)


def bucket_quantize_blockwise_mask_zero(x, bits, mode, ebit=8, small_block="FC", block_dim="B"):

    nonzero_mask = x.ne(0)

    x_nz = x[nonzero_mask]                 

    if x_nz.numel() == 0:                 
        return x.clone()

    x_mean = x_nz.mean()
    x_std  = torch.clamp(x_nz.std(unbiased=False), min=1e-10)
    
    q_values = Q_VALUES_TABLE[bits].to(x.device)
    
    q_values_data = q_values * x_std + x_mean
    edges_data    = 0.5 * (q_values_data[1:] + q_values_data[:-1])

    indices = torch.bucketize(x_nz, edges_data, right=False)
    x_q_nz  = q_values_data[indices]

    q_min = q_values_data[q_values_data >  x_nz.min()].min()
    q_max = q_values_data[q_values_data <  x_nz.max()].max()
    x_q_nz = torch.clamp(x_q_nz, min=q_min.item(), max=q_max.item())

    x_q = x.clone()          
    x_q[nonzero_mask] = x_q_nz 

    return x_q


def add_r_(data):
    r = torch.rand_like(data)
    data.add_(r)
    
quantize_block = BlockRounding.apply
quantize_block_ReLU = BlockRounding_ReLU.apply
