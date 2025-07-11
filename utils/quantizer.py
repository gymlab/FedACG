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
    
def plot_tensor_distribution(x, title='Weight Distribution', step=None, save_dir='/home/gymlab/projects/FedACG-1/fig/weight_hybrid_t50%'):
    os.makedirs(save_dir, exist_ok=True)

    timestamp = time.time()
    prefix = f'step_{step}' if step is not None else f'time_{timestamp:.6f}'
    
    x_flat = x.detach().cpu().numpy().flatten()
    plt.figure(figsize=(6, 4))
    plt.hist(x_flat, bins=200, color='skyblue', edgecolor='black')
    plt.title(title)
    plt.xlabel('Value')
    plt.ylabel('Frequency')
    plt.grid(True)
    plt.savefig(os.path.join(save_dir, f'{prefix}_{title}.png'))
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
    # 6: torch.tensor([ -2.099, -1.836, -1.659, -1.524, -1.411, -1.314, -1.228, -1.15, -1.079, -1.013, -0.95, -0.892, -0.836, -0.783, -0.732, -0.682, -0.635, -0.589, -0.544, -0.5,
    # -0.457, -0.414, -0.373, -0.332, -0.292, -0.252, -0.213, -0.174, -0.135, -0.096, -0.058, -0.019, 0.019, 0.058, 0.096, 0.135, 0.174, 0.213, 0.252, 0.292,
    # 0.332, 0.373, 0.414, 0.457, 0.5, 0.544, 0.589, 0.635, 0.682, 0.732, 0.783, 0.836, 0.892, 0.95, 1.013, 1.079, 1.15, 1.228, 1.314, 1.411, 1.524, 
    # 1.659, 1.836, 2.099]),
    
    6: torch.tensor([-2.154, -1.863, -1.676, -1.534, -1.418, -1.318, -1.23, -1.15, -1.078, -1.01,
    -0.947, -0.887, -0.831, -0.776, -0.725, -0.674, -0.626, -0.579, -0.533, -0.489,
    -0.445, -0.402, -0.36, -0.319, -0.278, -0.237, -0.197, -0.157, -0.118, -0.078,
    -0.039, 0, 0.038, 0.076, 0.114, 0.153, 0.191, 0.23, 0.269,
    0.309, 0.349, 0.389, 0.431, 0.473, 0.516, 0.56, 0.605, 0.651, 0.699,
    0.748, 0.799, 0.852, 0.908, 0.967, 1.03, 1.097, 1.169, 1.248, 1.335,
    1.434, 1.55, 1.691, 1.876, 2.166]),
    
    # 6: torch.tensor([-1.534, -1.418, -1.318, -1.23, -1.15, -1.078, -1.01,
    # -0.947, -0.887, -0.831, -0.776, -0.725, -0.674, -0.626, -0.579, -0.533, -0.489,
    # -0.445, -0.402, -0.36, -0.319, -0.278, -0.237, -0.197, -0.157, -0.118, -0.078,
    # -0.039, 0, 0.038, 0.076, 0.114, 0.153, 0.191, 0.23, 0.269,
    # 0.309, 0.349, 0.389, 0.431, 0.473, 0.516, 0.56, 0.605, 0.651, 0.699,
    # 0.748, 0.799, 0.852, 0.908, 0.967, 1.03, 1.097, 1.169, 1.248, 1.335,
    # 1.434, 1.55]),
    
#     # 10% -> 양쪽 3개씩 제외한 58개 levels
    # 6 : torch.tensor([-1.3200, -1.2400, -1.1700, -1.1000, -1.0300, -0.9800, -0.9200, -0.8700,
    #     -0.8100, -0.7700, -0.7200, -0.6700, -0.6300, -0.5900, -0.5400, -0.5000,
    #     -0.4600, -0.4200, -0.3800, -0.3500, -0.3100, -0.2700, -0.2300, -0.2000,
    #     -0.1600, -0.1300, -0.0900, -0.0500, -0.0200,  0.0200,  0.0500,  0.0900,
    #      0.1300,  0.1600,  0.2000,  0.2300,  0.2700,  0.3100,  0.3500,  0.3800,
    #      0.4200,  0.4600,  0.5000,  0.5400,  0.5900,  0.6300,  0.6700,  0.7200,
    #      0.7700,  0.8100,  0.8700,  0.9200,  0.9800,  1.0300,  1.1000,  1.1700,
    #      1.2400,  1.3200]),
    
    # # 20% -> 양쪽 6개씩 제외한 52개 levels
    # 6: torch.tensor([-1.3200, -1.2300, -1.1500, -1.0700, -1.0100, -0.9400, -0.8800, -0.8200,
    #     -0.7700, -0.7200, -0.6700, -0.6200, -0.5700, -0.5200, -0.4800, -0.4300,
    #     -0.3900, -0.3500, -0.3000, -0.2600, -0.2200, -0.1800, -0.1400, -0.1000,
    #     -0.0600, -0.0200,  0.0200,  0.0600,  0.1000,  0.1400,  0.1800,  0.2200,
    #      0.2600,  0.3000,  0.3500,  0.3900,  0.4300,  0.4800,  0.5200,  0.5700,
    #      0.6200,  0.6700,  0.7200,  0.7700,  0.8200,  0.8800,  0.9400,  1.0100,
    #      1.0700,  1.1500,  1.2300,  1.3200]),
    
    # 30% -> 양쪽 9개씩 제외한 46개 levels
    # 6: torch.tensor([-1.3200, -1.2200, -1.1300, -1.0500, -0.9700, -0.9000, -0.8400, -0.7700,
    #     -0.7100, -0.6600, -0.6000, -0.5500, -0.5000, -0.4400, -0.4000, -0.3500,
    #     -0.3000, -0.2500, -0.2100, -0.1600, -0.1100, -0.0700, -0.0200,  0.0200,
    #      0.0700,  0.1100,  0.1600,  0.2100,  0.2500,  0.3000,  0.3500,  0.4000,
    #      0.4400,  0.5000,  0.5500,  0.6000,  0.6600,  0.7100,  0.7700,  0.8400,
    #      0.9000,  0.9700,  1.0500,  1.1300,  1.2200,  1.3200]),
    
    # 40% -> 양쪽 12개씩 제외한 40개 levels
    # 6: torch.tensor([-1.3200, -1.2000, -1.1000, -1.0100, -0.9300, -0.8500, -0.7800, -0.7100,
    #     -0.6400, -0.5800, -0.5200, -0.4600, -0.4000, -0.3500, -0.2900, -0.2400,
    #     -0.1800, -0.1300, -0.0800, -0.0300,  0.0300,  0.0800,  0.1300,  0.1800,
    #      0.2400,  0.2900,  0.3500,  0.4000,  0.4600,  0.5200,  0.5800,  0.6400,
    #      0.7100,  0.7800,  0.8500,  0.9300,  1.0100,  1.1000,  1.2000,  1.3200]),
    
    # 50% -> 양쪽 15개씩 제외한 34개 levels
    # 6 : torch.tensor([-1.3200, -1.1800, -1.0700, -0.9600, -0.8700, -0.7800, -0.7000, -0.6300,
    #     -0.5500, -0.4800, -0.4100, -0.3500, -0.2800, -0.2200, -0.1600, -0.0900,
    #     -0.0300,  0.0300,  0.0900,  0.1600,  0.2200,  0.2800,  0.3500,  0.4100,
    #      0.4800,  0.5500,  0.6300,  0.7000,  0.7800,  0.8700,  0.9600,  1.0700,
    #      1.1800,  1.3200]),
    

    8: torch.tensor([-2.418, -2.154, -1.987, -1.863, -1.762, -1.676, -1.601, -1.534, -1.473, -1.418, -1.366, -1.318, -1.273, -1.23, -1.189, -1.15, -1.113, -1.078, -1.043, -1.01,
                     -0.978, -0.947, -0.917, -0.887, -0.858, -0.831, -0.803, -0.776, -0.75, -0.725, -0.699, -0.674, -0.65, -0.626, -0.602, -0.579, -0.556, -0.533, -0.511, -0.489,
                     -0.467, -0.445, -0.424, -0.402, -0.381, -0.36, -0.339, -0.319, -0.298, -0.278, -0.257, -0.237, -0.217, -0.197, -0.177, -0.157, -0.138, -0.118, -0.098, -0.078,
                     -0.059, -0.039, -0.02, 0, 0.019, 0.039, 0.058, 0.077, 0.097, 0.116, 0.135, 0.155, 0.174, 0.194, 0.214, 0.233, 0.253, 0.273, 0.293, 0.314, 0.334, 0.354, 0.375,
                     0.396, 0.417, 0.438, 0.459, 0.481, 0.502, 0.524, 0.547, 0.569, 0.592, 0.615, 0.639, 0.662, 0.687, 0.711, 0.736, 0.762, 0.788, 0.814, 0.842, 0.869, 0.898, 0.927,
                     0.957, 0.988, 1.02, 1.053, 1.087, 1.123, 1.16, 1.198, 1.239, 1.282, 1.327, 1.375, 1.426, 1.482, 1.542, 1.609, 1.683, 1.769, 1.87, 1.994, 2.16, 2.423])
}

LUT = {
    
    'E3M2' : torch.tensor([-28.0000, -24.0000, -20.0000, -16.0000, -14.0000, -12.0000, -10.0000,
         -8.0000,  -7.0000,  -6.0000,  -5.0000,  -4.0000,  -3.5000,  -3.0000,
         -2.5000,  -2.0000,  -1.7500,  -1.5000,  -1.2500,  -1.0000,  -0.8750,
         -0.7500,  -0.6250,  -0.5000,  -0.4375,  -0.3750,  -0.3125,  -0.2500,
         -0.2188,  -0.1875,  -0.1562,   0.0000,   0.1562,   0.1875,   0.2188,
          0.2500,   0.3125,   0.3750,   0.4375,   0.5000,   0.6250,   0.7500,
          0.8750,   1.0000,   1.2500,   1.5000,   1.7500,   2.0000,   2.5000,
          3.0000,   3.5000,   4.0000,   5.0000,   6.0000,   7.0000,   8.0000,
          10.0000,  12.0000,  14.0000,  16.0000,  20.0000,  24.0000,  28.0000]),

    'E2M3' : torch.tensor([-7.5000, -7.0000, -6.5000, -6.0000, -5.5000, -5.0000, -4.5000,
            -4.0000, -3.7500, -3.5000, -3.2500, -3.0000, -2.7500, -2.5000, -2.2500,
        -2.0000, -1.8750, -1.7500, -1.6250, -1.5000, -1.3750, -1.2500, -1.1250,
        -1.0000, -0.9375, -0.8750, -0.8125, -0.7500, -0.6875, -0.6250, -0.5625,
        0.0000,  0.5625,  0.6250,  0.6875,  0.7500,  0.8125,  0.8750,  0.9375,
        1.0000,  1.1250,  1.2500,  1.3750,  1.5000,  1.6250,  1.7500,  1.8750,
        2.0000,  2.2500,  2.5000,  2.7500,  3.0000,  3.2500,  3.5000,  3.7500,
        4.0000,  4.5000,  5.0000,  5.5000,  6.0000,  6.5000,  7.0000,  7.5000]),

    'E1M4' : torch.tensor([-3.8750, -3.7500, -3.6250, -3.5000, -3.3750, -3.2500, -3.1250,
                -3.0000, -2.8750, -2.7500, -2.6250, -2.5000, -2.3750, -2.2500, -2.1250,
                -2.0000, -1.9375, -1.8750, -1.8125, -1.7500, -1.6875, -1.6250, -1.5625,
                -1.5000, -1.4375, -1.3750, -1.3125, -1.2500, -1.1875, -1.1250, -1.0000,
            0.0000,  1.0000,  1.1250,  1.1875,  1.2500,  1.3125,  1.3750,  1.4375,
            1.5000,  1.5625,  1.6250,  1.6875,  1.7500,  1.8125,  1.8750,  1.9375,
            2.0000,  2.1250,  2.2500,  2.3750,  2.5000,  2.6250,  2.7500,  2.8750,
            3.0000,  3.1250,  3.2500,  3.3750,  3.5000,  3.6250,  3.7500,  3.8750]),
    
    'E3M1' : torch.tensor([-24.0000, -16.0000, -12.0000,  -8.0000,  -6.0000,  -4.0000,  -3.0000,
         -2.0000,  -1.5000,  -1.0000,  -0.7500,  -0.5000,  -0.3750,  -0.2500,
         -0.1250,   0.0000,   0.1250,   0.2500,   0.3750,   0.5000,
          0.7500,   1.0000,   1.5000,   2.0000,   3.0000,   4.0000,   6.0000,
          8.0000,  12.0000,  16.0000,  24.0000]),
    
    'E2M2' : torch.tensor([-7.0000, -6.0000, -5.0000, -4.0000, -3.5000, -3.0000, -2.5000, -2.0000,
        -1.7500, -1.5000, -1.2500, -1.0000, -0.8750, -0.6250, -0.5000,   0.0000,
         0.5000,  0.6250,  0.8750,  1.0000,  1.2500,  1.5000,  1.7500,
         2.0000,  2.5000,  3.0000,  3.5000,  4.0000,  5.0000,  6.0000,  7.0000]),

    'E1M3' : torch.tensor([-3.7500, -3.5000, -3.2500, -3.0000, -2.7500, -2.5000, -2.2500, -2.0000,
        -1.8750, -1.6250, -1.5000, -1.3750, -1.2500, -1.1250, -1.0000,  0.0000,
         1.0000,  1.1250,  1.2500,  1.3750,  1.5000,  1.6250,  1.8750,
         2.0000,  2.2500,  2.5000,  2.7500,  3.0000,  3.2500,  3.5000,  3.7500]),
    
    'E1M2' : torch.tensor([-3.5000, -3.0000, -2.5000, -2.0000, -1.5000, -1.0000, -0.5000,
         0.0000, 0.5000, 1.0000, 1.5000, 2.0000,  2.5000,  3.0000,  3.5000]),

    'E2M1' : torch.tensor([-6.0000, -4.0000, -3.0000, -2.0000, -1.5000, -1.0000, -0.5000,
          0.0000, 0.5000,  0.7500,  1.0000,  1.5000,  2.0000,  3.0000,  4.0000,  6.0000]),
    
    'E3M0' : torch.tensor([-16.0000,  -8.0000,  -4.0000,  -2.0000,  -1.0000,  -0.5000,  -0.2500,
         0.0000,  0.2500,   0.5000,   1.0000,   2.0000,   4.0000,  8.0000,  16.0000])
    
    

}

class BlockRounding(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, forward_bits, backward_bits, mode,
                small_block="None", block_dim="B", quant_flag = "DANUQ", lut_mode="E3M2", comp=False, k=5.0):
        ctx.backward_bits = backward_bits
        ctx.mode = mode
        ctx.quant_flag = quant_flag
        ctx.small_block = small_block
        ctx.block_dim = block_dim
        # DGE용
        ctx.k = k
        ctx.lut_mode = lut_mode
        ctx.comp = comp
        if lut_mode not in LUT:
            raise ValueError(f"Unsupported quantization format: {lut_mode}")
        lut = LUT[lut_mode]
        max_lut = torch.max(torch.abs(lut))
        ctx.delta = max_lut.item() / (2 ** forward_bits - 1)
        ctx.save_for_backward(x)
        if forward_bits == -1:
            return x
        if ctx.quant_flag == "DANUQ":
            return DANUQ_quantize(x, forward_bits, ctx.mode, small_block=ctx.small_block, block_dim=ctx.block_dim)
        elif ctx.quant_flag == "BFP":
            return block_quantize(x, forward_bits, ctx.mode, small_block=ctx.small_block, block_dim=ctx.block_dim)
        elif ctx.quant_flag == "occ":
            return occ(x, forward_bits, ctx.mode, small_block=ctx.small_block, block_dim=ctx.block_dim, lut_mode=ctx.lut_mode, comp=ctx.comp)
    @staticmethod
    def backward(ctx, grad_output):
        x, = ctx.saved_tensors
        if ctx.needs_input_grad[0]:
            if ctx.backward_bits != -1:
                if ctx.quant_flag == "DANUQ":
                    grad_input = block_quantize(grad_output, ctx.backward_bits, ctx.mode,
                                    small_block=ctx.small_block, block_dim=ctx.block_dim)
                elif ctx.quant_flag == "BFP":
                    grad_input = block_quantize(grad_output, ctx.backward_bits, ctx.mode,
                                    small_block=ctx.small_block, block_dim=ctx.block_dim)
                elif ctx.quant_flag == "occ":
                    grad_input = occ(grad_output, ctx.backward_bits, ctx.mode,
                                    small_block=ctx.small_block, block_dim=ctx.block_dim, lut_mode=ctx.lut_mode, comp=ctx.comp)
                # DGE gradient
                # delta = ctx.delta
                # k = ctx.k
                # x_scaled = x
                # grad_est = (1 / k) * torch.pow(torch.abs(x_scaled - delta / 2) + 1e-6, (1 / k - 1))
                # grad_est = torch.clamp(grad_est, max=3.0)
                # grad_input = grad_output * grad_est
            else:
                grad_input = grad_output
        return grad_input, None, None, None, None, None, None, None, None
    
class BlockRounding_ReLU(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, forward_bits, backward_bits, mode, mean, std,
                small_block="None", block_dim="B", quant_flag="DANUQ",
                lut_mode="E3M2", comp=False, k=5.0):
        ctx.backward_bits = backward_bits
        ctx.mode = mode
        ctx.mean = mean
        ctx.std = std
        ctx.small_block = small_block
        ctx.block_dim = block_dim
        ctx.quant_flag = quant_flag
        ctx.k = k
        ctx.lut_mode = lut_mode
        ctx.comp = comp
        if lut_mode not in LUT:
            raise ValueError(f"Unsupported quantization format: {lut_mode}")
        lut = LUT[lut_mode]
        max_lut = torch.max(torch.abs(lut))
        ctx.delta = max_lut.item() / (2 ** forward_bits - 1)
        ctx.save_for_backward(x)
        if forward_bits == -1:
            return x
        # return DANUQ_ReLU_quantize(x, forward_bits, mode, mean, std, small_block=small_block, block_dim=block_dim)
        if ctx.quant_flag == "DANUQ":
            return DANUQ_quantize(x, forward_bits, ctx.mode, small_block=ctx.small_block, block_dim=ctx.block_dim)
        elif ctx.quant_flag == "BFP":
            return block_quantize(x, forward_bits, ctx.mode, small_block=ctx.small_block, block_dim=ctx.block_dim)
        elif ctx.quant_flag == "occ":
            return occ(x, forward_bits, ctx.mode, small_block=ctx.small_block, block_dim=ctx.block_dim, lut_mode=ctx.lut_mode, comp=ctx.comp)
    @staticmethod
    def backward(ctx, grad_output):
        x, = ctx.saved_tensors
        if ctx.needs_input_grad[0]:
            if ctx.backward_bits != -1:
                if ctx.quant_flag == "DANUQ":
                    grad_input = block_quantize(grad_output, ctx.backward_bits, ctx.mode,
                                    small_block=ctx.small_block, block_dim=ctx.block_dim)
                elif ctx.quant_flag == "BFP":
                    grad_input = block_quantize(grad_output, ctx.backward_bits, ctx.mode,
                                    small_block=ctx.small_block, block_dim=ctx.block_dim)
                elif ctx.quant_flag == "occ":
                    grad_input = occ(grad_output, ctx.backward_bits, ctx.mode,
                                    small_block=ctx.small_block, block_dim=ctx.block_dim, lut_mode=ctx.lut_mode, comp=ctx.comp)
                ### DGE gradient
                # delta = ctx.delta
                # k = ctx.k
                # # DGE gradient estimation
                # grad_est = (1 / k) * torch.pow(torch.abs(x - delta / 2) + 1e-6, (1 / k - 1))
                # grad_est = torch.clamp(grad_est, max=3.0)
                # grad_input = grad_output * grad_est
            else:
                grad_input = grad_output
        return grad_input, None, None, None, None, None, None, None, None, None, None





# class BlockRounding(torch.autograd.Function):
#     @staticmethod
#     def forward(self, x, forward_bits, backward_bits, mode, small_block="None", block_dim="B", quant_flag = "DANUQ"):
#         self.backward_bits = backward_bits
#         self.mode = mode
#         self.quant_flag = quant_flag
#         self.small_block = small_block
#         self.block_dim = block_dim
#         if forward_bits == -1: return x
        
#         return occ(x, forward_bits, self.mode, small_block=self.small_block, block_dim=self.block_dim)
        
#         # if self.quant_flag == "DANUQ":
#         #     # return bucket_quantize_blockwise_mask_zero(x, forward_bits, self.mode, small_block=self.small_block, block_dim=self.block_dim)
#         #     # return DANUQ_quantize(x, forward_bits, self.mode, small_block=self.small_block, block_dim=self.block_dim)
#         #     return hybrid_quantize(x, forward_bits, self.mode, small_block=self.small_block, block_dim=self.block_dim)
            
#         # elif self.quant_flag == "BFP":
#         #     return block_quantize(x, forward_bits, self.mode, small_block=self.small_block, block_dim=self.block_dim)
        
    
#     @staticmethod
#     def backward(self, grad_output):
        
#         if self.needs_input_grad[0]:
#             if self.backward_bits != -1:
#                 # layer_name = self.__class__.__name__
#                 # save_grad_output_distribution(grad_output, title= layer_name)
                
#                 # DGE gradient estimation
#                 delta = self.delta
#                 k = self.k
#                 x_scaled = x  # if x was already scaled before occ, you may need to save x_scaled directly
#                 grad_est = (1 / k) * torch.pow(torch.abs(x_scaled - delta / 2) + 1e-6, (1 / k - 1))
#                 grad_input = grad_output * grad_est
                
#                 # grad_input = block_quantize(grad_output, self.backward_bits, self.mode,
#                 #                              small_block=self.small_block, block_dim=self.block_dim)
                
#                 # if self.quant_flag == "BFP":
#                 #     grad_input = block_quantize(grad_output, self.backward_bits, self.mode,
#                 #                              small_block=self.small_block, block_dim=self.block_dim)
    
#                 # else:
#                 #     # return hybrid_quantize(grad_output, self.backward_bits, self.mode, small_block=self.small_block, block_dim=self.block_dim)
        
#                 #     grad_input = bucket_quantize_blockwise_mask_zero(grad_output, self.backward_bits, self.mode, small_block=self.small_block, block_dim=self.block_dim)
                
#                 # if self.quant_flag == "BFP":
#                 #     grad_input = block_quantize(grad_output, self.backward_bits, self.mode,
#                 #                              small_block=self.small_block, block_dim=self.block_dim)
                    
#                 # elif self.quant_flag == "DANUQ":
#                 #     grad_input = DANUQ_quantize(grad_output, self.backward_bits, self.mode,
#                 #                             small_block=self.small_block, block_dim=self.block_dim)
                
#             else:
#                 # layer_name = self.__class__.__name__
#                 grad_input = grad_output
                
#         return grad_input, None, None, None, None, None, None
   

# class BlockRounding_ReLU(torch.autograd.Function):
#     @staticmethod
#     def forward(self, x, forward_bits, backward_bits, mode, mean, std, small_block="None", block_dim="B", quant_flag = "DANUQ"):
#         self.backward_bits = backward_bits
#         self.mode = mode
#         self.mean = mean
#         self.std = std
#         self.small_block = small_block
#         self.block_dim = block_dim
#         if forward_bits == -1: return x
        
#         self.quant_flag = quant_flag
        
#         return DANUQ_ReLU_quantize(x, forward_bits, self.mode, self.mean, self.std, small_block=self.small_block, block_dim=self.block_dim)
        
#         # return occ(x, forward_bits, self.mode, small_block=self.small_block, block_dim=self.block_dim)
        
    
#     @staticmethod
#     def backward(self, grad_output):
#         if self.needs_input_grad[0]:
#             if self.backward_bits != -1:
#                 # layer_name = self.__class__.__name__
#                 # save_grad_output_distribution(grad_output, title= layer_name)
#                 # grad_input = DANUQ_quantize(grad_output, self.backward_bits, self.mode, small_block=self.small_block, block_dim=self.block_dim)
#                 grad_input = block_quantize(grad_output, self.backward_bits, self.mode,
#                                              small_block=self.small_block, block_dim=self.block_dim)
#                 # grad_input = bucket_quantize_blockwise_mask_zero(grad_output, self.backward_bits, self.mode, small_block=self.small_block, block_dim=self.block_dim)
                
#                 # if self.quant_flag == "DANUQ":
#                 #     grad_input = bucket_quantize_blockwise_mask_zero(grad_output, self.backward_bits, self.mode, small_block=self.small_block, block_dim=self.block_dim)
#                 # elif self.quant_flag == "BFP":
#                 #     grad_input = block_quantize(grad_output, self.backward_bits, self.mode, small_block=self.small_block, block_dim=self.block_dim)     
                
#             else:
#                 # layer_name = self.__class__.__name__
#                 grad_input = grad_output
                
#         return grad_input, None, None, None, None, None, None, None, None
    
class BlockQuantizer(nn.Module):
    def __init__(self, wl_activate, wl_error, mode,
            small_block="None", block_dim="B", quant_flag = "BFP", lut_mode="E3M2", comp=False):
        super(BlockQuantizer, self).__init__()
        self.wl_activate = wl_activate
        self.wl_error = wl_error
        self.mode = mode
        self.small_block="None"
        self.block_dim="B"
        self.quant_flag= quant_flag
        self.lut_mode = lut_mode
        self.comp = comp
    def forward(self, x):
        return quantize_block(x, self.wl_activate,
                              self.wl_error, self.mode,
                              self.small_block, self.block_dim, self.quant_flag, self.lut_mode, self.comp)
        
class BlockQuantizer_ReLU(nn.Module):
    def __init__(self, wl_activate, wl_error, mode,
            small_block="None", block_dim="B", quant_flag = "DANUQ", lut_mode="E3M2", comp=False):
        super(BlockQuantizer_ReLU, self).__init__()
        self.wl_activate = wl_activate
        self.wl_error = wl_error
        self.mode = mode
        self.small_block="None"
        self.block_dim="B"
        self.quant_flag= quant_flag
        self.lut_mode = lut_mode
        self.comp = comp
    def forward(self, x, mean, std):
        return quantize_block_ReLU(x, self.wl_activate,
                              self.wl_error, self.mode, mean, std,
                              self.small_block, self.block_dim, self.quant_flag, self.lut_mode, self.comp)
        
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


def quantile_tail_quantize(x_n: torch.Tensor, mask: torch.Tensor, n_bins: int = 3):
    x_sel = x_n[mask]

    q = torch.linspace(0, 1, n_bins+1, device=x_sel.device)

    q_vals = torch.quantile(x_sel, q)
    edges = q_vals[1:-1]
    
    centroids = 0.5 * (q_vals[1:] + q_vals[:-1])  # n_bins-1개 중심값
    
    indices = torch.bucketize(x_sel, edges, right=False)
    x_sel_quantized = centroids[indices]

    return x_sel_quantized


def bfp_tail_quantize(x_tail: torch.Tensor,
                      bits: int,
                      mode: str,
                      ebit: int,
                      block_dim: str = "B"):

    return block_quantize(x_tail, bits, mode,
                          ebit=ebit,
                          small_block="None",  
                          block_dim=block_dim)


def hybrid_quantize(x: torch.Tensor,
                   bits: int,
                   mode: str,
                   ebit: int = 8,
                   small_block: str = "FC",
                   block_dim: str = "B") -> torch.Tensor:
    
    # plot_tensor_distribution(x, title="input")
    device = x.device
    lut = Q_VALUES_TABLE[bits].to(device)           
    lut_sorted, _ = torch.sort(lut)
    lut_edges = 0.5 * (lut_sorted[1:] + lut_sorted[:-1]) 

    q_min = lut_sorted[0].item()
    q_max = lut_sorted[-1].item()

    # tail_step = (lut_sorted[-1] - lut_sorted[-2]).item()   # 마지막 두 값 간격

    tail_levels = 6                                        # 한쪽 3 레벨

    dim_threshold = {"Conv": 2, "FC": 1, "None": 4}.get(small_block, 1)

    def quantize_block(inp: torch.Tensor):
        x_mean  = inp.mean()
        std  = torch.clamp(inp.std(unbiased=False), min=1e-10)
        out = torch.zeros_like(inp)

        x_n = (inp - x_mean) / std                 

        # 마스크
        mid_mask  = (x_n >= q_min) & (x_n <= q_max)
        low_mask  = x_n <  q_min
        high_mask = x_n >  q_max
        
        # x_min = x_n.min().item()
        # x_max = x_n.max().item()
        
        # epsilon = 1e-5
        # low_range  = (min(x_min, q_min - epsilon), q_min)
        # high_range = (q_max, max(x_max, q_max + epsilon))
        
        
        # def make_tail_centers(rmin, rmax, levels=3):
        #     delta = (rmax - rmin) / levels
        #     return torch.tensor(
        #         [rmin + (i + 0.5) * delta for i in range(levels)],
        #         device=inp.device
        #     )

        # low_centers  = make_tail_centers(*low_range)
        # high_centers = make_tail_centers(*high_range)

        if mid_mask.any():
            x_mid = x_n[mid_mask]
            # plot_tensor_distribution(x_mid, title="x_mid")
            idx = torch.bucketize(x_mid, lut_edges, right=False)
            
            if mode != "stochastic":
                nxt = torch.clamp(idx + 1, max=lut.numel() - 1)
                lo, up = lut[idx], lut[nxt]
                dl, du = (x_mid - lo).abs(), (up - x_mid).abs()
                p_up = dl / (dl + du + 1e-10)
                pick = torch.rand_like(p_up) < p_up
                out[mid_mask] = torch.where(pick, up, lo)
                # plot_tensor_distribution(out[mid_mask], title="x_mid_quant") 
            else:
                out[mid_mask] = lut[idx]
                # plot_tensor_distribution(out[mid_mask], title="x_mid_quant") 

        if low_mask.any():
            x_low = x_n[low_mask]
            # plot_tensor_distribution(x_low, title="x_low")
            
            x_low_min = x_low.min()
            x_low_max = x_low.max()
            
            low_edges = torch.linspace(x_low_min, x_low_max, tail_levels+1, device=x_low.device)
            
            low_idx = torch.bucketize(x_low, low_edges, right=False) - 1
            low_idx = torch.clamp(low_idx, 0, tail_levels - 1)
            low_bin_centers = 0.5 * (low_edges[:-1] + low_edges[1:])
            x_low_quantized = low_bin_centers[low_idx]
            out[low_mask] = x_low_quantized
            # plot_tensor_distribution(out[low_mask], title="x_low_quant") 
            
        if high_mask.any():
            x_high = x_n[high_mask]
            # plot_tensor_distribution(x_high, title="x_high")
            
            x_high_min = x_high.min()
            x_high_max = x_high.max()
            
            high_edges = torch.linspace(x_high_min, x_high_max, tail_levels+1, device=x_high.device)
            
            high_idx = torch.bucketize(x_high, high_edges, right=False) - 1
            high_idx = torch.clamp(high_idx, 0, tail_levels - 1)
            high_bin_centers = 0.5 * (high_edges[:-1] + high_edges[1:])
            x_high_quantized = high_bin_centers[high_idx]
            out[high_mask] = x_high_quantized
            # plot_tensor_distribution(out[high_mask], title="x_high_quant") 
                    
        # # BFP Tail 양자화                    
        # if low_mask.any():
        #     # x_low = x_n[low_mask]
        #     # plot_tensor_distribution(x_low, title="x_low")
        #     # idx = torch.clamp(((q_min - x_low) / tail_step).floor(),
        #     #                   0, tail_levels - 1).long()
        #     # out_low = q_min - (idx + 0.5) * tail_step   
        #     # out[low_mask] = out_low
        #     # plot_tensor_distribution(out[low_mask], title="x_low_quant") 
            
        #     # BFP Tail 양자화
        #     # out_low = bfp_tail_quantize(x_n[low_mask], bits, mode, ebit)
        #     # out[low_mask] = out_low
            
        #     x_low = x_n[low_mask].clamp(min=low_range[0], max=low_range[1] - 1e-6)
        #     idx = ((x_low - low_range[0]) / (low_range[1] - low_range[0]) * tail_levels).floor().clamp(0, tail_levels - 1).long()
        #     out_low = low_centers[idx]
        #     out[low_mask] = out_low
            
        # if high_mask.any():
        #     # x_high = x_n[high_mask]
        #     # plot_tensor_distribution(x_high, title="x_high")
        #     # idx = torch.clamp(((x_high - q_max) / tail_step).floor(),
        #     #                   0, tail_levels - 1).long()
        #     # out_high = q_max + (idx + 0.5) * tail_step
        #     # out[high_mask] = out_high
        #     # plot_tensor_distribution(out[high_mask], title="x_high_quant") 

        #     # out_high = bfp_tail_quantize(x_n[high_mask], bits, mode, ebit)
        #     # out[high_mask] = out_high
            
        #     x_high = x_n[high_mask].clamp(min=high_range[0] + 1e-6, max=high_range[1])
        #     idx = ((x_high - high_range[0]) / (high_range[1] - high_range[0]) * tail_levels).floor().clamp(0, tail_levels - 1).long()
        #     out_high = high_centers[idx]
        #     out[high_mask] = out_high

        # plot_tensor_distribution(out, title="out")

        #quantile tail 양자화
        # if low_mask.any():
        #     out[low_mask] = quantile_tail_quantize(x_n, low_mask, n_bins=tail_levels)
            
        # if high_mask.any():
        #     out[high_mask] = quantile_tail_quantize(x_n, high_mask, n_bins=tail_levels)
        
        result = out * std + x_mean
        # plot_tensor_distribution(result, title="out_scale")
        return result

    if x.dim() <= dim_threshold:
        return quantize_block(x)
    else:
        return x
    
def add_r_(data):
    r = torch.rand_like(data)
    data.add_(r)
    
def quantize_to_LUT(x: torch.Tensor, lut: torch.Tensor):
    
    lut = torch.sort(lut)[0].to(x.device)

    # midpoints between LUT values to define bins
    midpoints = (lut[:-1] + lut[1:]) / 2

    # 각 x의 값이 어느 bin에 속하는지를 구함
    indices = torch.bucketize(x, midpoints)

    return lut[indices]
   
def occ(x: torch.Tensor,
        bits: int,
        mode: str,
        ebit: int = 8,
        small_block: str = "FC",
        block_dim: str = "B",
        lut_mode: str = 'E3M2',
        alpha: float = 0.99,
        comp: bool = True) -> torch.Tensor:

    values = LUT[lut_mode]

    # absmax scaling
    scaling_factor = values.abs().max().item() / (x.abs().max().item() + 1e-6)
    x_scaled = x * scaling_factor
    
    # outlier threshold
    threshold = torch.quantile(x_scaled.abs(), alpha)
    x_clamped = torch.clamp(x_scaled, -threshold, threshold)

    x_quant = quantize_to_LUT(x_clamped, values)

    # sparse outlier matrix
    if comp:
        residual = x_scaled - x_clamped
        x_dequant = (x_quant + residual) / scaling_factor
    else:        
        x_dequant = x_quant / scaling_factor
    
    return x_dequant

quantize_block = BlockRounding.apply
quantize_block_ReLU = BlockRounding_ReLU.apply
