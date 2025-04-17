import torch
import torch.nn as nn
import math

Q_VALUES_TABLE = {
    # 2: torch.tensor([-0.7979, 0.7979]),
    # 3: torch.tensor([-1.224, 0, 0.7646, 1.7242]   ),
    4: torch.tensor([-2.6536, -1.9735, -1.508, -1.149, -0.8337, -0.5439, -0.2686, 0.,
            0.2686, 0.5439, 0.8337, 1.149, 1.508, 1.9735, 2.6536]),
    6: torch.tensor([-2.154, -1.863, -1.676, -1.534, -1.418, -1.318, -1.23, -1.15, -1.078, -1.01, -0.947, -0.887, -0.831, -0.776, -0.725, -0.674, -0.626, -0.579, -0.533, -0.489,
    -0.445, -0.402, -0.36, -0.319, -0.278, -0.237, -0.197, -0.157, -0.118, -0.078, -0.039, 0, 0.038, 0.076, 0.114, 0.153, 0.191, 0.23, 0.269, 0.309, 0.349, 0.389, 0.431, 0.473, 0.516,
    0.56, 0.605, 0.651, 0.699, 0.748, 0.799, 0.852, 0.908, 0.967, 1.03, 1.097, 1.169, 1.248, 1.335, 1.434, 1.55, 1.691, 1.876, 2.166]),
    8: torch.tensor([-2.418, -2.154, -1.987, -1.863, -1.762, -1.676, -1.601, -1.534, -1.473, -1.418, -1.366, -1.318, -1.273, -1.23, -1.189, -1.15, -1.113, -1.078, -1.043, -1.01,
                     -0.978, -0.947, -0.917, -0.887, -0.858, -0.831, -0.803, -0.776, -0.75, -0.725, -0.699, -0.674, -0.65, -0.626, -0.602, -0.579, -0.556, -0.533, -0.511, -0.489,
                     -0.467, -0.445, -0.424, -0.402, -0.381, -0.36, -0.339, -0.319, -0.298, -0.278, -0.257, -0.237, -0.217, -0.197, -0.177, -0.157, -0.138, -0.118, -0.098, -0.078,
                     -0.059, -0.039, -0.02, 0, 0.019, 0.039, 0.058, 0.077, 0.097, 0.116, 0.135, 0.155, 0.174, 0.194, 0.214, 0.233, 0.253, 0.273, 0.293, 0.314, 0.334, 0.354, 0.375,
                     0.396, 0.417, 0.438, 0.459, 0.481, 0.502, 0.524, 0.547, 0.569, 0.592, 0.615, 0.639, 0.662, 0.687, 0.711, 0.736, 0.762, 0.788, 0.814, 0.842, 0.869, 0.898, 0.927,
                     0.957, 0.988, 1.02, 1.053, 1.087, 1.123, 1.16, 1.198, 1.239, 1.282, 1.327, 1.375, 1.426, 1.482, 1.542, 1.609, 1.683, 1.769, 1.87, 1.994, 2.16, 2.423])
}
class BlockRounding(torch.autograd.Function):
    @staticmethod
    def forward(self, x, forward_bits, backward_bits, mode, small_block="None", block_dim="B"):
        self.backward_bits = backward_bits
        self.mode = mode
        if forward_bits == -1: return x
        self.small_block = small_block
        self.block_dim = block_dim
        # return block_quantize(x, forward_bits, self.mode, small_block=self.small_block, block_dim=self.block_dim)
        
        with torch.no_grad():
            q = DANUQ_quantize(x, forward_bits, self.mode, small_block=self.small_block, block_dim=self.block_dim)
        #     q = block_quantize(x, forward_bits, self.mode, small_block=self.small_block, block_dim=self.block_dim)
            
        return x + (q - x).detach()
    
    @staticmethod
    def backward(self, grad_output):
        if self.needs_input_grad[0]:
            if self.backward_bits != -1:
                # grad_input = block_quantize(grad_output, self.backward_bits, self.mode,
                                            # small_block=self.small_block, block_dim=self.block_dim)
                with torch.no_grad():
                    gq = DANUQ_quantize(grad_output, self.backward_bits, self.mode,
                                            small_block=self.small_block, block_dim=self.block_dim)
                    # gq = block_quantize(grad_output, self.backward_bits, self.mode,
                                            # small_block=self.small_block, block_dim=self.block_dim)
                grad_input = grad_output + (gq - grad_output).detach()
                
            else:
                grad_input = grad_output
        return grad_input, None, None, None, None, None, None
    
class BlockQuantizer(nn.Module):
    def __init__(self, wl_activate, wl_error, mode,
            small_block="None", block_dim="B"):
        super(BlockQuantizer, self).__init__()
        self.wl_activate = wl_activate
        self.wl_error = wl_error
        self.mode = mode
        self.small_block="None"
        self.block_dim="B"
    def forward(self, x):
        return quantize_block(x, self.wl_activate,
                              self.wl_error, self.mode,
                              self.small_block, self.block_dim)
        
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
        if block_dim == "B":
            max_entry = torch.max(torch.abs(data.view(data.size(0), -1)), 1)[0]
            max_exponent = torch.floor(torch.log2(max_entry))
            max_exponent = torch.clamp(max_exponent, -2 ** (ebit - 1), 2 ** (ebit - 1) - 1)
            max_exponent = max_exponent.view([data.size(0)] + [1 for _ in range(data.dim() - 1)])
        elif block_dim == "BC":
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
        x_std = inp.std(unbiased = False)
        
        # if x_std.min() <= 1e-5:
        #     print(x_std.min(), inp.shape)
        
        # x_std = torch.clamp(x_std, min= 1e-5)
        if x_std == 0:
            return inp
        
        x_normed = (inp - x_mean) / x_std
        indices = torch.bucketize(x_normed, edges, right=False)
        quantized_normed = q_values_sorted[indices]
        return quantized_normed * x_std + x_mean
    
    if x.dim() <= dim_threshold:
        return bucket_quantize_blockwise(x)
    
    else:
        if block_dim == "B":
            B = x.size(0)
            x_2d = x.view(B, -1)
            row_mean = x_2d.mean(dim=1, keepdim=True)
            row_std = x_2d.std(dim=1, keepdim=True, unbiased = False)
            zero     = row_std == 0
            row_std_safe = torch.where(zero, torch.ones_like(row_std), row_std)
            x_normed = torch.where(zero, torch.zeros_like(x_2d), (x_2d - row_mean)/row_std_safe)
            # if row_std.min() <= 1e-5:
            #     print(row_std.min(), x.shape)
                
            # row_std = torch.clamp(row_std, min=1e-5)
            # x_normed = (x_2d - row_mean) / row_std
            indices = torch.bucketize(x_normed, edges, right=False)
            quant_normed = q_values_sorted[indices]
            x_deq_2d = quant_normed * row_std_safe + row_mean
            return x_deq_2d.view_as(x)
        
        elif block_dim == "BC":
            B, C = x.size(0), x.size(1)
            x_2d = x.view(B*C, -1)
            row_mean = x_2d.mean(dim=1, keepdim=True)
            row_std = x_2d.std(dim=1, keepdim=True, unbiased = False)
            # print(min(row_std), x.shape)
            zero     = row_std == 0
            row_std_safe = torch.where(zero, torch.ones_like(row_std), row_std)
            # row_std = torch.clamp(row_std, min=1e-10)
            x_normed = torch.where(zero, torch.zeros_like(x_2d), (x_2d - row_mean)/row_std_safe)
            # x_normed = (x_2d - row_mean) / row_std
            indices = torch.bucketize(x_normed, edges, right=False)
            quant_normed = q_values_sorted[indices]
            x_deq_2d = quant_normed * row_std_safe + row_mean
            return x_deq_2d.view_as(x)
        
        else:
            raise ValueError("Invalid block_dim option: {}".format(block_dim))
def add_r_(data):
    r = torch.rand_like(data)
    data.add_(r)
quantize_block = BlockRounding.apply