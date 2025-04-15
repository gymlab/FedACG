import torch
import torch.nn as nn
import math

Q_VALUES_TABLE = {
    2: torch.tensor([-0.7979, 0.7979]),                     
    3: torch.tensor([-1.224, 0, 0.7646, 1.7242]	),       
    4: torch.tensor([-2.6536, -1.9735, -1.508, -1.149, -0.8337, -0.5439, -0.2686, 0.,
            0.2686, 0.5439, 0.8337, 1.149, 1.508, 1.9735, 2.6536]) 
}

class BlockRounding(torch.autograd.Function):
    @staticmethod
    def forward(self, x, forward_bits, backward_bits, mode, small_block="None", block_dim="B"):
        self.backward_bits = backward_bits
        self.mode = mode
        if forward_bits == -1: return x
        self.small_block = small_block
        self.block_dim = block_dim
        return quantize_block(x, forward_bits, self.mode, small_block=self.small_block, block_dim=self.block_dim)

    @staticmethod
    def backward(self, grad_output):
        if self.needs_input_grad[0]:
            if self.backward_bits != -1:
                grad_input = quantize_block(grad_output, self.backward_bits, self.mode,
                                            small_block=self.small_block, block_dim=self.block_dim)
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

def bucket_quantize(x, q_values):

    x_mean = x.mean()
    x_std = x.std()

    if x_std < 1e-12:
        return torch.zeros_like(x) 

    x_normed = (x - x_mean) / x_std


    q_values = torch.sort(q_values).values
    q_values = q_values.to(x.device)
    edges = 0.5 * (q_values[1:] + q_values[:-1])
    edges = edges.to(x.device)

    indices = torch.bucketize(x_normed, edges, right=False)

    quantized_normed = q_values[indices]
    dequantized_x = quantized_normed * x_std + x_mean

    return dequantized_x


def DANUQ_quantize(x: torch.Tensor, bits, mode, ebit=8, small_block = "FC", block_dim = "B"):

    if bits not in Q_VALUES_TABLE:
        raise ValueError(f"Unsupported bits value: {bits}")

    q_values = Q_VALUES_TABLE[bits]

    if small_block == "Conv":
        dim_threshold = 2
    elif small_block == "FC":
        dim_threshold = 1
    elif small_block == "None":
        dim_threshold = 4
    else:
        raise ValueError("Invalid small_block option: {}".format(small_block))

    out = torch.zeros_like(x)

    if x.dim() <= dim_threshold:
        out = bucket_quantize(x, q_values)
    else:
        if block_dim == "B":
            bsz = x.size(0)
            for b in range(bsz):
                out[b] = bucket_quantize(x[b], q_values)
        elif block_dim == "BC":
            bsz = x.size(0)
            ch = x.size(1)
            for b in range(bsz):
                for c in range(ch):
                    out[b, c] = bucket_quantize(x[b, c], q_values)
        else:
            raise ValueError("Invalid block_dim option: {}".format(block_dim))

    return out

def add_r_(data):
    r = torch.rand_like(data)
    data.add_(r)


quantize_block = BlockRounding.apply