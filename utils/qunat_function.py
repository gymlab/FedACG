from models.quant import *
from models.layers import norm
from models.quant import quantization
import copy

def AQD_update(model, args):

    for name, param in model.named_parameters():
        if hasattr(args.quantizer, 'keyword'):
            if 'first-last' in args.quantizer.keyword and name == 'conv1.weight':
                first_quant_conv = quantization(args, 'wt', [param.shape[1], param.shape[0], param.shape[2], param.shape[2]], groups=1)
                param.data.copy_(first_quant_conv(param.data)) 
            elif name != "conv1.weight" and ("conv1.weight" in name or "conv2.weight" in name):
                layer_quant_conv = quantization(args, 'wt', [param.shape[1], param.shape[0], param.shape[2], param.shape[2]], groups=1)
                param.data.copy_(layer_quant_conv(param.data))
            elif "downsample.0.weight" in name:
                quant_conv1x1 = quantization(args, 'wt', [param.shape[1], param.shape[0], param.shape[2], param.shape[2]], groups=1)
                param.data.copy_(quant_conv1x1(param.data))
            elif 'first-last' in args.quantizer.keyword and name == 'fc.weight':
                last_quant_linear = quantization(args, 'wt', [param.shape[1], param.shape[0], param.shape[2], param.shape[2]], groups=1)
                param.data.copy_(last_quant_linear(param.data))


class ___WSQConv2d(nn.Module):
    bit1 = [0.7979]
    bit2 = [0.5288, 0.9816]
    bit3 = [0.4510, 0.7481, 0.9882]
    bit4 = [0.2960, 0.5567, 0.7088, 1.1286]
    bit5 = [0.2455, 0.4734, 0.5989, 0.9206, 0.9904]
    bit6 = [0.2219, 0.3354, 0.4478, 0.8548, 0.8936, 0.9315]
    bit7 = [0.1494, 0.2239, 0.2986, 0.41059, 0.5865, 1.1834, 1.7924]
    bit8 = [0.0498, 0.0991, 0.203, 0.3355, 0.5280, 0.9925, 1.3935, 1.4585]

    def __init__(self, n_bits=1, clip_prob=-1):
        super(___WSQConv2d, self).__init__()
        
        self.alpha = torch.tensor(getattr(self, f'bit{n_bits}'), dtype=torch.float32)
        self.clip_prob = clip_prob
        
        # Generate all combinations of b_k in {-1, 1} for 2^(M-1) terms
        b_combinations = torch.cartesian_prod(*[torch.tensor([-1., 1.]) for _ in range(len(self.alpha))])
        if len(self.alpha) == 1:
            b_combinations = b_combinations.unsqueeze(-1)
        q_values = torch.sum(b_combinations * self.alpha, dim=1)
        self.q_values = torch.sort(q_values).values
        self.edges = 0.5 * (self.q_values[1:] + self.q_values[:-1])
        
    def clip_by_prob(self, x):

        original_shape = x.shape
        x_flat = x.view(x.size(0), -1)
        abs_x_flat = x_flat.abs()
        topk = max(1, int(self.clip_prob * abs_x_flat.size(1)))
        thresholds = torch.topk(abs_x_flat, topk, dim=1, largest=True, sorted=True).values[:, -1].view(-1, 1)

        # Clip values in parallel
        clipped_x_flat = torch.where(
            x_flat > thresholds, thresholds,
            torch.where(x_flat < -thresholds, -thresholds, x_flat)
        )

        # Reshape back to the original shape
        clipped_x = clipped_x_flat.view(original_shape)

        return clipped_x
    
    def forward(self, x, global_x):
        with torch.no_grad():
            x = x - global_x    # residual
            if self.clip_prob > 0:
                x = self.clip_by_prob(x)
            x_mean = x.mean(dim=1, keepdim=True).mean(dim=2, keepdim=True).mean(dim=3, keepdim=True)
            x = x - x_mean
            raw_x_std = x.view(x.size(0), -1).std(dim=1).view(-1, 1, 1, 1)
            x_std = raw_x_std + 1e-12
            x = x / x_std.expand_as(x)

            indices = torch.bucketize(x, self.edges, right=False)
            quantized_x = self.q_values[indices]
            dequantized_x = quantized_x * x_std + x_mean
            
            # masking
            norm_mask = raw_x_std.squeeze(-1).squeeze(-1).squeeze(-1) < 1e-12
            dequantized_x[norm_mask] = 0           
            
            updated_global_x = global_x + dequantized_x
            
        return updated_global_x


class __WSQConv2d(nn.Module):
    bit2 = [-1., 0.2842, 0.2842, 1.]
    bit3 = [-1., -0.6453, -0.3673, -0.1196, 0.1196, 0.3673, 0.6453, 1.]
    bit4 = [-1., -0.8190, -0.6652, -0.5280, -0.4016, -0.2824, -0.1677, -0.0557,
            0.0557, 0.1677, 0.2824, 0.4016, 0.5280, 0.6652, 0.8190, 1.]

    def __init__(self, n_bits=1, clip_prob=-1):
        super(__WSQConv2d, self).__init__()

        q_values = torch.tensor(getattr(self, f'bit{n_bits}'), dtype=torch.float32)
        self.q_values = torch.sort(q_values).values
        self.edges = 0.5 * (self.q_values[1:] + self.q_values[:-1])

    def forward(self, x, global_x):
        residual = x - global_x
        residual_flatten = residual.view(-1)
        
        absmax = torch.max(torch.abs(residual_flatten))
        if absmax < 1e-12:
            return torch.zeros_like(residual_flatten), torch.zeros_like(residual), global_x.clone()

        normalized_residual = residual / absmax
        indices = torch.bucketize(normalized_residual, self.edges, right=False)
        quantized_residual = self.q_values[indices]
        
        updated_global_tensor = global_x + quantized_residual * absmax

        return updated_global_tensor
    

class _WSQConv2d(nn.Module):
    bit1 = [-0.7979, 0.7979]
    bit2 = [-1.0691, 0., 1.0691, 1.9445]
    bit3 = [-2.7247, -1.6053, -0.7947, 0., 0.7947, 1.6053, 2.7247, 5.1247]
    bit4 = [-3.8686, -2.7114, -1.9828, -1.4906, -1.1238, -0.7620, -0.3952, 0., 
            0.3952, 0.7620, 1.1238, 1.4906, 1.9828, 2.7114, 3.8686, 4.5972]

    def __init__(self, n_bits=1, clip_prob=-1):
        super(_WSQConv2d, self).__init__()
        
        q_values = torch.tensor(getattr(self, f'bit{n_bits}'), dtype=torch.float32)
        self.q_values = torch.sort(q_values).values
        self.edges = 0.5 * (self.q_values[1:] + self.q_values[:-1])
    
    def forward(self, x, global_x):
        with torch.no_grad():
            x = x - global_x    # residual
            x_mean = x.mean().view(-1, 1, 1, 1)
            x = x - x_mean
            x_std = x.std().view(-1, 1, 1, 1)
            x = x / x_std.expand_as(x)

            indices = torch.bucketize(x, self.edges, right=False)
            quantized_x = self.q_values[indices]
            dequantized_x = quantized_x * x_std + x_mean
            
            updated_global_x = global_x + dequantized_x
            
        return updated_global_x
    
    
class WSQConv2d(nn.Module):
    # bit4 = [-0.7135, -0.5307, -0.4055, -0.3090, -0.2242, -0.1462, -0.0722, 0.,
    #         0.0619, 0.1250, 0.1904,	0.2598, 0.3356, 0.4215, 0.5291, 0.7122]
    # bit4 = [-0.8587, -0.6386, -0.4880, -0.3718, -0.2698, -0.1760, -0.0869, 0.,
    #         0.0745, 0.1504, 0.2291, 0.3127, 0.4039, 0.5073, 0.6368, 0.8572]
    bit4 = [-2.6536, -1.9735, -1.508, -1.149, -0.8337, -0.5439, -0.2686, 0.,
            0.2303, 0.4648, 0.7081, 0.9663, 1.2481, 1.5676, 1.9679, 2.6488]

    def __init__(self, n_bits=1, clip_prob=0.001):
        super(WSQConv2d, self).__init__()
        
        q_values = torch.tensor(getattr(self, f'bit{n_bits}'), dtype=torch.float32)
        self.q_values = torch.sort(q_values).values
        self.edges = 0.5 * (self.q_values[1:] + self.q_values[:-1])
        self.clip_prob = clip_prob
    
    def forward(self, x, global_x, std):
        with torch.no_grad():
            x = x - global_x    # residual
            
            # clip: V11
            x_abs = torch.abs(x)
            k = int((1 - self.clip_prob) * x_abs.numel())
            clip_threshold = torch.kthvalue(x_abs.view(-1), k).values
            x_clipped = torch.clamp(x, min=-clip_threshold, max=clip_threshold)

            x_std = x_clipped.std().view(1, 1, 1, 1)
            x = x_clipped / x_std
            # x = x_clipped / clip_threshold

            indices = torch.bucketize(x, self.edges, right=False)
            quantized_x = self.q_values[indices]
            dequantized_x = x_std * quantized_x
            
            updated_global_x = global_x + dequantized_x
            
        return updated_global_x


def WSQ_update(model, global_model, args):
    
    g_params = dict(global_model.named_parameters())
    
    for name, param in model.named_parameters():
        if hasattr(args.quantizer, 'keyword'):
            if 'first-last' in args.quantizer.keyword and name == 'conv1.weight':
                first_quant_conv = WSQConv2d(n_bits=args.quantizer.wt_bit, clip_prob=args.quantizer.wt_clip_prob)
                param.data.copy_(first_quant_conv(param.data, g_params[name].data)) 
            elif name != "conv1.weight" and ("conv1.weight" in name or "conv2.weight" in name):
                layer_quant_conv = WSQConv2d(n_bits=args.quantizer.wt_bit, clip_prob=args.quantizer.wt_clip_prob)
                param.data.copy_(layer_quant_conv(param.data, g_params[name].data)) 
            elif "downsample.0.weight" in name:
                quant_conv1x1 = WSQConv2d(n_bits=args.quantizer.wt_bit, clip_prob=args.quantizer.wt_clip_prob)
                param.data.copy_(quant_conv1x1(param.data, g_params[name].data))
                

class WSQGConv2d(nn.Module):
    # bit4 = [-2.6536, -1.9735, -1.508, -1.149, -0.8337, -0.5439, -0.2686, 0.,
    #         0.2303, 0.4648, 0.7081, 0.9663, 1.2481, 1.5676, 1.9679, 2.6488]
    bit1 = [-0.7979, 0.7979]
    bit2 = [-1.224, 0, 0.7646, 1.7242]	
    bit4 = [-2.7327, -2.0691, -1.6181, -1.2563, -0.9424, -0.6568, -0.3881, 0.1284,
            0.1284, 0.3881, 0.6568, 0.9424, 1.2563, 1.6181, 2.0691, 2.7327]


    def __init__(self, n_bits=1, clip_prob=-1):
        super(WSQGConv2d, self).__init__()
        
        q_values = torch.tensor(getattr(self, f'bit{n_bits}'), dtype=torch.float32)
        self.q_values = torch.sort(q_values).values
        self.edges = 0.5 * (self.q_values[1:] + self.q_values[:-1])
        self.clip_prob = clip_prob
    
    def forward(self, x, global_x, std):
        with torch.no_grad():
            x = x - global_x    # residual
            
            # clip: V11
            if self.clip_prob > 0:
                x_abs = torch.abs(x)
                k = int((1 - self.clip_prob) * x_abs.numel())
                clip_threshold = torch.kthvalue(x_abs.view(-1), k).values
                x = torch.clamp(x, min=-clip_threshold, max=clip_threshold)
            
            local_std = x.std()
            x = x / std
            # x = x_clipped / clip_threshold

            indices = torch.bucketize(x, self.edges, right=False)
            quantized_x = self.q_values[indices]
            dequantized_x = std * quantized_x
            
            updated_global_x = global_x + dequantized_x
            
        return updated_global_x, local_std


def WSQG_update(model, global_model, args):
    
    g_params = dict(global_model.named_parameters())
    
    global_std_values = {}
    for name, param in global_model.named_parameters():
        if "global_std" in name:
            # 앞부분이 같은 "weight" 키를 찾아서 std를 가져옴
            weight_name = name.replace("global_std", "weight")
            global_std_values[weight_name] = param.data    # weight에서 저장한 std 값 가져오기

    local_std_values = {}
    for name, param in model.named_parameters():
        if hasattr(args.quantizer, 'keyword'):
            if 'first-last' in args.quantizer.keyword and name == 'conv1.weight':
                first_quant_conv = WSQGConv2d(n_bits=args.quantizer.wt_bit, clip_prob=args.quantizer.wt_clip_prob)
                updated_param, local_std = first_quant_conv(param.data, g_params[name].data, global_std_values[name])
                param.data.copy_(updated_param)
                local_std_values[name] = local_std
            elif name != "conv1.weight" and ("conv1.weight" in name or "conv2.weight" in name):
                layer_quant_conv = WSQGConv2d(n_bits=args.quantizer.wt_bit, clip_prob=args.quantizer.wt_clip_prob)
                updated_param, local_std = layer_quant_conv(param.data, g_params[name].data, global_std_values[name])
                param.data.copy_(updated_param)
                local_std_values[name] = local_std
            elif "downsample.0.weight" in name:
                quant_conv1x1 = WSQGConv2d(n_bits=args.quantizer.wt_bit, clip_prob=args.quantizer.wt_clip_prob)
                updated_param, local_std = quant_conv1x1(param.data, g_params[name].data, global_std_values[name])
                param.data.copy_(updated_param)
                local_std_values[name] = local_std

    for name, param in model.named_parameters():
        if "local_std" in name:
            # 앞부분이 같은 "weight" 키를 찾아서 std를 가져옴
            weight_name = name.replace("local_std", "weight")
            if weight_name in local_std_values:
                std = local_std_values[weight_name]  # weight에서 저장한 std 값 가져오기
                param.data.copy_(std)



class WSQLGConv2d(nn.Module):
    # bit4 = [-2.6536, -1.9735, -1.508, -1.149, -0.8337, -0.5439, -0.2686, 0.,
    #         0.2303, 0.4648, 0.7081, 0.9663, 1.2481, 1.5676, 1.9679, 2.6488]
    bit1 = [-0.7979, 0.7979]
    bit2 = [-1.224, 0, 0.7646, 1.7242]	
    bit4 = [-2.7327, -2.0691, -1.6181, -1.2563, -0.9424, -0.6568, -0.3881, 0.1284,
            0.1284, 0.3881, 0.6568, 0.9424, 1.2563, 1.6181, 2.0691, 2.7327]


    def __init__(self, n_bits=1, clip_prob=-1):
        super(WSQGConv2d, self).__init__()
        
        q_values = torch.tensor(getattr(self, f'bit{n_bits}'), dtype=torch.float32)
        self.q_values = torch.sort(q_values).values
        self.edges = 0.5 * (self.q_values[1:] + self.q_values[:-1])
        self.clip_prob = clip_prob
    
    def forward(self, x, global_x, std):
        with torch.no_grad():
            x = x - global_x    # residual
            
            # clip: V11
            if self.clip_prob > 0:
                x_abs = torch.abs(x)
                k = int((1 - self.clip_prob) * x_abs.numel())
                clip_threshold = torch.kthvalue(x_abs.view(-1), k).values
                x = torch.clamp(x, min=-clip_threshold, max=clip_threshold)
            
            local_std = x.std()
            std = 0.9 * std + 0.1 * local_std
            x = x / std
            # x = x_clipped / clip_threshold

            indices = torch.bucketize(x, self.edges, right=False)
            quantized_x = self.q_values[indices]
            dequantized_x = std * quantized_x
            
            updated_global_x = global_x + dequantized_x
            
        return updated_global_x, local_std


def WSQLG_update(model, global_model, args):
    
    g_params = dict(global_model.named_parameters())
    
    global_std_values = {}
    for name, param in global_model.named_parameters():
        if "global_std" in name and name != "conv1.global_std":
            # 앞부분이 같은 "weight" 키를 찾아서 std를 가져옴
            weight_name = name.replace("global_std", "weight")
            global_std_values[weight_name] = param.data    # weight에서 저장한 std 값 가져오기

    local_std_values = {}
    for name, param in model.named_parameters():
        if hasattr(args.quantizer, 'keyword'):
            if 'first-last' in args.quantizer.keyword and name == 'conv1.weight':
                first_quant_conv = WSQGConv2d(n_bits=args.quantizer.wt_bit, clip_prob=args.quantizer.wt_clip_prob)
                updated_param, local_std = first_quant_conv(param.data, g_params[name].data, global_std_values[name])
                param.data.copy_(updated_param)
                local_std_values[name] = local_std
            elif name != "conv1.weight" and ("conv1.weight" in name or "conv2.weight" in name):
                layer_quant_conv = WSQGConv2d(n_bits=args.quantizer.wt_bit, clip_prob=args.quantizer.wt_clip_prob)
                updated_param, local_std = layer_quant_conv(param.data, g_params[name].data, global_std_values[name])
                param.data.copy_(updated_param)
                local_std_values[name] = local_std
            elif "downsample.0.weight" in name:
                quant_conv1x1 = WSQGConv2d(n_bits=args.quantizer.wt_bit, clip_prob=args.quantizer.wt_clip_prob)
                updated_param, local_std = quant_conv1x1(param.data, g_params[name].data, global_std_values[name])
                param.data.copy_(updated_param)
                local_std_values[name] = local_std
                
    momentum = args.quantizer.momentum

    for name, param in model.named_parameters():
        if "global_std" in name and name != "conv1.global_std":
            weight_name = name.replace("global_std", "weight")
            param.data.copy_((1. - momentum) * global_std_values[weight_name] + momentum * local_std_values[weight_name])

               
def quantize_and_dequantize(tensor, global_tensor, bit_width, lr=1.0, flag = False, quantization_weight=None):
    residual = tensor - global_tensor
    
    # BFP 양자화
    if flag is True:
        # E(w) : formula (2) 계산
        max_val = torch.max(torch.abs(residual))
        exponent = torch.floor(torch.log2(max_val + 1e-6))
        exponent = torch.clamp(exponent, -(2 ** (bit_width -1)), 2 ** (bit_width - 1) - 1)

        theta = 2 ** (exponent + 2 - bit_width)

        # Q(x) : formula (3) 계산
        floor_val = torch.floor(residual / theta)
        ceil_val = torch.ceil(residual / theta)
        prob = (residual / theta) - floor_val
        quantized = torch.where(torch.rand_like(prob) < prob, ceil_val, floor_val)
        
        quantized_actual = quantized * theta
        
        lower_bound = -2 ** (exponent + 1)
        upper_bound = 2** (exponent + 1) - 2 ** (exponent + 2 - bit_width)
        dequantized = torch.clamp(quantized_actual, lower_bound, upper_bound)
        
        weighted_dequantized = dequantized * quantization_weight
        updated_global_tensor = global_tensor + weighted_dequantized * lr
    
    else:
        original_shape = residual.shape
        residual_flatten = residual.view(-1)
        
        norm = torch.norm(residual_flatten)
        if norm < 1e-12:
            return torch.zeros_like(residual_flatten), torch.zeros_like(residual), global_tensor.clone()

        levels = 2 ** (bit_width - 1)
        abs_ratio = torch.abs(residual_flatten) / norm
        scaled_ratio = torch.clamp(abs_ratio * (levels), 0, levels)
        lower_index = torch.floor(scaled_ratio).long()
        upper_index = torch.clamp(lower_index + 1, 0, levels)
        p_upper = scaled_ratio - lower_index
        random_values = torch.rand_like(abs_ratio)
        selected_index = torch.where(random_values < p_upper, upper_index, lower_index)
        quantized_values = torch.arange(0, levels + 1) / (levels)
        selected_levels = quantized_values[selected_index]
        quantized_flatten = norm * torch.sign(residual_flatten) * selected_levels

        dequantized_tensor = quantized_flatten.view(original_shape)
        
        updated_global_tensor = global_tensor + lr * dequantized_tensor # lr 어떻게 설정할지.. 일단 1로

    return updated_global_tensor


class NormalFloat(nn.Module):
    bit1 = [-1., 1.]
    bit2 = [-1., 0., 0.3339, 1.]
    bit4 = [-1.0000, -0.6962, -0.5257, -0.3946, -0.2849, -0.1892, -0.0931, 0.0000,
                        0.0796, 0.1603, 0.2453, 0.3487, 0.4622, 0.5952, 0.7579, 1.0000]

    def __init__(self, n_bits=1):
        super(NormalFloat, self).__init__()

        q_values = torch.tensor(getattr(self, f'bit{n_bits}'), dtype=torch.float32)
        self.q_values = torch.sort(q_values).values
        self.edges = 0.5 * (self.q_values[1:] + self.q_values[:-1])

    def forward(self, x, global_x):
        residual = x - global_x
        residual_flatten = residual.view(-1)
        
        absmax = torch.max(torch.abs(residual_flatten))
        if absmax < 1e-12:
            return torch.zeros_like(residual_flatten), torch.zeros_like(residual), global_x.clone()

        normalized_residual = residual / absmax
        indices = torch.bucketize(normalized_residual, self.edges, right=False)
        quantized_residual = self.q_values[indices]
        
        updated_global_tensor = global_x + quantized_residual * absmax

        return updated_global_tensor


def NF_update(model, global_model, args):
    
    g_params = dict(global_model.named_parameters())
        
    for name, param in model.named_parameters():
        if hasattr(args.quantizer, 'keyword'):
            if 'first-last' in args.quantizer.keyword and name == 'conv1.weight':
                first_quant_conv = NormalFloat(n_bits=args.quantizer.wt_bitb)
                param.data.copy_(first_quant_conv(param.data, g_params[name].data)) 
            elif name != "conv1.weight" and ("conv1.weight" in name or "conv2.weight" in name):
                layer_quant_conv = NormalFloat(n_bits=args.quantizer.wt_bit)
                param.data.copy_(layer_quant_conv(param.data, g_params[name].data)) 
            elif "downsample.0.weight" in name:
                quant_conv1x1 = NormalFloat(n_bits=args.quantizer.wt_bit)
                param.data.copy_(quant_conv1x1(param.data, g_params[name].data)) 
                
                
class E2M1(nn.Module):
    bit1 = [-1., 1.]
    bit2 = [-1.5, 0., 1.5]
    bit4 = [-6., -4., -3., -2., -1.5, -1., -0.5, 0., 0.5, 1., 1.5, 2., 3., 4., 6.]

    def __init__(self, n_bits=1, clip_prob=-1):
        super(E2M1, self).__init__()

        q_values = torch.tensor(getattr(self, f'bit{n_bits}'), dtype=torch.float32)
        self.q_values = torch.sort(q_values).values
        self.q_values = self.q_values / torch.max(self.q_values)
        self.edges = 0.5 * (self.q_values[1:] + self.q_values[:-1])
        self.clip_prob = clip_prob

    def forward(self, x, global_x):
        residual = x - global_x
        residual_flatten = residual.view(-1)
        
         # clip: V11
        if self.clip_prob > 0:
            x_abs = torch.abs(x)
            k = int((1 - self.clip_prob) * x_abs.numel())
            clip_threshold = torch.kthvalue(x_abs.view(-1), k).values
            x = torch.clamp(x, min=-clip_threshold, max=clip_threshold)

        absmax = torch.max(torch.abs(residual_flatten))
        if absmax < 1e-12:
            return torch.zeros_like(residual_flatten), torch.zeros_like(residual), global_x.clone()

        normalized_residual = residual / absmax
        indices = torch.bucketize(normalized_residual, self.edges, right=False)
        quantized_residual = self.q_values[indices]
        
        updated_global_tensor = global_x + quantized_residual * absmax

        return updated_global_tensor


def E2M1_update(model, global_model, args):
    
    g_params = dict(global_model.named_parameters())
        
    for name, param in model.named_parameters():
        if hasattr(args.quantizer, 'keyword'):
            if 'first-last' in args.quantizer.keyword and name == 'conv1.weight':
                first_quant_conv = E2M1(n_bits=args.quantizer.wt_bitb, clip_prob=args.quantizer.wt_clip_prob)
                param.data.copy_(first_quant_conv(param.data, g_params[name].data)) 
            elif name != "conv1.weight" and ("conv1.weight" in name or "conv2.weight" in name):
                layer_quant_conv = E2M1(n_bits=args.quantizer.wt_bit, clip_prob=args.quantizer.wt_clip_prob)
                param.data.copy_(layer_quant_conv(param.data, g_params[name].data)) 
            elif "downsample.0.weight" in name:
                quant_conv1x1 = E2M1(n_bits=args.quantizer.wt_bit, clip_prob=args.quantizer.wt_clip_prob)
                param.data.copy_(quant_conv1x1(param.data, g_params[name].data)) 


def PAQ_update(model, global_model, args):
    s = args.quantizer.wt_bit

    lr = getattr(args, 'global_PAQ_lr', 1.0)

    g_params = dict(global_model.named_parameters())
    
    
    for name, param in model.named_parameters():
        if hasattr(args.quantizer, 'keyword'):
            if 'first-last' in args.quantizer.keyword and name == 'conv1.weight':
                global_param = g_params[name]
                updated_local = quantize_and_dequantize(param.data, global_param.data, s, lr)
                param.data.copy_(updated_local)
                
            elif name != "conv1.weight" and ("conv1.weight" in name or "conv2.weight" in name):
                global_param = g_params[name]
                updated_local = quantize_and_dequantize(param.data, global_param.data, s, lr)
                param.data.copy_(updated_local)
                
            elif "downsample.0.weight" in name:
                global_param = g_params[name]
                updated_local = quantize_and_dequantize(param.data, global_param.data, s, lr)
                param.data.copy_(updated_local)
            
            elif 'first-last' in args.quantizer.keyword and name == 'fc.weight':
                global_param = g_params[name]
                updated_local = quantize_and_dequantize(param.data, global_param.data, s, lr)
                param.data.copy_(updated_local)
                
    return model


def compute_q_i(residual_model, after_residual_model):

    max_ratio = 0.0 

    for param_name in residual_model:
        p_new = after_residual_model[param_name] 
        p_orig = residual_model[param_name]       

        orig_norm_sq = p_orig.pow(2).sum().item()
        if orig_norm_sq == 0.0:

            ratio = 0.0
        else:
            diff_norm_sq = (p_new - p_orig).pow(2).sum().item()
            ratio = diff_norm_sq / orig_norm_sq

        if ratio > max_ratio:
            max_ratio = ratio

    return max_ratio

def compute_p_i(q_list):

    inv_terms = [1.0 / (1.0 + q) for q in q_list] 
    denom = sum(inv_terms)  
    if denom == 0:
        return [1.0 / len(q_list)] * len(q_list)
    
    p_list = [term / denom for term in inv_terms]
    return p_list

def compute_parameter_residuals(model, global_model):

    residual_model = {}  

    for (param_name, param), (param_name_global, param_global) in zip(model.named_parameters(), global_model.named_parameters()):
   
        residual = param - param_global
        residual_model[param_name] = residual.clone()
    
    return residual_model


def HQ_update(self, model, global_model, args):
    s = args.quantizer.wt_bit
    lr = 1.0

    flag = False
    quantization_weight = 1.0
    
    if 'BFP' in self.args.quantizer.keyword:
        flag = True
        quantization_weight = 1.0
        
    g_params = dict(global_model.named_parameters())

    residual_model = compute_parameter_residuals(model, global_model)
    
    for name, param in model.named_parameters():
        if hasattr(args.quantizer, 'keyword'):
            if 'first-last' in args.quantizer.keyword and name == 'conv1.weight':
                global_param = g_params[name]
                updated_local = quantize_and_dequantize(param.data, global_param.data, s, lr, flag, quantization_weight)
                param.data.copy_(updated_local)
                
            elif name != "conv1.weight" and ("conv1.weight" in name or "conv2.weight" in name):
                global_param = g_params[name]
                updated_local = quantize_and_dequantize(param.data, global_param.data, s, lr, flag, quantization_weight)
                param.data.copy_(updated_local)
                
            elif "downsample.0.weight" in name:
                global_param = g_params[name]
                updated_local = quantize_and_dequantize(param.data, global_param.data, s, lr, flag, quantization_weight)
                param.data.copy_(updated_local)
            
            elif 'first-last' in args.quantizer.keyword and name == 'fc.weight':
                global_param = g_params[name]
                updated_local = quantize_and_dequantize(param.data, global_param.data, s, lr, flag, quantization_weight)
                param.data.copy_(updated_local)

    after_resiudal_model = compute_parameter_residuals(model, global_model)
    local_error = compute_q_i(residual_model, after_resiudal_model)
    
    return local_error