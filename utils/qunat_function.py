from models.quant import *
from models.layers import norm
from models.quant import quantization

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


class WSQConv2d(nn.Module):
    bit1 = [0.7979]
    bit2 = [0.5288, 0.9816]
    bit3 = [0.4510, 0.7481, 0.9882]
    bit4 = [0.2960, 0.5567, 0.7088, 1.1286]
    bit8 = [0.0498, 0.0991, 0.203, 0.3355, 0.5280, 0.9925, 1.3935, 1.4585]

    def __init__(self, n_bits=1, clip_prob=-1):
        super(WSQConv2d, self).__init__()
        
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
