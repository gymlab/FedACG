from models.quant import *
from models.layers import norm

def AQD_update(model, args):

    for name, param in model.named_parameters():
        if hasattr(args.quantizer, 'keyword'):
            if 'first-last' in args.quantizer.keyword and name == 'conv1.weight':
                first_quant_conv = quant_conv(param.shape[0], param.shape[1], kernel_size=param.shape[2], args=args)
                param.data.copy_(first_quant_conv(param.data)) 
            elif name != "conv1.weight" and ("conv1.weight" in name or "conv2.weight" in name):
                layer_quant_conv = quant_conv(param.shape[0], param.shape[1], kernel_size=param.shape[2], args=args)
                param.data.copy_(layer_quant_conv(param.data))
            elif "downsample.0.weight" in name:
                quant_conv1x1 = quant_conv(param.shape[0], param.shape[1], kernel_size=1, args=args)
                param.data.copy_(quant_conv1x1(param.data))
            elif 'first-last' in args.quantizer.keyword and name == 'fc.weight':
                last_quant_linear = quant_linear(args.model.last_feature_dim, args.num_classes, bias=True, args=args)
                param.data.copy_(last_quant_linear(param.data))


class WSQConv2d(nn.Conv2d):
    bit1 = [0.7979]
    bit2 = [0.5288, 0.9816]
    bit3 = [0.4510, 0.7481, 0.9882]
    bit4 = [0.2960, 0.5567, 0.7088, 1.1286]
    bit8 = [0.05, 0.1, 0.2, 0.3375, 0.5250, 0.9875, 1.3875, 1.45]

    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True, rho=1e-3, n_bits=1):
        super(WSQConv2d, self).__init__(in_channels, out_channels, kernel_size, stride,
                 padding, dilation, groups, bias)
        
        self.alpha = torch.tensor(getattr(self, f'bit{n_bits}'), dtype=torch.float32)
        self.rho = rho

        # Generate all combinations of b_k in {-1, 1} for 2^(M-1) terms
        b_combinations = torch.cartesian_prod(*[torch.tensor([-1., 1.]) for _ in range(len(self.alpha))])
        if len(self.alpha) == 1:
            b_combinations = b_combinations.unsqueeze(-1)
        q_values = torch.sum(b_combinations * self.alpha, dim=1)
        self.q_values = torch.sort(q_values).values
        self.edges = 0.5 * (self.q_values[1:] + self.q_values[:-1])
        
    def forward(self, x):
        with torch.no_grad():
            x_mean = x.mean(dim=1, keepdim=True).mean(dim=2, keepdim=True).mean(dim=3, keepdim=True)
            x = x - x_mean
            x_std = x.view(x.size(0), -1).std(dim=1).view(-1, 1, 1, 1) + 1e-5
            x = x / x_std.expand_as(x)

            indices = torch.bucketize(x, self.edges, right=False)
            quantized_x = self.q_values[indices]
            quantized_x = quantized_x * x_std + x_mean
        return quantized_x
        

def WSQ_update(model, args):
    
    for name, param in model.named_parameters():
        if hasattr(args.quantizer, 'keyword'):
            if 'first-last' in args.quantizer.keyword and name == 'conv1.weight':
                first_quant_conv = WSQConv2d(param.shape[1], param.shape[0], kernel_size=param.shape[2], n_bits=args.quantizer.wt_bit)
                param.data.copy_(first_quant_conv(param.data)) 
            elif name != "conv1.weight" and ("conv1.weight" in name or "conv2.weight" in name):
                layer_quant_conv = WSQConv2d(param.shape[1], param.shape[0], kernel_size=param.shape[2], n_bits=args.quantizer.wt_bit)
                param.data.copy_(layer_quant_conv(param.data))
            elif "downsample.0.weight" in name:
                quant_conv1x1 = WSQConv2d(param.shape[1], param.shape[0], kernel_size=1, n_bits=args.quantizer.wt_bit)
                param.data.copy_(quant_conv1x1(param.data))
            elif 'first-last' in args.quantizer.keyword and name == 'fc.weight':
                last_quant_linear = quant_linear(args.model.last_feature_dim, args.num_classes, bias=True, n_bits=args.quantizer.wt_bit)
                param.data.copy_(last_quant_linear(param.data))
