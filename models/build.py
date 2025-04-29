from utils import get_numclasses
from utils.registry import Registry
import models
from utils.quantizer import *
#
ENCODER_REGISTRY = Registry("ENCODER")
ENCODER_REGISTRY.__doc__ = """
Registry for encoder
"""

__all__ = ['get_model', 'build_encoder']

# def get_model(args,trainset = None):
#     num_classes = get_numclasses(args, trainset)
#     print("=> Creating model '{}'".format(args.arch))
#     print("Model Option")
#     print(" 1) use_pretrained =", args.use_pretrained)
#     print(" 2) No_transfer_learning =", args.No_transfer_learning)
#     print(" 3) use_bn =", args.use_bn)
#     print(" 4) use_pre_fc =", args.use_pre_fc)
#     print(" 5) use_bn_layer =", args.use_bn_layer)
#     model = models.__dict__[args.arch](args, num_classes=num_classes, l2_norm=args.l2_norm, use_pretrained = args.use_pretrained, transfer_learning = not(args.No_transfer_learning), use_bn = args.use_bn, use_pre_fc = args.use_pre_fc, use_bn_layer = args.use_bn_layer)
#     #model = models.__dict__[args.arch](num_classes=num_classes, l2_norm=args.l2_norm, use_pretrained = args.use_pretrained, transfer_learning = not(args.No_transfer_learning), use_bn = args.use_bn, use_pre_fc = args.use_pre_fc, use_bn_layer = args.use_bn_layer)
#     return model

def build_encoder(args):

    num_classes = get_numclasses(args)

    if args.verbose:
        print(ENCODER_REGISTRY)

    print(f"=> Creating model '{args.model.name}, pretrained={args.model.pretrained}'")
    
    # Activation 양자화
    if args.quantizer.LPT_name == 'LPT':
        quant_model = lambda: BlockQuantizer( args.quantizer.quantization_bits, args.quantizer.quantization_bits, args.quantizer.quant_type,
                                             args.quantizer.small_block, args.quantizer.block_dim, 'DANUQ')
        
        quant_model2 = lambda: BlockQuantizer_ReLU( args.quantizer.quantization_bits, args.quantizer.quantization_bits, args.quantizer.quant_type,
                                             args.quantizer.small_block, args.quantizer.block_dim)
        
        quant_model3 = lambda: BlockQuantizer( args.quantizer.quantization_bits, args.quantizer.quantization_bits, args.quantizer.quant_type,
                                             args.quantizer.small_block, args.quantizer.block_dim, "BFP")
        
        quant_function = quant_model()
        quant_function2 = quant_model2()
        quant_function3 = quant_model3()
        
        server_model = lambda: BlockQuantizer(-1 , -1, args.quantizer.quant_type,
                                             args.quantizer.small_block, args.quantizer.block_dim)
        
        server_model2 = lambda: BlockQuantizer_ReLU(-1 , -1, args.quantizer.quant_type,
                                             args.quantizer.small_block, args.quantizer.block_dim)
        
        server_model3 = lambda: BlockQuantizer(-1 , -1, args.quantizer.quant_type,
                                             args.quantizer.small_block, args.quantizer.block_dim)
         
        server_quant_function = server_model()
        server_quant_function2 = server_model2()
        server_quant_function3 = server_model3()
        
    
    # if args.quantizer.get("quant", False):
    #     args.model["quant"] = quant_function
    # else:
    #     args.model["quant"] = None
    
    
    if args.quantizer.LPT_name == 'LPT':
        encoder = ENCODER_REGISTRY.get(args.model.name)(args, num_classes, quant = quant_function, quant2 = quant_function2, quant3 = quant_function3, **args.model) if len(args.model.name) > 0 else None
        eval_encoder= ENCODER_REGISTRY.get(args.model.name)(args, num_classes, quant = server_quant_function, quant2 = server_quant_function2, quant3 = server_quant_function3, **args.model) if len(args.model.name) > 0 else None
    else:
        encoder = ENCODER_REGISTRY.get(args.model.name)(args, num_classes, **args.model) if len(args.model.name) > 0 else None
        eval_encoder = ENCODER_REGISTRY.get(args.model.name)(args, num_classes, **args.model) if len(args.model.name) > 0 else None
    
    return encoder, eval_encoder
