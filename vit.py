import timm

print([m for m in timm.list_models(pretrained=True) if "deit_tiny" in m])
print(timm.list_models('vit_tiny*')) 
print(timm.list_models('deit_tiny*'))