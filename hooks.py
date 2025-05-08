# hooks.py

# Global dictionary to store feature maps
feature_maps = {}

def register_hook(model, layer_name, key_name):
    layer = dict([*model.named_modules()])[layer_name]
    def hook_fn(module, input, output):
        feature_maps[key_name] = output
    layer.register_forward_hook(hook_fn)
