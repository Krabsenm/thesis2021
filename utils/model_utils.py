import tensorflow as tf

def set_conv_trainable_false(model):
    for layer in model.layers:
        if 'conv' in layer.name:
            layer.trainable = False
    return model
