import tensorflow as tf

@tf.custom_gradient
def gradient_reversal(x, lamb):
    y = tf.identity(x)
    def grad(dy):
        return [tf.negative(dy)*lamb, tf.negative(dy)*lamb]
    return y, grad

@tf.custom_gradient
def gradient_scaling(x, lamb):
    y = tf.identity(x)
    def grad(dy):
        return [dy*lamb, dy*lamb]
    return y, grad

class GradientReversalLayer(tf.keras.layers.Layer):
    def __init__(self, lamb=0.0, name='grl', undo_grl=False):
        super().__init__(name=name)
        self.lamb = lamb
        if undo_grl:
            self.call_function = gradient_scaling
        else:
            self.call_function = gradient_reversal

    def call(self, x):
        return self.call_function(x, self.lamb)

    def set_lambda(self, lamb):
        self.lamb = lamb     #.assign(lamb)

    def get_lambda(self):
        return self.lamb #.numpy()

    def get_config(self):
        """
        Return configuration for Layer
        """
        conf = super().get_config()
        conf.update({"lamb":self.lamb})
        return conf