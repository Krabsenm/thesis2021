import tensorflow as tf

@tf.custom_gradient
def gradient_scaling(x, lamb):
    y = tf.identity(x)
    def grad(dy):
        return [dy*lamb, dy*lamb]
    return y, grad

class GradientNull(tf.keras.layers.Layer):
    def __init__(self, lamb=0.0, name='gradientNull'):
        super().__init__(name=name)
        self.lamb = lamb

    def call(self, x):
        return gradient_scaling(x, self.lamb)

    def set_lambda(self, lamb):
        self.lamb = lamb 

    def get_lambda(self):
        return self.lamb

