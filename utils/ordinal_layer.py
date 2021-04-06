import tensorflow as tf

class CoralOrdinal(tf.keras.layers.Layer):

  def __init__(self, num_classes, activation = None, **kwargs):
    """ Ordinal output layer, which produces ordinal logits by default.
    
    Args:
      num_classes: how many ranks (aka labels or values) are in the ordinal variable.
      activation: (Optional) Activation function to use. The default of None produces
        ordinal logits, but passing "ordinal_softmax" will cause the layer to output
        a probability prediction for each label.
    """
    
    if 'input_shape' not in kwargs and 'input_dim' in kwargs:
      kwargs['input_shape'] = (kwargs.pop('input_dim'),)

    # Pass any additional keyword arguments to Layer() (i.e. name, dtype)
    super(CoralOrdinal, self).__init__(**kwargs)
    self.num_classes = num_classes
    self.activation = activation
    
  def build(self, input_shape):

    # Single fully-connected neuron - this is the latent variable.
    num_units = 1

    self.fc = self.add_weight(shape = (input_shape[-1], num_units),
                              name = self.name + "_latent",
                              initializer = 'glorot_uniform',
                              dtype = tf.float32,
                              trainable = True)
                              
    # num_classes - 1 bias terms, defaulting to 0.
    self.linear_1_bias = self.add_weight(shape = (self.num_classes - 1, ),
    # Need a unique name if there are multiple coral_ordinal layers.
                                         name = self.name + "_bias",
                                         initializer = 'zeros',
                                         # Not sure if this is necessary:
                                         dtype = tf.float32,
                                         trainable = True)

  # This defines the forward pass.
  def call(self, inputs):
    fc_inputs = tf.matmul(inputs, self.fc)

    logits = fc_inputs + self.linear_1_bias
    
    if self.activation is None:
      outputs = logits
    else:
      outputs = self.activation(logits)

    return outputs
  
  # This allows for serialization supposedly.
  def get_config(self):
    config = super(CoralOrdinal, self).get_config()
    config.update({'num_classes': self.num_classes})
    return config