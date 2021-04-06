from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard,ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import CategoricalCrossentropy, BinaryCrossentropy
from tensorflow.keras.losses import MeanSquaredError
import tensorflow as tf
import numpy as np
#from ordinal_loss import OrdinalCrossEntropy

import tensorflow as tf
import numpy as np

# The outer function is a constructor to create a loss function using a certain number of classes.
class OrdinalCrossEntropy(tf.keras.losses.Loss):
  
  def __init__(self,
               num_classes = None,
               importance = None,
               from_type = "ordinal_logits",
               name = "ordinal_crossent", **kwargs):
    """ Cross-entropy loss designed for ordinal outcomes.
    
    Args:
      num_classes: (Optional) how many ranks (aka labels or values) are in the ordinal variable.
        If not provided it will be automatically determined by on the prediction data structure.
      importance: (Optional) importance weights for each binary classification task.
      from_type: one of "ordinal_logits" (default), "logits", or "probs".
        Ordinal logits are the output of a CoralOrdinal() layer with no activation.
        (Not yet implemented) Logits are the output of a dense layer with no activation.
        (Not yet implemented) Probs are the probability outputs of a softmax or ordinal_softmax layer.
    """
    super(OrdinalCrossEntropy, self).__init__(name = name, **kwargs)
    
    self.num_classes = num_classes
      
    #self.importance_weights = importance
    if importance is None:
      self.importance_weights = tf.ones(self.num_classes - 1, dtype = tf.float32)
    else:
      self.importance_weights = tf.cast(importance, dtype = tf.float32)
      
    self.from_type = from_type

  def label_to_levels(self,label):
    # Original code that we are trying to replicate:
    # levels = [1] * label + [0] * (self.num_classes - 1 - label)
    
    num_ones = tf.cast(tf.argmax(label, axis=1), tf.int32)
    label_vec = tf.ones(shape = (num_ones), dtype = tf.int32)
    num_zeros = self.num_classes - 1 - num_ones
    zero_vec = tf.zeros(shape = (num_zeros), dtype = tf.int32)
    
    levels = tf.concat([label_vec, zero_vec], axis = 0)
    return tf.cast(levels, tf.float32)
    
  def call(self, y_true, y_pred):

    # Ensure that y_true is the same type as y_pred (presumably a float).
    #y_pred = ops.convert_to_tensor_v2(y_pred)
    y_pred = tf.convert_to_tensor(y_pred)
    y_true = tf.cast(y_true, y_pred.dtype)
    
    if self.num_classes is None:
      # Determine number of classes based on prediction shape.
      if self.from_type == "ordinal_logits":
        # Number of classes = number of columns + 1
        self.num_classes = y_pred.shape[1] + 1
      else:
        self.num_classes = y_pred.shape[1]

    # Convert each true label to a vector of ordinal level indicators.
    tf_levels = tf.map_fn(self.label_to_levels, y_true)
    
    
    if self.from_type == "ordinal_logits":
      #return ordinal_loss(y_pred, tf_levels, self.importance_weights)
      return ordinal_loss(y_pred, tf_levels, self.importance_weights)
    elif self.from_type == "probs":
      raise Exception("not yet implemented")
    elif self.from_type == "logits":
      raise Exception("not yet implemented")
    else:
      raise Exception("Unknown from_type value " + self.from_type +
                      " in OrdinalCrossEntropy()")
    
#def ordinal_loss(logits, levels, importance):
def ordinal_loss(logits, levels, importance):
    val = (-tf.reduce_sum((tf.math.log_sigmoid(logits) * levels
                      + (tf.math.log_sigmoid(logits) - logits) * (1 - levels)) * importance,
           axis = 1))
    return tf.reduce_mean(val)


def configure_optimizer_loss_2nd(optimizer_name='adam', names=[], learning_rate=0.001):
     return config_optimizer(optimizer_name, learning_rate), {name:config_loss_2nd(name) for name in names}


def configure_optimizer_loss(optimizer_name='adam', task='classification', loss_name='categorical_crossentropy', learning_rate=0.001):
     return config_optimizer(optimizer_name, learning_rate), config_loss_2nd(loss_name)

def config_optimizer(name='adam', learning_rate=0.0001):
     return Adam(learning_rate=learning_rate, beta_1=0.9, beta_2=0.999, epsilon=1e-07, amsgrad=False,name=name)


def config_loss_2nd(name):
  if name == 'race':
    return CategoricalCrossentropy(name=name)
  else:
    return BinaryCrossentropy(name=name) 
    

def config_loss(task = 'classification', name='categorical_crossentropy'):
     if task == 'classification':
          return CategoricalCrossentropy(name=name)
     elif task == 'ordinal':
          return OrdinalCrossEntropy(7) 
     else:
          return MeanSquaredError(name='mean_squared_error')
