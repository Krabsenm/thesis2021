import tensorflow as tf

def MeanAbsoluteErrorLabels(y_true, y_pred):
  # Assume that y_pred is cumulative logits from our CoralOrdinal layer.
  
  # Predict the label as in Cao et al. - using cumulative probabilities
  #cum_probs = tf.map_fn(tf.math.sigmoid, y_pred)
  
  # Calculate the labels using the style of Cao et al.
  # above_thresh = tf.map_fn(lambda x: tf.cast(x > 0.5, tf.float32), cum_probs)
  
  # Skip sigmoid and just operate on logit scale, since logit > 0 is
  # equivalent to prob > 0.5.
  above_thresh = tf.map_fn(lambda x: tf.cast(x > 0., tf.float32), y_pred)
  
  # Sum across columns so that we estimate how many cumulative thresholds are passed.
  labels_v2 = tf.reduce_sum(above_thresh, axis = 1)
  
  # This can convert to an integer, which will mess with the calculations.
  # labels_v2 = tf.cast(labels_v2, y_true.dtype)
  
  return tf.reduce_mean(tf.abs(y_true - labels_v2), axis = -1)
