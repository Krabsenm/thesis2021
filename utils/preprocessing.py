import tensorflow as tf



@tf.function
def preprocess_input(x):
    x = tf.subtract(x,0.5)
    return x

#@tf.function
#def preprocess_input(x, version=1):
#    x = tf.unstack(x, axis=2)
#    if version == 1:
#        x[0] = tf.subtract(x[0],0.36704) #93.5940
#        x[1] = tf.subtract(x[1],0.41083) #104.7624
#        x[2] = tf.subtract(x[2],0.50661) #129.1863
#    elif version == 2:
#        x[0] = tf.subtract(x[0],0.35881) #91.4953
#        x[1] = tf.subtract(x[1],0.40738) #103.8827
#        x[2] = tf.subtract(x[2],0.51408) #131.0912
#    return tf.stack(x,axis=2)