import numpy as np
import tensorflow as tf



class MeanAbsoluteDifference(tf.keras.losses.Loss):

    def __init__(self,name, EO_weights, maximize=True):
        if maximize:
            self.maximize=-1
        else: 
            self.maximize=1
        self.EO_weights = EO_weights # weights calculated from frequency of positive and negative from each class and protected attribute
        super(MeanAbsoluteDifference, self).__init__(name=name)

    def apply_weights(self,Z):
        return self.EO_weights[Z[0]]

    def call(self, y_true, y_pred):
        def remap(a):
            if a > 1: 
                return 1
            else:
                return 0

        w = np.asarray(list(map(self.apply_weights, y_true)))

        Z = tf.cast(tf.map_fn(remap,y_true), dtype=tf.float32)

        wL = tf.multiply(w,tf.math.abs(Z-y_pred))

        return self.maximize*tf.math.reduce_mean(wL)