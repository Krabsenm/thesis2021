import numpy as np
import tensorflow as tf

def pos(Y):
    return np.sum(np.round(Y)).astype(np.float32)

def neg(Y):
    return np.sum(np.logical_not(np.round(Y))).astype(np.float32)

def PR(Y):
    return pos(Y) / (pos(Y) + neg(Y))

def NR(Y):
    return neg(Y) / (pos(Y) + neg(Y))

def TP(Y, Ypred):
    return np.sum(np.multiply(Y, np.round(Ypred))).astype(np.float32)

def FP(Y, Ypred):
    return np.sum(np.multiply(np.logical_not(Y), np.round(Ypred))).astype(np.float32)

def TN(Y, Ypred):
    return np.sum(np.multiply(np.logical_not(Y), np.logical_not(np.round(Ypred)))).astype(np.float32)

def FN(Y, Ypred):
    return np.sum(np.multiply(Y, np.logical_not(np.round(Ypred)))).astype(np.float32)


#note: TPR + FNR = 1; TNR + FPR = 1
def TPR(Y, Ypred):
    return TP(Y, Ypred) / pos(Y)

def FPR(Y, Ypred):
    return FP(Y, Ypred) / neg(Y)

def TNR(Y, Ypred):
    return TN(Y, Ypred) / neg(Y)

def FNR(Y, Ypred):
    return FN(Y, Ypred) / pos(Y)


class EO_Accuracy(tf.keras.metrics.Metric):

    def __init__(self, name='eo_accuracy', **kwargs):
      super(EO_Accuracy, self).__init__(name=name, **kwargs)
      self.ba =  tf.keras.metrics.BinaryAccuracy()
      self.eo_accuracy = self.add_weight(name='eo_a', initializer='zeros')

    def update_state(self, y_true, y_pred, sample_weight=None):
        def remap(a:tf.int32):
            if a > 1: 
                return 1
            return 0

        self.ba.update_state(tf.map_fn(remap,y_true, dtype=tf.int32), y_pred)
        self.eo_accuracy.assign(self.ba.result())

    def result(self):
        return self.eo_accuracy

    def reset_states(self):
        super(EO_Accuracy,self).reset_states()
        self.ba.reset_states()