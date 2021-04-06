from tensorflow.keras.callbacks import Callback
import numpy as np


class GradientReversalLayerCallback(Callback):
    
    def __init__(self, grl_no = None, gamma = 10, epochs = None):
        super().__init__()
        self.grl_no = grl_no
        self.epochs = epochs
        self.gamma = gamma

    def calculate_lambda(self,epoch):
        return (2/(1+np.exp(-self.gamma*epoch/self.epochs)))-1

#    def on_train_batch_begin(self, batch, logs=None):
#        if batch % 2: # odd
#            self.model.layers[self.grl_no].set_lambda(1.0)
#        else:
#            self.model.layers[self.grl_no].set_lambda(0.0)

    def on_epoch_end(self, epoch, logs=None):
        if self.grl_no != None:
            self.model.layers[self.grl_no].set_lambda(self.calculate_lambda(epoch))
            print("gradient revelsal layer lambda: {:.2f}".format(self.model.layers[self.grl_no].get_lambda()))


