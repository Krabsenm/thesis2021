from tensorflow.keras.callbacks import Callback
import numpy as np

class GradientNullShuffleCallback(Callback):
    
    def __init__(self, even = None, odd = None):
        super().__init__()
        self.even = even
        self.odd = odd

    def on_train_batch_begin(self, batch, logs=None):
        if batch % 2: # odd
            self.model.layers[self.odd].set_lambda(1)
            self.model.layers[self.even].set_lambda(0)
        else:
            self.model.layers[self.odd].set_lambda(0)
            self.model.layers[self.even].set_lambda(1)