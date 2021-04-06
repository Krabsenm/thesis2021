from tensorflow.keras.callbacks import Callback
import numpy as np
import tensorflow as tf
from sklearn.metrics import confution_matrix

class ValidationCallback(Callback):

    def __init__(
        self, dataset:tf.data.Dataset, utility_task="age", bias_task="gender"):
        super().__init__()
        self._utility_task = utility_task
        self._bias_task = bias_task

        # Add output extraction to dataset
        self._ys = {}
        def get_outputs_py(age, gender):
            self._ys["age"] = np.concatenate([self._ys.get("age", []), age.numpy()])
            self._ys["gender"] = np.concatenate([self._ys.get("gender", []), gender.numpy()])

        @tf.function
        def get_outputs(X, y):
            tf.py_function(get_outputs_py, [y["age"], y["gender"], ()])
            return (X, y)
        self._val_dataset = dataset.map(get_outputs, num_parallel_calls=1)

    def on_epoch_end(self, epoch, logs=None):
        ### predict entire validation dataset
        logs = logs or {}

        # Clear output and predict
        self._ys.clear()
        val_pred = self.model.predict(self._val_dataset, verbose=True)
        if len(self.model.outputs) == 1:
            val_pred = [val_pred]
        num_pred = len(val_pred[0])

        # Get prediction ys from ys on object.
        for key in self._ys.keys():
            self._ys[key] = self._ys[key][:num_pred]


        def per_class_cm(y_true, y_pred, y_bias):
            y_true_c = [y_true[_class] for _class in y_bias]
            y_pred_c = [y_pred[_class] for _class in y_bias]
            
            cm_c = [confution_matrix(yt,yp) for yt,yp in zip(y_true_c, y_pred_c)]

            return cm_c

        # Define pr auc metric and set name
        def pr_auc(y_pred, y_true):
            y_true = y_true.astype(int)
            if len(np.bincount(y_true)) < 2:
                return np.nan
            if len(y_pred) != len(y_true):
                raise ValueError("y_pred and y_true needs to be the same length.")
            try:
                precision, recall, _ = precision_recall_curve(y_true, y_pred, pos_label=1)
                return auc(recall, precision)
            except ValueError as _:
                return np.nan
        pr_auc.name = "PR_AUC"

        # For all metrics (ROC and PR) compute validation AUCs
        loss_weights_list = []
        if isinstance(self.model.loss_weights, dict):
            for name in self.model.output_names:
                loss_weights_list.append(self.model.loss_weights.get(name, 1.))
        elif isinstance(self.model.loss_weights, list):
            loss_weights_list.extend(self.model.loss_weights)

        updated_matrics = {}
        total_loss = 0
        for key, key_pred, loss_weight in zip(self.model.output_names, val_pred, loss_weights_list):
            key_loss = tf.keras.backend.eval(tf.reduce_mean(self.model.loss(self._ys[key], key_pred)))
            total_loss += key_loss * loss_weight
            updated_matrics["val_%s_loss" % key] = key_loss



        y_bias = self._ys[self._bias_task]
        
        y_bias = np.flip(np.transpose(y_bias),axis=1)

        cm_per_class = per_class_cm(self._ys[self._utility_task],val_pred[self._utility_task], y_bias)

        #################################################################################################
        #                                   I got to here                                               #
        #################################################################################################

        # Remove all nan keys
        updated_matrics = {key:val for key, val in updated_matrics.items() if not np.isnan(val)}

        print(" - ".join(["%s: %0.4f" % (key, val) for key, val in updated_matrics.items()]))

        # Add metrics to history
        logs.update(updated_matrics)
        super().on_epoch_end(epoch, logs)