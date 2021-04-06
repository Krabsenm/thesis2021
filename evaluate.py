from tensorflow.keras import Model
import tensorflow as tf
from utils import show_batch
import pandas as pd


def evaluate_model(model:Model, test:tf.data.Dataset, steps:int, task:str, labels:list) -> list:

    if task == 'ordinal':
        tensor_probs = model.predict(test, verbose=1, steps=steps)
        probs_df = pd.DataFrame(tensor_probs.numpy())
        labels_preds = probs_df.idxmax(axis = 1)
        result = np.mean(labels == labels_preds)
    else:
        result = model.evaluate(test, verbose=1, steps=steps)
        names = model.metrics_names

        output = [[name, value] for name, value in zip(names, result)]
    return output



def predict_and_show(model:Model, test:tf.data.Dataset, steps:int) -> list:

    x,y = next(iter(test))


    y_pred = model.predict(x)

    show_batch(x, y, y_pred)
