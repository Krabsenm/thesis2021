
import tensorflow as tf 
#import load_dataset.file_path_load as mkdir
from datetime import datetime
import os

def set_up_callbacks(output_dir, schedule='None'):

    callbacks = []

    lr_schedule = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', 
                                                       factor=0.10,
                                                       patience=5, 
                                                       min_lr=0.0005,
                                                       mode='min', 
                                                       #cooldown=5,
                                                       verbose=1)

    if schedule != 'None':
        callbacks.append(lr_schedule)


    weight_file = "weights_{val_loss:.4f}.hdf5"

    save_weights = os.path.join(output_dir, weight_file)

    # safe best weights 
    checkpointCallback = tf.keras.callbacks.ModelCheckpoint(save_weights, 
                                                            monitor='val_loss', 
                                                            verbose=1, 
                                                            save_freq='epoch',
                                                            save_weights_only=True,
                                                            save_best_only=True)

    callbacks.append(checkpointCallback)

    # stop training if model doesn't improve 
    #earlystoppingCallback = tf.keras.callbacks.EarlyStopping(monitor='val_loss',
    #                                                        patience=15, 
    #                                                        verbose=1)

    # save tensorboard log data
    tensorboardCallback = tf.keras.callbacks.TensorBoard(log_dir=output_dir)

    callbacks.append(tensorboardCallback)

    return callbacks


def set_up_callbacks_adv(output_dir, schedule):

    callbacks = []

    lr_schedule = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_age_loss', 
                                                       factor=0.05,
                                                       patience=5, 
                                                       min_lr=0.0005,
                                                       mode='min', 
                                                       #cooldown=5,
                                                       verbose=1)

    if schedule != 'None':
        callbacks.append(lr_schedule)


    weight_file = "weights_{val_age_loss:.4f}.hdf5"

    save_weights = os.path.join(output_dir, weight_file)

    # safe best weights 
    checkpointCallback = tf.keras.callbacks.ModelCheckpoint(save_weights, 
                                                            monitor='val_age_loss', 
                                                            verbose=1, 
                                                            save_freq='epoch',
                                                            save_weights_only=True,
                                                            save_best_only=True)

    callbacks.append(checkpointCallback)

    # stop training if model doesn't improve 
    #earlystoppingCallback = tf.keras.callbacks.EarlyStopping(monitor='val_loss',
    #                                                        patience=15, 
    #                                                        verbose=1)

    # save tensorboard log data
    tensorboardCallback = tf.keras.callbacks.TensorBoard(log_dir=output_dir)

    callbacks.append(tensorboardCallback)

    return callbacks