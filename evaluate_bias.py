import argparse
import datetime
import logging
import json
import math
import glob
import os
import sys
from functools import partial
import pandas as pd
from datetime import datetime as dtdt

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.metrics import CategoricalAccuracy
from tensorflow.keras.metrics import MeanSquaredError, MeanAbsoluteError

from models import build_model, load_weights
from datasets import get_tf_dataset, split_dataset, get_dataset, calculate_stats
from utils import init_gpus, str_to_list_fn, configure_optimizer_loss
from callbacks import set_up_callbacks
from evaluate import evaluate_model

def init_ws():
    # get list of subdirs in working directory
    dirs = next(os.walk('.'))[1]

    # remove hidden filders
    dirs = [dir for dir in dirs if not dir[0] == '.']

    # add subdirs to system path
    for dir in dirs:
        sys.path.append(os.path.abspath(dir))


def get_output_dir(args):
    """
    Build a output directory based on script arguments
    Only really usefull for I3D trainers.
    """

    output_dir = 'bias_' + args.bias_check + '_' + dtdt.now().strftime("%H%M_%d%m")

    if 'adversarial' in args.output_dir:
        output_dir = 'adversarial_' + output_dir

    if args.output_dir == 'test':
        output_dir = 'test_' + output_dir

    output_dir = os.path.join('bias_models', output_dir)

    for i in range(10):
        output_dir_test = output_dir if i == 0 else output_dir + "_" + str(i)
        try:
            os.makedirs(output_dir_test)
            break
        except FileExistsError as ex:
            if i == 9:
                raise ex
    output_dir = output_dir_test
    return output_dir


def main(args):

    # add project dirs to system path
    init_ws()

    init_gpus()

#######################################################################
#                    build dataset                                    #
#######################################################################

    x_data, y_data = get_dataset(args.data_path,args.dataset, args.labels, args.task)

    train_x, train_y, validate_x, validate_y, test_x, test_y = split_dataset(x_data, y_data, args.data_split)

    train_dataset, train_steps         = get_tf_dataset(images=train_x   , labels=train_y   , channels=args.channels, task=args.task, batch_size=args.batch_size, image_size=args.img_size, pre_process=args.preprocess, augmentation=args.augmentation, train=True)

    validate_dataset, validation_steps = get_tf_dataset(images=validate_x, labels=validate_y, channels=args.channels, task=args.task, batch_size=args.batch_size, image_size=args.img_size, pre_process=args.preprocess,  augmentation=[], train=False)

    test_dataset, test_steps           = get_tf_dataset(images=test_x    , labels=test_y    , channels=args.channels, task=args.task, batch_size=args.batch_size, image_size=args.img_size, pre_process=args.preprocess,  augmentation=[], train=False)

#######################################################################
#                    build model                                      #
#######################################################################

    output_dir = get_output_dir(args)

    model = build_model(name=args.model, classification_name=args.labels, input_tensor=(args.img_size, args.img_size, args.channels), task=args.task, nclasses=[2], weights=None, regularization=None, regularization_alpha=0.0)

    if args.only_predict:
        model.load_weights(args.weights) # make sure error if non-matching weights are loaded
    else:
        load_weights(model, args.weights) #

    for l in model.layers: # MK
        if l.name != 'gender':
            l.trainable = False  # freeze model except new output

    optimizer, losses = configure_optimizer_loss(task=args.task)

    callbacks = set_up_callbacks(output_dir, args.lr_schedule)

    metrics = [CategoricalAccuracy(name='categorical_accuracy')] if args.task == 'classification' else [MeanSquaredError(name='mean_squared_error'), MeanAbsoluteError(name='mean_absolute_error')]

    model.compile(optimizer=optimizer, loss=losses, metrics=metrics)


#######################################################################
#                    train model                                      #
#######################################################################

    if not args.only_predict:
        model.fit(train_dataset,
              epochs=args.max_epoch, 
              verbose=1, 
              callbacks=callbacks, 
              validation_data=validate_dataset, 
              class_weight=None,
              sample_weight=None, 
              initial_epoch=0, 
              steps_per_epoch=train_steps,
              validation_freq=1,
              validation_steps=validation_steps)

#######################################################################
#                    evaluate model                                   #
#######################################################################

    results = evaluate_model(model, test_dataset, test_steps, args.task, args.labels)
    df = pd.DataFrame(results, columns=["metrics", "score"])
    df.to_csv(os.path.join(output_dir,'results.csv'))



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Training a model')
 
    # Dataset parameters
    parser.add_argument('--data_path', type=str,default="/media/slow-storage/krabsen/imdb_crop", help='Path to folder containing data files')
    parser.add_argument('--dataset', type=str, default="clean_age_gender_dataset_110920.csv", help='Path to csv file with dataset file paths and labels')
    parser.add_argument('--labels', type=str_to_list_fn(str,','), help='labels for model', default='gender')
    parser.add_argument('--name', type=str, help='Name to add to model path')
    parser.add_argument('--data_split', type=str_to_list_fn(float,','), default="0.7,0.15,0.15", help='train, validation,test split')
    parser.add_argument("--generate_dataset_stats", action="store_true", default=False, help="calculates stats of the dataset (not implemented)")
    parser.add_argument("--augmentation", type=str_to_list_fn(str,','), default="flip, color,bright_constrast", help="augmentation transformations to apply")
    parser.add_argument('--channels', type=int, default=3, help="will images be color or grayscale")
    parser.add_argument('--img_size', type=int, default=224, help="image width and height - only square image support")
    parser.add_argument('--number_classes', type=int, default=2, help="age = 12")
    parser.add_argument('--preprocess', action='store_true', default=True, help="will preprocess images")

    # Training parameters
    parser.add_argument("--output_dir", default="", help="Directory for the trained model, automatically defined if not given.")
    parser.add_argument('--optimizer', default='adam', help='optimizer')
#    parser.add_argument('--loss', default='categorical_crossentropy', help='loss function')
#    parser.add_argument("--lr", type=float, default=0.0001, help="Learning rate")
    parser.add_argument('--max_epoch', type=int, default=48, help='max epoch')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size')
    parser.add_argument("--multli_gpu", action="store_true", default=False)
    parser.add_argument("--lr_schedule", default="plateau", help="Either None, plateau")
    
    # Model parameters
    parser.add_argument('--model', type=str, default="resnet", help="Model type to use, vgg_face or resnet50.")
    parser.add_argument('--task', type=str, default="classification", choices=['classification', 'regression'], help='which task to train for, classification or regression')
#    parser.add_argument('--nclasses', type=int, default=12, help="number of classes, age=12")
    parser.add_argument("--weights", default=None, type=str, help="pretrained weights for model")
    parser.add_argument("--bias_check", type=str, default='Gender', help="adds the corresponding output to the model which is checked for corresponding bias")#,  #MK 
    parser.add_argument("--only_predict", action="store_true", default=False)
    args = parser.parse_args()

    try:
        main(args)
    except:
        raise   # Reraise exception if there is one

