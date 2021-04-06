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


import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.metrics import CategoricalAccuracy
from tensorflow.keras.metrics import MeanSquaredError, MeanAbsoluteError

from models import build_model_adversarial
from utils import MeanAbsoluteErrorLabels
from datasets import get_tf_dataset, split_dataset, get_dataset, calculate_stats
from utils import setup_output_directories, init_gpus, str_to_list_fn, configure_optimizer_loss
from callbacks import set_up_callbacks
from callbacks import GradientReversalLayerCallback
from evaluate import evaluate_model

def init_ws():
    # get list of subdirs in working directory
    dirs = next(os.walk('.'))[1]

    # remove hidden filders
    dirs = [dir for dir in dirs if not dir[0] == '.']

    # add subdirs to system path
    for dir in dirs:
        sys.path.append(os.path.abspath(dir))


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
    if not args.skip_training:
        output_dir = setup_output_directories(args, args.output_dir)

    model = build_model_adversarial(name=args.model, classification_name=args.labels, input_tensor=(args.img_size, args.img_size, args.channels), task=args.task, nclasses=args.number_classes, weights=args.weights, regularization=args.regularization, regularization_alpha=args.regularization_alpha, const_grl=args.const_grl, capacity=args.capacity)
    if not args.skip_training:
        callbacks = set_up_callbacks(output_dir, args.lr_schedule)
    if not args.skip_training:
        if args.const_grl == 0.0:
            def getLayer(model):
                for i, layer in enumerate(model.layers):
                    if 'grl' in layer.name:
                        return i
                return None

            grl_no = getLayer(model)

            callbacks.insert(1, GradientReversalLayerCallback(grl_no=grl_no, epochs=args.max_epoch))

    optimizer, losses = configure_optimizer_loss(task=args.task, learning_rate=args.lr)

    metrics = [CategoricalAccuracy(name='categorical_accuracy')] if args.task == 'classification' else [MeanAbsoluteErrorLabels] if args.task == 'ordinal' else [MeanSquaredError(name='mean_squared_error'), MeanAbsoluteError(name='mean_absolute_error')]

    model.compile(optimizer=optimizer, loss=losses, loss_weights=[0.667, 0.333], metrics=metrics)

#######################################################################
#                    train model                                      #
#######################################################################
    if not args.skip_training:
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
    else: 
        if args.weights_pretrained != None:
            model.load_weights(args.weights_pretrained)
            output_dir = args.output_dir
        else:
            print("no pretrained weights, exits")
            try:
                sys.exit()
            except:
                raise

#######################################################################
#                    evaluate model                                   #
#######################################################################

    results = evaluate_model(model, test_dataset, test_steps, args.task, test_y[args.labels[0]])
    df = pd.DataFrame(results, columns=["metrics", "score"])
    df.to_csv(os.path.join(output_dir,'results.csv'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Training a model')
 
    # Dataset parameters
    parser.add_argument('--data_path', type=str, default='/media/slow-storage/DataFolder/krabsen/imdb_crop',help='Path to folder containing data files')
    parser.add_argument('--dataset', type=str, default='clean_age_gender_dataset_110920.csv', help='Path to csv file with dataset file paths and labels')
    parser.add_argument('--labels', type=str_to_list_fn(str,','), help='labels for model', default='age,gender')
    parser.add_argument('--test', default="", type=str, help='Name to add to model path')
    parser.add_argument('--data_split', type=str_to_list_fn(float,','), default="0.7,0.15,0.15", help='train, validation,test split')
    parser.add_argument("--generate_dataset_stats", action="store_true", default=False, help="calculates stats of the dataset (not implemented)")
    parser.add_argument("--augmentation", type=str_to_list_fn(str,','), default="flip,color,bright_constrast", help="augmentation transformations to apply")
    parser.add_argument('--channels', type=int, default=3, help="will images be color or grayscale")
    parser.add_argument('--img_size', type=int, default=224, help="image width and height - only square image support")
    parser.add_argument('--number_classes', type=str_to_list_fn(int,','), default='7,2', help="age = 12")
    parser.add_argument('--preprocess', action='store_true', default=True, help="will preprocess images")
    parser.add_argument('--data_fraction', type=float, default=1.0, help='fraction of dataset to use')

    # Training parameters
    parser.add_argument("--output_dir", default="adversarial_models", help="Directory for the trained model, automatically defined if not given.")
    parser.add_argument('--optimizer', default='adam', help='optimizer')
#    parser.add_argument('--loss', default='categorical_crossentropy', help='loss function')
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument('--max_epoch', type=int, default=48, help='max epoch')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size')
    parser.add_argument("--multi_gpu", action="store_true", default=False)
    parser.add_argument("--lr_schedule", default="plateau", help="Either None, plateau")
    parser.add_argument("--skip_training", action="store_true", default=False, help="expects that weights have been loaded and goes to evaluate")
    

    
    # Model parameters
    parser.add_argument('--model', type=str, default="mobilenet", choices=['resnet', 'resnet101','vggface', 'mobilenet'],help="Model type to use, vggface or resnet50.")
    parser.add_argument('--task', type=str, default="classification", choices=['classification', 'regression', 'ordinal'], help='which task to train for, classification or regression')
#    parser.add_argument('--nclasses', type=int, default=12, help="number of classes, age=12")
    parser.add_argument("--weights", default='imagenet', type=str, choices=[None, 'imagenet'], help="Weights to initialize the model: None or imagenet")
    parser.add_argument("--weights_pretrained", default=None, type=str, help="path to weights .hdf5 file")
    parser.add_argument('--regularization', default=None, type=str, choices=[None, 'l1', 'l2'], help="regularization will be added to all supported layers")
    parser.add_argument('--regularization_alpha', default=0.01, type=float, help="regularization hyperparameter")
    parser.add_argument('--const_grl', default=0.0, type=float, help="regularization hyperparameter")
    parser.add_argument('--capacity', default=512, type=int, help="number of weights in adversary")
    args = parser.parse_args()

    try:
        main(args)
    except:
        raise   # Reraise exception if there is one

