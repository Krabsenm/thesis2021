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

from models import build_model
from datasets import get_tf_dataset, split_dataset, get_dataset, calculate_stats
from utils import setup_output_directories, init_gpus, str_to_list_fn, configure_optimizer_loss
from callbacks import set_up_callbacks

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

    _, _, _, _, test_x, test_y = split_dataset(x_data, y_data, args.data_split)

    test_dataset, test_steps = get_tf_dataset(images=test_x    , labels=test_y    , channels=args.channels, task=args.task, batch_size=args.batch_size, image_size=args.img_size, pre_process=args.preprocess,  augmentation=[], train=False)

#######################################################################
#                    build model                                      #
#######################################################################


    model = build_model(name=args.model, classification_name=args.labels, input_tensor=(args.img_size, args.img_size, args.channels), task=args.task, nclasses=[args.number_classes], weights=args.weights, regularization=args.regularization, regularization_alpha=args.regularization_alpha)

    model.load_weights(args.weights)

#######################################################################
#                    predict model                                    #
#######################################################################

    age_predictions = np.asarray([[1]])
    age_labels = np.asarray([[1]])
    gender_labels = np.asarray([[1]])
    for i,(x,(age_label,gender_label)) in enumerate(test_dataset):
        print("predicting batch: {}".format(i))

        age_prediction = model.predict(x)

        age_prediction = np.round(age_prediction*96)
        age_label = np.round(age_label*96)
        
        age_label = np.expand_dims(age_label, axis=1)
        gender_label = np.expand_dims(gender_label, axis=1)

        age_predictions = np.concatenate([age_predictions, age_prediction])
        age_labels = np.concatenate([age_labels, age_label]) 
        gender_labels = np.concatenate([gender_labels, gender_label])

    age_predictions = age_predictions[1:]
    age_labels = age_labels[1:]
    gender_labels = gender_labels[1:]

#######################################################################
#                    save results                                     #
#######################################################################

    data = np.concatenate([age_labels, gender_labels, age_predictions], axis=1)
    df = pd.DataFrame(data=data, columns=['age_labels', 'gender_labels', 'age_predictions'])

    df.to_csv(os.path.join(os.path.dirname(args.weights),'results_eval.csv'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Training a model')
 
    # Dataset parameters
    parser.add_argument('--data_path', help='Path to folder containing data files')
    parser.add_argument('--dataset', help='Path to csv file with dataset file paths and labels')
    parser.add_argument('--labels', type=str_to_list_fn(str,','), help='labels for model', default='age,gender')
    parser.add_argument('--name', type=str, help='Name to add to model path')
    parser.add_argument('--data_split', type=str_to_list_fn(float,','), default="0.7,0.15,0.15", help='train, validation,test split')
    parser.add_argument("--generate_dataset_stats", action="store_true", default=False, help="calculates stats of the dataset (not implemented)")
    parser.add_argument("--augmentation", type=str_to_list_fn(str,','), default="flip, color,bright_constrast", help="augmentation transformations to apply")
    parser.add_argument('--channels', type=int, default=3, help="will images be color or grayscale")
    parser.add_argument('--img_size', type=int, default=224, help="image width and height - only square image support")
    parser.add_argument('--number_classes', type=int, default=12, help="age = 12")
    parser.add_argument('--preprocess', action='store_true', default=False, help="will preprocess images")
    parser.add_argument('--batch_size', type=int, default=32, help='batch size')
    parser.add_argument('--regularization', default=None, type=str, choices=[None, 'l1', 'l2'], help="regularization will be added to all supported layers")
    parser.add_argument('--regularization_alpha', default=0.01, type=float, help="regularization hyperparameter")


    # Model parameters
    parser.add_argument('--model', type=str, default="resnet", help="Model type to use, vgg_face or resnet50.")
    parser.add_argument('--task', type=str, default="classification", choices=['classification', 'regression'], help='which task to train for, classification or regression')
#    parser.add_argument('--nclasses', type=int, default=12, help="number of classes, age=12")
    parser.add_argument("--weights", default=None, type=str, help="path to weights for trained model /trained_models/resnet...")
    
    args = parser.parse_args()

    try:
        main(args)
    except:
        raise   # Reraise exception if there is one

