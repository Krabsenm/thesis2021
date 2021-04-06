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
from tensorflow.keras.metrics import BinaryAccuracy
from tensorflow.keras.metrics import MeanSquaredError, MeanAbsoluteError

from models import build_model_2nd
from utils import MeanAbsoluteErrorLabels
from datasets import get_tf_dataset_2nd, get_and_split_dataset, calculate_stats, get_and_split_utk, insert_bias
from utils import setup_output_directories_2nd, init_gpus, str_to_list_fn, configure_optimizer_loss_2nd
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


def main(args):

    # add project dirs to system path
    init_ws()

    init_gpus()

#######################################################################
#                    build dataset                                    #
#######################################################################
    if 'celeba' in args.dataset:
        train_x, validate_x, test_x, train_y, validate_y, test_y = get_and_split_dataset(args.data_path, args.dataset, args.labels + [args.protected_attr])
        num_class = None
    elif 'utk' in args.dataset:
        train_x, validate_x, test_x, train_y, validate_y, test_y = get_and_split_utk(args.data_path, args.labels)

    if args.amplify_bias:
        train_x, train_y = insert_bias(train_x, train_y)


    train_dataset, train_steps         = get_tf_dataset_2nd(images=train_x   , labels=train_y, number_classes = num_class,   batch_size=args.batch_size, image_size=(args.img_h, args.img_w), augmentation=args.augmentation, train=True)

    validate_dataset, validation_steps = get_tf_dataset_2nd(images=validate_x, labels=validate_y, number_classes = num_class, batch_size=args.batch_size, image_size=(args.img_h, args.img_w),  augmentation=[], train=False)

    test_dataset, test_steps           = get_tf_dataset_2nd(images=test_x,     labels=test_y,  number_classes = num_class,  batch_size=args.batch_size, image_size=(args.img_h, args.img_w),  augmentation=[], train=False)


#######################################################################
#                    build model                                      #
#######################################################################

    output_dir = setup_output_directories_2nd(args, args.output_dir)

    model = build_model_2nd(name=args.model, classification_name=args.labels, number_classes=args.number_classes, input_tensor=(args.img_h, args.img_w, 3), weights=args.weights)

    optimizer, losses = configure_optimizer_loss_2nd(names=args.labels, learning_rate=args.lr)

    callbacks = set_up_callbacks(output_dir, args.lr_schedule)

#    metrics = [BinaryAccuracy(name='accuracy')]

    model.compile(optimizer=optimizer, loss=losses, metrics=['accuracy'])

#######################################################################
#                    train model                                      #
#######################################################################

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

    results = model.predict(test_dataset)
    df = pd.DataFrame(results, columns=args.labels)
    df.to_csv(os.path.join(output_dir,'results.csv'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Training a model')
 
    # Dataset parameters
    parser.add_argument('--data_path', type=str, default='/media/slow-storage/DataFolder/krabsen/',help='Path to folder containing data files')
    parser.add_argument('--dataset', type=str, default='list_attr_celeba.txt', help='Path to csv file with dataset file paths and labels')
    parser.add_argument('--labels', type=str_to_list_fn(str,','), help='labels for model', default='Attractive,Eyeglasses,Mouth_Slightly_Open,Pointy_Nose,Smiling')
    parser.add_argument('--protected_attr', type=str, help='labels for model', default='Male', choices=['Male','Pale_Skin','Young'])
    parser.add_argument('--test', default="", type=str, help='Name to add to model path')
    parser.add_argument("--augmentation", type=str_to_list_fn(str,','), default="flip,color,bright_constrast", help="augmentation transformations to apply")
    parser.add_argument('--img_h', type=int, default=218, help="image height")
    parser.add_argument('--img_w', type=int, default=178, help="image width")

    # Training parameters
    parser.add_argument("--output_dir", default="trained_models_2nd", help="Directory for the trained model, automatically defined if not given.")
    parser.add_argument('--optimizer', default='adam', help='optimizer')
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument('--max_epoch', type=int, default=24, help='max epoch')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size')
    parser.add_argument("--multi_gpu", action="store_true", default=False)
    parser.add_argument("--lr_schedule", default="plateau", help="Either None, plateau")
    parser.add_argument("--amplify_bias", action="store_true", default=False)
    parser.add_argument('--number_classes', type=str_to_list_fn(int,','), default='1', help="age = 12")


    # Model parameters
    parser.add_argument('--model', type=str, default="mobilenet", choices=['resnet', 'mobilenet'],help="Model type to use, vggface or resnet50.")
    parser.add_argument("--weights", default=None, type=str, choices=[None, 'imagenet'], help="Weights to initialize the model: None or imagenet")
    args = parser.parse_args()

    try:
        main(args)
    except:
        raise   # Reraise exception if there is one


































