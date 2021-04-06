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

from models import build_model_adversarial_2nd, build_model_adversarial_shuffle
from utils import MeanAbsoluteErrorLabels, MeanAbsoluteDifference, FN,TN, FP,TP, EO_Accuracy 
from datasets import get_tf_dataset_2nd, get_and_split_dataset, calculate_stats, get_and_split_utk, insert_bias
from utils import setup_output_directories, init_gpus, str_to_list_fn, configure_optimizer_loss_2nd
from callbacks import set_up_callbacks
from callbacks import GradientNullShuffleCallback, GradientReversalLayerCallback
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

        if args.adversarial_loss == 'MAD':
            def reclass(Z, Y):
                remap = lambda z,y : 0 + 1 * (1 - z)* y + 2*z*(1-y) + 3*z*y 
                return np.asarray(list(map(remap, Z,Y)))

            train_y[args.protected_attr]    = reclass(train_y[args.protected_attr], train_y[args.labels[0]])
            validate_y[args.protected_attr] = reclass(validate_y[args.protected_attr], validate_y[args.labels[0]])
            test_y[args.protected_attr]     = reclass(test_y[args.protected_attr], test_y[args.labels[0]])  

    else:
        train_x, validate_x, test_x, train_y, validate_y, test_y = get_and_split_utk(args.data_path, args.labels + [args.protected_attr])
        num_class = {'race':5}

    if args.amplify_bias:
        train_x, train_y = insert_bias(train_x, train_y)


    train_dataset, train_steps         = get_tf_dataset_2nd(images=train_x   , labels=train_y,    batch_size=args.batch_size, image_size=(args.img_h, args.img_w), number_classes=num_class, augmentation=args.augmentation, train=True)

    validate_dataset, validation_steps = get_tf_dataset_2nd(images=validate_x, labels=validate_y, batch_size=args.batch_size, image_size=(args.img_h, args.img_w), number_classes= num_class, augmentation=[], train=False)

    test_dataset, test_steps           = get_tf_dataset_2nd(images=test_x,     labels=test_y,     batch_size=args.batch_size, image_size=(args.img_h, args.img_w), number_classes=num_class, augmentation=[], train=False)

#######################################################################
#                    build model                                      #
#######################################################################
    if not args.skip_training:
        output_dir = setup_output_directories(args, args.output_dir)

    model = build_model_adversarial_2nd(name=args.model, classification_name=args.labels, number_classes=args.number_classes, protected_attr=args.protected_attr, input_tensor=(args.img_h, args.img_w, 3), weights=args.weights, const_grl=args.const_grl, capacity=args.capacity)

    def getLayer(model, key):
        for i, layer in enumerate(model.layers):
            if key in layer.name:
                return i
            return None    

    if not args.skip_training:
        callbacks = set_up_callbacks(output_dir, args.lr_schedule)
    if not args.skip_training:
        if args.const_grl == 0.0:
            grl_no = getLayer(model, 'grl')
            callbacks.insert(1, GradientReversalLayerCallback(grl_no=grl_no, epochs=args.max_epoch))

    optimizer, losses = configure_optimizer_loss_2nd(names=args.labels + [args.protected_attr])

    metrics = {args.labels[0]:tf.keras.metrics.BinaryAccuracy()}
    weights = None 
    if args.adversarial_loss == 'MAD':
        ttl = float(len(train_y[args.protected_attr]))
        Z0Y0 = len(train_y[args.protected_attr][train_y[args.protected_attr] == 0]) / ttl
        Z0Y1 = len(train_y[args.protected_attr][train_y[args.protected_attr] == 1]) / ttl
        Z1Y0 = len(train_y[args.protected_attr][train_y[args.protected_attr] == 2]) / ttl
        Z1Y1 = len(train_y[args.protected_attr][train_y[args.protected_attr] == 3]) / ttl
        losses[args.protected_attr] = MeanAbsoluteDifference(name=args.protected_attr, EO_weights= [1/Z0Y0, 1/Z0Y1,1/Z1Y0,1/Z1Y1], maximize=False)
        metrics[args.protected_attr] = EO_Accuracy()
#    metrics = ['accuracy']
    elif args.adversarial_loss == 'DP_MAD':
        ttl = float(len(train_y[args.protected_attr]))
        weights = {}
        weights[args.labels[0]] = {1:0.5, 0:0.5}
        weights[args.protected_attr] =  {1: 1/(len(train_y[args.protected_attr][train_y[args.protected_attr] == 1]) / ttl), 0: 1/(len(train_y[args.protected_attr][train_y[args.protected_attr] == 0]) / ttl)}
        losses[args.protected_attr] = tf.keras.losses.MeanAbsoluteError()
        metrics[args.protected_attr] = tf.keras.metrics.BinaryAccuracy()

    model.compile(optimizer=optimizer, loss=losses, metrics=metrics) 

#######################################################################
#                    train model                                      #
#######################################################################
    if not args.skip_training:
        model.fit(train_dataset,
                  epochs=args.max_epoch, 
                  verbose=1, 
                callbacks=callbacks, 
                validation_data=validate_dataset, 
                class_weight=weights,
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
    parser.add_argument('--data_path', type=str, default='/media/slow-storage/DataFolder/krabsen',help='Path to folder containing data files')
    parser.add_argument('--dataset', type=str, default='list_attr_celeba.txt', help='Path to csv file with dataset file paths and labels')
    parser.add_argument('--labels', type=str_to_list_fn(str,','), help='labels for model', default='Attractive,Eyeglasses,Mouth_Slightly_Open,Pointy_Nose,Smiling')
    parser.add_argument('--protected_attr', type=str, help='labels for model', default='Male', choices=['Male','Pale_Skin','Young'])
    parser.add_argument('--test', default="", type=str, help='Name to add to model path')
    parser.add_argument('--data_split', type=str_to_list_fn(float,','), default="0.7,0.15,0.15", help='train, validation,test split')
    parser.add_argument("--generate_dataset_stats", action="store_true", default=False, help="calculates stats of the dataset (not implemented)")
    parser.add_argument("--augmentation", type=str_to_list_fn(str,','), default="flip,color,bright_constrast", help="augmentation transformations to apply")
    parser.add_argument('--channels', type=int, default=3, help="will images be color or grayscale")
    parser.add_argument('--img_h', type=int, default=218, help="image height")
    parser.add_argument('--img_w', type=int, default=178, help="image width")
    parser.add_argument('--number_classes', type=str_to_list_fn(int,','), default='1,1,1,1,1', help="age = 12")
    parser.add_argument('--preprocess', action='store_true', default=True, help="will preprocess images")
    parser.add_argument('--data_fraction', type=float, default=1.0, help='fraction of dataset to use')

    # Training parameters
    parser.add_argument("--output_dir", default="adversarial_models_2nd", help="Directory for the trained model, automatically defined if not given.")
    parser.add_argument('--optimizer', default='adam', help='optimizer')
#    parser.add_argument('--loss', default='categorical_crossentropy', help='loss function')
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument('--max_epoch', type=int, default=10, help='max epoch')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size')
    parser.add_argument("--multi_gpu", action="store_true", default=False)
    parser.add_argument("--lr_schedule", default="plateau", help="Either None, plateau")
    parser.add_argument("--skip_training", action="store_true", default=False, help="expects that weights have been loaded and goes to evaluate")
    parser.add_argument("--adversarial_loss", type=str, default='MAD', help="set to MAD for Equal Odds Mean Absolute Difference adversarial loss")
    parser.add_argument("--amplify_bias", action="store_true", default=False)
    parser.add_argument("--gradient_shuffle", action="store_true", default=False)
    
    # Model parameters
    parser.add_argument('--model', type=str, default="mobilenet", choices=['resnet', 'resnet101','vggface', 'mobilenet'],help="Model type to use, vggface or resnet50.")
    parser.add_argument('--task', type=str, default="classification", choices=['classification', 'regression', 'ordinal'], help='which task to train for, classification or regression')
#    parser.add_argument('--nclasses', type=int, default=12, help="number of classes, age=12")
    parser.add_argument("--weights", default=None, type=str, choices=[None, 'imagenet'], help="Weights to initialize the model: None or imagenet")
    parser.add_argument("--ampl", default=None, type=str, choices=[None, 'imagenet'], help="Weights to initialize the model: None or imagenet")
    parser.add_argument("--weights_pretrained", default=None, type=str, help="path to weights .hdf5 file")
    parser.add_argument('--regularization', default=None, type=str, choices=[None, 'l1', 'l2'], help="regularization will be added to all supported layers")
    parser.add_argument('--regularization_alpha', default=0.01, type=float, help="regularization hyperparameter")
    parser.add_argument('--const_grl', default=1.0, type=float, help="regularization hyperparameter")
    parser.add_argument('--capacity', default=1024, type=int, help="number of weights in adversary")
    args = parser.parse_args()

    try:
        main(args)
    except:
        raise   # Reraise exception if there is one

