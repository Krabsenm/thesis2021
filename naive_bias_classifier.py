import argparse
import logging
import math
import os
import os.path as opath

import numpy as np
import pandas as pd

from sklearn.dummy import DummyClassifier
from sklearn.metrics import auc, precision_recall_curve, roc_auc_score, accuracy_score

from datasets import get_tf_dataset, split_dataset, get_dataset, calculate_stats
from utils import str_to_list_fn

def evaluate(FLAGS):

    x_data, y_data = get_dataset(args.data_dir,args.dataset, [args.labels], 'classification')

    X_train, Y_train, X_val, Y_val, X_hold_out, Y_hold_out = split_dataset(x_data, y_data, args.data_split)


    if FLAGS.dataset_train == 'train':
        dataset_train = (X_train, Y_train[FLAGS.bias_check])
    elif FLAGS.dataset_train == 'validation':
        dataset_train = (X_val, Y_val[FLAGS.bias_check])
    if FLAGS.dataset_train == 'hold_out':
        dataset_train = (X_hold_out, Y_hold_out[FLAGS.bias_check])


    if FLAGS.dataset_test == 'train':
        dataset_test = (X_train, Y_train[FLAGS.bias_check])
    elif FLAGS.dataset_test == 'validation':
        dataset_test = (X_val, Y_val[FLAGS.bias_check])
    if FLAGS.dataset_test == 'hold_out':
        dataset_test = (X_hold_out, Y_hold_out[FLAGS.bias_check])

    models = []
    for strategy in ['most_frequent','stratified', 'uniform']:
        models.append(DummyClassifier(strategy=strategy))

    accuracies = []
    for model in models:
        model.fit(dataset_train[0], dataset_train[1])
        accuracies.append(accuracy_score(dataset_test[1], model.predict(dataset_test[0])))

    print('Naive classification results for {}:'.format(args.bias_check))
    print('The model is trained on: {}'.format(args.dataset_train))
    print('The model is tested on: {}'.format(args.dataset_test))
    print('classification accuracy for most frequent: {:.3f}'.format(accuracies[0]))
    print('classification accuracy for stratified: {:.3f}'.format(accuracies[1]))
    print('classification accuracy for uniform: {:.3f}'.format(accuracies[2]))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default='/media/slow-storage/krabsen/imdb_crop/', type=str)
    parser.add_argument("--dataset", default='clean_age_gender_dataset_110920.csv', type=str)
    parser.add_argument("--bias_check", type=str, default='age', help="adds the corresponding output to the model which is checked for corresponding bias")
    parser.add_argument("--labels", type=str, default='age', help="adds the corresponding output to the model which is checked for corresponding bias")
    parser.add_argument('--data_split', type=str_to_list_fn(float,','), default="0.7,0.15,0.15", help='train, validation,test split')
    parser.add_argument("--naive_strategy", type=str, default='most_frequent', choices=['most_frequent','stratified', 'uniform'], help='which naive model to use: most_frequent: constant prediction of majority class, stratified: uniformly samples from training dataset labels each class, uniform: uniformly predicts on each class'  )
    parser.add_argument("--dataset_train", type=str, default='train', choices=['train','validation', 'hold_out'], help='which dataset to fit on')
    parser.add_argument("--dataset_test", type=str, default='hold_out', choices=['train','validation', 'hold_out'], help='which dataset to test on')
    
    pd.options.display.float_format = '{:,.3f}'.format
    args = parser.parse_args()
    evaluate(args)
