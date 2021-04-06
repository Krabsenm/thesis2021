import tensorflow as tf
import numpy as np
import pandas as pd
import os
from utils import age_encode
from utils import age_encode_2 
from utils import age_encode_utk
from utils import age_encode_regression
from sklearn.model_selection import train_test_split



def get_and_split_utk(data_path, labels):

    def loadUTKFace(datapath):
        images = os.listdir(datapath)
        y = np.asarray([x.split('_')[:3] for x in images], dtype=np.int32)
        y[:,0] = np.asarray(age_encode_utk(y[:,0]))
        x = [os.path.join(datapath, img) for img in images]
        return x,y

    x,y = loadUTKFace(data_path)
    
    X_train, X_val, y_train, y_val = train_test_split(x, y, test_size=0.1,stratify=y[:,0:1], random_state=42)
    
    X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.111,stratify=y_train[:,0:1], random_state=42)

    decode = {'age':0, 'gender':1, 'race':2}

    if len(labels) != 3:
        keys = list(decode.keys())
        for key in keys:
            if key not in labels:
                decode.pop(key)

    if len(decode) == 0:
        raise ValueError("no labels left ")
        
    y_train = {key:y_train[:,decode[key]] for key in decode.keys()}
    y_val   = {key:y_val[:,decode[key]] for key in decode.keys()}
    y_test  = {key:y_test[:,decode[key]] for key in decode.keys()}

    return X_train, X_val, X_test, y_train, y_val, y_test



def get_and_split_dataset(data_path, dataset, labels):
    # read csv file containing dataset filepaths and labels
    dataset = pd.read_csv(os.path.join(data_path, dataset), delim_whitespace=True, encoding = "ISO-8859-1")
    dataset = dataset.replace(-1, 0)
    dataset = dataset[['Path'] + labels]

    absolute_path = lambda x : os.path.join(data_path, 'img_align_celeba', x) 
    dataset.loc[:, 'Path'] = dataset['Path'].apply(absolute_path).to_numpy()

    # predefined split provided by publication [.8,.1,.1]
    splits = pd.read_csv(os.path.join(data_path, 'list_eval_partition.txt'), delim_whitespace=True, encoding = "ISO-8859-1")
    dataset['Split'] = splits['0']
    
    train = dataset[dataset['Split'] == 0]
    val = dataset[dataset['Split']   == 1]
    test = dataset[dataset['Split']  == 2]

    # get images
    x_train = train['Path'].to_numpy()
    x_val = val['Path'].to_numpy()
    x_test = test['Path'].to_numpy()

    train = train.drop(columns='Path')
    val   = val.drop(columns='Path')
    test  = test.drop(columns='Path')

    train = train.drop(columns='Split')
    val   = val.drop(columns='Split')
    test  = test.drop(columns='Split')

    # extract desired labels
    y_train = {label:train[label].to_numpy() for label in train.columns}
    y_val   = {label:val[label].to_numpy() for label in val.columns}
    y_test  = {label:test[label].to_numpy() for label in test.columns}

    return x_train, x_val, x_test, y_train, y_val, y_test 


def get_dataset(data_path, dataset, labels, task= 'classification'):
    # read csv file containing dataset filepaths and labels
    dataset = pd.read_csv(os.path.join(data_path, dataset), encoding = "ISO-8859-1")

    # get image relative filepaths 
    x = dataset['full_path']

    # append to absolute filepaths
    x = [os.path.join(data_path, img) for img in x]

    # extract desired labels
    y = {label:dataset[label].to_numpy() for label in labels}

    # encode labels to desired format
    if 'age' in y:
        if task == 'classification' or task == 'ordinal':
            y['age'] = age_encode(y['age'])
        elif task == 'regression':
            y['age'] = age_encode_regression(y['age'])

    return x,y