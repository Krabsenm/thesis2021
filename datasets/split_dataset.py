import numpy as np

def split_dataset_2nd(x_data, y_data, data_split):


    train_y = {}
    val_y = {}
    test_y = {}

    # training set
    train_x = x_data[0:num_train]  

    for key in y_data.keys():
        train_y[key] = y_data[key][0:num_train]

    # validation set
    val_x = x_data[num_train:num_train+num_val]

    for key in y_data.keys():
        val_y[key] = y_data[key][num_train:num_train+num_val]

    # validation set
    test_x = x_data[num_train+num_val:]
    for key in y_data.keys():
        test_y[key] = y_data[key][num_train+num_val:]

    return train_x, train_y, val_x, val_y, test_x, test_y


def split_dataset(x_data, y_data, data_split, stratify=False):

    if stratify:
        print('not yet implemented')


    # calculate split (does there need to be taken more care in splitting the images?)
    num_samples = len(x_data)
    num_train   = int(np.ceil(num_samples*data_split[0]))
    num_val     = int(np.floor(num_samples*data_split[1]))

    train_y = {}
    val_y = {}
    test_y = {}

    # training set
    train_x = x_data[0:num_train]  

    for key in y_data.keys():
        train_y[key] = y_data[key][0:num_train]

    # validation set
    val_x = x_data[num_train:num_train+num_val]

    for key in y_data.keys():
        val_y[key] = y_data[key][num_train:num_train+num_val]

    # validation set
    test_x = x_data[num_train+num_val:]
    for key in y_data.keys():
        test_y[key] = y_data[key][num_train+num_val:]

    return train_x, train_y, val_x, val_y, test_x, test_y