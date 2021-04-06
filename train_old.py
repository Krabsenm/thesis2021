import tensorflow as tf
import numpy as np
import pandas as pd
import models.models as mo
import utils.label_encoder as enc
from utils.preprocessing import preprocess_input

import os
from datetime import datetime

# GPU settings 
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

#dir_path = '/media/slow-storage/krabsen/imdb_crop'
dir_path = '/media/slow-storage/DataFolder/krabsen/imdb_crop'

data = 'clean_age_gender_dataset.csv'

dataset = pd.read_csv(os.path.join(dir_path, data), encoding = "ISO-8859-1")

x = dataset['full_path']

x = [os.path.join(dir_path, img) for img in x]

# get labels and encode into groups 
y = dataset['age'].to_numpy()
y = enc.age_encode(y)


# calculate split (does there need to be taken more care in splitting the images?)
num_samples = len(y)
num_train   = int(np.ceil(num_samples*0.7))
num_val     = int(np.floor(num_samples*0.15))
num_test    = num_samples - num_val - num_train

# training set
train_x = x[0:num_train]  
train_y = y[0:num_train]

# validation set
val_x = x[num_train:num_train+num_val]
val_y = y[num_train:num_train+num_val]

# validation set
test_x = x[num_train+num_val:]
test_y = y[num_train+num_val:]

train = tf.data.Dataset.from_tensor_slices((train_x, train_y))
val = tf.data.Dataset.from_tensor_slices((val_x, val_y))
test = tf.data.Dataset.from_tensor_slices((test_x, test_y))

train = train.shuffle(buffer_size=num_train)
train = train.repeat()

#model = mo.make_vgg()
model = make_vggface()
#model.summary()

optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001, name='Adam')

model.compile(optimizer=optimizer, loss=tf.keras.losses.CategoricalCrossentropy(), metrics=['accuracy'])

@tf.function
def decode(x,y, h,w):
    x_decoded = decode_img(x, h, w)
    return (x_decoded, tf.one_hot(y, 12))

@tf.function
def decode_img(filename, h, w):
    # convert the compressed string to a 3D uint8 tensor
    jpeg_str = tf.io.read_file(filename)
    img = tf.image.decode_jpeg(jpeg_str, channels=3)
    img = preprocess_input(img) ## vggface preprocessing
    # Use `convert_image_dtype` to convert to floats in the [0,1] range.
    img = tf.image.convert_image_dtype(img, tf.float32)
    # resize the image to the desired size.
    return tf.image.resize(img, [h, w])

decoder = lambda x,y: decode(x,y,224,224)

train   = train.map(decoder, num_parallel_calls=4)
val     = val.map(decoder)
test    = test.map(decoder)

train = train.batch(32)
val = val.batch(32)
test = test.batch(32)

train = train.prefetch(1)

RUNTIME = datetime.now().strftime("%H%M_%d%m") # e.g. 1307_2702 

OUTPUT_DIR = os.path.join('trained_models', RUNTIME)

if not os.path.isdir(OUTPUT_DIR):
    os.mkdir(OUTPUT_DIR)

CHECKPOINT_DIR = os.path.join(OUTPUT_DIR, 'weights_{val_loss:.2f}.hdf5')

# safe best weights 
checkpointCallback = tf.keras.callbacks.ModelCheckpoint(CHECKPOINT_DIR, 
                                                        monitor='val_loss', 
                                                        verbose=1,
                                                        save_freq='epoch',
                                                        save_best_only=True)

# stop training if model doesn't improve 
#earlystoppingCallback = tf.keras.callbacks.EarlyStopping(monitor='val_loss',  
#                                                        patience=patience, 
#                                                            verbose=1)

# save tensorboard log data
tensorboardCallback = tf.keras.callbacks.TensorBoard(log_dir=OUTPUT_DIR)

lr_schedule = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.1,patience=5, min_lr=0.00001, verbose=1)

model.fit(train,
         epochs=50,
         verbose=1,
         validation_data=val,
         callbacks = [tensorboardCallback, checkpointCallback,lr_schedule],
         steps_per_epoch=int(num_train/32))


model.evaluate(test)