import tensorflow as tf
from utils import preprocess_input
from utils import augmentation_fn
AUTOTUNE = tf.data.experimental.AUTOTUNE

def get_tf_dataset_2nd(images, labels, number_classes, batch_size=32, image_size=(224,224), augmentation= ['color', 'flip', 'bright_constrast'], train=False):

    dataset = tf.data.Dataset.from_tensor_slices((images, labels))

    if train:
        dataset = dataset.repeat()
        dataset = dataset.shuffle(buffer_size=len(images))
    
    decode_x = lambda x: decode_c3(x,image_size[0], image_size[1])

    if number_classes != None: 
        decoder = lambda x,y: (decode_x(x), decode_classification(y, number_classes))
    else:
        decoder = lambda x,y: (decode_x(x), y)

    augment = lambda x,y: augmentation_fn(x,y,augmentation, 3)

    dataset = dataset.map(decoder, num_parallel_calls=AUTOTUNE)

    dataset = dataset.map(preprocess, num_parallel_calls=AUTOTUNE) 

    if train:
        dataset = dataset.map(augment, num_parallel_calls=AUTOTUNE)
    
    dataset = dataset.batch(batch_size)

    dataset= dataset.prefetch(1)

    steps_pr_epoch = int(len(images)/batch_size) 

    return dataset, steps_pr_epoch

def get_tf_dataset(images, labels, channels=3, task='classification', nclasses={'age':7, 'gender':2}, batch_size=32, image_size=224, augmentation= ['color', 'flip', 'bright_constrast'], pre_process=False, train=False):

    dataset = tf.data.Dataset.from_tensor_slices((images, labels))

    if train:
        dataset = dataset.repeat()
        dataset = dataset.shuffle(buffer_size=len(images))
    
    if task == 'classification':
        decode_y = lambda y: decode_classification(y,nclasses)
    else:
        decode_y = lambda y: decode_regression(y)

    if channels == 3:
        decode_x = lambda x: decode_c3(x,image_size,image_size)
    else:
        decode_x = lambda x: decode_c1(x,image_size,image_size)

    decoder = lambda x,y: (decode_x(x), decode_y(y))

    augment = lambda x,y: augmentation_fn(x,y,augmentation, channels)

    dataset = dataset.map(decoder, num_parallel_calls=AUTOTUNE)

    if pre_process:
        dataset = dataset.map(preprocess, num_parallel_calls=AUTOTUNE) 

    if train:
        dataset = dataset.map(augment, num_parallel_calls=AUTOTUNE)
    
    dataset = dataset.batch(batch_size)

    dataset= dataset.prefetch(1)

    steps_pr_epoch = int(len(images)/batch_size) 

    return dataset, steps_pr_epoch

@tf.function
def decode(x,y,dx,dy):
    return dx(x), dy(y)

@tf.function
def decode_c3(x, h,w):
    return decode_img2(x, h, w)

@tf.function
def decode_c1(x, h,w):
    x_decoded = decode_img(x, h, w)
    x_decoded = tf.image.rgb_to_grayscale(x_decoded)
    return x_decoded

#@tf.function
#def decode_classification(y, nclasses):
#    return tf.one_hot(y['age'], nclasses)

@tf.function
def decode_classification(y, nclasses):
    return [tf.one_hot(y[key], nclasses[key]) for key in y.keys()]

@tf.function
def decode_classification_adv(y, nclasses):
    return tuple([y['gender'] , tf.one_hot(y['race'], 4)])


#@tf.function
#def decode_regression(y):
#    return y['age']

@tf.function
def decode_regression(y):
    return tuple([y[key] for key in y.keys()])

@tf.function
def preprocess(x,y):
    x = preprocess_input(x)
    return (x, y)

@tf.function
def decode_img(filename, h, w):
    # convert the compressed string to a 3D uint8 tensor
    jpeg_str = tf.io.read_file(filename)
    img = tf.image.decode_jpeg(jpeg_str, channels=3)
#    img = preprocess_input(img) ## vggface preprocessing
    # Use `convert_image_dtype` to convert to floats in the [0,1] range.
    img = tf.image.convert_image_dtype(img, tf.float32)
    # resize the image to the desired size.
    return tf.image.resize(img, [h, w])

@tf.function
def decode_img2(filename, h, w):
    # convert the compressed string to a 3D uint8 tensor
    jpeg_str = tf.io.read_file(filename)
    img = tf.image.decode_jpeg(jpeg_str, channels=3)
    img = tf.dtypes.cast(img, tf.float32)
    # scale [-1 1]
    img /= 127.5
    img -= 1.
    # resize the image to the desired size.
    return tf.image.resize(img, [h, w])





