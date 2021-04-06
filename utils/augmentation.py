import tensorflow as tf 
import numpy as np

@tf.function
def augmentation_fn(x:tf.Tensor,y:tf.Tensor, transformations:list, channels:int) -> (tf.Tensor, tf.Tensor):
    
    # both vertical and horizontal flips can occur 
    if 'flip' in transformations:
        x = random_flip(x)
    
    if ('color' in transformations and channels==3): 
        x = random_color(x)


    if 'bright_constrast' in transformations:
        x = random_bright_contrast(x)

    return x,y

@tf.function
def random_flip(x: tf.Tensor) -> tf.Tensor:

    x = tf.image.random_flip_left_right(x)
    x = tf.image.random_flip_up_down(x)
    return x

# randomly applies huw, brightness, saturation and contrast augmentations 
@tf.function
def random_color(x: tf.Tensor) -> tf.Tensor:
    x = tf.image.random_hue(x, 0.08*2)
    x = tf.image.random_saturation(x, 0.6, 1.6*2)
    return x

@tf.function
def random_bright_contrast(x: tf.Tensor) -> tf.Tensor:
    x = tf.image.random_brightness(x, 0.05*2)
    x = tf.image.random_contrast(x, 0.7, 1.3*2)
    return x


@tf.function
def random_zoom(x: tf.Tensor) -> tf.Tensor:

    # Generate 20 crop settings, ranging from a 1% to 20% crop.
    scales = list(np.arange(0.8, 1.0, 0.01))
    boxes = np.zeros((len(scales), 4))

    for i, scale in enumerate(scales):
        x1 = y1 = 0.5 - (0.5 * scale)
        x2 = y2 = 0.5 + (0.5 * scale)
        boxes[i] = [x1, y1, x2, y2]

    def random_crop(img):
        # Create different crops for an image
        crops = tf.image.crop_and_resize([img], boxes=boxes, box_indices=np.zeros(len(scales)), crop_size=(32, 32))
        # Return a random crop
        return crops[tf.random.uniform(shape=[], minval=0, maxval=len(scales), dtype=tf.int32)]

    choice = tf.random.uniform(shape=[], minval=0., maxval=1., dtype=tf.float32)

    # Only apply cropping 10% of the time
    return tf.cond(choice < 0.1, lambda: x, lambda: random_crop(x))