import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np 


def show_batch(x, y, y_pred):

    x = tf.unstack(x,0)

    






#def show_batch(dataset, n_images, samples_per_image):
#    output = np.zeros((32 * n_images, 32 * samples_per_image, 3))
#
#    row = 0
#    for images in dataset.repeat(samples_per_image).batch(n_images):
#        output[:, row*32:(row+1)*32] = np.vstack(images.numpy())
#        row += 1
#
#    plt.figure()
#    plt.imshow(output)
#    plt.show()
