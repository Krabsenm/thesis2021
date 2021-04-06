import tensorflow as tf


#def init_gpus():

    # GPU settings 
#    gpus = tf.config.experimental.list_physical_devices('GPU')
#    for gpu in gpus:
#        tf.config.experimental.set_memory_growth(gpu, True)


def init_gpus():
    tf.config.experimental_run_functions_eagerly(True)
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
 #       try:
            # Currently, memory growth needs to be the same across GPUs
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
#            logical_gpus = tf.config.experimental.list_logical_devices('GPU')
#            logging.getLogger("train").info("Physical GPUs: %d Logical GPUs: %d" % (len(gpus), len(logical_gpus)))
#        except RuntimeError as e:
            # Memory growth must be set before GPUs have been initialized
#            logging.exception(e)