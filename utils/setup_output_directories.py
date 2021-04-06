import os 
import sys
from datetime import datetime

def setup_output_directories_2nd(args, output_dir):
    
    if not os.path.isdir(output_dir):
        raise NameError('output dir doesnt exist: \s ', output_dir)
    
    output_dir = os.path.abspath(output_dir)
 

    instance_dir = os.path.join(output_dir, args.test)
    runtime = datetime.now().strftime("%H%M_%d%m")

    instance_dir = '{}-{}'.format(instance_dir, runtime)

    if not os.path.isdir(instance_dir):
        os.mkdir(instance_dir)

    return instance_dir



def setup_output_directories(args, output_dir):
    
    if not os.path.isdir(output_dir):
        raise NameError('output dir doesnt exist: \s ', output_dir)
    
    output_dir = os.path.abspath(output_dir)

    instance_dir = os.path.join(output_dir, '%smodel_%s_%s_%s_%s_data_%s_%s_c%d_%s_train_b%d_lrS_%s_e%s' % (
                                args.test, args.model, args.task, args.weights, args.regularization,                                                    # model params
                                ''.join(args.labels), ''.join([x[0] for x in args.augmentation]), args.channels, 'prep' if args.preprocess else '', # dataset params
                                args.batch_size, args.lr_schedule, args.max_epoch))                                                         # train params

    runtime = datetime.now().strftime("%H%M_%d%m")

    instance_dir = '{}-{}'.format(instance_dir, runtime)

    if not os.path.isdir(instance_dir):
        os.mkdir(instance_dir)

    return instance_dir



  

