import tensorflow as tf 
import numpy as np
from tensorflow.keras.layers import Flatten, Dense, Input, GlobalAveragePooling2D, \
    GlobalMaxPooling2D, Activation, Conv2D, MaxPooling2D, BatchNormalization, \
    AveragePooling2D, Reshape, Permute, multiply
from tensorflow.keras.models import Model
from tensorflow.keras.applications import ResNet50V2
from tensorflow.keras.applications import ResNet101V2
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications import VGG16
from utils import add_regularization
from utils import CoralOrdinal
from utils import ordinal_softmax
from utils import GradientReversalLayer
from utils import GradientNull


def build_model_2nd(name, classification_name, input_tensor, weights, number_classes):

    if name == 'resnet50':
        basemodel = make_resnet(input_tensor=input_tensor, weights=weights) 
    elif name == 'mobilenet':
        basemodel = make_mobilenet(input_tensor=input_tensor, weights=weights)
    else:
        raise NameError('unknown model name %s', name)

    _input = tf.keras.layers.Input(input_tensor, dtype = tf.float32)

    x = basemodel(_input)
   
    out = []
    for n,name in zip(number_classes,classification_name):
        if n > 1:
            out.append(Dense(n,name=name, activation='softmax')(x))
        else:
            out.append(Dense(1,name=name, activation='sigmoid')(x))
    
    model = Model(inputs=[_input], outputs=out)
    
    return model


def build_model_adversarial_2nd(name, classification_name, protected_attr, input_tensor, weights, number_classes, const_grl, capacity):

    if name == 'resnet50':
        basemodel = make_resnet(input_tensor=input_tensor, weights=weights) 
    elif name == 'mobilenet':
        basemodel = make_mobilenet(input_tensor=input_tensor, weights=weights)
    else:
        raise NameError('unknown model name %s', name)

    _input = tf.keras.layers.Input(input_tensor, dtype = tf.float32)

    x = basemodel(_input)
   
    out = []

    # utility output
    for n,name in zip(number_classes,classification_name):
        out.append(Dense(n,name=name, activation='softmax' if n > 1 else 'sigmoid')(x))   

    # adversarial head
    x = GradientReversalLayer(lamb = const_grl, undo_grl=False)(x)
    x = Dense(capacity, activation='relu')(x)
    x = Dense(capacity, activation='relu')(x)
    out.append(Dense(1,name=protected_attr, activation='sigmoid')(x))    
    
    model = Model(inputs=[_input], outputs=out)
    
    return model


def build_model_adversarial_shuffle(name, classification_name, protected_attr, input_tensor, weights, number_classes, const_grl, capacity):

    if name == 'resnet50':
        basemodel = make_resnet(input_tensor=input_tensor, weights=weights) 
    elif name == 'mobilenet':
        basemodel = make_mobilenet(input_tensor=input_tensor, weights=weights)
    else:
        raise NameError('unknown model name %s', name)

    _input = tf.keras.layers.Input(input_tensor, dtype = tf.float32)

    x = basemodel(_input)
   
    out = []

    # utility output
    for n,name in zip(number_classes,classification_name):
        b = GradientNull(name='gradientNullEven')(x)
        out.append(Dense(n,name=name, activation='softmax' if n > 1 else 'sigmoid')(b))   

    # adversarial head
    x = GradientNull(name='gradientNullOdd')(x)
    x = GradientReversalLayer(lamb = const_grl, undo_grl=False)(x)
    x = Dense(capacity, activation='relu')(x)
    x = Dense(capacity, activation='relu')(x)
    out.append(Dense(1,name=protected_attr, activation='sigmoid')(x))    
    
    model = Model(inputs=[_input], outputs=out)
    
    return model



def build_model(name, classification_name, input_tensor, task,nclasses, weights, regularization, regularization_alpha):

    if name == 'vggface':
        basemodel = make_vgg(input_tensor=input_tensor, weights=weights) 

    elif name == 'resnet':
        basemodel = make_resnet(input_tensor=input_tensor, weights=weights) 
#        prep = lambda i: tf.keras.applications.resnet_v2.preprocess_input(i)
    elif name == 'resnet101':
        basemodel = make_resnet101(input_tensor=input_tensor, weights=weights) 
#        prep = lambda i: tf.keras.applications.resnet_v2.preprocess_input(i)
    elif name == 'mobilenet':
        basemodel = make_mobilenet(input_tensor=input_tensor, weights=weights)
#        prep = lambda i: tf.keras.applications.mobilenet.preprocess_input(i) 
    else:
        raise NameError('unknown model name %s', name)

    if regularization == 'l2':
        basemodel = add_regularization(basemodel,tf.keras.regularizers.l2(regularization_alpha))
    elif regularization == 'l1':
        basemodel = add_regularization(basemodel,tf.keras.regularizers.l1(regularization_alpha))

    _input = tf.keras.layers.Input(input_tensor, dtype = tf.float32)
#    x = prep(_input)
    x = basemodel(_input)
   
    out = []
    if task == 'classification':
        for name, nclass in zip(classification_name,nclasses):
            out.append(Dense(nclass,name=name, activation='softmax')(x))
    elif task == 'ordinal':
        for name, nclass in zip(classification_name,nclasses):
            out.append(CoralOrdinal(nclass,name=name, activation=ordinal_softmax)(x))
    else:
        for name in classification_name:
            out.append(Dense(1,name=name, activation=None)(x)) #activation is linear

    model = Model(inputs=[_input], outputs=out)
    
    return model

def build_model_adversarial(name, classification_name, input_tensor, task,nclasses, weights, regularization, regularization_alpha, const_grl, capacity):

    if name == 'vggface':
        basemodel = make_vggface(input_tensor=input_tensor, weights=weights) 

    elif name == 'resnet':
        basemodel = make_resnet(input_tensor=input_tensor, weights=weights) 
#        prep = lambda i: tf.keras.applications.resnet_v2.preprocess_input(i)
    elif name == 'resnet101':
        basemodel = make_resnet101(input_tensor=input_tensor, weights=weights) 
#        prep = lambda i: tf.keras.applications.resnet_v2.preprocess_input(i)
    elif name == 'mobilenet':
        basemodel = make_mobilenet(input_tensor=input_tensor, weights=weights)
#        prep = lambda i: tf.keras.applications.mobilenet.preprocess_input(i) 
    else:
        raise NameError('unknown model name %s', name)

    if regularization == 'l2':
        basemodel = add_regularization(basemodel,tf.keras.regularizers.l2(regularization_alpha))
    elif regularization == 'l1':
        basemodel = add_regularization(basemodel,tf.keras.regularizers.l1(regularization_alpha))

    _input = tf.keras.layers.Input(input_tensor, dtype = tf.float32)
#    x = prep(_input)
    x = basemodel(_input)
   
    out = []

    out.append(Dense(nclasses[0],name=classification_name[0], activation='softmax')(x))

    x = GradientReversalLayer(lamb = const_grl, undo_grl=False)(x)
    x = Dense(capacity, activation='relu')(x)
    x = Dense(capacity, activation='relu')(x)
    out.append(Dense(nclasses[1],name=classification_name[1], activation='softmax')(x))
    
    model = Model(inputs=[_input], outputs=out)
    
    return model

def make_vggface(input_tensor, weights):
    img_input = Input(shape=input_tensor)
    
     # Block 1
    x = Conv2D(64, (3, 3), activation='relu', padding='same', name='conv1_1')(img_input)
    x = Conv2D(64, (3, 3), activation='relu', padding='same', name='conv1_2')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='pool1')(x)

    # Block 2
    x = Conv2D(128, (3, 3), activation='relu', padding='same', name='conv2_1')(x)
    x = Conv2D(128, (3, 3), activation='relu', padding='same', name='conv2_2')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='pool2')(x)

    # Block 3
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='conv3_1')(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='conv3_2')(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='conv3_3')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='pool3')(x)

    # Block 4
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='conv4_1')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='conv4_2')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='conv4_3')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='pool4')(x)

        # Block 5
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='conv5_1')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='conv5_2')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='conv5_3')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='pool5')(x)

    # top 
    x = Flatten(name='flatten')(x)
    x = Dense(4096, name='fc6')(x)
    x = Activation('relu', name='fc6/relu')(x)
    x = Dense(4096, name='fc7')(x)
    x = Activation('relu', name='fc7/relu')(x)
    model = Model(img_input, x, name='vggface_vgg16')
    
    return model

def make_resnet(input_tensor=(224,224,3), weights=None):
    return ResNet50V2(include_top=False,
                      weights=weights,
                      input_shape=input_tensor,
                      pooling='avg')

def make_resnet101(input_tensor=(224,224,3), weights=None):
    return ResNet101V2(include_top=False,
                      weights=weights,
                      input_shape=input_tensor,
                      pooling='avg')

def make_mobilenet(input_tensor=(224,224,3), weights=None):
    return MobileNetV2(include_top=False,
                       alpha=1.0,
                       weights=weights,
                       input_shape=input_tensor,
                       pooling='avg')

def make_vgg(input_tensor=(224,224,3), weights=None):
    return MobileNetV2(include_top=False,
                       weights=weights,
                       input_shape=input_tensor,
                       pooling='avg')




