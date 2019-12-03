import numpy as np
import tensorflow as tf
from keras.models import Model
from keras.layers import Input, MaxPool2D,  Conv2D, Dropout, concatenate, Reshape, Lambda
from keras.initializers import TruncatedNormal
from keras.regularizers import l2
import parameter as para


def SqueezeDet():
    input_layer = Input(shape=(para.IMAGE_HEIGHT, para.IMAGE_WIDTH, 3), name="input")
    conv1 = Conv2D(filters=64, kernel_size=(3, 3), strides=(2, 2), padding="SAME", activation='relu', use_bias=True, kernel_initializer=TruncatedNormal(stddev=0.001), kernel_regularizer=l2(0.001))(input_layer)

    pool1 = MaxPool2D(pool_size=(3,3), strides=(2, 2), padding='SAME', name="pool1")(conv1)
    fire3 = fire_layers(2, pool1, 16, 64, 64)
    pool3 = MaxPool2D(pool_size=(3, 3), strides=(2, 2), padding='SAME', name='pool3')(fire3)

    fire5 = fire_layers(4, pool3, 32, 128, 128)
    pool5 = MaxPool2D(pool_size=(3, 3), strides=(2, 2), padding='SAME', name="pool5")(fire5)

    fire7 = fire_layers(6, pool5, 48, 192, 192)
    fire9 = fire_layers(8, fire7, 64, 256, 256)

    fire11 = fire_layers(10, fire9, 96, 384, 384)
    dropout11 = Dropout(rate=0.5, name='drop11')(fire11)

    # Number of classes, 1 confidence score and 4 bounding box corners per anchor
    num_output = para.ANCHOR_PER_GRID * (para.NUM_CLASSES + 1 + 4)

    preds = Conv2D(name='conv12', filters=num_output, kernel_size=(3, 3), strides=(1, 1), activation=None, padding="SAME", 
                   use_bias=True, kernel_initializer=TruncatedNormal(stddev=0.001), 
                   kernel_regularizer=l2(0.001))(dropout11)
    pred_reshaped = Reshape((para.ANCHORS, -1))(preds)
    # Wrap in lambda layer, making y_pred and y_true have the same dimensions
    pred_padded = Lambda(pad)(pred_reshaped)
    model = Model(inputs=input_layer, outputs=pred_padded)
    return model

def fire_layers(name_idx, inputs, squeeze, exp1, exp2):
    fire = fire_layer(name='fire' + str(name_idx), pre_layer=inputs, sqz=squeeze, exp1=exp1, exp2=exp2)
    return fire_layer(name='fire' + str(name_idx+1), pre_layer=fire, sqz=squeeze, exp1=exp1, exp2=exp2)

def fire_layer(name, pre_layer, sqz, exp1, exp2, stdd=0.01, weight_decay=0.001):
    """
    Return a keras fire layer
    :param name: name of the layer
    :param pre_layer: input layer
    :param sqz: number of filters for squeezing
    :param exp1: number of filter for expand 1x1
    :param exp2: number of filter for expand 3x3
    :param stdd: standard deviation used for intialization
    :weight_decay: regularization parameter
    :return: a keras fire layer
    """
    squeeze = Conv2D(name = name + '/squeeze', filters=sqz, kernel_size=(1, 1), strides=(1, 1), use_bias=True, 
                   padding='SAME', kernel_initializer=TruncatedNormal(stddev=stdd), activation="relu",
                   kernel_regularizer=l2(weight_decay))(pre_layer)
    expand1 = Conv2D(name = name + '/expand1', filters=exp1, kernel_size=(1, 1), strides=(1, 1), use_bias=True,
                   padding='SAME',  kernel_initializer=TruncatedNormal(stddev=stdd), activation="relu",
                   kernel_regularizer=l2(weight_decay))(squeeze)
    expand2 = Conv2D(name = name + '/expand2',  filters=exp2, kernel_size=(3, 3), strides=(1, 1), use_bias=True,
                   padding='SAME', kernel_initializer=TruncatedNormal(stddev=stdd), activation="relu",
                   kernel_regularizer=l2(weight_decay))(squeeze)
    return concatenate([expand1, expand2], axis=3)

def pad(pre_layer):
    """
    Pad the network output so that y_pred and y_true have the same dimensions.
    """
    padding = np.zeros((3,2))
    padding[2,1] = 4
    return tf.pad(pre_layer, padding ,"CONSTANT")

