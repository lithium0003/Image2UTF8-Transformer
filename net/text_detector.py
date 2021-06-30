import tensorflow as tf
import numpy as np

width = 128
height = 128

def hard_swish(x):
    return x * tf.nn.relu6(x+3) / 6

from keras.utils.generic_utils import get_custom_objects

get_custom_objects().update({'hard_swish': hard_swish})

def FeatureBlock():
    inputs = {
        'image': tf.keras.Input(shape=(height,width,3)),
    }
    base_model = tf.keras.applications.EfficientNetB2(include_top=False, weights='imagenet', activation=hard_swish)
    x = base_model(inputs['image'])
    x = tf.keras.layers.DepthwiseConv2D(4)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation(hard_swish)(x)
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(256)(x)
    outputs = x
    return tf.keras.Model(inputs, outputs, name='FeatureBlock')

def SimpleDecoderBlock():
    embedded = tf.keras.Input(shape=(256,))

    dense_id1 = tf.keras.layers.Dense(104)(embedded)
    dense_id2 = tf.keras.layers.Dense(100)(embedded)

    return tf.keras.Model(embedded, 
        {
            'id1': dense_id1,
            'id2': dense_id2,
        }, 
        name='SimpleDecoderBlock')


if __name__ == '__main__':
    encoder = FeatureBlock()
    encoder.summary()

    decoder = SimpleDecoderBlock()
    decoder.summary()
