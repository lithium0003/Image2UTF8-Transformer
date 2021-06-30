#!/usr/bin/env python3

import os
import tensorflow as tf
import numpy as np
import coremltools


#
# for float16 nagative overflow patch
#
from tensorflow.python.keras.utils import tf_utils

class Softmax(tf.keras.layers.Layer):
  """Softmax activation function.
  Example without mask:
  >>> inp = np.asarray([1., 2., 1.])
  >>> layer = tf.keras.layers.Softmax()
  >>> layer(inp).numpy()
  array([0.21194157, 0.5761169 , 0.21194157], dtype=float32)
  >>> mask = np.asarray([True, False, True], dtype=bool)
  >>> layer(inp, mask).numpy()
  array([0.5, 0. , 0.5], dtype=float32)
  Input shape:
    Arbitrary. Use the keyword argument `input_shape`
    (tuple of integers, does not include the samples axis)
    when using this layer as the first layer in a model.
  Output shape:
    Same shape as the input.
  Args:
    axis: Integer, or list of Integers, axis along which the softmax
      normalization is applied.
  Call arguments:
    inputs: The inputs, or logits to the softmax layer.
    mask: A boolean mask of the same shape as `inputs`. Defaults to `None`. The
      mask specifies 1 to keep and 0 to mask.
  Returns:
    softmaxed output with the same shape as `inputs`.
  """

  def __init__(self, axis=-1, **kwargs):
    super(Softmax, self).__init__(**kwargs)
    self.supports_masking = True
    self.axis = axis

  def call(self, inputs, mask=None):
    if mask is not None:
      # Since mask is 1.0 for positions we want to keep and 0.0 for
      # masked positions, this operation will create a tensor which is 0.0 for
      # positions we want to attend and -1e.9 for masked positions.
      inputs *= tf.cast(mask, inputs.dtype)
      adder = (1.0 - tf.cast(mask, inputs.dtype)) * (-5)

      # Since we are adding it to the raw scores before the softmax, this is
      # effectively the same as removing these entirely.
      inputs += adder
    if isinstance(self.axis, (tuple, list)):
      if len(self.axis) > 1:
        return tf.math.exp(inputs - tf.math.reduce_logsumexp(
            inputs, axis=self.axis, keepdims=True))
      else:
        return tf.nn.softmax(inputs, axis=self.axis[0])
    return tf.nn.softmax(inputs, axis=self.axis)

  def get_config(self):
    config = {'axis': self.axis}
    base_config = super(Softmax, self).get_config()
    return dict(list(base_config.items()) + list(config.items()))

  @tf_utils.shape_type_conversion
  def compute_output_shape(self, input_shape):
    return input_shape

_build_attention_org_func = tf.keras.layers.MultiHeadAttention._build_attention
def _build_attention(self, rank):
    _build_attention_org_func(self, rank)
    self._softmax = Softmax(axis=self._softmax.axis)

tf.keras.layers.MultiHeadAttention._build_attention = _build_attention

import net

def convert1():
    detector_save_dir = './result/encoder/'
    save_dir = './result/step1/'
    if os.path.exists(detector_save_dir):
        encoder = tf.keras.models.load_model(detector_save_dir)
        encoder.summary()
        print('loaded from freezed model')
    elif os.path.exists(save_dir):
        checkpoint_dir = save_dir
        encoder = net.FeatureBlock()
        encoder.summary()
        decoder = net.SimpleDecoderBlock()
        decoder.summary()
        inputs = {
            'image': tf.keras.Input(shape=(128,128,3)),
        }
        feature_out = encoder(inputs)
        outputs = decoder(feature_out)
        model = tf.keras.Model(inputs, outputs, name='SimpleEncodeDecoder')
        checkpoint = tf.train.Checkpoint(model=model)
        last = tf.train.latest_checkpoint(checkpoint_dir)
        checkpoint.restore(last).expect_partial()
        if not last is None:
            init_epoch = int(os.path.basename(last).split('-')[1])
            print('loaded %d epoch'%init_epoch)
    else:
        exit()

    mlmodel = coremltools.convert(encoder,
            inputs=[coremltools.ImageType()])
    mlmodel.save("ImageEncoder.mlmodel")

def convert2():
    transformer = net.TextTransformer(vocab_size=256)
    if os.path.exists('result/transformer2_weights'):
        print('load from transformer2_weights')
        transformer.load_weights('result/transformer2_weights/ckpt')
    elif os.path.exists('result/transformer_weights'):
        print('load from transformer_weights')
        transformer.load_weights('result/transformer_weights/ckpt')

    embedded = tf.keras.Input(shape=(None,256))
    decoder_input = tf.keras.Input(shape=(None,), dtype=tf.int32)
    inputs = {'encoder': embedded, 'decoder': decoder_input }
    outputs = transformer((embedded, decoder_input))
    model = tf.keras.Model(inputs=inputs, outputs=outputs, name='TextTransformer')

    checkpoint_dir = 'result/step2/'
    checkpoint = tf.train.Checkpoint(model=model)
    last = tf.train.latest_checkpoint(checkpoint_dir)
    checkpoint.restore(last).expect_partial()
    if not last is None:
        init_epoch = int(os.path.basename(last).split('-')[1])
        print('loaded %d epoch'%init_epoch)
    else:
        init_epoch = 0

    inputs = {'encoder': embedded }
    outputs = transformer.encoder(embedded)
    model = tf.keras.Model(inputs=inputs, outputs=outputs, name='TextTransformerEncoder')
    model.summary()

    # import logging
    # logging.basicConfig(filename='debug.log', level=logging.DEBUG)

    input1 = coremltools.TensorType(name='input_1', shape=(1, 32, 256))
    mlmodel = coremltools.convert(model, inputs=[input1])
    mlmodel.save("ImageTransformerEncoder.mlmodel")

    encoder_output = tf.keras.Input(shape=(None,512))
    inputs = {'encoder': embedded, 'decoder': decoder_input, 'encoder_output': encoder_output }
    outputs = transformer.decoder((decoder_input, encoder_output, embedded))
    model = tf.keras.Model(inputs=inputs, outputs=outputs, name='TextTransformerDecoder')
    model.summary()

    input1 = coremltools.TensorType(name='input_1', shape=(4, 32, 256))
    input2 = coremltools.TensorType(name='input_2', shape=(4, 128), dtype=int)
    input3 = coremltools.TensorType(name='input_3', shape=(4, 32, 512))
    mlmodel = coremltools.convert(model, inputs=[input1,input2,input3])
    mlmodel.save("ImageTransformerDecoder.mlmodel")
    
if __name__ == '__main__':
    convert1()
    convert2()

