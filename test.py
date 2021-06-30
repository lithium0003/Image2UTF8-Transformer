#!/usr/bin/env python3

import tensorflow as tf
physical_devices = tf.config.list_physical_devices('GPU')
try:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
except:
    # Invalid device or cannot modify virtual devices once initialized.
    pass

import numpy as np
import os, time, csv, copy
import tqdm
import datetime
import subprocess

import net

class SimpleEncodeDecoder:
    def __init__(self):
        transformer_dir = 'result/transformer2_weights/ckpt'
        if os.path.exists(transformer_dir):
            self.transformer = net.TextTransformer(vocab_size=256+2)
            self.transformer.load_weights('result/transformer2_weights/ckpt')

            embedded = tf.keras.Input(shape=(None,256))
            decoder_input = tf.keras.Input(shape=(None,), dtype=tf.int64)
            inputs = {'encoder': embedded, 'decoder': decoder_input }
            outputs = self.transformer((embedded, decoder_input))
            self.model = tf.keras.Model(inputs=inputs, outputs=outputs, name='TextTransformer')

        else:
            checkpoint_dir = 'result/step2/'

            self.transformer = net.TextTransformer(vocab_size=256+2)
            embedded = tf.keras.Input(shape=(None,256))
            decoder_input = tf.keras.Input(shape=(None,), dtype=tf.int64)
            inputs = {'encoder': embedded, 'decoder': decoder_input }
            outputs = self.transformer((embedded, decoder_input))
            self.model = tf.keras.Model(inputs=inputs, outputs=outputs, name='TextTransformer')
            checkpoint = tf.train.Checkpoint(model=self.model)

            last = tf.train.latest_checkpoint(checkpoint_dir)
            checkpoint.restore(last).expect_partial()

            if not last is None:
                self.init_epoch = int(os.path.basename(last).split('-')[1])
                print('loaded %d epoch'%self.init_epoch)
            else:
                self.init_epoch = 0

            self.model.summary()

        self.encoder = tf.keras.models.load_model('result/encoder', compile=False)
        self.encoder.call = tf.function(self.encoder.call, experimental_relax_shapes=True)
        self.encoder.summary()

        self.target_font = 'data/jpfont/ipaexm.ttf'

    def run_transformer(self, encoder_input):
        decoder_input = tf.constant([256], dtype=tf.int32)
        output = tf.expand_dims(decoder_input, 0)
        encoder_input = tf.expand_dims(encoder_input, 0)
        MAX_LENGTH = 1024
        SEARCH_N = 4

        predictions = self.transformer(
            (encoder_input, output))
        predictions = tf.nn.softmax(predictions, axis=-1)
        predictions = predictions[:, -1, :]
        top_probs, top_idxs = tf.math.top_k(predictions, k=SEARCH_N)
        predicted_ids = top_idxs[0,:]
        scores = top_probs[0, :]

        predicted_ids = predicted_ids[:, tf.newaxis]
        output = tf.tile(tf.expand_dims(decoder_input, 0), [SEARCH_N, 1])
        output = tf.concat([output, predicted_ids], axis=-1)
        encoder_input = tf.tile(encoder_input, [SEARCH_N, 1, 1])

        for i in range(1, MAX_LENGTH):
            end_mask = output[:,-1] == 256+1
            if tf.math.reduce_all(end_mask):
                break
            scores = tf.tile(scores[:, tf.newaxis], [1, SEARCH_N])

            predictions = self.transformer(
                (encoder_input, output))

            predictions = tf.nn.softmax(predictions, axis=-1)
            predictions = predictions[:, -1, :]
            top_probs, top_idxs = tf.math.top_k(predictions, k=SEARCH_N)
            scores = tf.where(end_mask, scores, scores * top_probs)
            scores = tf.reshape(scores, [-1])
            end_mask = tf.tile(end_mask[:,tf.newaxis], [1, SEARCH_N])
            end_mask = tf.reshape(end_mask, [-1])
            predicted_ids = tf.where(end_mask, 256+1, tf.reshape(top_idxs,[-1]))
            top_probs, top_idxs = tf.math.top_k(scores, k=SEARCH_N)
            output = tf.reshape(tf.tile(output[:,tf.newaxis,:], [1,SEARCH_N,1]), [SEARCH_N*SEARCH_N, -1])
            predicted_ids = predicted_ids[:, tf.newaxis]
            output = tf.concat([output, predicted_ids], axis=-1)
            output = tf.gather_nd(output, top_idxs[:,tf.newaxis])
            scores = top_probs
            
        ret = []
        for i in range(SEARCH_N):     
            predict = output[i,:]
            score = scores[i].numpy()
            pred_bytes = predict[1:].numpy()
            end_idx = np.where(pred_bytes == 257)[0].tolist()
            if len(end_idx) > 0:
                pred_bytes = pred_bytes[:end_idx[0]]
            pred_bytes = pred_bytes.tolist()
            pred_text = bytes(pred_bytes).decode("utf-8", "backslashreplace")
            ret.append((score,pred_text))
        return ret

    def sub_load(self, value):
        proc = subprocess.Popen([
            './data/load_font/load_font',
            self.target_font,
            '96',
            ], stdin=subprocess.PIPE, stdout=subprocess.PIPE)

        ret = []
        for c in value:
            if c.isspace():
                ret.append(None)
                continue       
            charbuf = c.encode("utf-32-le")
            proc.stdin.write(charbuf[:4])
            proc.stdin.flush()
            result = proc.stdout.read(48)

            rows = int.from_bytes(result[:8], 'little')
            width = int.from_bytes(result[8:16], 'little')
            left = int.from_bytes(result[16:24], 'little', signed=True)
            top = int.from_bytes(result[24:32], 'little', signed=True)
            advance_x = int.from_bytes(result[32:40], 'little', signed=True)
            advance_y = int.from_bytes(result[40:48], 'little', signed=True)

            if rows == 0 or width == 0:
                continue

            buffer = proc.stdout.read(rows*width)
            img = np.frombuffer(buffer, dtype='ubyte').reshape(rows,width)
            
            ret.append({
                    'rows': rows, 
                    'width': width, 
                    'left': left,
                    'top': top,
                    'advx': advance_x / 64,
                    'image': img
                })
        proc.stdin.close()
        return ret

    def make_features(self, target_data):
        height = 128
        width = 128
        fontsize = 96
        image_batch = len(target_data)

        image = np.zeros([image_batch, height, width, 3], dtype=np.float32)

        print('make images')
        for i, item in enumerate(target_data):
            if item is None:
                continue
            im = item['image']
            w = item['width']
            h = item['rows']
            t = item['top']
            l = item['left']

            img = np.zeros([height, width], dtype=np.float32)
            margin = (width - fontsize) / 2
            tile_top = min(max(0, int(height - margin - t)), height - h)
            tile_left = min(max(0, int(margin + l)), width - w)
            img[tile_top:tile_top+h,tile_left:tile_left+w] = im[:h,:w]

            img = 1 - img / 255.
            image[i,:,:,0] = img * 255.
            image[i,:,:,1] = img * 255.
            image[i,:,:,2] = img * 255.

        print('run encoder')
        inputs = {
            'image': tf.constant(image, dtype=tf.float32), 
        }
        embedding = self.encoder(inputs,training=False)
        return embedding

    def make_pad(self, embeddeing):
        encoder_inputs = tf.cast(tf.concat([
            tf.ones([1,256]) * 10.,
            embeddeing,
            tf.ones([1,256]) * -10,
        ], axis=0), tf.float32)
        return encoder_inputs

def test_model(font):
    encoder = SimpleEncodeDecoder()
    if font:
        encoder.target_font = font

    while True:
        val = input('Enter input text: ')
        if not val:
            break
            
        target_data = encoder.sub_load(val)
        embedding = encoder.make_features(target_data)            
        encoder_inputs = encoder.make_pad(embedding)
        predictions = encoder.run_transformer(encoder_inputs)
        for p, text in predictions:
            print('p=',p,'prediction:',text)

if __name__ == '__main__':
    import sys

    if len(sys.argv) > 1:
        test_model(sys.argv[1])
    else:
        test_model(None)
