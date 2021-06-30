#!/usr/bin/env python3

import tensorflow as tf
physical_devices = tf.config.list_physical_devices('GPU')
try:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
except:
    # Invalid device or cannot modify virtual devices once initialized.
    pass

import numpy as np
import os, time, csv, copy, sys
import tqdm
import datetime
import random
from timeit import default_timer as timer
from multiprocessing import Pool
import pickle
from queue import Queue
from threading import Thread

import net

running = True

def calc_traindata():
    global running

    if os.path.exists('result/encoder'):
        encoder = tf.keras.models.load_model('result/encoder')
        encoder.summary()
    else:
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

        last = tf.train.latest_checkpoint('result/step1/')
        checkpoint.restore(last).expect_partial()
        if not last is None:
            epoch = int(os.path.basename(last).split('-')[1])
            print('loaded %d epoch'%epoch)

    @tf.function(input_signature=(tf.TensorSpec(shape=[None,128,128], dtype=tf.float32),tf.TensorSpec(shape=[], dtype=tf.string)))
    def call_encoder(images, target_text):
        images = net.image_add_noise(images)
        embeddeing = encoder(images)
        decoder_true, decoder_task, encoder_inputs = data.convert_text(target_text, embeddeing)
        return decoder_true, decoder_task, encoder_inputs

    def call_construct(images, text):
        result = []
        for item, random_noise in images:
            result.append(net.construct_image(item, random_noise))
        return result, text

    data = net.RandomTransformerData()
    q_train1 = Queue(256)
    q_test1 = Queue(256)
    q_train2 = Queue(256)
    q_test2 = Queue(256)
    def run_process1(q, bs, training):
        global running
        count = 0
        text_cache = ''
        while running:
            if count <= 0:
                text_cache = data.get_newtext()
                data.font_load(text_cache, training)
                count = 5000
            count -= 1
            args = []
            for _ in range(bs):
                text = data.get_random_text(text_cache)
                #print(text, file=sys.stderr)
                target_text, target_data = data.get_target_text(text, training)
                if training:
                    random_noise = random.choices(data.random_noise, k=len(target_data))
                else:
                    random_noise = [None] * len(target_data)
                args.append((zip(target_data, random_noise),target_text))
            q.put([call_construct(*arg) for arg in args])

    def run_process2(q_in, q_out):
        global running
        while running:
            texts = []
            encoders = []
            decoder_tasks = []
            decoder_trues = []
            for result, target_text in q_in.get():
                images = np.stack(result)
                images = tf.constant(images, tf.float32)
                decoder_true, decoder_task, encoder_inputs = call_encoder(images, target_text)
                texts.append(target_text)
                encoders.append(encoder_inputs)
                decoder_tasks.append(decoder_task)
                decoder_trues.append(decoder_true)
            texts = tf.stack(texts)
            encoders = tf.stack(encoders)
            decoder_tasks = tf.stack(decoder_tasks)
            decoder_trues = tf.stack(decoder_trues)
            obj = (texts, encoders, decoder_tasks, decoder_trues)
            q_out.put(obj)
            q_in.task_done()

    train_thread1 = None
    test_thread1 = None
    print('Ready to input.')
    while True:
        try:
            user = input()
        except EOFError:
            break
        try:
            val, training = user.split(',')
            i = int(val)
            training = int(training)
            training = bool(training)
        except ValueError:
            i = 0

        if training:
            if not train_thread1:
                train_thread1 = Thread(target=run_process1, args=(q_train1, i, True), daemon=True)
                train_thread1.start()
                train_thread2 = Thread(target=run_process2, args=(q_train1, q_train2), daemon=True)
                train_thread2.start()
        else:
            if not test_thread1:
                test_thread1 = Thread(target=run_process1, args=(q_test1, i, False), daemon=True)
                test_thread1.start()
                test_thread2 = Thread(target=run_process2, args=(q_test1, q_test2), daemon=True)
                test_thread2.setDaemon(True)
                test_thread2.start()

        if i == 0:
            running = False
            break

        try:
            if training:
                obj = q_train2.get()
                q_train2.task_done()
            else:
                obj = q_test2.get()
                q_test2.task_done()
            pickle.dump(obj, sys.stdout.buffer, protocol=-1)
        except:
            running = False
            raise

if __name__ == '__main__':
    calc_traindata()
