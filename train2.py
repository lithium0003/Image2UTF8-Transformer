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
import signal

import net

isRunning = True

def is_alive():
    return isRunning

def handler(signum, frame):
    global isRunning
    print('Signal handler called with signal', signum)
    isRunning = False

signal.signal(signal.SIGUSR1, handler)

class WarmupExponentialDecay(tf.keras.optimizers.schedules.LearningRateSchedule):
  def __init__(
      self,
      start_learing_rate,
      initial_learning_rate,
      warmup_steps,
      decay_steps,
      decay_rate,
      name=None):
    super().__init__()
    self.start_learing_rate = start_learing_rate
    self.initial_learning_rate = initial_learning_rate
    self.warmup_steps = warmup_steps
    self.decay_steps = decay_steps
    self.decay_rate = decay_rate
    self.name = name

  def __call__(self, step):
    with tf.name_scope(self.name or "WarmupExponentialDecay") as name:
      initial_learning_rate = tf.convert_to_tensor(
          self.initial_learning_rate, name="initial_learning_rate")
      start_learing_rate = tf.convert_to_tensor(
          self.start_learing_rate, name="start_learing_rate")
      dtype = initial_learning_rate.dtype
      warmup_steps = tf.cast(self.warmup_steps, dtype)
      decay_steps = tf.cast(self.decay_steps, dtype)
      decay_rate = tf.cast(self.decay_rate, dtype)

      global_step_recomp = tf.cast(step, dtype)
      p = (global_step_recomp - warmup_steps) / decay_steps
      return tf.where(global_step_recomp < warmup_steps,
        start_learing_rate + (initial_learning_rate - start_learing_rate) * global_step_recomp / warmup_steps,
        initial_learning_rate * tf.math.pow(decay_rate, p),
        name=name
      )

  def get_config(self):
    return {
        "start_learing_rate": self.start_learing_rate,
        "initial_learning_rate": self.initial_learning_rate,
        "warmup_steps": self.warmup_steps,
        "decay_steps": self.decay_steps,
        "decay_rate": self.decay_rate,
        "name": self.name
    }

def padded_cross_entropy_loss(logits, labels, masks, smoothing=0.05, vocab_size=256):
  """Calculate cross entropy loss while ignoring padding.
  Args:
    logits: Tensor of size [batch_size, length_logits, vocab_size]
    labels: Tensor of size [batch_size, length_labels]
    masks: Tensor of size [batch_size, length_labels], 1 means vaild, 0 means masked
    smoothing: Label smoothing constant, used to determine the on and off values
    vocab_size: int size of the vocabulary
  Returns:
    Returns the cross entropy loss and weight tensors: float32 tensors with
      shape [batch_size, max(length_logits, length_labels)]
  """
  with tf.name_scope("loss"):
    # Calculate smoothing cross entropy
    with tf.name_scope("smoothing_cross_entropy"):
      confidence = 1.0 - smoothing
      low_confidence = (1.0 - confidence) / tf.cast(vocab_size - 1, tf.float32)
      soft_targets = tf.one_hot(
          tf.cast(labels, tf.int32),
          depth=vocab_size,
          on_value=confidence,
          off_value=low_confidence)
      xentropy = tf.nn.softmax_cross_entropy_with_logits(
          logits=logits, labels=soft_targets)

      # Calculate the best (lowest) possible value of cross entropy, and
      # subtract from the cross entropy loss.
      normalizing_constant = -(
          confidence * tf.math.log(confidence) +
          tf.cast(vocab_size - 1, tf.float32) * low_confidence *
          tf.math.log(low_confidence + 1e-20))
      xentropy -= normalizing_constant

    weights = tf.cast(masks == True, tf.float32)
    return xentropy * weights, weights


class SimpleEncodeDecoder:
    def __init__(self):
        self.save_dir = './result/step2/'
        checkpoint_dir = self.save_dir

        self.max_epoch = 300
        self.steps_per_epoch = 1000
        self.batch_size = 16

        lr = WarmupExponentialDecay(1e-6, 5e-5, 15e3, 1e5, 0.5)
        self.optimizer = tf.keras.optimizers.Adam(lr)

        self.transformer = net.TextTransformer(vocab_size=256)
        embedded = tf.keras.Input(shape=(None,256))
        decoder_input = tf.keras.Input(shape=(None,), dtype=tf.int32)
        decoder_true = tf.keras.Input(shape=(None,), dtype=tf.int32)
        inputs = {'encoder': embedded, 'decoder': decoder_input, 'decoder_true': decoder_true }
        outputs = self.transformer((embedded, decoder_input))
        self.model = tf.keras.Model(inputs=inputs, outputs=outputs, name='TextTransformer')
        checkpoint = tf.train.Checkpoint(optimizer=self.optimizer, model=self.model)

        last = tf.train.latest_checkpoint(checkpoint_dir)
        checkpoint.restore(last)
        self.manager = tf.train.CheckpointManager(
                checkpoint, directory=checkpoint_dir, max_to_keep=2)
        if not last is None:
            self.init_epoch = int(os.path.basename(last).split('-')[1])
            print('loaded %d epoch'%self.init_epoch)
        else:
            self.init_epoch = 0

        self.model.summary()

        self.summary_writer_train = tf.summary.create_file_writer(
                os.path.join(self.save_dir, "train"))
        self.summary_writer_test = tf.summary.create_file_writer(
                os.path.join(self.save_dir, "test"))

    def save_weight(self):
        save_dir = 'result/transformer_weights/ckpt'
        self.transformer.save_weights(save_dir)

    def loss_func(self, inputs, outputs):
        label = inputs['decoder_true']
        mask = inputs['decoder'] > 255
        xentropy, weights = padded_cross_entropy_loss(outputs, label, mask)
        loss = tf.reduce_sum(xentropy) / tf.reduce_sum(weights)
        return loss
    
    @tf.function
    def train_substep(self, inputs, step):
        with tf.GradientTape() as tape1:
            outputs = self.model(inputs, training=True)
            loss = self.loss_func(inputs, outputs)

        model_gradients = tape1.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(model_gradients, self.model.trainable_variables))

        with self.summary_writer_train.as_default():
            tf.summary.scalar('loss', loss, step=step)
            tf.summary.scalar('decayed_lr', self.optimizer._decayed_lr(tf.float32), step=step)

    @tf.function
    def test_substep(self, inputs, step):
        outputs = self.model(inputs)
        loss = self.loss_func(inputs, outputs)

        with self.summary_writer_test.as_default():
            tf.summary.scalar('loss', loss, step=step)

    def evaluate(self, step):
        LOOP_COUNT = 4
        LEN_CANDIDATE = 4
        result_text = tf.constant([['true','order','p','predict']])
        for _ in range(4):
            decoder_input = tf.constant([257], dtype=tf.int32)
            output = tf.expand_dims(decoder_input, 0)
            inputs = self.data.generate_data()
            encoder_input = tf.expand_dims(inputs['encoder'], 0)

            predictions = self.transformer(
                (encoder_input, output))
            predictions = tf.nn.softmax(predictions, axis=-1)
            predictions = predictions[0, 0, :]
            values, indices = tf.math.top_k(predictions, k=LEN_CANDIDATE)
            print(values.numpy(), indices.numpy())
            pred_lens = tf.where(indices > self.data.max_decoderlen, self.data.max_decoderlen, indices)
            pred_lens = pred_lens.numpy()
            print(pred_lens)
            max_len = max(pred_lens)
            decoder_input = [tf.constant([256] * pred_len + [0] * (max_len - pred_len), dtype=tf.int32) for pred_len in pred_lens]
            output = tf.stack(decoder_input, 0)
            encoder_input = tf.tile(encoder_input, [LEN_CANDIDATE, 1, 1])
            for i in range(0, LOOP_COUNT):
                #print(output)
                predictions = self.transformer(
                    (encoder_input, output))
                predictions = tf.nn.softmax(predictions, axis=-1)
                pred_idx = tf.math.argmax(predictions, axis=-1)
                #print(pred_idx)
                bs, ss, ch = predictions.get_shape()
                t1 = tf.tile(tf.range(ss,dtype=pred_idx.dtype)[tf.newaxis, :, tf.newaxis], multiples=(bs,1,1))
                t2 = tf.concat((t1, tf.expand_dims(pred_idx, -1)),axis=-1)
                t3 = tf.tile(tf.range(tf.cast(bs, dtype=pred_idx.dtype))[:, tf.newaxis, tf.newaxis], (1,ss,1))
                idx = tf.concat((t3, t2), -1)
                scores = tf.gather_nd(predictions, idx)
                if i + 1 < LOOP_COUNT:
                    fix_count = np.maximum(np.array(pred_lens) * (i + 1) / LOOP_COUNT, 1)
                    output = []
                    for k in range(LEN_CANDIDATE):
                        values, indices = tf.math.top_k(scores[k,:pred_lens[k]], k=fix_count[k])
                        th_value = values[-1]
                        masked_input = tf.where(scores[k,:pred_lens[k]] < th_value, 256, pred_idx[k,:pred_lens[k]])
                        masked_input = tf.concat([
                            tf.cast(masked_input, dtype=tf.int32),
                            tf.constant([0] * (max_len - pred_lens[k]), dtype=tf.int32),
                        ],axis=0)
                        output.append(masked_input)
                    output = tf.stack(output, 0)
                else:
                    output = []
                    mean_scores = []
                    for k in range(LEN_CANDIDATE):
                        output.append(pred_idx[k,:pred_lens[k]].numpy())
                        mean_scores.append(tf.math.exp(tf.math.reduce_mean(tf.math.log(scores[k,:pred_lens[k]]))).numpy())

            order = np.argsort(np.argsort(-np.array(mean_scores))) + 1

            for i in range(LEN_CANDIDATE):
                pred_bytes = output[i]
                score = mean_scores[i]
                pred_bytes = pred_bytes.tolist()
                pred_text = bytes(pred_bytes).decode("utf-8", "backslashreplace")
                input_text = inputs['text']
                if pred_text == input_text:
                    print(input_text, '%d'%order[i], '*%f'%score , pred_text)
                    result_text = tf.concat([result_text, tf.constant([[input_text, '%d'%order[i], '*%f'%score , pred_text]])], axis=0)
                else:
                    print(input_text, '%d'%order[i], ' %f'%score , pred_text)
                    result_text = tf.concat([result_text, tf.constant([[input_text, '%d'%order[i], '%f'%score , pred_text]])], axis=0)

        with self.summary_writer_test.as_default():
            tf.summary.text("predict", result_text, step=step)

    def fit(self, train_ds, test_ds, init_epoch=0):
        for epoch in range(init_epoch, self.max_epoch):
            if not is_alive():
                print('Break!')
                return

            print("Epoch: ", epoch + 1)
            with tqdm.tqdm(total=self.steps_per_epoch) as pbar:
                for n, inputs in train_ds.take(self.steps_per_epoch).enumerate():
                    self.train_substep(inputs, n + epoch * self.steps_per_epoch)
                    pbar.update(1)
            self.manager.save()
            
            print("Test: ", epoch + 1)
            with tqdm.tqdm(total=self.steps_per_epoch // 10) as pbar:
                for n, inputs in test_ds.take(self.steps_per_epoch // 10).enumerate():
                    self.test_substep(inputs, n * 10 + epoch * self.steps_per_epoch)
                    pbar.update(1)

            print('Evaluate')
            self.evaluate(epoch)

    def train(self):
        self.data = net.RandomTextFeatureData()
        self.fit(self.data.train_data(self.batch_size), self.data.test_data(self.batch_size), init_epoch=self.init_epoch)

def train():
    encoder = SimpleEncodeDecoder()
    encoder.train()
    encoder.save_weight()

if __name__ == '__main__':
    train()
