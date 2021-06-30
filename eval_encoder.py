#!/usr/bin/env python3

import tensorflow as tf
physical_devices = tf.config.list_physical_devices('GPU')
try:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
except:
    # Invalid device or cannot modify virtual devices once initialized.
    pass

import numpy as np
import os, time, csv
import tqdm
import umap
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import datetime
import signal

import net

from matplotlib import rcParams
rcParams['font.family'] = 'sans-serif'
rcParams['font.sans-serif'] = ['Hiragino Maru Gothic Pro', 'Yu Gothic', 'Meirio', 'Takao', 'IPAexGothic', 'IPAPGothic', 'Noto Sans CJK JP']

import net

class SimpleEncodeDecoder:
    def __init__(self):
        self.save_dir = './result/step1/'
        self.result_dir = './result/plot/'
        os.makedirs(self.result_dir, exist_ok=True)
        checkpoint_dir = self.save_dir

        self.max_epoch = 300
        self.steps_per_epoch = 1000
        self.batch_size = 64

        lr = tf.keras.optimizers.schedules.ExponentialDecay(1e-3, 1e5, 0.5)
        self.optimizer = tf.keras.optimizers.Adam(lr)

        self.encoder = net.FeatureBlock()
        self.encoder.summary()
        self.decoder = net.SimpleDecoderBlock()
        self.decoder.summary()
        inputs = {
            'image': tf.keras.Input(shape=(128,128,3)),
        }
        feature_out = self.encoder(inputs)
        outputs = self.decoder(feature_out)
        self.model = tf.keras.Model(inputs, outputs, name='SimpleEncodeDecoder')
        checkpoint = tf.train.Checkpoint(optimizer=self.optimizer,
                            model=self.model)

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

    def eval(self):
        self.data = net.FontData()
        print("Plot: ", self.init_epoch + 1)
        acc = self.make_plot(self.data.test_data(self.batch_size), (self.init_epoch + 1))
        print('acc', acc)

    @tf.function
    def eval_substep(self, inputs):
        input_data = {
            'image': inputs['input'],
        }
        feature = self.encoder(input_data)
        outputs = self.decoder(feature)
        target_id = inputs['index']
        target_id1 = inputs['idx1']
        target_id2 = inputs['idx2']
        pred_id1 = tf.nn.softmax(outputs['id1'], -1)
        pred_id2 = tf.nn.softmax(outputs['id2'], -1)
        return {
            'feature': feature,
            'pred_id1': pred_id1,
            'pred_id2': pred_id2,
            'target_id': target_id,
            'target_id1': target_id1,
            'target_id2': target_id2,
        }

    def make_plot(self, test_ds, epoch):
        result = []
        labels = []
        with open(os.path.join(self.result_dir,'test_result-%d.txt'%epoch),'w') as txt:
            correct_count = 0
            failed_count = 0
            with tqdm.tqdm(total=len(self.data.test_keys)) as pbar:
                for inputs in test_ds:
                    pred = self.eval_substep(inputs)
                    result += [pred['feature']]
                    labels += [pred['target_id']]
                    for i in range(pred['target_id1'].shape[0]):
                        txt.write('---\n')
                        target = pred['target_id'][i].numpy()
                        txt.write('target: id %d = %s\n'%(target, self.data.glyphs[target-1]))
                        predid1 = np.argmax(pred['pred_id1'][i])
                        predid2 = np.argmax(pred['pred_id2'][i])
                        predid = predid1 * 100 + predid2
                        if predid == 0:
                            txt.write('predict: id %d nothing (p=%f)\n'%(predid, pred['pred_id1'][i][predid1] * pred['pred_id2'][i][predid2]))
                        elif predid > self.data.id_count + 1:
                            txt.write('predict: id %d nothing (p=%f)\n'%(predid, pred['pred_id1'][i][predid1] * pred['pred_id2'][i][predid2]))
                        else:
                            txt.write('predict: id %d = %s (p=%f)\n'%(predid, self.data.glyphs[predid-1], pred['pred_id1'][i][predid1] * pred['pred_id2'][i][predid2]))
                        if target == predid:
                            txt.write('Correct!\n')
                            correct_count += 1
                        else:
                            txt.write('Failed!\n')
                            failed_count += 1
                        pbar.update(1)
            acc = correct_count / (correct_count + failed_count)
            txt.write('==============\n')
            txt.write('Correct = %d\n'%correct_count)
            txt.write('Failed = %d\n'%failed_count)
            txt.write('accuracy = %f\n'%acc)

        result = np.concatenate(result)
        labels = np.concatenate(labels)
        print('run UMAP')

        X_reduced = umap.UMAP(metric='cosine').fit_transform(result)
        fig, ax = plt.subplots(figsize=(50, 50))
        ax.scatter(X_reduced[:, 0], X_reduced[:, 1], c=labels, cmap=plt.get_cmap('hsv'))
        
        print('plot UMAP')
        for i, label in enumerate(labels):
            ax.annotate(self.data.glyphs[label-1], (X_reduced[i,0], X_reduced[i,1]))

        plt.savefig(os.path.join(self.result_dir,'test_result-%d.png'%epoch), dpi=300)
        plt.close('all')
        return acc

def eval():
    encoder = SimpleEncodeDecoder()
    encoder.eval()

if __name__ == '__main__':
    eval()
