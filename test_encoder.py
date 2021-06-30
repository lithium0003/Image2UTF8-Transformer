#!/usr/bin/env python3

import tensorflow as tf
physical_devices = tf.config.list_physical_devices('GPU')
try:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
except:
    # Invalid device or cannot modify virtual devices once initialized.
    pass

import numpy as np
import os, time, csv, copy, glob
import tqdm
import datetime
import random
import subprocess
from PIL import Image

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from matplotlib import rcParams
rcParams['font.family'] = 'sans-serif'
rcParams['font.sans-serif'] = ['Hiragino Maru Gothic Pro', 'Yu Gothic', 'Meirio', 'Takao', 'IPAexGothic', 'IPAPGothic', 'Noto Sans CJK JP']

import net

def sub_load(args):
    proc = subprocess.Popen([
        './data/load_font/load_font',
        args[0],
        '96',
        ], stdin=subprocess.PIPE, stdout=subprocess.PIPE)

    ret = {}
    for c in args[1]:
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
        
        ret[(args[0],c)] = {
                'rows': rows, 
                'width': width, 
                'left': left,
                'top': top,
                'advx': advance_x / 64,
                'image': img
            }
    proc.stdin.close()
    return ret

def sub_load_image(path):
    dirnames = glob.glob(os.path.join(path, '*'))
    ret = {}
    for d in dirnames:
        c_code = os.path.basename(d)
        char = str(bytes.fromhex(c_code), 'utf-8')
        count = 0
        for f in glob.glob(os.path.join(d, '*.png')):
            rawim = np.asarray(Image.open(f).convert('L'))
            ylines = np.any(rawim < 255, axis=1)
            content = np.where(ylines)[0]
            rows = content[-1] - content[0] + 1
            top = 128 - 16 - content[0]
            y = content[0]
            xlines = np.any(rawim < 255, axis=0)
            content = np.where(xlines)[0]
            width = content[-1] - content[0] + 1
            left = content[0] - 16
            x = content[0]

            if rows == 0 or width == 0:
                continue

            img = 255 - rawim[y:y+rows,x:x+width]

            ret[('hand%06d'%count,char)] = {
                    'rows': rows, 
                    'width': width, 
                    'left': left,
                    'top': top,
                    'advx': 96.0,
                    'image': img
                }
            count += 1
    return ret

def random_lineimage():
    width = 1024
    height = 1024
    img = np.zeros([width, height])

    for _ in range(100):
        th = random.random() * np.pi
        x0 = random.randrange(0, width)
        y0 = random.randrange(0, height)

        for d in range(max(width,height)):
            x = int(d * np.cos(th) + x0)
            y = int(d * np.sin(th) + y0)
            if x < 0 or x >= width or y < 0 or y >= height:
                break
            img[y, x] = 255
            
        for d in range(max(width,height)):
            x = int(d * np.cos(th + np.pi) + x0)
            y = int(d * np.sin(th + np.pi) + y0)
            if x < 0 or x >= width or y < 0 or y >= height:
                break
            img[y, x] = 255

    return img

class SimpleEncodeDecoder:
    def __init__(self):
        encoder_dir = './result/encoder/'
        if os.path.exists(encoder_dir):
            self.encoder = tf.keras.models.load_model(encoder_dir)
            self.encoder.summary()
        else:
            checkpoint_dir = 'result/step1/'

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
            checkpoint = tf.train.Checkpoint(model=self.model)

            last = tf.train.latest_checkpoint(checkpoint_dir)
            checkpoint.restore(last).expect_partial()

            if not last is None:
                self.init_epoch = int(os.path.basename(last).split('-')[1])
                print('loaded %d epoch'%self.init_epoch)
            else:
                self.init_epoch = 0

            self.model.summary()

        self.target_font = 'data/jpfont/ipaexm.ttf'
        self.handimg_cache = {}
        self.handimg_cache.update(sub_load_image('data/handwritten'))

        self.random_noise = [random_lineimage() for _ in range(10)]

    def run_encode(self, glyph, hand=False):
        if not hand:
            img_cache = sub_load([self.target_font, glyph])
        width = 128
        height = 128
        fontsize = 96

        for g in glyph:
            min_pixel = int(fontsize * 0.9)
            max_pixel = int(fontsize * 1.1)
            tile_size = random.randint(min_pixel, max_pixel)

            image = np.zeros([height, width], dtype=np.float32)

            angle = 7.5 * np.random.normal() / 180 * np.pi
            angle = np.clip(angle, -np.pi, np.pi)
            pad_x = np.random.normal() * width * 0.02
            pad_y = np.random.normal() * height * 0.02

            if not g.isspace():
                if hand:
                    item = random.choice([self.handimg_cache[x] for x in self.handimg_cache if x[1] == g])
                else:
                    item = img_cache[(self.target_font, g)]
                ratio = tile_size / fontsize
                if item['width'] * ratio > width:
                    ratio = width / item['width']
                if item['rows'] * ratio > height:
                    ratio = height / item['rows']
                w = max(1, int(item['width'] * ratio))
                h = max(1, int(item['rows'] * ratio))
                t = item['top'] * ratio
                l = item['left'] * ratio

                margin = (width - fontsize) / 2
                pad_x = np.clip(pad_x, -(margin + l), width - w - l - margin)
                pad_y = np.clip(pad_y, -(height - (margin + t)), margin - (h - t))

                im = np.asarray(Image.fromarray(item['image']).resize((w,h)))
                tile_top = min(max(0, int(height - margin - t)), height - h)
                tile_left = min(max(0, int(margin + l)), width - w)
                image[tile_top:tile_top+h,tile_left:tile_left+w] = im[:h,:w]
                im = Image.fromarray(image).rotate(angle / np.pi * 180, resample=Image.BILINEAR, translate=(pad_x, pad_y))

                if random.random() < 0.5:
                    im = np.maximum(im, (np.random.random([height, width]) < 0.1) * 255)
                if random.random() < 0.5:
                    x = random.randrange(0, self.random_noise[0].shape[1] - 128)
                    y = random.randrange(0, self.random_noise[0].shape[0] - 128)
                    im = np.maximum(im, random.choice(self.random_noise)[y:y+128,x:x+128])

                image = np.asarray(im) / 255.

            img = image[...,np.newaxis]
            if random.uniform(0.,1.) < 0.5:
                fg_c = random.uniform(-1., 1.)
                bk_c = random.uniform(-1., 1.)
                if abs(fg_c - bk_c) < 1.0:
                    d = fg_c - bk_c
                    if d < 0:
                        d = -1.0 - d
                    else:
                        d = 1.0 - d
                    fg_c += d / 2
                    bk_c -= d / 2
                fgimg = np.array([fg_c,fg_c,fg_c]).reshape(1,1,1,-1)
                bkimg = np.array([bk_c,bk_c,bk_c]).reshape(1,1,1,-1)
            else:
                fg_c = np.random.uniform(-1., 1., size=[3])
                bk_c = np.random.uniform(-1., 1., size=[3])
                if np.all(np.abs(fg_c - bk_c) < 1.0):
                    ind = np.argmax(np.abs(fg_c - bk_c))
                    d = (fg_c - bk_c)[ind]
                    if d < 0:
                        d = -1.0 - d
                    else:
                        d = 1.0 - d
                    fg_c += d / 2
                    bk_c -= d / 2
                fgimg = fg_c.reshape(1,1,1,-1)
                bkimg = bk_c.reshape(1,1,1,-1)
            fgimg = fgimg + np.random.normal(size=[1, height, width, 3]) * random.gauss(0., 0.1)
            bkimg = bkimg + np.random.normal(size=[1, height, width, 3]) * random.gauss(0., 0.1)
            fgimg = fgimg.astype(np.float32)
            bkimg = bkimg.astype(np.float32)
            image = fgimg * img + bkimg * (1 - img)
            image = np.clip(image, -1., 1.)
            image = image * 127.5 + 127.5

            embedded = self.encoder(image)
            embedded = embedded.numpy().reshape([16,-1])
            print(embedded.max(), embedded.min())
            print(g)

            plt.subplot(1,2,1)
            plt.imshow(embedded, interpolation='none', vmin=-5, vmax=5, aspect='equal')
            plt.title(g)
            plt.subplot(1,2,2)
            plt.imshow(image[0,...] / 255., aspect='equal')
            lastfile = sorted(glob.glob('test-*.png'))
            if lastfile:
                i = int(os.path.splitext(lastfile[-1])[0].split('-')[1]) + 1
            else:
                i = 0
            plt.savefig('test-%08d.png'%i, bbox_inches='tight', pad_inches=0)
            plt.close('all')


def test_model():
    encoder = SimpleEncodeDecoder()

    while True:
        val = input('Enter input text: ')
        if not val:
            break

        encoder.run_encode(val)

def test_handwritten():
    encoder = SimpleEncodeDecoder()

    while True:
        val = input('Enter input text: ')
        if not val:
            break

        encoder.run_encode(val, hand=True)

if __name__ == '__main__':
    import sys

    if len(sys.argv) < 2:
        test_model()
    elif sys.argv[1] == 'hand':
        test_handwritten()
    else:
        test_model()
