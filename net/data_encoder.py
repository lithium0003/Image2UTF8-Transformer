import tensorflow as tf
physical_devices = tf.config.list_physical_devices('GPU')
try:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
except:
    # Invalid device or cannot modify virtual devices once initialized.
    pass

import numpy as np
from PIL import Image
import random
import tqdm
import glob, os
import itertools
import operator
from collections import defaultdict
import csv
import time
from multiprocessing import Pool, Process, JoinableQueue
import queue
import multiprocessing
import subprocess

class BaseData:
    def __init__(self):
        self.load_idmap()

    def load_idmap(self):
        self.glyph_id = {}
        self.glyphs = {}
        self.glyph_type = {}

        with open('data/id_map.csv','r') as f:
            reader = csv.reader(f)
            for row in reader:
                self.glyph_id[row[1]] = int(row[0])
                self.glyphs[int(row[0])] = row[1]
                self.glyph_type[int(row[0])] = int(row[2])

        self.id_count = len(self.glyph_id)
        self.kanji2_offset = min([k for k in range(self.id_count) if self.glyph_type[k] == 8])
        self.kanji3_offset = min([k for k in range(self.id_count) if self.glyph_type[k] == 9])
        self.kanji4_offset = min([k for k in range(self.id_count) if self.glyph_type[k] == 10])

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

class FontData(BaseData):
    def __init__(self):
        super().__init__()

        self.img_cache = {}
        print('loading handwrite image')
        self.img_cache.update(sub_load_image('data/handwritten'))

        self.random_noise = []
        for _ in range(10):
            self.random_noise.append(random_lineimage())

        print('loading enfont')
        enfont_files = sorted(glob.glob('data/enfont/*.ttf') + glob.glob('data/enfont/*.otf'))
        en_glyphs = [self.glyphs[key] for key in self.glyphs.keys() if self.glyph_type[key] in [0,1,2,6]]
        items = [(f, en_glyphs) for f in enfont_files]
        total = len(enfont_files)
        with Pool() as pool:
            dicts = tqdm.tqdm(pool.imap_unordered(sub_load, items), total=total)
            for dictitem in dicts:
                self.img_cache.update(dictitem)

        print('loading jpfont')
        jpfont_files = sorted(glob.glob('data/jpfont/*.ttf') + glob.glob('data/jpfont/*.otf'))
        items = [(f, list(self.glyphs.values())) for f in jpfont_files]
        total = len(jpfont_files)
        with Pool() as pool:
            dicts = tqdm.tqdm(pool.imap_unordered(sub_load, items), total=total)
            for dictitem in dicts:
                self.img_cache.update(dictitem)

        self.count = 0
        self.image_keys = list(self.img_cache.keys())
        self.test_keys = self.get_test_keys()
        self.train_keys = self.get_train_keys()

    def get_test_keys(self):
        def fontname(fontpath):
            return os.path.splitext(os.path.basename(fontpath))[0]

        keys = self.image_keys
        test_keys = list(filter(lambda x: fontname(x[0]).startswith('Noto'), keys))
        return test_keys

    def get_train_keys(self):
        def fontname(fontpath):
            return os.path.splitext(os.path.basename(fontpath))[0]

        keys = self.image_keys
        train_keys = list(filter(lambda x: not fontname(x[0]).startswith('Noto'), keys))
        return train_keys

    def test_data(self, batch_size):
        def process(i):
            return self.construct_alphatext(keys, aug=False)

        keys = self.test_keys
        ds = tf.data.Dataset.range(len(keys))
        ds = ds.map(process, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        ds = ds.batch(batch_size)
        ds = ds.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
        return ds

    def validate_data(self, batch_size):
        keys = self.test_keys
        return self.prob_images(keys, batch_size)

    def train_data(self, batch_size):
        keys = self.train_keys
        return self.prob_images(keys, batch_size)

    def prob_images(self, keys, batch_size):
        def process(i):
            return self.construct_alphatext(keys)

        ds = tf.data.Dataset.range(1)
        ds = ds.repeat()
        ds = ds.map(process, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        ds = ds.batch(batch_size)
        ds = ds.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
        return ds

    def construct_alphatext(self, keys, aug=True):
        width = 128
        height = 128
        fontsize = 96

        def load_image():
            if aug:
                key = random.choice(keys)
            else:
                key = keys[self.count % len(keys)]
                self.count += 1
            idx = self.glyph_id[key[1]]

            min_pixel = int(fontsize * 0.9)
            max_pixel = int(fontsize * 1.1)
            tile_size = random.randint(min_pixel, max_pixel)
            if not aug:
                tile_size = fontsize

            image = np.zeros([height, width], dtype=np.float32)
            if aug and random.random() < 0.01:
                if random.uniform(0.,1.) < 0.5:
                    image = np.maximum(image, (np.random.random([height, width]) < 0.1) * 255)
                if random.uniform(0.,1.) < 0.5:
                    x = random.randrange(0, self.random_noise[0].shape[1] - 128)
                    y = random.randrange(0, self.random_noise[0].shape[0] - 128)
                    image = np.maximum(image, random.choice(self.random_noise)[y:y+128,x:x+128])
                return image / 255., 0, 0, 0

            angle = 7.5 * np.random.normal() / 180 * np.pi
            angle = np.clip(angle, -np.pi, np.pi)
            pad_x = np.random.normal() * width * 0.1
            pad_y = np.random.normal() * height * 0.1
            if not aug:
                angle = 0
                pad_x = 0
                pad_y = 0

            item = self.img_cache[key]

            ratio = tile_size / fontsize
            if item['width'] * ratio > width:
                ratio = width / item['width']
            if item['rows'] * ratio > height:
                ratio = height / item['rows']
            w = max(1, int(item['width'] * ratio))
            h = max(1, int(item['rows'] * ratio))
            t = item['top'] * ratio
            l = item['left'] * ratio
            im = np.asarray(Image.fromarray(item['image']).resize((w,h)))
            margin = (width - fontsize) / 2
            pad_x = np.clip(pad_x, -(margin + l), width - w - l - margin)
            pad_y = np.clip(pad_y, -(height - (margin + t)), margin - (h - t))
            tile_top = min(max(0, int(height - margin - t)), height - h)
            tile_left = min(max(0, int(margin + l)), width - w)
            image[tile_top:tile_top+h,tile_left:tile_left+w] = im[:h,:w]
            im = Image.fromarray(image).rotate(angle / np.pi * 180, resample=Image.BILINEAR, translate=(pad_x, pad_y))
            im = np.array(im)

            # random noise
            if aug:
                if random.uniform(0.,1.) < 0.5:
                    im = np.maximum(im, (np.random.random([height, width]) < 0.1) * 255)
                if random.uniform(0.,1.) < 0.5:
                    x = random.randrange(0, self.random_noise[0].shape[1] - 128)
                    y = random.randrange(0, self.random_noise[0].shape[0] - 128)
                    im = np.maximum(im, random.choice(self.random_noise)[y:y+128,x:x+128])
            
            return im / 255., idx + 1, (idx + 1) // 100, (idx + 1) % 100

        image, index, idx1, idx2 = tf.py_function(
            func=load_image, 
            inp=[], 
            Tout=[tf.float32, tf.int32, tf.int32, tf.int32])
        image = tf.ensure_shape(image, [height, width])
        index = tf.ensure_shape(index, [])
        idx1 = tf.ensure_shape(idx1, [])
        idx2 = tf.ensure_shape(idx2, [])

        return self.sub_constructimage(image, index, idx1, idx2)

    @tf.function
    def sub_constructimage(self, image, index, idx1, idx2):
        width = 128
        height = 128

        img = image[...,tf.newaxis]

        if tf.random.uniform([], 0., 1.) < 0.5:
            fg_c =  tf.random.uniform([], -1., 1.)
            bk_c =  tf.random.uniform([], -1., 1.)
            if tf.math.abs(fg_c - bk_c) < 1.0:
                d = fg_c - bk_c
                if d < 0:
                    d = -1.0 - d
                else:
                    d = 1.0 - d
                fg_c += d / 2
                bk_c -= d / 2
            fgimg = tf.stack([fg_c, fg_c, fg_c])[tf.newaxis, tf.newaxis, :]
            bkimg = tf.stack([bk_c, bk_c, bk_c])[tf.newaxis, tf.newaxis, :]
        else:
            fg_c =  tf.random.uniform([3], -1., 1.)
            bk_c =  tf.random.uniform([3], -1., 1.)
            if tf.math.reduce_sum(tf.math.abs(fg_c - bk_c)) < 3.0:
                ind = tf.math.argmax(tf.math.abs(fg_c - bk_c))
                d = (fg_c - bk_c)[ind]
                if d < 0:
                    d = -1.0 - d
                else:
                    d = 1.0 - d
                fg_c += d / 2
                bk_c -= d / 2
            fgimg = fg_c[tf.newaxis,tf.newaxis,:]
            bkimg = bk_c[tf.newaxis,tf.newaxis,:]
        fgimg += tf.random.normal([height, width, 3]) * tf.random.normal([]) * 0.1
        bkimg += tf.random.normal([height, width, 3]) * tf.random.normal([]) * 0.1

        image = fgimg * img + bkimg * (1 - img)

        image = tf.clip_by_value(image, -1.0, 1.0)
        image = image * 127.5 + 127.5

        return {
            'input': image, 
            'index': index,
            'idx1': idx1,
            'idx2': idx2,
        }

if __name__ == '__main__':
    import matplotlib.pyplot as plt

    data = FontData()
    ds = data.train_data(4)

    for d in ds:
        img = d['input']
        for i in range(img.shape[0]):
            img1 = img[i,:,:,:].numpy()
            img1 = img1 / 255.
            
            plt.figure(figsize=(5,5))
            plt.imshow(img1)

            plt.plot([16,16],[0,128],'k')
            plt.plot([0,128],[128-16,128-16],'k')

            plt.show()