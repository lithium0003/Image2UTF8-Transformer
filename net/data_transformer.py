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
import glob, os, sys
import csv
import time
import subprocess
from multiprocessing import Pool, Process, JoinableQueue
import queue
import datetime

from timeit import default_timer as timer

import json
import urllib.parse
import urllib.request
import zipfile
import io
import csv
import re
from html.parser import HTMLParser

UNICODE_WHITESPACE_CHARACTERS = [
    "\u0009", # character tabulation
    "\u000a", # line feed
    "\u000b", # line tabulation
    "\u000c", # form feed
    "\u000d", # carriage return
    "\u0020", # space
    "\u0085", # next line
    "\u00a0", # no-break space
    "\u1680", # ogham space mark
    "\u2000", # en quad
    "\u2001", # em quad
    "\u2002", # en space
    "\u2003", # em space
    "\u2004", # three-per-em space
    "\u2005", # four-per-em space
    "\u2006", # six-per-em space
    "\u2007", # figure space
    "\u2008", # punctuation space
    "\u2009", # thin space
    "\u200A", # hair space
    "\u2028", # line separator
    "\u2029", # paragraph separator
    "\u202f", # narrow no-break space
    "\u205f", # medium mathematical space
    "\u3000", # ideographic space
]

####################################################
# wikipedia
####################################################

# Wikipedia API
WIKI_en_URL = "https://en.wikipedia.org/w/api.php?"
WIKI_jp_URL = "https://ja.wikipedia.org/w/api.php?"

# 記事を1件、ランダムに取得するクエリのパラメータを生成する
def set_url_random():
    params = {
        'action': 'query',
        'format': 'json',
        'list': 'random', #ランダムに取得
        'rnnamespace': 0, #標準名前空間を指定する
        'rnlimit': 1 #結果数の上限を1にする(Default: 1)
    }
    return params

# 指定された記事の内容を取得するクエリのパラメータを生成する
def set_url_extract(pageid):
    params = {
        'action': 'query',
        'format': 'json',
        'prop': 'extracts',
        'pageids': pageid, #記事のID
        'explaintext': '',
    }
    return params

#ランダムな記事IDを取得
def get_random_wordid(en=True):
    pageid = None
    while pageid is None:
        try:
            WIKI_URL = WIKI_en_URL if en else WIKI_jp_URL
            request_url = WIKI_URL + urllib.parse.urlencode(set_url_random())
            html = urllib.request.urlopen(request_url)
            html_json = json.loads(html.read().decode('utf-8'))
            pageid = (html_json['query']['random'][0])['id']
        except Exception as e:
            print ("get_random_word: Exception Error: ", e, file=sys.stderr)
            time.sleep(10)
    return pageid

def get_word_content(pageid, en=True):
    explaintext = None
    while explaintext is None:
        try:
            WIKI_URL = WIKI_en_URL if en else WIKI_jp_URL
            request_url = WIKI_URL + urllib.parse.urlencode(set_url_extract(pageid))
            html = urllib.request.urlopen(request_url)
            html_json = json.loads(html.read().decode('utf-8'))
            explaintext = html_json['query']['pages'][str(pageid)]['extract']
        except Exception as e:
            print ("get_word_content: Exception Error: ", e, file=sys.stderr)
            time.sleep(10)
    return explaintext

######################################################
# 青空文庫
######################################################

code_list = {}
with open('data/codepoints.csv') as f:
    reader = csv.reader(f)
    for row in reader:
        d1,d2,d3 = row[0].split('-')
        d1 = int(d1)
        d2 = int(d2)
        d3 = int(d3)
        code_list['%d-%02d-%02d'%(d1,d2,d3)] = chr(int(row[1], 16))

def get_aozora_urls():
    aozora_csv_url = 'https://www.aozora.gr.jp/index_pages/list_person_all_extended_utf8.zip'

    xhtml_urls = []
    html = urllib.request.urlopen(aozora_csv_url)
    with zipfile.ZipFile(io.BytesIO(html.read())) as myzip:
        with myzip.open('list_person_all_extended_utf8.csv') as myfile:
            reader = csv.reader(io.TextIOWrapper(myfile))
            idx = -1
            for row in reader:
                if idx < 0:
                    idx = [i for i, x in enumerate(row) if 'URL' in x]
                    idx = [i for i in idx if 'HTML' in row[i]]
                    if len(idx) == 0:
                        exit()
                    idx = idx[0]
                    continue
                if row[idx].startswith('https://www.aozora.gr.jp/cards/'):
                    xhtml_urls.append(row[idx])
    return xhtml_urls

class MyHTMLParser(HTMLParser):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.main = False
        self.count = 0
        self.startpos = (-1,-1)
        self.endpos = (-1,-1)

    def handle_starttag(self, tag, attrs):
        if tag == 'div':
            if self.main:
                self.count += 1
            elif ('class', 'main_text') in attrs:
                self.main = True
                self.startpos = self.getpos()
        
    def handle_endtag(self, tag):
        if tag == 'div':
            if self.main:
                if self.count == 0:
                    self.endpos = self.getpos()
                else:
                    self.count -= 1

def get_aozora_contents(url):
    contents = None
    while contents is None:
        try:
            html = urllib.request.urlopen(url)
            contents = html.read().decode('cp932')
        except Exception as e:
            print ("get_word_content: Exception Error: ", e, file=sys.stderr)
            time.sleep(10)
    parser = MyHTMLParser()
    parser.feed(contents)
    maintext = []
    for lineno, line in enumerate(contents.splitlines()):
        if parser.startpos[0] == lineno + 1:
            maintext.append(line[parser.startpos[1]:])
        elif parser.startpos[0] < lineno + 1 <= parser.endpos[0]:
            if parser.endpos[0] == lineno + 1:
                if parser.endpos[1] == 0:
                    pass
                else:
                    maintext.append(line[:parser.endpos[1]])
            else:
                maintext.append(line)
    maintext = '\n'.join(maintext)
    maintext = re.sub(r'<ruby><rb>(.*?)</rb>.*?</ruby>', r'\1', maintext)
    m = True
    while m:
        m = re.search(r'<img .*?/(\d-\d\d-\d\d)\.png.*?>', maintext)
        if m:
            maintext = maintext[:m.start()] + code_list[m.group(1)] + maintext[m.end():]
    maintext = re.sub(r'<span class="notes">.*?</span>', r'', maintext)
    maintext = re.sub(r'<[^>]*?>', r'', maintext)
    return maintext



########################################################

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

def construct_image(item, random_noise):
    width = 128
    height = 128
    fontsize = 96
    if item is not None:
        min_pixel = int(fontsize * 0.9)
        max_pixel = int(fontsize * 1.1)
        tile_size = random.randint(min_pixel, max_pixel)

        angle = 7.5 * np.random.normal() / 180 * np.pi
        angle = np.clip(angle, -np.pi, np.pi)
        pad_x = np.random.normal() * width * 0.02
        pad_y = np.random.normal() * height * 0.02

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
        img = np.zeros([height, width], dtype=np.float32)
        img[tile_top:tile_top+h,tile_left:tile_left+w] = im[:h,:w]
        im = Image.fromarray(img).rotate(angle / np.pi * 180, resample=Image.BILINEAR, translate=(pad_x, pad_y))
        im = np.array(im) 
    else:
        im = np.zeros([height, width], dtype=np.float32)

    if random_noise is not None and random.uniform(0.,1.) < 0.5:
        im = np.maximum(im, (np.random.random([height, width]) < 0.1) * 255)
    if random_noise is not None and random.uniform(0.,1.) < 0.5:
        x = random.randrange(0, random_noise.shape[1] - 128)
        y = random.randrange(0, random_noise.shape[0] - 128)
        im = np.maximum(im, random_noise[y:y+128,x:x+128])

    return np.asarray(im) / 255.

@tf.function(input_signature=(tf.TensorSpec(shape=[None, 128, 128], dtype=tf.float32),))
def image_add_noise(image):
    shape = tf.shape(image)
    image_batch = shape[0]
    height = shape[1]
    width = shape[2]
    img = image[...,tf.newaxis]

    if tf.random.uniform([], 0., 1.) < 0.5:
        fg_c = tf.random.uniform([], -1., 1.)
        bk_c = tf.random.uniform([], -1., 1.)
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
        fg_c = tf.random.uniform([3], -1., 1.)
        bk_c = tf.random.uniform([3], -1., 1.)
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
    return image

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

class RandomTransformerData(BaseData):
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

        self.max_encoderlen = 32
        self.max_decoderlen = 4 * self.max_encoderlen
        
        self.image_keys = list(self.img_cache.keys())
        self.testkey = self.test_keys()
        self.trainkey = self.train_keys()
        
        self.random_texts = list(self.glyph_id.keys())

        self.test_fontname = list(filter(lambda x: os.path.splitext(os.path.basename(x))[0].startswith('Noto'), jpfont_files))
        self.train_fontname = [x for x in jpfont_files if x not in self.test_fontname]

        self.aozora_urls = get_aozora_urls()

    def test_keys(self):
        def fontname(fontpath):
            return os.path.splitext(os.path.basename(fontpath))[0]

        keys = self.image_keys
        test_keys = list(filter(lambda x: fontname(x[0]).startswith('Noto'), keys))
        result = {}
        for k in test_keys:
            result.setdefault(k[1], [])
            result[k[1]].append(k)
        return result

    def train_keys(self):
        def fontname(fontpath):
            return os.path.splitext(os.path.basename(fontpath))[0]

        keys = self.image_keys
        train_keys = list(filter(lambda x: not fontname(x[0]).startswith('Noto'), keys))
        result = {}
        for k in train_keys:
            result.setdefault(k[1], [])
            result[k[1]].append(k)
        return result

    def get_newtext(self):
        text_cache = ''

        results = []

        for _ in range(5):
            random.shuffle(self.random_texts)
            results.append(''.join(self.random_texts))

        jyosuu = [
                '丁', '両', '人', '位', '体', '俵', '個', '具', '冊', '刀', 
                '切', '刎', '剣', '包', '匹', '区', '卓', '双', '反', '口', 
                '句', '台', '叺', '名', '品', '喉', '基', '壺', '対', '封', 
                '尊', '尾', '局', '巻', '帖', '席', '帳', '幅', '座', '張', 
                '戦', '戸', '手', '把', '拍', '振', '挺', '掛', '斤', '服', 
                '本', '条', '杯', '枚', '果', '枝', '架', '柄', '柱', '株', 
                '棟', '棹', '機', '灯', '片', '献', '玉', '球', '番', '畳', 
                '発', '着', '石', '票', '筋', '篇', '粒', '組', '羽', '脚', 
                '腰', '腹', '膳', '艇', '艘', '菓', '葉', '行', '貫', '貼', 
                '足', '躯', '軒', '輪', '輿', '通', '連', '部', '錠', '門', 
                '隻', '面', '頁', '領', '頭', '顆', '首', '騎', '齣',
                ]
        for _ in range(5000):
            n = random.randint(0, 10000)
            results.append('%d%s'%(n, random.choice(jyosuu)))

        for _ in range(5000):
            k = random.randint(1, 10)
            n = random.randint(0, 10 ** k)
            if random.random() < 0.5:
                results.append('%d円'%n)
            else:
                results.append('{:,}円'.format(n))

        for _ in range(5000):
            dt = datetime.datetime.fromtimestamp(random.randint(0, 10000000000))
            if random.random() < 0.5:
                results.append('%s'%dt)
            else:
                results.append(dt.strftime('%Y年%m月%d日 %H時%M分%S秒'))

        count = 0
        while count < 50000:
            url = random.choice(self.aozora_urls)
            contents = get_aozora_contents(url)
            count += len(contents)
            results += contents.splitlines()
            time.sleep(0.01)
        
        en = True
        count = 0
        while count < 50000:
            pageid = get_random_wordid(en)
            extract = get_word_content(pageid, en)
            count += len(extract)
            results += extract.splitlines()
            time.sleep(0.01)

        en = False
        count = 0
        while count < 50000:
            pageid = get_random_wordid(en)
            extract = get_word_content(pageid, en)
            count += len(extract)
            results += extract.splitlines()
            time.sleep(0.01)

        contents = [s for s in results if s.strip()]
        text_cache = ' '.join(contents)
        return text_cache

    def get_random_text(self, text_cache):
        n = len(text_cache)
        textlen = random.randint(1, min(n, self.max_encoderlen))
        
        k = n - textlen
        if k > 0:
            i = random.randrange(0, k)
        else:
            i = 0
        
        text = text_cache[i:i+textlen]
        return text

    def font_load(self, text, train=False):
        text = set(list(text))
        if train:
            keys = self.trainkey
        else:
            keys = self.testkey
        new_target = set()
        for c in text:
            # ignore white space
            if c in UNICODE_WHITESPACE_CHARACTERS:
                continue
            ks = keys.get(c)
            if ks is not None:
                continue
            else:
                new_target.add(c)
        new_target = list(new_target)
        if len(new_target) == 0:
            return

        #print(new_target, file=sys.stderr)

        font_files = self.train_fontname if train else self.test_fontname
        items = [(f, new_target) for f in font_files]
        with Pool() as pool:
            dicts = pool.imap_unordered(sub_load, items)
            for dictitem in dicts:
                self.img_cache.update(dictitem)

        self.image_keys = list(self.img_cache.keys())
        self.testkey = self.test_keys()
        self.trainkey = self.train_keys()

    def encode_func(self, s):
        decoder_input = tf.io.decode_raw(s, tf.uint8)
        decoder_input = tf.cast(decoder_input, dtype=tf.int32)
        decoder_len = tf.shape(decoder_input)[0]
        if tf.random.uniform([]) < 0.1:
            decoder_true = tf.concat([
                tf.cast([decoder_len], dtype=tf.int32),
                tf.zeros([self.max_decoderlen - 1], dtype=tf.int32)
            ], axis=0)
            decoder_task = tf.concat([
                tf.constant([257], dtype=tf.int32),
                tf.zeros([self.max_decoderlen - 1], dtype=tf.int32)
            ], axis=0)
        else:
            p = tf.random.uniform([decoder_len])
            mask_p = tf.random.uniform([], tf.math.reduce_min(p), 1.0)
            mask_input = tf.where(p < mask_p, 256, decoder_input)
            decoder_true = tf.pad(decoder_input, [[0, self.max_decoderlen - decoder_len]])
            decoder_task = tf.pad(mask_input, [[0, self.max_decoderlen - decoder_len]])
        return decoder_true, decoder_task

    def get_target_text(self, text, train=True):
        target_text = ''
        target_data = []
        if train:
            keys = self.trainkey
        else:
            keys = self.testkey
        for t in text:
            if t in UNICODE_WHITESPACE_CHARACTERS:
                target_data.append(None)
                target_text += t
                continue

            ks = keys.get(t)
            if ks is not None:
                k = random.choice(ks)
                target_data.append(self.img_cache[k])
                target_text += t
            else:
                target_data.append(None)
                target_text += ' '

        return target_text, target_data

    @tf.function(input_signature=(tf.TensorSpec(shape=[], dtype=tf.string), tf.TensorSpec(shape=[None, 256], dtype=tf.float32)))
    def convert_text(self, target_text, embeddeing):
        decoder_true, decoder_task = self.encode_func(target_text)

        s_len = tf.strings.length(target_text,unit="UTF8_CHAR")
        encoder_inputs = tf.cast(tf.concat([
            embeddeing[:s_len,...],
            tf.zeros([self.max_encoderlen - s_len, 256]),
        ], axis=0), tf.float32)
        return decoder_true, decoder_task, encoder_inputs

