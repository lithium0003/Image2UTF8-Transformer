#!/usr/bin/env python3

import numpy as np
import subprocess
import csv
import glob
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def sub_load(font, key):
    proc = subprocess.run([
        'load_font/load_font',
        font,
        '256',
        str(int.from_bytes(key.encode("utf-32-le"), 'little'))
        ], stdout=subprocess.PIPE)

    result = proc.stdout
    if len(result) == 0:
        return None

    rows = int.from_bytes(result[:8], 'little')
    width = int.from_bytes(result[8:16], 'little')
    left = int.from_bytes(result[16:24], 'little', signed=True)
    top = int.from_bytes(result[24:32], 'little', signed=True)
    advance_x = int.from_bytes(result[32:40], 'little', signed=True)
    advance_y = int.from_bytes(result[40:48], 'little', signed=True)

    if rows == 0 or width == 0:
        return None

    img = np.frombuffer(result[48:], dtype='ubyte').reshape(rows,width)
    return {
            'rows': rows, 
            'width': width, 
            'left': left,
            'top': top,
            'advx': advance_x / 64,
            'image': img
            }


def process():
    glyph_id = {}
    glyphs = {}
    glyph_type = {}

    with open('id_map.csv','r') as f:
        reader = csv.reader(f)
        for row in reader:
            glyph_id[row[1]] = int(row[0])
            glyphs[int(row[0])] = row[1]
            glyph_type[int(row[0])] = int(row[2])

    enfont_files = sorted(glob.glob('enfont/*.ttf') + glob.glob('enfont/*.otf'))
    en_glyphs = [glyphs[key] for key in glyphs.keys() if glyph_type[key] in [0,1,2,6]]
    
    x_len = int(len(en_glyphs) ** 0.5) + 1
    y_len = len(en_glyphs) // x_len + 1

    os.makedirs('test', exist_ok=True)

    for j,font in enumerate(enfont_files):
        print('%d/%d'%(j,len(enfont_files)),font)
        if os.path.exists(os.path.join('test',os.path.splitext(os.path.basename(font))[0]+'.png')):
            continue
        plt.figure(facecolor='black')
        for i,g in enumerate(en_glyphs):
            item = sub_load(font, g)
            if item is None:
                continue
            plt.subplot(x_len,y_len,i+1)
            plt.imshow(item['image'], cmap='gray')
            plt.axis('off')
        plt.savefig(os.path.join('test',os.path.splitext(os.path.basename(font))[0]+'.png'), facecolor='black')
        plt.close()

if __name__ == '__main__':
    process()
