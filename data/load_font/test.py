import sys
import subprocess
import numpy as np
import matplotlib.pyplot as plt

def process(font, size, chars):
    proc = subprocess.Popen([
        './load_font',
        font,
        size,
        ], stdin=subprocess.PIPE, stdout=subprocess.PIPE)

    for c in chars:
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

        print('rows:', rows,
                'width:', width,
                'left:', left,
                'top:', top,
                'advx:', advance_x / 64,
                'advy:', advance_y / 64)

        buffer = proc.stdout.read(rows*width)
        img = np.frombuffer(buffer, dtype='ubyte').reshape(rows,width)
        plt.imshow(img, cmap='gray')
        plt.show()

if __name__ == '__main__':
    process(*sys.argv[1:])
