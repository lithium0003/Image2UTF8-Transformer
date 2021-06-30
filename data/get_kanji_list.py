import urllib.request
import re

def get_kanjis(url):
    html = urllib.request.urlopen(url).read().decode('utf-8')
    charcodes = re.findall(r'&#x([0-9A-F]+);', html)
    chars = [bytes.decode(int(c, 16).to_bytes(4, 'little'), 'utf-32') for c in charcodes]
    return chars

if __name__ == '__main__':
    kanji1 = get_kanjis('http://www13.plala.or.jp/bigdata/jis_1.html')
    kanji2 = get_kanjis('http://www13.plala.or.jp/bigdata/jis_2.html')
    kanji3 = get_kanjis('http://www13.plala.or.jp/bigdata/jis_3.html')
    kanji4 = get_kanjis('http://www13.plala.or.jp/bigdata/jis_4.html')

    with open('1st_kanji.txt', 'w') as f:
        f.write(''.join(kanji1) + '\n')

    with open('2nd_kanji.txt', 'w') as f:
        f.write(''.join(kanji2) + '\n')

    with open('3rd_kanji.txt', 'w') as f:
        f.write(''.join(kanji3) + '\n')

    with open('4th_kanji.txt', 'w') as f:
        f.write(''.join(kanji4) + '\n')
