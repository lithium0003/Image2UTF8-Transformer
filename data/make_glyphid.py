#!/usr/bin/env python3

import glob, os
import itertools
import operator
import csv

keys = [chr(ord('0') + i) for i in range(10)]
keys += [chr(ord('A') + i) for i in range(26)]
keys += [chr(ord('a') + i) for i in range(26)]
keys += [chr(i) for i in range(ord('ぁ'), ord('ゖ')+1)]
keys += [chr(i) for i in range(ord('ァ'), ord('ヺ')+1)]

with open('kanji_list.txt','r') as f:
    lines = f.read().splitlines()

for line in lines:
    keys += list(line.strip())

with open('other_list.txt','r') as f:
    lines = f.read().splitlines()

for line in lines:
    keys += list(line.strip())

with open('2nd_kanji.txt','r') as f:
    lines = f.read().splitlines()

for line in lines:
    keys += list(line.strip())

with open('3rd_kanji.txt','r') as f:
    lines = f.read().splitlines()

for line in lines:
    keys += list(line.strip())

with open('4th_kanji.txt','r') as f:
    lines = f.read().splitlines()

for line in lines:
    keys += list(line.strip())

num_targets = [chr(ord('0') + i) for i in range(10)]
upper_targets = [chr(ord('A') + i) for i in range(26)]
lower_targets = [chr(ord('a') + i) for i in range(26)]
hira_targets = [chr(i) for i in range(ord('ぁ'), ord('ゖ')+1)]
kata_targets = [chr(i) for i in range(ord('ァ'), ord('ヺ')+1)]

kanji_targets = []
with open('kanji_list.txt','r') as f:
    lines = f.read().splitlines()

for line in lines:
    kanji_targets += list(line.strip())

with open('other_list.txt','r') as f:
    lines = f.read().splitlines()

en_other_targets = list(lines[0].strip())

kanji2_targets = []
with open('2nd_kanji.txt','r') as f:
    lines = f.read().splitlines()

for line in lines:
    kanji2_targets += list(line.strip())

kanji3_targets = []
with open('3rd_kanji.txt','r') as f:
    lines = f.read().splitlines()

for line in lines:
    kanji3_targets += list(line.strip())

kanji4_targets = []
with open('4th_kanji.txt','r') as f:
    lines = f.read().splitlines()

for line in lines:
    kanji4_targets += list(line.strip())

key_map = {}
for k in keys:
    if k in num_targets:
        key_map[k] = 0
    elif k in upper_targets:
        key_map[k] = 1
    elif k in lower_targets:
        key_map[k] = 2
    elif k in hira_targets:
        key_map[k] = 3
    elif k in kata_targets:
        key_map[k] = 4
    elif k in kanji_targets:
        key_map[k] = 5
    elif k in kanji2_targets:
        key_map[k] = 8
    elif k in kanji3_targets:
        key_map[k] = 9
    elif k in kanji4_targets:
        key_map[k] = 10
    elif k in en_other_targets:
        key_map[k] = 6
    else:
        key_map[k] = 7

with open('id_map.csv','w') as f:
    writer = csv.writer(f)
    for i,k in enumerate(keys):
        writer.writerow([i,k,key_map[k]])
