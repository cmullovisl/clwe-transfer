#!/usr/bin/env python
import sys
import fasttext
import numpy as np

FASTTEXT = '/home/cmullov/repositories/fastText.git/fasttext'

model_path  = sys.argv[1]
matrix_path = sys.argv[2]

FTM = fasttext.load_model(model_path)

with open(matrix_path, 'r') as f:
    next(f)
    R = [[float(x) for x in line.split()] for line in f]
R = np.array(R)

words = sys.stdin.read().strip().split('\n')
x_full = [FTM[w] for w in words]
x_full = np.array(x_full)
# x_full = FTM['Wort'].reshape(1, -1)
x = np.dot(x_full, R.T)
x /= np.linalg.norm(x, axis=1)[:, np.newaxis] + 1e-8

for w, vec in zip(words, x):
    print(w, ' '.join(['{:.4f}'.format(f) for f in vec]))
