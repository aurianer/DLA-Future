#!/usr/bin/env python3

import os
import re
import numpy as np

np.set_printoptions(precision=3, suppress=True, linewidth=200)

A = np.load('build/matA.npy')
Z = np.load('build/Z.npy')

Ts = []
for entry in os.listdir('build'):
    m = re.match('T\d+\.npy', entry)
    if not m:
        continue

    Ts.append(np.load(os.path.join('build', entry)))

n = A.shape[0]
nb = Ts[0].shape[0]

def getstep(j, A=A):
    j_r = j * nb
    Ai = A[j_r:,j_r-nb:j_r]
    At = A[j_r:,j_r:]

    return (Ai, At)
