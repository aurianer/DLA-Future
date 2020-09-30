#!/usr/bin/env python3

import os
import re
import numpy as np

np.set_printoptions(precision=7, suppress=True, linewidth=200)

A = np.load('matA.npy')
Z = np.load('Z.npy')

Ts = {}
for entry in os.listdir('.'):
    m = re.match('T(\d+)\.npy', entry)
    if not m:
        continue

    Ts[int(m.group(1))] = np.load(entry)

n = A.shape[0]
nb = Ts[0].shape[0]

print('Ts', Ts)
print('A\n', A)

H = np.zeros(A.shape)

for index, j_r in enumerate(range(nb, n, nb)):
    print("new panel", int(j_r/nb)-1)

    Ai = A[j_r:,j_r-nb:j_r]
    At = A[j_r:,j_r:]

    print('A\n', A)
    print('Ai\n', Ai)
    print('At\n', At)

    Q,R = np.linalg.qr(Ai)
    V,ts = np.linalg.qr(Ai, mode='raw')

    V = V.T
    V = np.tril(V)

    for t in range(V.shape[1]): V[t, t] = 1
    print('V\n', V)

    H[j_r:,j_r-nb:j_r] = V

    # update Ai
    A[j_r:j_r+nb,j_r-nb:j_r] = R

    # COMPUTE T-factor
    print('ts', ts)
    T = Ts[index]
    print('T', T)
    try:
        for t in range(T.shape[0]): assert(T[t,t] - ts[t] < 1e-3)
    except Exception as e:
        print(e)
        print(ts)
        print(T)
        exit(1)

    # UPDATE TRAILING MATRIX
    W = V @ T
    print('W\n', W)

    X = At @ V @ T
    print('X\n', X)

    W2 = W.T @ X
    print('W2\n', W2)

    X = X - .5 * V @ W2
    print('X\n', X)

    At = At - X @ V.T - V @ X.T

    # in-place computation
    A[j_r:,j_r:] = At
    A[j_r+nb:,j_r-nb:j_r] = 0

    print('A\n', np.tril(A))

print('H\n', H)
print('Z\n', np.tril(Z))

H_computed = np.tril(Z - A)
print("diff Z - A\n", H_computed)
H_final = H - H_computed
print("diff Hs\n", H_final)
print("diff Hs tril\n", np.linalg.norm(np.tril(H_final[nb:,:], -1)))
