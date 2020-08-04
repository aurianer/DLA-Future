#!/usr/bin/env python3

import numpy as np

np.set_printoptions(precision=2, suppress=True, linewidth=200)

A = np.array([-0.680413, 0.98429, 0.194989, 0.807373, 0.390099, -0.405084, 0.579434, 0.693761, 0.0973265, -0.600378, -0.565286, 0.973016, 0.712034, 0.890715, -0.622667, 0.98429, -0.920862, 0.0845699, 0.588186, -0.174282, 0.456824, 0.652098, 0.642126, 0.486423, 0.0271842, 0.563069, -0.982732, 0.471538, -0.372389, -0.98384, 0.194989, 0.0845699, -0.88568, -0.630542, 0.639491, 0.82364, -0.905684, 0.0238092, 0.146788, 0.98097, 0.404143, -0.805374, -0.71633, -0.0240075, 0.115704, 0.807373, 0.588186, -0.630542, -0.0700127, 0.371002, -0.902682, -0.648251, 0.297784, 0.0441707, -0.484276, -0.0899823, 0.113947, 0.618568, -0.120186, -0.338333, 0.390099, -0.174282, 0.639491, 0.371002, 0.396426, -0.620341, -0.64614, -0.336987, -0.826764, -0.998609, 0.477812, 0.563477, -0.85635, -0.360855, -0.490775, -0.405084, 0.456824, 0.82364, -0.902682, -0.620341, 0.206051, 0.1245, 0.405521, -0.981253, -0.709126, 0.957982, -0.81073, -0.132733, 0.00945453, 0.210769, 0.579434, 0.652098, -0.905684, -0.648251, -0.64614, 0.1245, -0.287324, -0.232913, 0.211601, -0.551944, 0.410279, 0.106039, 0.569973, 0.0346615, -0.734196, 0.693761, 0.642126, 0.0238092, 0.297784, -0.336987, 0.405521, -0.232913, 0.537454, 0.543235, -0.228147, 0.0380699, 0.154316, 0.460751, 0.57539, 0.119612, 0.0973265, 0.486423, 0.146788, 0.0441707, -0.826764, -0.981253, 0.211601, 0.543235, 0.34891, 0.404378, 0.445204, 0.554902, -0.100663, 0.432267, -0.767614, -0.600378, 0.0271842, 0.98097, -0.484276, -0.998609, -0.709126, -0.551944, -0.228147, 0.404378, -0.748283, 0.313134, -0.618927, -0.540507, -0.0413885, 0.977473, -0.565286, 0.563069, 0.404143, -0.0899823, 0.477812, 0.957982, 0.410279, 0.0380699, 0.445204, 0.313134, 0.0997272, 0.941728, 0.581627, 0.764039, 0.969975, 0.973016, -0.982732, -0.805374, 0.113947, 0.563477, -0.81073, 0.106039, 0.154316, 0.554902, -0.618927, 0.941728, -0.035361, 0.695076, 0.468352, -0.879306, 0.712034, 0.471538, -0.71633, 0.618568, -0.85635, -0.132733, 0.569973, 0.460751, -0.100663, -0.540507, 0.581627, 0.695076, -0.940902, 0.262572, 0.981353, 0.890715, -0.372389, -0.0240075, -0.120186, -0.360855, 0.00945453, 0.0346615, 0.57539, 0.432267, -0.0413885, 0.764039, 0.468352, 0.262572, -0.444989, -0.75755, -0.622667, -0.98384, 0.115704, -0.338333, -0.490775, 0.210769, -0.734196, 0.119612, -0.767614, 0.977473, 0.969975, -0.879306, 0.981353, -0.75755, -0.785965, ]).reshape(15, 15).T

Ts = [
        np.array([1.35718, 0, 0, -0.4262, 1.13243, 0, -0.0134119, 0.855456, 1.46529, ]).reshape(3, 3).T,
        np.array([1.41802, 0, 0, 0.266689, 1.20166, 0, 0.410361, -0.662587, 1.75502, ]).reshape(3, 3).T,
        np.array([1.06313, 0, 0, -0.0213741, 1.3702, 0, -0.180712, 0.0511637, 1.17342, ]).reshape(3, 3).T,
        np.array([1.55996, 0, 0, -0.354257, 1.22437, 0, 0, 0, 0, ]).reshape(3, 3).T,

]

Z = np.array([-0.680413, 0.98429, 0.194989, -2.26042, 0.12716, -0.132044, 0.188877, 0.226144, 0.0317253, -0.195704, -0.184265, 0.317172, 0.2321, 0.290344, -0.202969, 0.98429, -0.920862, 0.0845699, -0.185077, 2.05851, -0.239769, -0.217083, -0.200443, -0.198141, -0.076579, -0.302668, 0.526782, -0.125289, 0.256058, 0.354719, 0.194989, 0.0845699, -0.88568, 1.45627, 0.127553, -1.5785, -0.148207, 0.281688, 0.158674, 0.273286, 0.110196, -0.239071, -0.0581917, 0.165525, -0.252307, 0.807373, 0.588186, -0.630542, 1.57188, -0.352811, -0.276397, -1.61069, -0.107938, -0.280155, 0.162257, -0.383761, -0.130592, 0.17607, 0.042764, 0.31112, 0.390099, -0.174282, 0.639491, 0.371002, -0.97335, 0.284755, -0.468244, -1.57244, 0.274499, -0.320636, -0.0556038, 0.326843, -0.292399, 0.294195, 0.451925, -0.405084, 0.456824, 0.82364, -0.902682, -0.620341, -0.397804, -0.996416, 0.351124, -0.996537, 0.125491, -0.0757974, 0.292025, 0.171775, 0.0398769, 0.041495, 0.579434, 0.652098, -0.905684, -0.648251, -0.64614, 0.1245, -0.452353, -0.0282079, -1.07513, -1.12311, 0.52458, -0.242653, -0.637242, -0.346778, 0.144357, 0.693761, 0.642126, 0.0238092, 0.297784, -0.336987, 0.405521, -0.232913, -0.334455, 0.709755, -1.4598, 1.20834, 0.325409, 0.462509, 0.299708, -0.223633, 0.0973265, 0.486423, 0.146788, 0.0441707, -0.826764, -0.981253, 0.211601, 0.543235, -0.41314, 0.141987, 0.824372, -1.09656, -0.117354, -0.760075, 0.336059, -0.600378, 0.0271842, 0.98097, -0.484276, -0.998609, -0.709126, -0.551944, -0.228147, 0.404378, -0.990869, -0.142869, 0.0452626, -0.372235, 0.431713, -0.309368, -0.565286, 0.563069, 0.404143, -0.0899823, 0.477812, 0.957982, 0.410279, 0.0380699, 0.445204, 0.313134, 0.30717, 0.569999, -0.669289, 0.55448, 0.795927, 0.973016, -0.982732, -0.805374, 0.113947, 0.563477, -0.81073, 0.106039, 0.154316, 0.554902, -0.618927, 0.941728, 0.639746, -0.690549, 0.16982, 0.364833, 0.712034, 0.471538, -0.71633, 0.618568, -0.85635, -0.132733, 0.569973, 0.460751, -0.100663, -0.540507, 0.581627, 0.695076, -1.02209, 0.24242, -0.819112, 0.890715, -0.372389, -0.0240075, -0.120186, -0.360855, 0.00945453, 0.0346615, 0.57539, 0.432267, -0.0413885, 0.764039, 0.468352, 0.262572, 0.438578, -0.735544, -0.622667, -0.98384, 0.115704, -0.338333, -0.490775, 0.210769, -0.734196, 0.119612, -0.767614, 0.977473, 0.969975, -0.879306, 0.981353, -0.75755, -0.0975829, ]).reshape(15, 15).T

n = A.shape[0]
nb = Ts[0].shape[0]

print('A\n', A)

H = np.zeros(A.shape)

for index, j_r in enumerate(range(nb, n, nb)):
    #if index > 0:
    #    break

    print("new panel", j_r)

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

    print('ts', ts)
    T = Ts[index]
    print('T', T)
    for t in range(T.shape[0]): assert(T[t,t] - ts[t] < 1e-3)

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
    A[j_r:j_r+nb,j_r-nb:j_r] = R
    A[j_r+nb:,j_r-nb:j_r] = 0

    print('A\n', np.tril(A))

print('H\n', H)
print('Z\n', np.tril(Z))
print("diff Z - A\n", np.tril(Z - A))
