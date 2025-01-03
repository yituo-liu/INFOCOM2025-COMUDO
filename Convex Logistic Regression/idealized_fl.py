import torch
import torchvision
from torch.utils.data import DataLoader, Subset
import numpy as np
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import matplotlib.pyplot as plt
import random

def Idealized_FL(N, T, D, d, C, Data, Label, Testdata, Testlabels):
    alpha = 0.01
    xtc = {t: {c: np.zeros(d) for c in range(0, C)} for t in range(0, T)}
    for c in range(10):
        xtc[0][c] = np.ones(d) * 1e-7
    xtnc = {t: {n: {c: np.zeros(d) for c in range(0, C)} for n in range(N)} for t in range(0, T)}

    for t in range(0, T - 1):
        GD_tnc_f = np.zeros((d * C, N))
        for n in range(N):
            for i in range(D):
                d_tni = Data[t][n][i]
                b_tni = Label[t][n][i]
                p_tni = np.zeros(10)
                for k in range(10):
                    p_tni[k] = np.exp(d_tni @ xtc[t][k])
                hsum_tn = np.sum(p_tni)

                for c in range(0, C):
                    idx = slice(d * c, d * (c + 1))
                    GD_tnc_f[idx, n] -= (1 / D) * ((b_tni == c) - p_tni[c] / hsum_tn) * d_tni

        for c in range(0, C):
            xtc[t + 1][c] = np.zeros(d)

        for c in range(0, C):
            for n in range(N):
                idx = slice(d * c, d * (c + 1))

                xtnc[t + 1][n][c] = xtc[t][c] - alpha * GD_tnc_f[idx, n]
                xtc[t + 1][c] += xtnc[t + 1][n][c]

            xtc[t + 1][c] = xtc[t + 1][c] / N

        print(f'Idealized_FL, t:{t + 1}')

    # Test Accuracy
    Accuracy = []
    for t in range(T):
        wrong = 0
        for i in range(1000):
            d_i = Testdata[:, i]
            h_ti = np.zeros(10)
            for k in range(10):
                h_ti[k] = np.exp(d_i @ xtc[t][k])
            hsum_ti = np.sum(h_ti)
            TorF = (h_ti / hsum_ti).argmax()
            if Testlabels[i] != TorF:
                wrong += 1
        At = (1 - wrong / 1000) * 100
        if t == 0:
            Accuracy.append(At)
        else:
            Accuracy.append((Accuracy[t - 1] * t + At) / (t + 1))
        print(f'Accuracy: {Accuracy[t]}, t:{t + 1}')
    return Accuracy