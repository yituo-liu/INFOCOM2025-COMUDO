import torch
import torchvision
from torch.utils.data import DataLoader, Subset
import numpy as np
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import matplotlib.pyplot as plt
import random

def Func_dBm2W(dBm):
    return 10**((dBm - 30) / 10)

def Func_W2dBm(W):
    return 10 * np.log10(W) + 30

def OTARCI(N, T, D, d, C, Data, Label, H, Z, Testdata, Testlabels, par):
    alpha = 0.01
    gamma = 7.4e-7
    Lambda = {t: {c: np.zeros((10, 784)) for c in range(10)} for t in range(par['T'])}
    eta = 3e-17
    U = 5e8
    Pnc_bar = Func_dBm2W(16) / 1000

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
                H_tnc = np.abs(H[t + 1][c][n, :])
                e_tnc = (gamma * H_tnc / ((H_tnc ** 2) + eta))
                Lambda[t + 1][c][n, :] = np.minimum(e_tnc, Pnc_bar * U)
                xtnc[t + 1][n][c] = xtc[t][c] - alpha * GD_tnc_f[idx, n]
                xtc[t + 1][c] += H_tnc * Lambda[t + 1][c][n, :] * xtnc[t + 1][n][c]

        for c in range(0, C):
            if par['add_noise'] == 1:
                xtc[t + 1][c] = xtc[t + 1][c] / N / gamma + Z[t + 1][c] / N / gamma
        print(f'OTA-RCI, t:{t + 1}')

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

    # Power Constraint
    Power_Separate = np.zeros((500, 10, 10))
    TX_Pt = 0
    for n in range(10):
        for c in range(10):
            H_tnc = H[0][c][n, :]
            e_tnc = (gamma * H_tnc / (np.abs(H_tnc) ** 2) + eta)
            Lambda[0][c][n, :] = np.minimum(e_tnc, Pnc_bar * U)
            Power_Separate[0][n][c] = np.linalg.norm(Lambda[0][c][n, :] * xtc[0][c]) ** 2
            Power_Constraint = np.linalg.norm(Lambda[0][c][n, :] * xtc[0][c]) ** 2 - 10 ** (-1.4) / C * par['slot']
            if Power_Constraint < 0:
                Power_Constraint = 0
            TX_Pt += Power_Constraint
    Pow1 = TX_Pt / (N * par['slot']) / 10 ** (-1.4) * 1e-3
    Power = []
    Power.append(Pow1)
    Power_dBm = []
    Power_dBm.append(Func_W2dBm(Power[0]))
    for t in range(T - 1):
        TX_Pt = 0
        for n in range(10):
            for c in range(10):
                Power_Separate[t + 1][n][c] = np.linalg.norm(
                    Lambda[t + 1][c][n, :] * (xtnc[t + 1][n][c] - xtc[t][c])) ** 2
                Power_Constraint = np.linalg.norm(
                    Lambda[t + 1][c][n, :] * (xtnc[t + 1][n][c] - xtc[t][c])) ** 2 - 10 ** (-1.4) / C * par['slot']
                if Power_Constraint < 0:
                    Power_Constraint = 0
                TX_Pt += Power_Constraint
        Power.append((Power[t] * (t + 1) + TX_Pt / (N * par['slot']) / 10 ** (-1.4) * 1e-3) / (t + 2))
        Power_dBm.append(Func_W2dBm(Power[t + 1]))
        print('Power: {}, t: {}'.format(Power_dBm[t + 1], t + 2))
    return Accuracy, Power_dBm