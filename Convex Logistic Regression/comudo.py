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

def COMUDO(N, T, D, d, C, Data, Label, H, Z, Testdata, Testlabels, par):
    alpha = 0.01
    Gamma = 1.2e-2
    Lambda = 2e-6
    eta = 1e-3
    V = 20
    Pnc_bar = Func_dBm2W(16) / C * par['slot']

    xtc = {t: {c: np.zeros(d) for c in range(0, C)} for t in range(0, T)}
    for c in range(10):
        xtc[0][c] = np.ones(d) * 1e-7
    xtnc = {t: {n: {c: np.zeros(d) for c in range(0, C)} for n in range(N)} for t in range(0, T)}
    Qtnc = {t: np.ones((N, C)) * V for t in range(0, T)}
    gtnc = {t: np.zeros((N, C)) for t in range(0, T)}
    gtnc2 = {t: np.zeros((N, C)) for t in range(0, T)}

    for t in range(0, T-1):
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
                    idx = slice(d * c, d * (c+1))
                    GD_tnc_f[idx, n] -= (1 / D) * ((b_tni == c) - p_tni[c] / hsum_tn) * d_tni

        GD_tnc_g = np.zeros((d * C, N))
        for n in range(N):
            for c in range(0, C):
                idx = slice(d * c, d * (c+1))
                H_tnc = H[t][c][n, :]
                e_tnc = (1 / (np.abs(H_tnc)**2))
                GD_tnc_g[idx, n] = 2 * Lambda**2 * e_tnc

        for c in range(0, C):
            xtc[t + 1][c] = np.zeros(d)

        for c in range(0, C):
            for n in range(N):
                idx = slice(d * c, d * (c+1))
                H_tnc = H[t][c][n, :]
                e_tnc = (1 / (np.abs(H_tnc)**2))
                H_tnc2 = H[t + 1][c][n, :]
                e_tnc2 = (1 / (np.abs(H_tnc2)**2))
                gtnc[t][n, c] = (Lambda**2) * np.linalg.norm(alpha * np.sqrt(e_tnc) * GD_tnc_f[idx, n])**2 - Pnc_bar
                if gtnc[t][n, c] <= 0:
                    gtnc[t][n, c] = 0

                if gtnc[t][n, c] <= 0:
                    A_tnc = np.ones(d)
                    B_tnc = xtc[t][c] - alpha * GD_tnc_f[idx, n]
                else:
                    A_tnc = np.ones(d) + alpha * Gamma * Qtnc[t][n, c] * GD_tnc_g[idx, n]
                    B_tnc = (np.ones(d) + alpha * Gamma * Qtnc[t][n, c] * GD_tnc_g[idx, n]) * xtc[t][c] - alpha * GD_tnc_f[idx, n]

                xtnc[t + 1][n][c] = (1 / A_tnc) * B_tnc
                xtc[t + 1][c] += xtnc[t + 1][n][c]

                gtnc2[t + 1][n, c] = (Lambda**2) * np.linalg.norm(np.sqrt(e_tnc2) * (xtnc[t + 1][n][c] - xtc[t][c]))**2 - Pnc_bar
                if gtnc2[t + 1][n, c] <= 0:
                    gtnc2[t + 1][n, c] = 0

                Qtnc[t + 1][n, c] = max(V, (1 - eta) * Qtnc[t][n, c] + Gamma * gtnc2[t + 1][n, c])

            if par['add_noise'] == 1:
                xtc[t + 1][c] = xtc[t + 1][c] / N + (1 / Lambda) * Z[t + 1][c] / N
        print(f'COMUDO, t:{t+1}')

    # Test Accuracy
    Accuracy = []
    for t in range(T):
        At = 0
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
            Accuracy.append((Accuracy[t-1] * t + At) / (t + 1))
        print(f'Accuracy: {Accuracy[t]}, t:{t + 1}')
    # Power Constraint
    Power_Separate = np.zeros((500, 10, 10))
    TX_Pt = 0
    for n in range(10):
        for c in range(10):
            H_tnc = H[0][c][n, :]
            e_tnc = (1 / (np.abs(H_tnc) ** 2))
            Power_Separate[0][n][c] = np.linalg.norm(Lambda * np.sqrt(e_tnc) * xtc[0][c]) ** 2
            Power_Constraint = np.linalg.norm(Lambda * np.sqrt(e_tnc) * xtc[0][c]) ** 2 - 10**(-1.4)/C*par['slot']
            if Power_Constraint < 0:
                Power_Constraint = 0
            TX_Pt += Power_Constraint
    Pow1 = TX_Pt / (N * par['slot']) / 10**(-1.4)
    Power = []
    Power.append(Pow1)
    Power_dBm = []
    Power_dBm.append(Func_W2dBm(Power[0]*1e-3))
    for t in range(T-1):
        TX_Pt = 0
        for n in range(10):
            for c in range(10):
                H_tnc = H[t+1][c][n, :]
                e_tnc = (1 / (np.abs(H_tnc) ** 2))
                Power_Separate[t+1][n][c] = np.linalg.norm(Lambda * np.sqrt(e_tnc) * (xtnc[t+1][n][c] - xtc[t][c])) ** 2
                Power_Constraint = np.linalg.norm(Lambda * np.sqrt(e_tnc) * (xtnc[t+1][n][c] - xtc[t][c])) ** 2 - 10**(-1.4)/C*par['slot']
                if Power_Constraint < 0:
                    Power_Constraint = 0
                TX_Pt += Power_Constraint
        Power.append((Power[t] * (t + 1) + TX_Pt / (N * par['slot']) / 10**(-1.4)) / (t + 2))
        Power_dBm.append(Func_W2dBm(Power[t+1]*1e-3))
        print('Power: {}, t: {}'.format(Power_dBm[t+1], t + 2))
    return Accuracy, Power_dBm