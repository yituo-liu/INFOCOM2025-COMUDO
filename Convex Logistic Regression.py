import torch
import torchvision
from torch.utils.data import DataLoader, Subset
import numpy as np
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import matplotlib.pyplot as plt
import random

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

# Random Seed
set_seed(42)


# Function
def Func_dBm2W(dBm):
    return 10**((dBm - 30) / 10)

def Func_W2dBm(W):
    return 10 * np.log10(W) + 30

def normalize(vector):
    return vector / np.linalg.norm(vector)

# System Parameters
sys = {
    'N': 10,  # number of nodes
    'T': 500,  # time horizon
    'D_min': 20,  # minimum data samples
    'D_max': 20,  # maximum data samples
    'D_vary': 1,  # batch data
    'd': 784,  # data dimension
    'C': 10,  # number of classes
    'D': 20   # batch size
}

# Systems Generation
Case = {
    'nonIID_Gen': 0,  # 1: generate non IID data
    'Test_Gen': 0,  # generate test data
    'Test_SampleSize': 1000,  # test data size
    'MAC_Gen': 0  # generate MAC
}

# Communication Parameters
com = {
    's': sys['d'] * sys['C'],  # subchannels
    'BW': 15 * (10**3),  # 15 k for each subchannel
    'N0': -174,  # noise spectral density
    'NF': 10,  # noise figure
    'psi': 8,  # shadowing
    'dis': 500,  # distance of user to BS
    'Halpha': 0.997,  # Gauss Markov constant
    'gamma': 37 # path loss coefficient
}

com['noise_dBm'] = com['N0'] + 10 * np.log10(com['BW']) + com['NF']  # noise (dBm) per subchannel 15 kHz: -122 dBm
com['noise'] = Func_dBm2W(com['noise_dBm'])  # noise (W) per subchannel

# Simulation parameters
par = {
    'noniid': 1,  # non iid data
    'dis': 500,  # user distance
    'Pnbar_dBm': 16,  # average power limit
    'Halpha': com['Halpha'],  # channel correlation
    'add_noise': 1,  # add noise to global model
    'T': sys['T'],  # run time
    'channel': 1000,  # number of sub channels
    'slot': sys['d'] * sys['C'] / 1000,  # number of time slots
}

# DataSet
dataset_tranform = torchvision.transforms.Compose([torchvision.transforms.ToTensor(), torchvision.transforms.Normalize((0.1307,), (0.3081,))])

train_set = torchvision.datasets.MNIST(root="./dataset", train=True, transform=dataset_tranform, download=True)
test_set = torchvision.datasets.MNIST(root="./dataset", train=False, transform=dataset_tranform, download=True)

# Divided into 10 classes
class_indices = {i: [] for i in range(10)}
for idx, (image, label) in enumerate(train_set):
    class_indices[label].append(idx)

device_data_loaders = []
length_sample = []
for i in range(10):
    indices = class_indices[i]
    data_loader = Subset(train_set, indices)
    length_sample.append(len(data_loader))
    device_data_loaders.append(data_loader)

Data = np.zeros((500, 10, 20, 784))
Label = np.zeros((500, 10, 20))

for t in range(0, sys['T']):
    for n in range(sys['N']):
        Sample = np.random.choice(length_sample[n], 20, replace=False)
        for idx in Sample:
            img, tar = device_data_loaders[n][idx]
            Data[t][n] = img.view(-1).numpy()
            Label[t][n] = tar

# TestSample
test_samples = []
test_labels = []
for i in range(1000):
    img, label = test_set[i]
    test_samples.append(img.view(-1).numpy())
    test_labels.append(label)

Testdata = np.array(test_samples).T
Testlabels = np.array(test_labels)

# Channel Generate
def compute_epsilon(distance, gamma, psi):
    db_values = -31.54 - gamma * np.log10(distance) - np.sqrt(psi) * np.random.randn()
    epsilon_com = 10 ** (db_values / 10)
    return epsilon_com

com['PL'] = [compute_epsilon(com['dis'], com['gamma'], com['psi']) for _ in range(10)]

H = np.zeros((500, 10, 10, 784), dtype=complex)
Z = np.zeros((500, 10, 784))
for c in range(0, 10):
    H[0][c] = np.sqrt(np.diag(com['PL']) / 2) @ (np.random.randn(10, 784) + 1j * np.random.randn(10, 784))
    Z[0][c] = np.sqrt(com['noise']) * np.random.randn(784)

for t in range(1, sys['T']):
    for c in range(0, 10):
        Hz = np.sqrt((1 - com['Halpha']**2) * np.diag(com['PL']) / 2) @ (np.random.randn(10, 784) + 1j * np.random.randn(10, 784))
        H[t][c] = 0.997 * H[t - 1][c] + Hz
        Z[t][c] = np.sqrt(com['noise']) * np.random.randn(784)

N = sys['N']
T = par['T']
D = sys['D']
d = sys['d']
C = sys['C']

def Idealized_FL():
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

            if par['add_noise'] == 1:
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

def COMUDO():
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
            # TX_Pt += np.linalg.norm(Lambda * np.sqrt(e_tnc) * xtc[0][c]) ** 2
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

def OTALPC():
    alpha = 0.01
    Lambda = np.zeros(500)

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

        Pt = 0
        for n in range(10):
            for c in range(10):
                H_tnc = H[t + 1][c][n, :]
                e_tnc = (1 / (np.abs(H_tnc) ** 2))
                Pt += np.linalg.norm(np.sqrt(e_tnc) * (xtnc[t + 1][n][c] - xtc[t][c])) ** 2
        Lambda[t + 1] = np.sqrt(Func_dBm2W(16) * N * par['slot'] / Pt)

        for c in range(0, C):
            if par['add_noise'] == 1:
                xtc[t + 1][c] = xtc[t + 1][c] / N + (1 / Lambda[t + 1]) * Z[t + 1][c] / N
        print(f'OTA-LPC, t:{t + 1}')

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
            e_tnc = (1 / (np.abs(H_tnc) ** 2))
            Power_Separate[0][n][c] = np.linalg.norm(5.7e-7 * np.sqrt(e_tnc) * xtc[0][c]) ** 2
            Power_Constraint = np.linalg.norm(5.7e-7 * np.sqrt(e_tnc) * xtc[0][c]) ** 2 - 10 ** (-1.4) / C * par['slot']
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
                H_tnc = H[t + 1][c][n, :]
                e_tnc = (1 / (np.abs(H_tnc) ** 2))
                Power_Separate[t + 1][n][c] = np.linalg.norm(
                    Lambda[t + 1] * np.sqrt(e_tnc) * (xtnc[t + 1][n][c] - xtc[t][c])) ** 2
                Power_Constraint = np.linalg.norm(
                    Lambda[t + 1] * np.sqrt(e_tnc) * (xtnc[t + 1][n][c] - xtc[t][c])) ** 2 - 10 ** (-1.4) / C * par['slot']
                if Power_Constraint < 0:
                    Power_Constraint = 0
                TX_Pt += Power_Constraint
        Power.append((Power[t] * (t + 1) + TX_Pt / (N * par['slot']) / 10 ** (-1.4) * 1e-3) / (t + 2))
        Power_dBm.append(Func_W2dBm(Power[t + 1]))
        print('Power: {}, t: {}'.format(Power_dBm[t + 1], t + 2))
    return Accuracy, Power_dBm

def OTARCI():
    alpha = 0.01
    gamma = 7.4e-7
    Lambda = {t: {c: np.zeros((10, 784)) for c in range(10)} for t in range(sys['T'])}
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

def OMUAA():
    alpha = 0.01
    Gamma = 1e-3
    Lambda = 8e-7
    Pnc_bar = Func_dBm2W(16) / C * par['slot']

    xtc = {t: {c: np.zeros(d) for c in range(0, C)} for t in range(0, T)}
    for c in range(10):
        xtc[0][c] = np.ones(d) * 1e-7
    xtnc = {t: {n: {c: np.zeros(d) for c in range(0, C)} for n in range(N)} for t in range(0, T)}
    Qtnc = {t: np.zeros((N, C)) for t in range(0, T)}
    gtnc2 = {t: np.zeros((N, C)) for t in range(0, T)}

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

        GD_tnc_g = np.zeros((d * C, N))
        for n in range(N):
            for c in range(0, C):
                idx = slice(d * c, d * (c + 1))
                H_tnc = H[t][c][n, :]
                e_tnc = (1 / (np.abs(H_tnc) ** 2))
                GD_tnc_g[idx, n] = 2 * Lambda ** 2 * e_tnc

        for c in range(0, C):
            xtc[t + 1][c] = np.zeros(d)

        for c in range(0, C):
            for n in range(N):
                idx = slice(d * c, d * (c + 1))
                H_tnc = H[t][c][n, :]
                e_tnc = (1 / (np.abs(H_tnc) ** 2))
                xtnc[t + 1][n][c] = xtc[t][c] - (alpha / (np.ones(d) + alpha * Gamma * Qtnc[t][n, c] * GD_tnc_g[idx, n])) * GD_tnc_f[idx, n]
                xtc[t + 1][c] += xtnc[t + 1][n][c]

                gtnc2[t][n, c] = (Lambda ** 2) * np.linalg.norm(
                    np.sqrt(e_tnc) * (xtnc[t + 1][n][c] - xtc[t][c])) ** 2 - Pnc_bar

                Qtnc[t + 1][n, c] = max(0, Qtnc[t][n, c] + gtnc2[t][n, c])

            if par['add_noise'] == 1:
                xtc[t + 1][c] = xtc[t + 1][c] / N + (1 / Lambda) * Z[t + 1][c] / N

        print(f'OMUAA, t:{t + 1}')

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
            e_tnc = (1 / (np.abs(H_tnc) ** 2))
            Power_Separate[0][n][c] = np.linalg.norm(Lambda * np.sqrt(e_tnc) * xtc[0][c]) ** 2
            Power_Constraint = np.linalg.norm(Lambda * np.sqrt(e_tnc) * xtc[0][c]) ** 2 - 10 ** (-1.4) / C * par['slot']
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
                H_tnc = H[t + 1][c][n, :]
                e_tnc = (1 / (np.abs(H_tnc) ** 2))
                Power_Separate[t + 1][n][c] = np.linalg.norm(
                    Lambda * np.sqrt(e_tnc) * (xtnc[t + 1][n][c] - xtc[t][c])) ** 2
                Power_Constraint = np.linalg.norm(
                    Lambda * np.sqrt(e_tnc) * (xtnc[t + 1][n][c] - xtc[t][c])) ** 2 - 10 ** (-1.4) / C * par['slot']
                if Power_Constraint < 0:
                    Power_Constraint = 0
                TX_Pt += Power_Constraint
        Power.append((Power[t] * (t + 1) + TX_Pt / (N * par['slot']) / 10 ** (-1.4) * 1e-3) / (t + 2))
        Power_dBm.append(Func_W2dBm(Power[t + 1]))
        print('Power: {}, t: {}'.format(Power_dBm[t + 1], t + 2))
    return Accuracy, Power_dBm

def OCOMSP():
    alpha = 0.01
    gammaMSP = 1e-5
    etaMSP = 1e2
    Lambda = 5.75e-7  # 5.7e-7
    U = 1e-15
    Pnc_bar = Func_dBm2W(16) / C * par['slot']

    xtc = {t: {c: np.zeros(d) for c in range(0, C)} for t in range(0, T)}
    for c in range(10):
        xtc[0][c] = np.ones(d) * 1e-7
    xtnc = {t: {n: {c: np.zeros(d) for c in range(0, C)} for n in range(N)} for t in range(0, T)}
    Qtnc = {t: np.zeros(C) for t in range(0, T)}
    gtnc = {t: np.zeros((N, C)) for t in range(0, T)}

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

        GD_tnc_g = np.zeros((d * C, N))
        for n in range(N):
            for c in range(0, C):
                idx = slice(d * c, d * (c + 1))
                H_tnc = H[t][c][n, :]
                e_tnc = (1 / (np.abs(H_tnc) ** 2))
                GD_tnc_g[idx, n] = 2 * Lambda ** 2 * e_tnc

        for c in range(0, C):
            xtc[t + 1][c] = np.zeros(d)

        for c in range(0, C):
            for n in range(N):
                idx = slice(d * c, d * (c + 1))
                H_tnc = H[t + 1][c][n, :]
                e_tnc = (1 / (np.abs(H_tnc) ** 2))

                xtnc[t + 1][n][c] = xtc[t][c] - alpha * GD_tnc_f[idx, n] - U * Qtnc[t][c] * GD_tnc_g[idx, n]
                xtc[t + 1][c] += xtnc[t + 1][n][c]

                gtnc[t + 1][n, c] = (Lambda ** 2) * np.linalg.norm(
                    np.sqrt(e_tnc) * (xtnc[t + 1][n][c] - xtc[t][c])) ** 2 - Pnc_bar

            Qtnc[t + 1][c] = max(Qtnc[t][c] + gammaMSP * (np.sum(gtnc[t][:, c]) - gammaMSP * etaMSP * Qtnc[t][c]), 0)

            if par['add_noise'] == 1:
                xtc[t + 1][c] = xtc[t + 1][c] / N + (1 / Lambda) * Z[t + 1][c] / N
        print(f'OCOMSP, t:{t + 1}')

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
            e_tnc = (1 / (np.abs(H_tnc) ** 2))
            Power_Separate[0][n][c] = np.linalg.norm(Lambda * np.sqrt(e_tnc) * xtc[0][c]) ** 2
            Power_Constraint = np.linalg.norm(Lambda * np.sqrt(e_tnc) * xtc[0][c]) ** 2 - 10 ** (-1.4) / C * par['slot']
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
                H_tnc = H[t + 1][c][n, :]
                e_tnc = (1 / (np.abs(H_tnc) ** 2))
                Power_Separate[t + 1][n][c] = np.linalg.norm(
                    Lambda * np.sqrt(e_tnc) * (xtnc[t + 1][n][c] - xtc[t][c])) ** 2
                Power_Constraint = np.linalg.norm(
                    Lambda * np.sqrt(e_tnc) * (xtnc[t + 1][n][c] - xtc[t][c])) ** 2 - 10 ** (-1.4) / C * par['slot']
                if Power_Constraint < 0:
                    Power_Constraint = 0
                TX_Pt += Power_Constraint
        Power.append((Power[t] * (t + 1) + TX_Pt / (N * par['slot']) / 10 ** (-1.4) * 1e-3) / (t + 2))
        Power_dBm.append(Func_W2dBm(Power[t + 1]))
        print('Power: {}, t: {}'.format(Power_dBm[t + 1], t + 2))
    return Accuracy, Power_dBm

Accuracy_Summary = np.zeros((5, 500))
Power_dBm_Summary = np.zeros((5, 500))

Accuracy_Ideal = Idealized_FL()
Accuracy_Summary[0], Power_dBm_Summary[0] = COMUDO()
Accuracy_Summary[1], Power_dBm_Summary[1] = OTALPC()
Accuracy_Summary[2], Power_dBm_Summary[2] = OTARCI()
Accuracy_Summary[3], Power_dBm_Summary[3] = OMUAA()
Accuracy_Summary[4], Power_dBm_Summary[4] = OCOMSP()

# Accuracy Plot
plt.figure(1)
plt.plot(Accuracy_Ideal, label="Idealized_FL")
plt.plot(Accuracy_Summary[0], label="COMUDO")
plt.plot(Accuracy_Summary[1], label="OTALPC")
plt.plot(Accuracy_Summary[2], label="OTARCI")
plt.plot(Accuracy_Summary[3], label="OMUAA")
plt.plot(Accuracy_Summary[4], label="OCOMSP")
plt.xlabel("time")
plt.ylabel("accuracy %")
plt.legend()
plt.show(block=False)

# Power Constraint Plot
plt.figure(2)
plt.plot(Power_dBm_Summary[0], label="HODOO")
plt.plot(Power_dBm_Summary[1], label="OTALPC")
plt.plot(Power_dBm_Summary[2], label="OTARCI")
plt.plot(Power_dBm_Summary[3], label="OMUAA")
plt.plot(Power_dBm_Summary[4], label="OCOMSP")
plt.xlabel("time")
plt.ylabel("power dBm")
plt.legend()
plt.show()