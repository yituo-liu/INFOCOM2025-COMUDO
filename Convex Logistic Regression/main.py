import torch
import torchvision
from torch.utils.data import DataLoader, Subset
import numpy as np
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import matplotlib.pyplot as plt
import random

from idealized_fl import Idealized_FL
from comudo import COMUDO
from otalpc import OTALPC
from otarci import OTARCI
from omuaa import OMUAA
from ocomsp import OCOMSP

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
T = sys['T']
D = sys['D']
d = sys['d']
C = sys['C']

# Simulation
Accuracy_Summary = np.zeros((5, 500))
Power_dBm_Summary = np.zeros((5, 500))

Accuracy_Ideal = Idealized_FL(N, T, D, d, C, Data, Label, Testdata, Testlabels)
Accuracy_Summary[0], Power_dBm_Summary[0] = COMUDO(N, T, D, d, C, Data, Label, H, Z, Testdata, Testlabels, par)
Accuracy_Summary[1], Power_dBm_Summary[1] = OTALPC(N, T, D, d, C, Data, Label, H, Z, Testdata, Testlabels, par)
Accuracy_Summary[2], Power_dBm_Summary[2] = OTARCI(N, T, D, d, C, Data, Label, H, Z, Testdata, Testlabels, par)
Accuracy_Summary[3], Power_dBm_Summary[3] = OMUAA(N, T, D, d, C, Data, Label, H, Z, Testdata, Testlabels, par)
Accuracy_Summary[4], Power_dBm_Summary[4] = OCOMSP(N, T, D, d, C, Data, Label, H, Z, Testdata, Testlabels, par)

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
plt.plot(Power_dBm_Summary[0], label="COMUDO")
plt.plot(Power_dBm_Summary[1], label="OTALPC")
plt.plot(Power_dBm_Summary[2], label="OTARCI")
plt.plot(Power_dBm_Summary[3], label="OMUAA")
plt.plot(Power_dBm_Summary[4], label="OCOMSP")
plt.xlabel("time")
plt.ylabel("power dBm")
plt.legend()
plt.show()