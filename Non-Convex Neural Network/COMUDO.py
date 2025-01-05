import copy
import torch
import torchvision
from torch.nn import Conv2d, MaxPool2d, Flatten, Linear, Sequential, ReLU, Softmax, CrossEntropyLoss
from torch.utils.data import DataLoader, Subset
import torch.nn as nn
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

# Seed Setting
set_seed(44)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Channel Generation
distances = np.linspace(500, 500, 10)
gamma = 37.0
psi = 8.0
Halpha = torch.tensor(0.997).cuda()

def compute_epsilon(distance, gamma, psi):
    db_values = -31.54 - gamma * np.log10(distance) - np.sqrt(psi) * np.random.randn()
    epsilon_com = 10 ** (db_values / 10)
    return epsilon_com

epsilon_device = [compute_epsilon(distance, gamma, psi) for distance in distances]

h = [
    {
        'model1.0.weight': torch.randn(10, 1, 7, 7, dtype=torch.complex64)*np.sqrt(epsilon/2),
        'model1.0.bias': torch.randn(10, dtype=torch.complex64)*np.sqrt(epsilon/2),
        'model1.3.weight': torch.randn(10, 4840, dtype=torch.complex64)*np.sqrt(epsilon/2),
        'model1.3.bias': torch.randn(10, dtype=torch.complex64)*np.sqrt(epsilon/2),
    }
    for epsilon in epsilon_device
]

h_gpu = []
for model_params in h:
    model_params_gpu = {k: v.to(device) for k, v in model_params.items()}
    h_gpu.append(model_params_gpu)

# Noise Generation
N0 = -174
NF = 10
BW = 15e3
noise_dBm = N0 + 10 * np.log10(BW) + NF
noise = 10 ** (noise_dBm/10) * 1e-3

Z = {
        'model1.0.weight': torch.randn(10, 1, 7, 7)*np.sqrt(noise),
        'model1.0.bias': torch.randn(10)*np.sqrt(noise),
        'model1.3.weight': torch.randn(10, 4840)*np.sqrt(noise),
        'model1.3.bias': torch.randn(10)*np.sqrt(noise),
    }

Z_gpu = {k: v.to(device) for k, v in Z.items()}

# To GPU
epsilon_device = torch.tensor(epsilon_device, dtype=torch.float32).to(device)

# Train and Test Sample
dataset_tranform = torchvision.transforms.Compose([torchvision.transforms.ToTensor(), torchvision.transforms.Normalize((0.1307,), (0.3081,))])

train_set = torchvision.datasets.MNIST(root="./dataset", train=True, transform=dataset_tranform, download=True)
test_set = torchvision.datasets.MNIST(root="./dataset", train=False, transform=dataset_tranform, download=True)

# Divided to 10 Classes
class_indices = {i: [] for i in range(10)}
for idx, (image, label) in enumerate(train_set):
    class_indices[label].append(idx)

device_data_loaders = []
length_sample = []

for i in range(10):
    indices = class_indices[i]
    subset = Subset(train_set, indices)

    data_loader = subset
    length_sample.append(len(data_loader))
    device_data_loaders.append(data_loader)

ind_n = {t: {n: [] for n in range(10)} for t in range(2000)}
for t in range(2000):
    for i in range(10):
        ind_n[t][i] = np.random.choice(length_sample[i], 20, replace=False)

subset_indices = list(range(1000))
test_subset = Subset(test_set, subset_indices)
test_dataloader = DataLoader(test_subset, batch_size=64, drop_last=False)

# Model aggregation
def FedAvg(w):
    weights_avg = copy.deepcopy(w[0])
    for k in weights_avg.keys():
        for i in range(1, len(w)):
            weights_avg[k] += w[i][k]
        weights_avg[k] = torch.div(weights_avg[k], len(w))
    return weights_avg

def FedAvg_Noise(w,Z,Lambda):
    weights_avg = copy.deepcopy(w[0])
    for k in weights_avg.keys():
        for i in range(1, len(w)):
            weights_avg[k] += w[i][k]
        weights_avg[k] += torch.div(Z[k], Lambda)
        weights_avg[k] = torch.div(weights_avg[k], len(w))
    return weights_avg

# Channel Tune
def Auto_tune(Round, T, Power):
    Proportion = 0.15
    DetectionT = T * Proportion
    if Round > DetectionT:
        if (Power[-1] - Power[-2]) >= 0.1:
            Adjustment = 1
        else:
            Adjustment = 0
    else:
        Adjustment = 0
    return Adjustment

# Model structure
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.model1 = Sequential(
            Conv2d(1, 10, 7),
            ReLU(),
            Flatten(),
            Linear(4840, 10),
            Softmax(dim=1)
        )
    def forward(self, x):
        x = self.model1(x)
        return x

##
glob_model = Net()
glob_model.to(device)

local_models = [Net().to(device) for i in range(10)]

loss = CrossEntropyLoss()
loss.to(device)

local_optimizers = []

# Initialize model
for i in range(10):
    local_models[i].load_state_dict(glob_model.state_dict())
    optimizer = torch.optim.SGD(local_models[i].parameters(), lr=0.02)
    local_optimizers.append(optimizer)

# Parameters
Lambda = torch.tensor(2e-6, device=device)
Gamma = torch.tensor(1.5e-3, device=device)
Eta = torch.tensor(1e-3, device=device)
V = torch.tensor(1, device=device)
Qtn = {i: torch.tensor(1.0, device=device) for i in range(10)}
Gtn = {i: torch.tensor(0.0, device=device) for i in range(10)}
Pnbar = torch.tensor((10 ** (-1.4)) * 48910/1000, device=device)
Qtn_Last = {i: torch.tensor(1.0, device=device) for i in range(10)}

T = 2000
plot_accuracy = []
plot_loss = []

transmit_power_history = [0]
transmit_power_history_dBm = [0]
Power_Separate = np.zeros((2001, 10))

for t in range(2000):
    transmit_power = 0
    weights_local = []
    loss_local = []

    for i in range(10):
        local_models[i].load_state_dict(glob_model.state_dict())
    # Device i starts training
    for i in range(10):
        local_models[i].train()
        indices = ind_n[t][i]
        images = []
        targets = []
        for idx in indices:
            img, tar = device_data_loaders[i][idx]
            images.append(img)
            targets.append(tar)

        images = torch.stack(images)
        targets = torch.tensor(targets)

        sampled_images = images.to(device)
        sampled_targets = targets.to(device)

        outputs = local_models[i](sampled_images)
        result_loss = loss(outputs, sampled_targets)
        local_optimizers[i].zero_grad()
        result_loss.backward()
        # Calculate Gtn
        g_sum = torch.tensor(0, dtype=torch.float32, device=device)
        for param_tensor, param in local_models[i].named_parameters():
            if param.grad is not None:
                h_data = h_gpu[i][param_tensor]
                h_abs_inverse = 1.0 / torch.abs(h_data)
                g = torch.sum((Lambda * 0.02 * param.grad * h_abs_inverse) ** 2)
                g_sum += g
        Gtn[i] = g_sum - Pnbar

        # Model update
        if Gtn[i] <= 0:
            local_optimizers[i].step()
        elif Gtn[i] > 0:
            with torch.no_grad():
                for param_tensor, param in local_models[i].named_parameters():
                    if param.grad is not None:
                        h_data = h_gpu[i][param_tensor]
                        h_abs_inverse = 1.0 / torch.abs(h_data)
                        lr = 0.02/(1 + 0.02 * Gamma * Qtn[i] * (2 * (Lambda * h_abs_inverse) ** 2))
                        param.data -= lr * param.grad

        weights_local.append(copy.deepcopy(local_models[i].state_dict()))
        loss_local.append(result_loss)

        # Calculate transmission power
        gn = torch.tensor(0, device=device)
        state_dict = local_models[i].state_dict()
        for param_tensor in state_dict:
            param_data = state_dict[param_tensor]
            glob_data = glob_model.state_dict()[param_tensor]
            h_data = h_gpu[i][param_tensor]
            # Update Channel State h
            Hz = torch.sqrt(epsilon_device[i] * (1 - Halpha ** 2) / 2) * torch.randn(size=h_data.shape, dtype=torch.complex64).to(device)
            h_gpu[i][param_tensor] = Halpha * h_data + Hz

            h_abs_inverse = 1.0 / torch.abs(h_gpu[i][param_tensor])
            power_compute = torch.sum((Lambda * (param_data - glob_data) * h_abs_inverse) ** 2)
            power = power_compute.item()
            transmit_power = transmit_power + power
            gn = gn + power_compute
        cons = torch.maximum(gn - Pnbar, torch.tensor(0, device=device))
        Qtn_Last[i] = Qtn[i]
        Qtn[i] = torch.maximum((1 - Eta) * Qtn[i] + Gamma * (cons), V)
        Power_Separate[t+1][i] = gn

    tranPower_toHistory = (transmit_power_history[t] * (t+1) + transmit_power/(10*48910/1000))/(t+2)
    transmit_power_history.append(tranPower_toHistory)
    transmit_power_history_dBm.append(np.log10(transmit_power_history[t+1]*1000) * 10)

    # Update Noise Z
    for param_tensor in Z_gpu:
        Z_gpu[param_tensor] = torch.randn(size=Z_gpu[param_tensor].shape, device=device) * torch.sqrt(torch.tensor(noise, device=device))

    # When the channel state h is too low, this mechanism is for waiting the next time slot to transmit
    Adjustment = Auto_tune(t + 1, T, transmit_power_history_dBm)
    if Adjustment == 1:
        transmit_power = 0
        for i in range(10):
            gn = torch.tensor(0, device=device)
            state_dict = local_models[i].state_dict()
            for param_tensor in state_dict:
                param_data = state_dict[param_tensor]
                glob_data = glob_model.state_dict()[param_tensor]
                h_data = h_gpu[i][param_tensor]
                # Update Channel State h
                Hz = torch.sqrt(epsilon_device[i] * (1 - Halpha ** 2) / 2) * torch.randn(size=h_data.shape,dtype=torch.complex64).to(device)
                h_gpu[i][param_tensor] = Halpha * h_data + Hz

                h_abs_inverse = 1.0 / torch.abs(h_gpu[i][param_tensor])
                power_compute = torch.sum((Lambda * (param_data - glob_data) * h_abs_inverse) ** 2)
                power = power_compute.item()
                transmit_power = transmit_power + power
                gn = gn + power_compute
            cons = torch.maximum(gn - Pnbar, torch.tensor(0, device=device))
            Qtn[i] = torch.maximum((1 - Eta) * Qtn_Last[i] + Gamma * (cons), V)
            Power_Separate[t + 1][i] = gn
        for param_tensor in Z_gpu:
            Z_gpu[param_tensor] = torch.randn(size=Z_gpu[param_tensor].shape, device=device) * torch.sqrt(torch.tensor(noise, device=device))
        tranPower_toHistory = (transmit_power_history[t] * (t + 1) + transmit_power / (10 * 48910 / 1000)) / (t + 2)
        transmit_power_history[-1] = tranPower_toHistory
        transmit_power_history_dBm[-1] = np.log10(transmit_power_history[-1] * 1000) * 10

    # Update Global Model
    weights_glob = FedAvg_Noise(weights_local, Z_gpu, Lambda)
    glob_model.load_state_dict(weights_glob)

    loss_avg = sum(loss_local)/len(loss_local)
    print("t: {}, Loss: {}, Power: {}".format(t+1, loss_avg, transmit_power_history_dBm[t+1]))

    # test
    if (t+1) % 1 == 0:
        glob_model.eval()
        total_accuracy = 0
        with torch.no_grad():
            for data in test_dataloader:
                imgs, targets = data
                imgs = imgs.to(device)
                targets = targets.to(device)
                outputs = glob_model(imgs)
                accuracy = (outputs.argmax(1) == targets).sum()
                total_accuracy = total_accuracy + accuracy
            if t == 0:
                plot_accuracy.append(torch.Tensor.cpu(total_accuracy / len(test_subset)) * 100)
                plot_loss.append(torch.Tensor.cpu(loss_avg))
            else:
                plot_accuracy.append((plot_accuracy[t-1] * t + torch.Tensor.cpu(total_accuracy/len(test_subset)) * 100) / (t+1))
                plot_loss.append((plot_loss[t - 1] * t + torch.Tensor.cpu(loss_avg)) / (t + 1))
        print("Average accuracy: {}".format(plot_accuracy[t]))


# Accuracy Plot
plt.figure(1)
plt.plot(plot_accuracy)
plt.xlabel("time")
plt.ylabel("accuracy %")
plt.show(block=False)

# Power Constraint Plot
Power_Constraint = np.zeros((2001))
for t in range(2000):
    TX_Pt = 0
    for i in range(10):
        powercons = Power_Separate[t + 1][i] - 10 ** (-1.4) * 48910 / 1000
        if powercons <= 0:
            powercons = 0
        TX_Pt = TX_Pt + powercons
    Power_Constraint[t+1] = (Power_Constraint[t] * (t+1) + TX_Pt / (10 * 48910 / 1000) / 10 ** (-1.4) * 1e-3) / (t+2)
    print(10 * np.log10(Power_Constraint[t] * 1e3))

Power_Constraint = 10 * np.log10(Power_Constraint * 1e3)

plt.figure(2)
plt.plot(Power_Constraint[0:t+2])
plt.xlabel("time")
plt.ylabel("power dBm")
plt.show()