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
plot_accuracy = []
plot_loss = []

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
        local_optimizers[i].step()

        weights_local.append(copy.deepcopy(local_models[i].state_dict()))

        loss_local.append(result_loss)

    # Update Global Model
    weights_glob = FedAvg(weights_local)
    glob_model.load_state_dict(weights_glob)

    loss_avg = sum(loss_local)/len(loss_local)
    print("t: {}, Loss: {}".format(t+1, loss_avg))

    # 测试
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
                plot_loss.append((plot_loss[t-1] * t + torch.Tensor.cpu(loss_avg)) / (t + 1))
        print("Average accuracy: {}".format(plot_accuracy[t]))

# Accuracy Plot
plt.figure(1)
plt.plot(plot_accuracy)
plt.xlabel("time")
plt.ylabel("accuracy %")
plt.show()

