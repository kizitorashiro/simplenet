import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import numpy as np
from model.simple_net import SimpleNet
from run.pytorch.fit import fit
from utils.utils import torch_seed
from dataloader.cifar10_loader import get_cifar10_dataloaders

if __name__ == '__main__':

    device = torch.device('mps' if torch.mps.is_available() else 'cpu')
    print(f'Device: {device}')

    num_epochs = 50
    lr = 0.001
    batch_size = 100

    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    n_output = len(classes)

    data_root = './data'

    train_loader, test_loader = get_cifar10_dataloaders(data_root, batch_size)

    torch_seed()
    net = SimpleNet(n_output).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=lr)
    history = np.zeros((0, 5))    
    history = fit(net, optimizer, criterion, num_epochs, train_loader, test_loader, device, history)
