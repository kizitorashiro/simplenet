import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from model.simple_net import SimpleNet
from run.pytorch.fit import fit
from utils.utils import torch_seed
from dataloader.cifar10_loader import get_cifar10_dataloaders
import wandb 
if __name__ == '__main__':


    config = dict(
        epochs=50,
        classes=10,
        batch_size=100,
        learning_rate=0.001,
        dataset="CIFAR10",
        architecture="CNN"  
    )


    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    n_output = len(classes)

    data_root = './data'

    train_loader, test_loader = get_cifar10_dataloaders(data_root, config['batch_size'])


    torch_seed()
    device = torch.device('mps' if torch.mps.is_available() else 'cpu')
    print(f'Device: {device}')
    net = SimpleNet(n_output).to(device)

    with wandb.init(project='simple_net', config=config) as run:
        run.watch(net, log='all', log_freq=100)
        fit(net, config, train_loader, test_loader, device, run)
