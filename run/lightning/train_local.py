import lightning as L
from utils.utils import torch_seed
from dataloader.cifar10_loader import get_cifar10_dataloaders

from .lit_saimple_net import LitSimpleNet

if __name__ == '__main__':

    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    n_output = len(classes)

    config = dict(
        epochs=50,
        classes=n_output,
        batch_size=100,
        learning_rate=0.001,
        dataset="CIFAR10",
        architecture="CNN"  
    )

    data_root = './data'

    train_loader, val_loader = get_cifar10_dataloaders(data_root, config['batch_size'])

    torch_seed()

    net = LitSimpleNet(num_classes=n_output, lr=config['learning_rate'])

    wandb_logger = L.pytorch.loggers.WandbLogger(project='simple_net', log_model=True, config=config)
    wandb_logger.watch(net, log="all")
    
    # trainer = L.Trainer(max_epochs=config['epochs'], logger=wandb_logger, strategy='ddp')
    trainer = L.Trainer(max_epochs=config['epochs'], logger=wandb_logger)


    trainer.fit(net, train_loader, val_loader)
