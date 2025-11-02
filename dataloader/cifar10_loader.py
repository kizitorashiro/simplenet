import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader

def get_cifar10_dataloaders(data_root, batch_size):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(0.5, 0.5)
    ])

    train_set = datasets.CIFAR10(
        root=data_root, train=True,
        download=False, transform=transform
    )

    val_set = datasets.CIFAR10(
        root=data_root, train=False,
        download=False, transform=transform
    )

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader
