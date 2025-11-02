import torchvision.datasets as datasets

def download_cifar10(data_root='./data'):
    datasets.CIFAR10(root=data_root, train=True, download=True)
    datasets.CIFAR10(root=data_root, train=False, download=True)

if __name__ == '__main__':
    download_cifar10()

