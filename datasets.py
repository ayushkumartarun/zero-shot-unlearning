import torchvision
import torchvision.transforms as tt
import tarfile
import os
from torchvision.datasets import ImageFolder
from torchvision.datasets.utils import download_url


def cifar10(root = './'):
    transform = tt.Compose([
        tt.ToTensor(),
        tt.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    dataset_url = "https://s3.amazonaws.com/fast-ai-imageclas/cifar10.tgz"
    download_url(dataset_url, '.')

    # Extract from archive
    with tarfile.open('./cifar10.tgz', 'r:gz') as tar:
        tar.extractall(path='./data')

    # Look into the data directory
    data_dir = os.path.join(root, 'data/cifar10')
    #print(os.listdir(data_dir))
    #classes = os.listdir(data_dir + "/train")
    
    #train_ds = torchvision.datasets.CIFAR10(root='./', train=True, download=True, transform=transform)
    #valid_ds = torchvision.datasets.CIFAR10(root='./', train=False, download=True, transform=transform)
    train_ds = ImageFolder(data_dir+'/train', transform)
    valid_ds = ImageFolder(data_dir+'/test', transform)
    return train_ds, valid_ds

def svhn(root = './'):
    transform = tt.Compose([
        tt.ToTensor(),
        tt.Normalize((0.4376821, 0.4437697, 0.47280442), (0.19803012, 0.20101562, 0.19703614))
    ])
    
    train_ds = torchvision.datasets.SVHN(root='./', train=True, download=True, transform=transform)
    valid_ds = torchvision.datasets.SVHN(root='./', train=False, download=True, transform=transform)
    
    return train_ds, valid_ds

def mnist(root = './'):
    transform = tt.Compose([
        tt.ToTensor(),
    ])
    
    train_ds = torchvision.datasets.MNIST(root='./', train=True, download=True, transform=transform)
    valid_ds = torchvision.datasets.MNIST(root='./', train=False, download=True, transform=transform)
    
    return train_ds, valid_ds