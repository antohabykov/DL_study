import torch
import torchvision
import torchvision.transforms as transforms

# for downloading 
downloading = torchvision.datasets.CIFAR10(root='./data', train=True,
                                            download=True)
downloading = torchvision.datasets.CIFAR10(root='./data', train=False,
                                            download=True)

def dataloaders(config):
    transform = transforms.Compose(
        [transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))])

    train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                                download=False, transform=transform)
    test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                            download=False, transform=transform)
    trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=config['batch_size'],
                                            num_workers=config['num_workers'])
    testloader = torch.utils.data.DataLoader(test_dataset, batch_size=config['batch_size'],
                                            num_workers=config['num_workers'])
    return trainloader, testloader




