import os
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch

def get_eurosat_dataloaders(data_dir='data/EuroSAT_RGB', batch_size=64, num_workers=2):
    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.3444, 0.3809, 0.4082], std=[0.1818, 0.1317, 0.1205])  # EuroSAT RGB mean/std
    ])

    dataset = datasets.ImageFolder(root=data_dir, transform=transform)

    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])


    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, test_loader, dataset.classes
