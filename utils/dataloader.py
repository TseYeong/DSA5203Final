import os
import copy
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split


def get_train_val_loaders(data_dir, val_ratio=0.2, batch_size=32):
    """
    Load and preprocess the dataset with data augmentation, and split into training and validation sets.

    Args:
        data_dir (str): Path to the training dataset.
        val_ratio (float): Ratio of validation data.
        batch_size (int): Number of samples per batch.

    Returns:
        tuple: train_loader, val_loader, custom_classes
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = datasets.ImageFolder(
        root=data_dir,
        transform=transforms.ToTensor()
    )

    means = torch.zeros(3).to(device)
    stds = torch.zeros(3).to(device)

    for img, _ in dataset:
        img = img.to(device)
        means += torch.mean(img, dim=(1, 2))
        stds += torch.mean(img, dim=(1, 2))

    means /= len(dataset)
    stds /= len(dataset)
    
    means = means.cpu()
    stds = stds.cpu()

    custom_classes = ['bedroom', 'Coast', 'Forest', 'Highway', 'industrial',
                      'Insidecity', 'kitchen', 'livingroom', 'Mountain', 'Office',
                      'OpenCountry', 'store', 'Street', 'Suburb', 'TallBuilding']
    custom_class_to_index = {cls: i for i, cls in enumerate(custom_classes)}

    dataset.class_to_idx = custom_class_to_index
    dataset.samples = [
        (path, custom_class_to_index[os.path.basename(os.path.dirname(path))])
        for (path, _) in dataset.samples
    ]

    train_transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=3),
        transforms.Resize(224),
        transforms.RandomRotation(5),
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(224, padding=4),
        transforms.ToTensor(),
        transforms.Normalize(means, stds)
    ])
    val_transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=3),
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize(means, stds)
    ])

    dataset.transforms = train_transform
    val_len = int(len(dataset) * val_ratio)
    train_len = len(dataset) - val_len
    train_set, val_set = random_split(dataset, [train_len, val_len])

    val_set = copy.deepcopy(val_set)
    val_set.dataset.transform = val_transform

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, custom_classes


def get_test_loader(data_dir, batch_size=32):
    """
    Load and preprocess the test dataset.

    Args:
        data_dir (str): Path to the test dataset.
        batch_size (int): Number of samples per batch.

    Returns:
        DataLoader: test_loader
    """
    test_transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=3),
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3)
    ])

    test_set = datasets.ImageFolder(
        root=data_dir,
        transform=test_transform
    )

    custom_classes = ['bedroom', 'Coast', 'Forest', 'Highway', 'industrial',
                      'Insidecity', 'kitchen', 'livingroom', 'Mountain', 'Office',
                      'OpenCountry', 'store', 'Street', 'Suburb', 'TallBuilding']
    custom_class_to_index = {cls: i + 1 for i, cls in enumerate(custom_classes)}
    test_set.class_to_idx = custom_class_to_index
    test_set.samples = [
        (path, custom_class_to_index[os.path.basename(os.path.dirname(path))])
        for (path, _) in test_set.samples
    ]

    test_loader = DataLoader(test_set, batch_size=batch_size)

    return test_loader, custom_class_to_index
