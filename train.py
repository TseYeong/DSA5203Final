import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from model.resnet import build_resnet
from utils.dataloader import get_train_val_loaders
from utils.eval import calculate_topk_accuracy


def train(train_dir, model_path, depth=18, lr=1e-3, epochs=15):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loader, val_loader, classes = get_train_val_loaders(train_dir)
    model = build_resnet(output_dim=len(classes), depth=depth).to(device)

    criterion = nn.CrossEntropyLoss().to(device)

    params = [
        {'params': model.conv1.parameters(), 'lr': lr / 10},
        {'params': model.bn1.parameters(), 'lr': lr / 10},
        {'params': model.layer1.parameters(), 'lr': lr / 8},
        {'params': model.layer2.parameters(), 'lr': lr / 6},
        {'params': model.layer3.parameters(), 'lr': lr / 4},
        {'params': model.layer4.parameters(), 'lr': lr / 2},
        {'params': model.fc.parameters()}
    ]
    optimizer = optim.Adam(params, lr=lr)

    STEPS_PER_EPOCH = len(train_loader)
    TOTAL_STEPS = epochs * STEPS_PER_EPOCH
    MAX_LRS = [p['lr'] for p in optimizer.param_groups]
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=MAX_LRS,
        total_steps=TOTAL_STEPS
    )

    best_val_loss = float('inf')
    for epoch in range(epochs):
        train_loss, train_acc_1, train_acc_5 = train_epoch(model, train_loader, optimizer,
                                                           criterion, scheduler, device)
        val_loss, val_acc_1, val_acc_5 = evaluate(model, val_loader, criterion, device)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), model_path)

        print(f"Epoch: {epoch + 1:02}")
        print(f'\tTrain Loss: {train_loss:.3f} | Train Acc @1: {train_acc_1*100:6.2f}% | '
              f'Train Acc @5: {train_acc_5*100:6.2f}%')
        print(f'\tValid Loss: {val_loss:.3f} | Valid Acc @1: {val_acc_1 * 100:6.2f}% | '
              f'Valid Acc @5: {val_acc_5 * 100:6.2f}%')

    return train_acc_1


def train_epoch(model, loader, optimizer, criterion, scheduler, device):
    """
    Trains the model for one epoch and returns the loss and top-1/top-5 accuracy.

    Args:
        model (nn.Module): The model to train.
        loader (DataLoader): Training data loader.
        optimizer (Optimizer): Optimizer.
        criterion (Loss): Loss function.
        scheduler (LRScheduler): Learning rate scheduler.
        device (torch.device): Computation device.

    Returns:
        Tuple[float, float, float]: avg_loss, top-1 acc, top-5 acc
    """
    model.train()
    epoch_loss = 0.0
    epoch_acc_1 = 0.0
    epoch_acc_5 = 0.0

    for (images, labels) in tqdm(loader, desc='Training'):
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs, _ = model(images)
        loss = criterion(outputs, labels)
        acc_1, acc_5 = calculate_topk_accuracy(outputs, labels)
        loss.backward()
        optimizer.step()
        scheduler.step()

        epoch_loss += loss.item()
        epoch_acc_1 += acc_1.item()
        epoch_acc_5 += acc_5.item()

    avg_loss = epoch_loss / len(loader)
    avg_acc_1 = epoch_acc_1 / len(loader)
    avg_acc_5 = epoch_acc_5 / len(loader)

    return avg_loss, avg_acc_1, avg_acc_5


def evaluate(model, loader, criterion, device):
    """
    Evaluate the model on validation or test set.

    Args:
        model (nn.Module): Trained model.
        loader (DataLoader): Evaluation data loader.
        criterion (Loss): Loss function.
        device (torch.device): Computation device.

    Returns:
        Tuple[float, float, float]: avg_loss, top-1 acc, top-5 acc
    """
    model.eval()
    epoch_loss = 0.0
    epoch_acc_1 = 0.0
    epoch_acc_5 = 0.0

    with torch.no_grad():
        for (images, labels) in loader:
            images, labels = images.to(device), labels.to(device)
            outputs, _ = model(images)
            loss = criterion(outputs, labels)
            acc_1, acc_5 = calculate_topk_accuracy(outputs, labels)

            epoch_loss += loss.item()
            epoch_acc_1 += acc_1.item()
            epoch_acc_5 += acc_5.item()

    avg_loss = epoch_loss / len(loader)
    avg_acc_1 = epoch_acc_1 / len(loader)
    avg_acc_5 = epoch_acc_5 / len(loader)

    return avg_loss, avg_acc_1, avg_acc_5
