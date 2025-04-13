import random
import numpy as np
import torch
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.optim as optim
from model.resnet import build_resnet
from utils.dataloader import get_train_val_loaders
from utils.lr_finder import LRFinder


def find_lr(train_dir, depth=18, start_lr=1e-7, end_lr=10, num_iter=100):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loader, _, _classes = get_train_val_loaders(train_dir)
    model = build_resnet(len(_classes), depth=depth).to(device)

    optimizer = optim.Adam(model.parameters(), lr=start_lr)
    criterion = nn.CrossEntropyLoss().to(device)

    lr_finder = LRFinder(model, optimizer, criterion, device)
    lrs, losses = lr_finder.range_test(train_loader, end_lr=end_lr, num_iter=num_iter)

    plot_lr_finder(lrs, losses)


def plot_lr_finder(lrs, losses, skip_start=5, skip_end=5):
    if skip_end == 0:
        lrs = lrs[skip_start:]
        losses = losses[skip_start:]
    else:
        lrs = lrs[skip_start:-skip_end]
        losses = losses[skip_start:-skip_end]

    fig = plt.figure(figsize=(16, 8))
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(lrs, losses)
    ax.set_xscale('log')
    ax.set_xlabel('Learning rate')
    ax.set_ylabel('Loss')
    ax.grid(True, 'both', 'x')
    plt.show()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--train_data_dir', type=str, required=True, help='Path to training data')
    parser.add_argument('--depth', type=int, default=18, help='ResNet depth (default: 18)')
    parser.add_argument('--start_lr', type=float, default=1e-7, help='Initial learning rate (default: 1e-7)')
    parser.add_argument('--end_lr', type=float, default=10, help='Maximum learning rate (default: 10)')
    parser.add_argument('--num_iter', type=int, default=100, help='Number of iterations (default: 100)')
    opt = parser.parse_args()

    SEED = 1234

    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    find_lr(
        train_dir=opt.train_data_dir,
        depth=opt.depth,
        start_lr=opt.start_lr,
        end_lr=opt.end_lr,
        num_iter=opt.num_iter
    )
