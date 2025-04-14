import random
import re

import numpy as np
import torch
import argparse
from train import train
from test import test


def main():
    SEED = 1234
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    parser = argparse.ArgumentParser()
    parser.add_argument('--phase', default='train', choices=['train', 'test'],
                        help='Choose to train or test the model.')
    parser.add_argument('--train_data_dir', default='./data/train/',
                        help='Path to training data directory.')
    parser.add_argument('--test_data_dir', default='./data/test/',
                        help='Path to testing data directory.')
    parser.add_argument('--model', default='resnet18',
                        choices=['resnet18', 'resnet34', 'resnet50', 'efficientnet'],
                        help='Which model architecture to use.')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--epochs', type=int, default=15, help='Number of training epochs')
    parser.add_argument('--dropout', type=float, default=0.0, help="Dropout rate")
    args = parser.parse_args()

    model_path = './saved_models/' + args.model + '.pth'
    if args.model != 'efficientnet':
        pattern = re.compile(r'(\d+)')
        match = pattern.search(args.model)
        depth = int(match.group(1))
    else:
        depth = 0

    if args.phase == 'train':
        acc = train(args.train_data_dir, model_path, args.model,
                    depth=depth, lr=args.lr, epochs=args.epochs, dropout=args.dropout)
        print(f"Train acc: {acc:.4f}")
    else:
        acc = test(args.test_data_dir, model_path, args.model, depth=depth, dropout=args.dropout)
        print(f"Test acc: {acc:.4f}")


if __name__ == "__main__":
    main()
