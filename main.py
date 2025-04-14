import random
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
    parser.add_argument('--model_path', default='./saved_models/resnet18.pth',
                        help='Path to save/load the model checkpoint.')
    parser.add_argument('--epochs', type=int, default=15, help='Number of training epochs')
    parser.add_argument('--depth', type=int, default=18, help="ResNet depth")
    parser.add_argument('--dropout', type=float, default=0.0, help="Dropout rate")
    args = parser.parse_args()

    if args.phase == 'train':
        acc = train(args.train_data_dir, args.model_path,
                    depth=args.depth, epochs=args.epochs, dropout=args.dropout)
        print(f"Train acc: {acc:.4f}")
    else:
        acc = test(args.test_data_dir, args.model_path, depth=args.depth, dropout=args.dropout)
        print(f"Test acc: {acc:.4f}")


if __name__ == "__main__":
    main()
