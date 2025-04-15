import torch
import torch.nn as nn
from tqdm import tqdm
from model.resnet import build_resnet
from model.efficientnet import build_efficientnet
from utils.dataloader import get_test_loader
from utils.eval import calculate_topk_accuracy


def test(test_dir, model_path, model, depth=18, dropout=0.0):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    test_loader, classes = get_test_loader(test_dir)

    if model == 'efficientnet':
        model = build_efficientnet(len(classes)).to(device)
    else:
        model = build_resnet(len(classes), depth=depth, dropout=dropout).to(device)

    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    criterion = nn.CrossEntropyLoss.to(device)

    total_loss = 0.0
    total_acc = 0.0

    with torch.no_grad():
        for (images, labels) in tqdm(test_loader, desc='Testing'):
            images, labels = images.to(device), labels.to(device)
            outputs, _ = model(images)
            loss = criterion(outputs, labels)
            acc_1, _ = calculate_topk_accuracy(outputs, labels)
            total_acc += acc_1.item()
            total_loss += loss.item()

    acc = total_acc / len(test_loader)
    loss = total_loss / len(test_loader)

    return acc, loss
