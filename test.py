import torch
from model.resnet import build_resnet
from utils.dataloader import get_test_loader


def test(test_dir, model_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    test_loader, classes = get_test_loader(test_dir)
    model = build_resnet(len(classes)).to(device)

    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    total_acc = 0.0

    with torch.no_grad():
        for (images, labels) in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs, _ = model(images)
            total_acc += (outputs.argmax(1) == labels).sum().item()

    acc = total_acc / len(test_loader)

    return acc
