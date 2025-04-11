import torch


def calculate_topk_accuracy(y_pred, y, k=5):
    """
    Computes top-1 and top-k accuracy.

    Args:
        y_pred (Tensor): Logits from the model [B, C].
        y (Tensor): Ground truth labels [B].
        k (int): The maximum top-k to compute.

    Returns:
        Tuple[Tensor, Tensor]: top-1 accuracy, top-k accuracy
    """
    with torch.no_grad():
        batch_size = y.size(0)
        _, top_pred = y_pred.topk(k, dim=1)
        top_pred = top_pred.t()
        correct = top_pred.eq(y.view(1, -1).expand_as(top_pred))
        correct_1 = correct[:1].reshape(-1).float().sum(0, keepdim=True)
        correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
        acc_1 = correct_1 / batch_size
        acc_k = correct_k / batch_size
    return acc_1, acc_k
