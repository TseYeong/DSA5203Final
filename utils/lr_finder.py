import torch
from torch.optim.lr_scheduler import _LRScheduler


class LRFinder:
    """
    Learning Rate Finder class to perform range test and find optimal learning rate.
    """
    def __init__(self, model, optimizer, criterion, device):
        self.optimizer = optimizer
        self.model = model
        self.criterion = criterion
        self.device = device

        torch.save(model.state_dict(), 'init_params.pt')

    def range_test(self, iterator, end_lr=10, num_iter=100, smooth_f=0.05, diverge_th=5):
        """
        Perform the learning rate range test.

        Args:
            iterator (DataLoader): Training data loader.
            end_lr (float): Maximum learning rate.
            num_iter (int): Number of iterations.
            smooth_f (float): Smoothing factor for loss.
            diverge_th (float): Threshold to stop if loss diverges.

        Returns:
            lrs (list): Learning rates used.
            losses (list): Corresponding smoothed losses.

        """
        lrs = []
        losses = []
        best_loss = float('inf')

        lr_scheduler = ExponentialLR(self.optimizer, end_lr, num_iter)
        iterator = IteratorWrapper(iterator)

        for iteration in range(num_iter):
            loss = self._train_batch(iterator)
            lr_scheduler.step()
            lrs.append(lr_scheduler.get_lr()[0])

            if iteration > 0:
                loss = smooth_f * loss + (1 - smooth_f) * losses[-1]

            if loss < best_loss:
                best_loss = loss

            losses.append(loss)

            if loss > diverge_th * best_loss:
                print("Stopping early, the loss has diverged")
                break

        self.model.load_state_dict(torch.load('init_params.pt'))

        return lrs, losses

    def _train_batch(self, iterator):
        self.model.train()
        self.optimizer.zero_grad()
        x, y = iterator.get_batch()
        x, y = x.to(self.device), y.to(self.device)
        y_pred, _ = self.model(x)
        loss = self.criterion(y_pred, y)
        loss.backward()
        self.optimizer.step()
        return loss.item()


class ExponentialLR(_LRScheduler):
    """
    Exponentially increases the learning rate between base_lr and end_lr.
    """
    def __init__(self, optimizer, end_lr, num_iter, last_epoch=-1):
        self.end_lr = end_lr
        self.num_iter = num_iter
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        curr_iter = self.last_epoch + 1
        r = curr_iter / self.num_iter
        return [base_lr * (self.end_lr / base_lr) ** r for base_lr in self.base_lrs]


class IteratorWrapper:
    """
    Wrapper to allow endless iteration over a DataLoader.
    """
    def __init__(self, iterator):
        self.iterator = iterator
        self._iterator = iter(iterator)

    def __next__(self):
        try:
            inputs, labels = next(self._iterator)
        except StopIteration:
            self._iterator = iter(self.iterator)
            inputs, labels = next(self._iterator)
        return inputs, labels

    def get_batch(self):
        return next(self)
