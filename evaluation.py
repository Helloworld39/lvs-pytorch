import torch


def dice_score(y: torch.Tensor, gt: torch.Tensor):
    batch_size = y.size()[0]
    y = y.view(batch_size, -1)
    gt = gt.view(batch_size, -1)

    intersection = (y * gt).sum(1)
    t1, t2 = y.sum(1), gt.sum(1)
    score = (2 * intersection + 1e-5) / (t1 + t2 + 1e-5)
    return score / batch_size
