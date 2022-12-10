import torch


def dice_score(y: torch.Tensor, gt: torch.Tensor):
    intersection = (y * gt).sum()
    t1, t2 = y.sum(), gt.sum()
    return (2 * intersection + 1e-5) / (t1 + t2 + 1e-5)

