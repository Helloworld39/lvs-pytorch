import torch.nn as nn


class DiceLoss(nn.Module):
    def __init__(self, weight=1, ep=1e-5):
        super().__init__()
        self.ep = ep
        self.weight = weight

    def forward(self, predict, gt):
        batch_size = predict.size()[0]
        predict = predict.view(batch_size, -1)
        gt = gt.view(batch_size, -1)
        intersection = (predict * gt).sum(1)
        t1, t2 = predict.sum(1), gt.sum(1)
        dice_score = (2 * intersection + self.ep) / (t1 + t2 + self.ep)
        dice_loss = self.weight * (1 - dice_score)
        return dice_loss
