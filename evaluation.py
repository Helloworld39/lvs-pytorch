import torch


def dice_score(y: torch.Tensor, gt: torch.Tensor, threshold=0.5):
    batch_size = y.size()[0]
    y = y.view(batch_size, -1)
    gt = gt.view(batch_size, -1)
    y[y < threshold] = 0.0
    y[y >= threshold] = 1.0

    intersection = (y * gt).sum(1)
    t1, t2 = y.sum(1), gt.sum(1)
    score = (2 * intersection + 1e-5) / (t1 + t2 + 1e-5)
    return score.sum(0).item() / batch_size


if __name__ == '__main__':
    import data

    dataset_name = 'msdc'
    dataset_root_dir = data.dataset_dir_manager(dataset_name, 'local', True)
    index_list = data.get_ct_index('msdc')
    predict_dir = './out/unet_msdc_150'

