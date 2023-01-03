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
    import cv2 as cv
    import os.path

    dataset_name = 'msdc'
    dataset_root_dir = data.dataset_dir_manager(dataset_name, 'local', True)
    index_list = data.get_ct_index('msdc')[-6:]
    predict_dir = './out/unet_msdc_150'
    target_dir = os.path.join(dataset_root_dir, 'labels')

    total_dice = 0
    for i in range(5):
        slice_num = index_list[i]
        true_pos, total_pos = 0, 0
        while slice_num < index_list[i+1]:
            pre = cv.imread(os.path.join(predict_dir, str(slice_num)+'.png'), cv.IMREAD_GRAYSCALE)
            tar = cv.imread(os.path.join(target_dir, str(slice_num)+'.png'), cv.IMREAD_GRAYSCALE)
            pre[pre != 0] = 1
            tar[tar != 0] = 1

            inter = (pre * tar).sum()
            t_num = tar.sum() + pre.sum()

            total_dice += (2 * inter + 1e-5) / (t_num + 1e-5)
            true_pos += inter
            total_pos += t_num
            slice_num += 1
        dice = (2 * true_pos + 1e-5) / (total_pos + 1e-5)
        print('CT Dice No.', i, ':', dice)

    print('Slice Avg Dice:', total_dice / (index_list[-1] - index_list[0]))
