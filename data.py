import math
import os
from typing import Union

import torch
import torchvision.transforms
import cv2 as cv
import numpy as np
import SimpleITK as sitk
from torch.utils.data import TensorDataset, DataLoader


def confirm_file_suffix(filename, suffix) -> bool:
    # 限制文件的后缀名
    for suf in suffix:
        if filename[-len(suf):] == suf:
            return True
    return False


def ct_resampling(ct_mat: sitk.Image):
    outsize = [0, 0, 0]
    input_size = ct_mat.GetSize()
    input_spacing = ct_mat.GetSpacing()
    transform = sitk.Transform()
    transform.SetIdentity()

    outsize[0] = round(input_size[0] * input_spacing[0] / 1.0)
    outsize[1] = round(input_size[1] * input_spacing[1] / 1.0)
    outsize[2] = round(input_size[2] * input_spacing[2] / 1.0)

    resampler = sitk.ResampleImageFilter()
    resampler.SetTransform(transform)
    resampler.SetInterpolator(sitk.sitkLinear)
    resampler.SetOutputOrigin(ct_mat.GetOrigin())
    resampler.SetOutputSpacing([1.0, 1.0, 1.0])
    resampler.SetOutputDirection(ct_mat.GetDirection())
    resampler.SetSize(outsize)
    return resampler.Execute(ct_mat)


def read_ct(src_dir, need_resample=False) -> np.ndarray:
    """
    读取CT图像的方法，
    :param src_dir:
    :param need_resample:
    :return:
    """
    if not confirm_file_suffix(os.path.basename(src_dir), ['nii', 'nii.gz']):
        print('应为nii文件')
        exit(-1)
    ct = sitk.ReadImage(src_dir)
    if need_resample:
        ct = ct_resampling(ct)
    ct_mat = sitk.GetArrayFromImage(ct)
    return ct_mat


def get_projection(mat, dim, angle) -> np.ndarray:
    rot_val = angle * math.pi / 180
    if dim == 1:
        mat = np.transpose(mat, (1, 0, 2))
    elif dim == 2:
        mat = np.transpose(mat, (2, 0, 1))
    rot = cv.getRotationMatrix2D((mat.shape[2] // 2, mat.shape[1] // 2), angle, 1.0)
    new_h = int(mat.shape[1] * abs(math.cos(rot_val)) + mat.shape[2] * abs(math.sin(rot_val)))
    new_w = int(mat.shape[1] * abs(math.sin(rot_val)) + mat.shape[2] * abs(math.cos(rot_val)))
    rot[0, 2] += (new_h - mat.shape[2]) // 2
    rot[1, 2] += (new_w - mat.shape[1]) // 2

    projection_arr = []
    for arrow_slice in mat:
        arrow_slice = cv.warpAffine(arrow_slice, rot, (new_w, new_h), borderMode=cv.BORDER_CONSTANT, borderValue=0)
        projection_arr.append(arrow_slice.max(0))
    projection_arr = np.array(projection_arr)
    projection_arr = cv.resize(projection_arr, (512, 512))
    return projection_arr


def normalization(ct_mat: np.ndarray, need_adjust=False) -> np.ndarray:
    if need_adjust:
        ct_mat[ct_mat < -160] = -160
        ct_mat[ct_mat > 240] = 240
    cv.normalize(ct_mat, ct_mat, 0, 255, cv.NORM_MINMAX)
    ct_mat = ct_mat.astype(np.uint8)
    return ct_mat


def get_ct_index(dataset_name: str) -> list:
    if dataset_name == 'msdc':
        return [1, 50, 116, 159, 200, 250, 297, 346, 383, 420, 456, 492, 537, 576, 619, 675, 734, 786, 835, 875, 918,
                957, 1065, 1125, 1263, 1308, 1352, 1397, 1438, 1524, 1550, 1591, 1625, 1655, 1697, 1735, 1768, 1805,
                1847, 1887, 1926, 1975, 2015, 2055, 2104, 2141, 2185, 2222, 2261, 2308, 2400, 2445, 2499, 2549, 2603,
                2648, 2684, 2722, 2761, 2817, 2864, 2909, 2954, 2990, 3031, 3072, 3226, 3271, 3308, 3354, 3391, 3436,
                3485, 3527, 3576, 3608, 3652, 3689, 3728, 3765, 3818, 3861, 3900, 3955, 4007, 4052, 4133, 4181, 4240,
                4288, 4343, 4387, 4431, 4484, 4601, 4651, 4697, 4841, 4891, 4941, 4989, 5047, 5105, 5156, 5333, 5385,
                5427, 5481, 5527, 5573, 5612, 5648, 5678, 5759, 5820, 5927, 5981, 6132, 6253, 6358, 6390, 6472, 6508,
                6551, 6583, 6636, 6707, 6738, 6784, 6824, 6878, 6980, 7022, 7056, 7094, 7138, 7198, 7228, 7280, 7310,
                7357, 7404, 7499, 7581, 7686, 7774, 7938, 7990, 8164, 8263, 8315, 8362, 8412, 8576, 8613, 8662, 8761,
                8806, 8847, 8889, 8923, 8968, 9008, 9059, 9111, 9149, 9181, 9226, 9330, 9501, 9536, 9570, 9635, 9659,
                9683, 9814, 9960, 10011, 10138, 10201, 10362, 10413, 10454, 10505, 10556, 10704, 10845, 10892, 11050,
                11181, 11216, 11390, 11473, 11618, 11645, 11675, 11715, 11879, 12043, 12069, 12114, 12257, 12301,
                12469, 12510, 12558, 12702, 12740, 12775, 12896, 13037, 13084, 13129, 13187, 13331, 13482, 13525,
                13562, 13614, 13775, 13895, 13936, 13980, 14029, 14203, 14357, 14466, 14521, 14570, 14612, 14753,
                14802, 14901, 15056, 15192, 15222, 15393, 15449, 15473, 15614, 15661, 15711, 15825, 15865, 15909,
                15998, 16149, 16192, 16297, 16377, 16518, 16696, 16734, 16895, 16945, 17126, 17284, 17345, 17387,
                17538, 17585, 17632, 17687, 17740, 17901, 17935, 18066, 18115, 18145, 18293, 18326, 18470, 18519,
                18657, 18791, 18828, 18980, 19021, 19155, 19202, 19229, 19363, 19397, 19427, 19561, 19609, 19757,
                19791, 19846, 19984, 20036, 20133, 20182, 20230, 20358, 20445, 20613, 20660, 20710, 20760, 20791,
                20838, 20952, 21121]
    elif dataset_name == 'company':
        return [1, 418, 712, 842, 1185, 1475, 1764, 2114, 2364, 2634, 2869, 2996, 3269, 3424, 3641, 4366, 4649, 4948,
                5379, 5875, 6180, 6555, 6780, 6998, 7219, 7448, 7695, 8898, 9274, 9731, 9936, 10252, 10489, 10890,
                11306, 11578, 11903, 12462, 12798, 13019, 13357, 13574, 13881, 13998, 14259, 14566, 14923, 15178,
                15345, 15746, 15991, 16224, 16354, 16579, 17155, 17440, 17770, 18103, 18430, 18611, 18723, 18940,
                19177, 19515, 19725, 20090, 20384, 20710, 21003, 21228, 21453, 21748, 21868, 22205, 22520, 22794,
                23087, 23508, 23620, 23725, 23955, 24195, 24642, 25085, 25558, 25713, 26053, 26327, 26704, 26984,
                27300, 27533, 27838, 27965, 28289]
    elif dataset_name == '3dircadb':
        return [1, 130, 302, 502, 593, 732, 867, 1018, 1142, 1253, 1375, 1507, 1767, 1889, 2002, 2127, 2282, 2401,
                2475, 2599, 2824]
    else:
        return []


def dataset_dir_manager(dataset_name, where='autodl', is_root=False) -> Union[str, tuple]:
    dataset_root = ''
    if dataset_name == 'msdc':
        if where == 'local':
            dataset_root = 'D:/Github/MSDC'
        else:
            dataset_root = '/root/autodl-tmp/msdc'
    elif dataset_name == '3dircadb':
        if where == 'local':
            dataset_root = 'D:/Github/3Dircadb1'
        else:
            dataset_root = '/root/autodl-tmp/3dircadb'
    else:
        print('没有该数据集', dataset_name)
        exit(-1)

    dataset_x_dir = os.path.join(dataset_root, 'images')
    dataset_y_dir = os.path.join(dataset_root, 'labels')

    if is_root:
        return dataset_root
    else:
        return dataset_x_dir, dataset_y_dir


def read_image_to_tensor(image_dir: str) -> torch.Tensor:
    mat = cv.imread(image_dir, cv.IMREAD_GRAYSCALE)
    to_tensor = torchvision.transforms.ToTensor()
    tsr = to_tensor(mat)
    return tsr


def create_4d_tensor_dataset(dataset_name, data_dir, start, end, **kwargs):
    """
    待优化的方法
    具体问题在于使用torch.cat()消耗的内存过高，使用的内存是数据所占内存的将近4倍，需要优化
    """
    input_type = 0
    memory_threshold = 1000
    if 'input_type' in kwargs:
        input_type = kwargs['input_type']
        if type(input_type) is not int:
            input_type = 0
    if 'memory_adjust' in kwargs:
        memory_threshold = kwargs['memory_adjust']
        if type(memory_threshold) is not int:
            memory_threshold = 1000
    x_dir, y_dir = data_dir
    x_list, y_list = [], []
    x_mat, y_mat = [], []
    for i in range(start, end):
        print('Slice: ', i, '/', end, end='\r')
        image_x_dir = os.path.join(x_dir, str(i)+'.png')
        image_y_dir = os.path.join(y_dir, str(i)+'.png')
        if input_type == 1:
            temp_x_dir = os.path.join(x_dir, 'image_pj', str(i)+'.png')
            x_arr = torch.cat([read_image_to_tensor(image_x_dir), read_image_to_tensor(temp_x_dir)], dim=0)
            y_arr = read_image_to_tensor(image_y_dir)
        elif input_type == 2:
            temp_x_dir = os.path.join(x_dir, 'image_pj_16', str(i) + '.png')
            x_arr = torch.cat([read_image_to_tensor(image_x_dir), read_image_to_tensor(temp_x_dir)], dim=0)
            y_arr = read_image_to_tensor(image_y_dir)
        else:
            x_arr = read_image_to_tensor(image_x_dir)
            y_arr = read_image_to_tensor(image_y_dir)
        x_list.append(x_arr)
        y_list.append(y_arr)
        if (i % memory_threshold) == 0 or i == (end-1):
            x_list = torch.stack(x_list, dim=0)
            x_mat.append(x_list)
            y_list = torch.stack(y_list, dim=0)
            y_mat.append(y_list)
            x_list, y_list = [], []
    x_mat, y_mat = torch.cat(x_mat, dim=0), torch.cat(y_mat, dim=0)

    dataset = TensorDataset(x_mat, y_mat)
    torch.save(dataset, dataset_name)
    print('已生成数据集', dataset_name)


def create_5d_tensor_dataset(dataset_name, data_dir, patch_size, start, end):
    """
    生成3D数据集，有待重写
    :param dataset_name:
    :param data_dir:
    :param patch_size:
    :param start:
    :param end:
    :return:
    """
    x_dir, y_dir = data_dir
    i, j = 0, start
    x_mat, y_mat = [], []
    x_patch, y_patch = [], []
    first_read = True
    while j < end:
        if i == 0:
            x_patch, y_patch = [], []
        image_x_dir = os.path.join(x_dir, str(j)+'.png')
        image_y_dir = os.path.join(y_dir, str(j)+'.png')
        x_patch.append(torch.unsqueeze(read_image_to_tensor(image_x_dir), 0))
        y_patch.append(torch.unsqueeze(read_image_to_tensor(image_y_dir), 0))

        i += 1
        if i == patch_size:
            i = 0
            x_patch = torch.cat(x_patch, dim=1)
            y_patch = torch.cat(y_patch, dim=1)
            if first_read:
                x_mat, y_mat = torch.unsqueeze(x_patch, dim=0), torch.unsqueeze(y_patch, dim=0)
                first_read = False
            else:
                x_mat = torch.cat([x_mat, torch.unsqueeze(x_patch, dim=0)], dim=0)
                y_mat = torch.cat([y_mat, torch.unsqueeze(y_patch, dim=0)], dim=0)

        j += 1
    print('剩余', i, '个切片未放入数据集')
    while i < patch_size:
        x_patch.append(torch.zeros((1, 1, 512, 512)))
        y_patch.append(torch.zeros((1, 1, 512, 512)))
        i += 1
    x_patch = torch.cat(x_patch, dim=1)
    y_patch = torch.cat(y_patch, dim=1)
    if first_read:
        x_mat, y_mat = torch.unsqueeze(x_patch, dim=0), torch.unsqueeze(y_patch, dim=0)
        first_read = False
    else:
        x_mat = torch.cat([x_mat, torch.unsqueeze(x_patch, dim=0)], dim=0)
        y_mat = torch.cat([y_mat, torch.unsqueeze(y_patch, dim=0)], dim=0)

    dataset = TensorDataset(x_mat, y_mat)
    torch.save(dataset, dataset_name)
    print('已生成数据集', dataset_name)


def data_loader(dataset_dir: str, batch_size=16, is_shuffled=False) -> ():
    def f():
        dataset = torch.load(dataset_dir)
        if is_shuffled:
            return DataLoader(dataset, batch_size, True, drop_last=True)
        else:
            return DataLoader(dataset, batch_size, False)
    return f


def create_compare_map(predict_dir, target_dir, dst_dir):
    if not os.path.exists(dst_dir):
        os.makedirs(dst_dir)
    for filename in os.listdir(predict_dir):
        pre = cv.imread(os.path.join(predict_dir, filename), cv.IMREAD_GRAYSCALE)
        tar = cv.imread(os.path.join(target_dir, filename), cv.IMREAD_GRAYSCALE)
        pre[pre != 0] = 1
        tar[tar != 0] = 2
        inter = pre + tar

        inter[inter == 1] = 70
        inter[inter == 2] = 200
        inter[inter == 3] = 128
        inter = cv.applyColorMap(inter, cv.COLORMAP_TURBO)

        cv.imwrite(os.path.join(dst_dir, filename), inter)
        print('已完成图片：', filename)


if __name__ == '__main__':
    create_compare_map('./out/unet_msdc_150', 'D:/GitHub/MSDC/labels', 'D:/GitHub/MSDC/compare/unet_nsdc_150')
