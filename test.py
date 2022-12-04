import os.path
import cv2 as cv
import numpy as np

from data import get_ct_index
index_list = get_ct_index('msdc')
list_length = len(index_list)

data_root = 'D:/Github/MSDC'
x_dir, y_dir = data_root+'/images', data_root+'/labels'
x_pj_dir, y_pj_dir = data_root+'/image_pj', data_root+'/label_pj'

x_patch, y_patch = [], []

for i in range(list_length-1):
    for j in range(index_list[i], index_list[i+1]):
        x = cv.imread(os.path.join(x_dir, str(j)+'.png'), cv.IMREAD_GRAYSCALE)
        y = cv.imread(os.path.join(y_dir, str(j)+'.png'), cv.IMREAD_GRAYSCALE)

        x_patch.append(x)
        y_patch.append(y)

    x_patch = np.array(x_patch)
    y_patch = np.array(y_patch)

    start_index, end_index = index_list[i], index_list[i+1]
    total_slice = x_patch.shape[0]
    i = 0
    patch_size = total_slice
    while True:
        if i + patch_size >= total_slice:
            x_pj = np.max(x_patch[i:, :, :], 0)
            y_pj = np.max(y_patch[i:, :, :], 0)
            while start_index < end_index:
                cv.imwrite(os.path.join(x_pj_dir, str(start_index)+'.png'), x_pj)
                cv.imwrite(os.path.join(y_pj_dir, str(start_index) + '.png'), y_pj)
                start_index += 1
                print(start_index)
            break
        else:
            temp_index = start_index + patch_size
            x_pj = np.max(x_patch[i: i+patch_size, :, :], 0)
            y_pj = np.max(y_patch[i: i+patch_size, :, :], 0)
            while start_index < temp_index:
                cv.imwrite(os.path.join(x_pj_dir, str(start_index)+'.png'), x_pj)
                cv.imwrite(os.path.join(y_pj_dir, str(start_index)+'.png'), y_pj)
                start_index += 1
                print(start_index)
        i += patch_size

    x_patch, y_patch = [], []
