import SimpleITK as sitk
import numpy as np
import cv2 as cv
import os


src_dir = 'D:/GitHub/Company/source'
src_img_dir = src_dir + '/images'
src_lab_dir = src_dir + '/portalvein'

out_index_start = 1
out_index_start_r = 1
dst_img_dir = 'D:/GitHub/Company/images'
dst_lab_dir = 'D:/GitHub/Company/labels'

for i in range(2):
    dcm_dir = os.path.join(src_img_dir, str(i+1))
    if not os.path.exists(dcm_dir):
        continue
    dcm_list = os.listdir(dcm_dir)
    for dcm in dcm_list:
        ct_slice = sitk.ReadImage(os.path.join(dcm_dir, dcm))
        ct_slice = sitk.GetArrayFromImage(ct_slice)
        ct_slice[ct_slice < -160] = -160
        ct_slice[ct_slice > 240] = 240
        cv.normalize(ct_slice, ct_slice, 0, 255, cv.NORM_MINMAX)
        ct_slice = ct_slice.astype(np.uint8)

        cv.imwrite(dst_img_dir+"/"+str(out_index_start)+'.png', ct_slice[0])
        out_index_start += 1

    ct_masks = sitk.ReadImage(os.path.join(src_lab_dir, str(i+1)+'.nii'))
    print(ct_masks.GetOrigin())
    ct_masks = sitk.GetArrayFromImage(ct_masks)
    cv.normalize(ct_masks, ct_masks, 0, 255, cv.NORM_MINMAX)
    for mask in ct_masks:
        cv.imwrite(dst_lab_dir+'/'+str(out_index_start_r)+'.png', mask)
        out_index_start_r += 1
