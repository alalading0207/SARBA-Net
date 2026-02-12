# Crop into patches of 256*256

import numpy as np
import os
from PIL import Image
import cv2


CropSize = 256
overpixel_heigh = 26   # Calculate the overlap in advance
overpixel_width = 25


def Opentif(fn):
    tif = Image.open(fn)
    tif_array = np.array(tif)
    return tif_array


if __name__ == '__main__':

    # crop sar
    img_dir = './whu-opt-sar/sar/'
    img_crop_dir = './whu-opt-sar/sar_crop/'

    # crop label
    lbl_dir = './whu-opt-sar/lbl_4/'
    lbl_crop_dir = './whu-opt-sar/lbl_4_crop/'

    file_name = [] 
    for root, dirs, files in os.walk(lbl_dir):
        for file in files:
            if file.lower().endswith(('.tif', '.tiff')):
                file_name.append(file)

    for k in range(len(file_name)):
        img = Opentif(img_dir + file_name[k]) 
        lbl = Opentif(lbl_dir + file_name[k])
        img_heigh, img_width = lbl.shape
                
        for row in range(16):
            start_y = max(0, row*(256-overpixel_heigh))
            end_y = start_y + 256
            if row == 15:   
                start_y = img_heigh - 256
                end_y = img_heigh

            for colum in range(24):   
                start_x = max(0, colum*(256-overpixel_width))
                end_x = start_x + 256
                if colum == 23:  
                    start_x = img_width - 256
                    end_x = img_width

                # crop
                img_patch = img[start_y:end_y, start_x:end_x]
                lbl_patch = lbl[start_y:end_y, start_x:end_x]

                # write
                file_name_without = file_name[k].rsplit(".", 1)[0]
                patch_name = f"{file_name_without}_{row}_{colum}.tif"

                img_crop_path = img_crop_dir +  patch_name
                lbl_crop_path = lbl_crop_dir +  patch_name

                cv2.imwrite(img_crop_path, img_patch)
                cv2.imwrite(lbl_crop_path, lbl_patch)

        print(f"Te image {file_name[k]} has been successfully cropped.")
        print(f" {k} /100 ")
