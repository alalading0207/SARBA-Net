# Reduce 100 original labels to binary labels (and 4-category labels).

from PIL import Image
import numpy as np
import os


lbl_path = './whu-opt-sar/lbl/'
file_name = []

for root, dirs, files in os.walk(lbl_path):
    for file in files:
        if file.lower().endswith(('.tif', '.tiff')):
            file_name.append(file)


# Define the category mapping relationship
# class_mapping = {0: 0, 10: 0, 20: 255, 30: 255, 40: 0, 50: 0, 60: 255, 70: 0}    # 2 classes for training
class_mapping = {0: 0, 10: 0, 20: 50, 30: 100, 40: 0, 50: 0, 60: 150, 70: 0}     # 4 classes for visualization
# lbl_path_change = './whu-opt-sar/lbl_2/'
lbl_path_change = './whu-opt-sar/lbl_4/'


for i in range(len(file_name)):
    name = lbl_path + file_name[i]
    image = Image.open(name)
    image_array = np.array(image)

    # Go through every pixel
    for j in range(image_array.shape[0]):
        for k in range(image_array.shape[1]):
            
            if image_array[j, k] in class_mapping:
                image_array[j, k] = class_mapping[image_array[j, k]]
            else:
                image_array[j, k] = 0


    modified_image = Image.fromarray(image_array)

    # save
    modified_name = lbl_path_change + file_name[i]
    modified_image.save(modified_name)

    print(f"The image {file_name[i]} has been processed and saved as {modified_name}")
    print(f" {i} /100 ")



