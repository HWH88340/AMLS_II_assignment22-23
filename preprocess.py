import cv2
import os
import numpy as np

def preprocess():
    # Set the path to the folder containing the images
    input_folder_path = './Datasets/cassava-leaf-disease-classification/train_images'

    # Set the path to the folder where you want to save the processed images
    output_folder_path = './Datasets/cassava-leaf-disease-classification/processed_images'


    def ResizePadding(img, fixed_side):
        h, w = img.shape[0], img.shape[1]
        scale = max(w, h) / float(fixed_side)  # 获取缩放比例
        new_w, new_h = int(w / scale), int(h / scale)
        resize_img = cv2.resize(img, (new_w, new_h))  # 按比例缩放

        # 计算需要填充的像素长度
        if new_w % 2 != 0 and new_h % 2 == 0:
            top, bottom, left, right = (fixed_side - new_h) // 2, (fixed_side - new_h) // 2, (
                        fixed_side - new_w) // 2 + 1, (
                                               fixed_side - new_w) // 2
        elif new_w % 2 == 0 and new_h % 2 != 0:
            top, bottom, left, right = (fixed_side - new_h) // 2 + 1, (fixed_side - new_h) // 2, (
                        fixed_side - new_w) // 2, (
                                               fixed_side - new_w) // 2
        elif new_w % 2 == 0 and new_h % 2 == 0:
            top, bottom, left, right = (fixed_side - new_h) // 2, (fixed_side - new_h) // 2, (fixed_side - new_w) // 2, (
                    fixed_side - new_w) // 2
        else:
            top, bottom, left, right = (fixed_side - new_h) // 2 + 1, (fixed_side - new_h) // 2, (
                        fixed_side - new_w) // 2 + 1, (
                                               fixed_side - new_w) // 2

        # 填充图像
        pad_img = cv2.copyMakeBorder(resize_img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=[0, 0, 0])

        return pad_img


    # Check if the directory already exists
    if not os.path.exists(output_folder_path):
        # Use os.makedirs to create the new directory
        os.makedirs(output_folder_path)
        print(f"Created directory: {output_folder_path}")
    else:
        print(f"Directory already exists: {output_folder_path}")

    # Loop over all the files in the input folder
    for filename in os.listdir(input_folder_path):
        # Load the image
        img = cv2.imread(os.path.join(input_folder_path, filename))

        new_img = ResizePadding(img, fixed_side=128)

        # Save the new image to the output folder
        output_path = os.path.join(output_folder_path, filename)
        cv2.imwrite(output_path, new_img)