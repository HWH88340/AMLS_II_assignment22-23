import cv2
import os
import numpy as np

def preprocess():
    # Set the path to the folder containing the images
    input_folder_path = './Datasets/cassava-leaf-disease-classification/train_images'

    # Set the path to the folder where you want to save the processed images
    output_folder_path = './Datasets/cassava-leaf-disease-classification/processed_images'


    def ResizePadding(image, fixed_side):
        height, width = image.shape[0], image.shape[1]
        scale = max(width, height) / float(fixed_side)  # Get the scale of image
        width_new, height_new = int(width / scale), int(height / scale)
        resize_image = cv2.resize(image, (width_new, height_new))  # Zoom in and out

        # Calculate the length of image
        if width_new % 2 != 0 and height_new % 2 == 0:
            top, bottom, left, right = (fixed_side - height_new) // 2, (fixed_side - height_new) // 2, (
                        fixed_side - width_new) // 2 + 1, (
                                               fixed_side - width_new) // 2
        elif width_new % 2 == 0 and height_new % 2 != 0:
            top, bottom, left, right = (fixed_side - height_new) // 2 + 1, (fixed_side - height_new) // 2, (
                        fixed_side - width_new) // 2, (
                                               fixed_side - width_new) // 2
        elif width_new % 2 == 0 and height_new % 2 == 0:
            top, bottom, left, right = (fixed_side - height_new) // 2, (fixed_side - height_new) // 2, (fixed_side - width_new) // 2, (
                    fixed_side - width_new) // 2
        else:
            top, bottom, left, right = (fixed_side - height_new) // 2 + 1, (fixed_side - height_new) // 2, (
                        fixed_side - width_new) // 2 + 1, (
                                               fixed_side - width_new) // 2

        # Padding the images
        pad_image = cv2.copyMakeBorder(resize_image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=[0, 0, 0])

        return pad_image


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
        image = cv2.imread(os.path.join(input_folder_path, filename))

        new_image = ResizePadding(image, fixed_side=128)

        # Save the new image to the output folder
        output_path = os.path.join(output_folder_path, filename)
        cv2.imwrite(output_path, new_image)