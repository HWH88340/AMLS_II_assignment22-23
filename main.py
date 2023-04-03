import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Flatten, Conv2D, MaxPooling2D
from sklearn.model_selection import train_test_split
import pandas as pd
import os
import numpy as np
import cv2

config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.compat.v1.InteractiveSession(config=config)
os.environ['CUDA_VISIBLE_DEVICES'] = '/device:GPU:0'

image_path = "./Datasets/cassava-leaf-disease-classification/processed_images"
datanames = os.listdir(image_path)
csv_data = pd.read_csv("./Datasets/cassava-leaf-disease-classification/train.csv")


# connect filename with label
def filename2label(datanames, csv_data, image_path):
    label_list = []
    image_list = []

    for file in datanames:
        digit_str_name = "".join(list(filter(str.isdigit, str(file))))
        digit_str_name = digit_str_name + ".jpg"
        row = csv_data.loc[csv_data['image_id'] == digit_str_name]
        label = row.iat[0, 1]
        label_list.append(label)

        single_image_path = image_path + "/" + file
        image = cv2.imread(single_image_path)
        image_list.append(image)
    return np.array(label_list), np.array(image_list)


label_list, image_list = filename2label(datanames, csv_data, image_path)
label_list = label_list.reshape(-1, 1)

# Load the CIFAR-10 dataset
x_train, x_test, y_train, y_test = train_test_split(image_list, label_list)



# Normalize the input data
x_train = x_train / 255.0
x_test = x_test / 255.0

# Define the VGG16 model
model = Sequential([
    Conv2D(64, (3, 3), activation='relu', padding='same', input_shape=(128, 128, 3)),
    Conv2D(64, (3, 3), activation='relu', padding='same'),
    MaxPooling2D((2, 2)),
    Conv2D(128, (3, 3), activation='relu', padding='same'),
    Conv2D(128, (3, 3), activation='relu', padding='same'),
    MaxPooling2D((2, 2)),
    Conv2D(256, (3, 3), activation='relu', padding='same'),
    Conv2D(256, (3, 3), activation='relu', padding='same'),
    Conv2D(256, (3, 3), activation='relu', padding='same'),
    MaxPooling2D((2, 2)),
    Conv2D(512, (3, 3), activation='relu', padding='same'),
    Conv2D(512, (3, 3), activation='relu', padding='same'),
    Conv2D(512, (3, 3), activation='relu', padding='same'),
    MaxPooling2D((2, 2)),
    Conv2D(512, (3, 3), activation='relu', padding='same'),
    Conv2D(512, (3, 3), activation='relu', padding='same'),
    Conv2D(512, (3, 3), activation='relu', padding='same'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(4096, activation='relu'),
    Dense(4096, activation='relu'),
    Dense(100, activation='softmax')
])

# Compile the model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Train the model
model.fit(x_train, y_train, epochs=300, validation_data=(x_test, y_test))

######
### 需要从dataframe里面，取出叶子的名字，并给它们排序，把序号存到label_list里面



#
# import cv2
# import os
# import numpy as np
#
# # Set the path to the folder containing the images
# input_folder_path = './Datasets/cassava-leaf-disease-classification/train_images'
#
# # Set the path to the folder where you want to save the processed images
# output_folder_path = './Datasets/cassava-leaf-disease-classification/processed_images'
#
#
# def ResizePadding(img, fixed_side):
#     h, w = img.shape[0], img.shape[1]
#     scale = max(w, h) / float(fixed_side)  # 获取缩放比例
#     new_w, new_h = int(w / scale), int(h / scale)
#     resize_img = cv2.resize(img, (new_w, new_h))  # 按比例缩放
#
#     # 计算需要填充的像素长度
#     if new_w % 2 != 0 and new_h % 2 == 0:
#         top, bottom, left, right = (fixed_side - new_h) // 2, (fixed_side - new_h) // 2, (
#                     fixed_side - new_w) // 2 + 1, (
#                                            fixed_side - new_w) // 2
#     elif new_w % 2 == 0 and new_h % 2 != 0:
#         top, bottom, left, right = (fixed_side - new_h) // 2 + 1, (fixed_side - new_h) // 2, (
#                     fixed_side - new_w) // 2, (
#                                            fixed_side - new_w) // 2
#     elif new_w % 2 == 0 and new_h % 2 == 0:
#         top, bottom, left, right = (fixed_side - new_h) // 2, (fixed_side - new_h) // 2, (fixed_side - new_w) // 2, (
#                 fixed_side - new_w) // 2
#     else:
#         top, bottom, left, right = (fixed_side - new_h) // 2 + 1, (fixed_side - new_h) // 2, (
#                     fixed_side - new_w) // 2 + 1, (
#                                            fixed_side - new_w) // 2
#
#     # 填充图像
#     pad_img = cv2.copyMakeBorder(resize_img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=[0, 0, 0])
#
#     return pad_img
#
#
# # Check if the directory already exists
# if not os.path.exists(output_folder_path):
#     # Use os.makedirs to create the new directory
#     os.makedirs(output_folder_path)
#     print(f"Created directory: {output_folder_path}")
# else:
#     print(f"Directory already exists: {output_folder_path}")
#
# # Loop over all the files in the input folder
# for filename in os.listdir(input_folder_path):
#     # Load the image
#     img = cv2.imread(os.path.join(input_folder_path, filename))
#
#     new_img = ResizePadding(img, fixed_side=128)
#
#     # Save the new image to the output folder
#     output_path = os.path.join(output_folder_path, filename)
#     cv2.imwrite(output_path, new_img)