import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Flatten, Conv2D, MaxPooling2D
from sklearn.model_selection import train_test_split
import pandas as pd
import os
import numpy as np
import cv2

def VGG19():
    config = tf.compat.v1.ConfigProto()
    config.gpu_options.allow_growth = True
    session = tf.compat.v1.InteractiveSession(config=config)
    os.environ['CUDA_VISIBLE_DEVICES'] = '/device:GPU:0'

    image_path = "./Datasets/cassava-leaf-disease-classification/processed_images"
    # image_path = "./Datasets/cassava-leaf-disease-classification/train_images"

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
        Conv2D(256, (3, 3), activation='relu', padding='same'),
        MaxPooling2D((2, 2)),
        Conv2D(512, (3, 3), activation='relu', padding='same'),
        Conv2D(512, (3, 3), activation='relu', padding='same'),
        Conv2D(512, (3, 3), activation='relu', padding='same'),
        Conv2D(512, (3, 3), activation='relu', padding='same'),
        MaxPooling2D((2, 2)),
        Conv2D(512, (3, 3), activation='relu', padding='same'),
        Conv2D(512, (3, 3), activation='relu', padding='same'),
        Conv2D(512, (3, 3), activation='relu', padding='same'),
        Conv2D(512, (3, 3), activation='relu', padding='same'),
        MaxPooling2D((2, 2)),
        Flatten(),
        Dense(4096, activation='relu'),
        Dense(4096, activation='relu'),
        Dense(5, activation='softmax')
    ])

    # Compile the model
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    # Train the model
    model.fit(x_train, y_train, batch_size=10, epochs=300, validation_data=(x_test, y_test))


