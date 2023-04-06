import tensorflow as tf
from keras import layers
from vit_keras import vit
import os
import pandas as pd
import numpy as np
import cv2
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

def ViT():
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

    # load data
    train_x, test_x, train_y, test_y = train_test_split(image_list, label_list)

    # Normalize pixel values to [0, 1]
    train_x = train_x.astype('float32') / 255.0
    test_x = test_x.astype('float32') / 255.0

    # Define the ViT model
    vit_model = vit.vit_b16(
        image_size=128,
        activation='softmax',
        pretrained=True,
        include_top=True,
        pretrained_top=False,
        classes=5
    )

    # Model summary
    # vit_model.summary()

    # Compile the model
    vit_model.compile(
        optimizer=tf.keras.optimizers.Adam(),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    # Train the model
    history = vit_model.fit(train_x, train_y, batch_size=10, epochs=1, validation_data=(test_x, test_y))

    # Evaluate the model
    vit_model.evaluate(test_x, test_y)

    # plot the loss curve
    plt.plot(history.history['loss'], label='train')
    plt.plot(history.history['val_loss'], label='validation')
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend()
    plt.show()