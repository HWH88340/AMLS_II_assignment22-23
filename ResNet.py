import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Flatten, Conv2D, MaxPooling2D
from sklearn.model_selection import train_test_split
from keras.models import Model
from keras.layers import Input, Dense, Conv2D, MaxPooling2D, BatchNormalization, Dropout, Add, Activation, AveragePooling2D, Flatten
import pandas as pd
import os
import numpy as np
import cv2
import matplotlib.pyplot as plt

def res_block(x, filters, repetitions):
    shortcut = x
    for i in range(repetitions):
        if i == 0:
            x = Conv2D(filters, (3, 3), padding='same')(x)
        else:
            x = Conv2D(filters, (3, 3), padding='same')(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = Conv2D(filters, (3, 3), padding='same')(x)
        x = BatchNormalization()(x)
        shortcut = Conv2D(filters, (1, 1), padding='same')(shortcut)
        shortcut = BatchNormalization()(shortcut)
        x = Add()([x, shortcut])
        x = Activation('relu')(x)
        shortcut = x
    return x

def ResNet():
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

    # Identify the training set and testing set
    x_train, x_test, y_train, y_test = train_test_split(image_list, label_list)

    # Normalize the input data
    x_train = x_train / 255.0
    x_test = x_test / 255.0

    # Define the ResNet-18 model
    input_tensor = Input(shape=(128, 128, 3))
    x = Conv2D(64, (7, 7), strides=(2, 2), padding='same')(input_tensor)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same')(x)

    x = res_block(x, 64, 2)
    x = res_block(x, 128, 2)
    x = res_block(x, 256, 2)
    x = res_block(x, 512, 2)

    x = AveragePooling2D(pool_size=(4, 4))(x)
    x = Flatten()(x)
    output_tensor = Dense(5, activation='softmax')(x)

    model = Model(inputs=input_tensor, outputs=output_tensor)

    # model.summary()

    # Compile the model
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    # Train the model
    history = model.fit(x_train, y_train, batch_size=10, epochs=1, validation_data=(x_test, y_test))

    # plot the loss curve
    plt.plot(history.history['loss'], label='train')
    plt.plot(history.history['val_loss'], label='validation')
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend()
    plt.show()




