# import tensorflow as tf
# from keras import layers
# from vit_keras import vit
# import os
# import pandas as pd
# import numpy as np
# import cv2
# from sklearn.model_selection import train_test_split
#
# def ViT():
#     image_path = "./Datasets/cassava-leaf-disease-classification/processed_images"
#     # image_path = "./Datasets/cassava-leaf-disease-classification/train_images"
#
#     datanames = os.listdir(image_path)
#     csv_data = pd.read_csv("./Datasets/cassava-leaf-disease-classification/train.csv")
#
#     # connect filename with label
#     def filename2label(datanames, csv_data, image_path):
#         label_list = []
#         image_list = []
#
#         for file in datanames:
#             digit_str_name = "".join(list(filter(str.isdigit, str(file))))
#             digit_str_name = digit_str_name + ".jpg"
#             row = csv_data.loc[csv_data['image_id'] == digit_str_name]
#             label = row.iat[0, 1]
#             label_list.append(label)
#
#             single_image_path = image_path + "/" + file
#             image = cv2.imread(single_image_path)
#             image_list.append(image)
#         return np.array(label_list), np.array(image_list)
#
#     label_list, image_list = filename2label(datanames, csv_data, image_path)
#     label_list = label_list.reshape(-1, 1)
#
#     # Load CIFAR-10 dataset
#     x_train, x_test, y_train, y_test = train_test_split(image_list, label_list)
#
#     # Normalize pixel values to [0, 1]
#     x_train = x_train.astype('float32') / 255.0
#     x_test = x_test.astype('float32') / 255.0
#
#     # Define the ViT model
#     vit_model = vit.vit_b16(
#         image_size=128,
#         activation='softmax',
#         pretrained=True,
#         include_top=True,
#         pretrained_top=False,
#         classes=5
#     )
#
#     # Compile the model
#     vit_model.compile(
#         optimizer=tf.keras.optimizers.Adam(),
#         loss='sparse_categorical_crossentropy',
#         metrics=['accuracy']
#     )
#
#     # Train the model
#     vit_model.fit(x_train, y_train, batch_size=10, epochs=300, validation_data=(x_test, y_test))
#
#     # Evaluate the model
#     vit_model.evaluate(x_test, y_test)


import numpy as np
import tensorflow as tf
import keras
from keras import layers
import tensorflow_addons as tfa
import matplotlib.pyplot as plt
import cv2
import os
import pandas as pd
from sklearn.model_selection import train_test_split


def ViT():
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


    ## Prepare the data
    num_classes = 5
    input_shape = (128, 128, 3)

    x_train, x_test, y_train, y_test = train_test_split(image_list, label_list)


    print(f"x_train shape: {x_train.shape} - y_train shape: {y_train.shape}")
    print(f"x_test shape: {x_test.shape} - y_test shape: {y_test.shape}")


    ## Configure the hyperparameters
    learning_rate = 0.001
    weight_decay = 0.0001
    batch_size = 10
    num_epochs = 300
    image_size = 72  # We'll resize input images to this size
    patch_size = 6  # Size of the patches to be extract from the input images
    num_patches = (image_size // patch_size) ** 2
    projection_dim = 64
    num_heads = 4
    transformer_units = [
        projection_dim * 2,
        projection_dim,
    ]  # Size of the transformer layers
    transformer_layers = 8
    mlp_head_units = [2048, 1024]  # Size of the dense layers of the final classifier

    ## Use data augmentation
    data_augmentation = keras.Sequential(
        [
            layers.Normalization(),
            layers.Resizing(image_size, image_size),
            layers.RandomFlip("horizontal"),
            layers.RandomRotation(factor=0.02),
            layers.RandomZoom(
                height_factor=0.2, width_factor=0.2
            ),
        ],
        name="data_augmentation",
    )
    # Compute the mean and the variance of the training data for normalization.
    data_augmentation.layers[0].adapt(x_train)


    ## Implement multilayer perceptron (MLP)
    def mlp(x, hidden_units, dropout_rate):
        for units in hidden_units:
            x = layers.Dense(units, activation=tf.nn.gelu)(x)
            x = layers.Dropout(dropout_rate)(x)
        return x

    ## Implement patch creation as a layer
    class Patches(layers.Layer):
        def __init__(self, patch_size):
            super().__init__()
            self.patch_size = patch_size

        def call(self, images):
            batch_size = tf.shape(images)[0]
            patches = tf.image.extract_patches(
                images=images,
                sizes=[1, self.patch_size, self.patch_size, 1],
                strides=[1, self.patch_size, self.patch_size, 1],
                rates=[1, 1, 1, 1],
                padding="VALID",
            )
            patch_dims = patches.shape[-1]
            patches = tf.reshape(patches, [batch_size, -1, patch_dims])
            return patches

    plt.figure(figsize=(4, 4))
    image = x_train[np.random.choice(range(x_train.shape[0]))]
    plt.imshow(image.astype("uint8"))
    plt.axis("off")

    resized_image = tf.image.resize(
        tf.convert_to_tensor([image]), size=(image_size, image_size)
    )
    patches = Patches(patch_size)(resized_image)
    print(f"Image size: {image_size} X {image_size}")
    print(f"Patch size: {patch_size} X {patch_size}")
    print(f"Patches per image: {patches.shape[1]}")
    print(f"Elements per patch: {patches.shape[-1]}")

    n = int(np.sqrt(patches.shape[1]))
    plt.figure(figsize=(4, 4))
    for i, patch in enumerate(patches[0]):
        ax = plt.subplot(n, n, i + 1)
        patch_img = tf.reshape(patch, (patch_size, patch_size, 3))
        plt.imshow(patch_img.numpy().astype("uint8"))
        plt.axis("off")

    ## Implement the patch encoding layer
    class PatchEncoder(layers.Layer):
        def __init__(self, num_patches, projection_dim):
            super().__init__()
            self.num_patches = num_patches
            self.projection = layers.Dense(units=projection_dim)
            self.position_embedding = layers.Embedding(
                input_dim=num_patches, output_dim=projection_dim
            )

        def call(self, patch):
            positions = tf.range(start=0, limit=self.num_patches, delta=1)
            encoded = self.projection(patch) + self.position_embedding(positions)
            return encoded

    ## Build the ViT model
    def create_vit_classifier():
        inputs = layers.Input(shape=input_shape)
        # Augment data.
        augmented = data_augmentation(inputs)
        # Create patches.
        patches = Patches(patch_size)(augmented)
        # Encode patches.
        encoded_patches = PatchEncoder(num_patches, projection_dim)(patches)

        # Create multiple layers of the Transformer block.
        for _ in range(transformer_layers):
            # Layer normalization 1.
            x1 = layers.LayerNormalization(epsilon=1e-6)(encoded_patches)
            # Create a multi-head attention layer.
            attention_output = layers.MultiHeadAttention(
                num_heads=num_heads, key_dim=projection_dim, dropout=0.1
            )(x1, x1)
            # Skip connection 1.
            x2 = layers.Add()([attention_output, encoded_patches])
            # Layer normalization 2.
            x3 = layers.LayerNormalization(epsilon=1e-6)(x2)
            # MLP.
            x3 = mlp(x3, hidden_units=transformer_units, dropout_rate=0.1)
            # Skip connection 2.
            encoded_patches = layers.Add()([x3, x2])

        # Create a [batch_size, projection_dim] tensor.
        representation = layers.LayerNormalization(epsilon=1e-6)(encoded_patches)
        representation = layers.Flatten()(representation)
        representation = layers.Dropout(0.5)(representation)
        # Add MLP.
        features = mlp(representation, hidden_units=mlp_head_units, dropout_rate=0.5)
        # Classify outputs.
        logits = layers.Dense(num_classes)(features)
        # Create the Keras model.
        model = keras.Model(inputs=inputs, outputs=logits)
        return model

    ## Compile, train, and evaluate the mode
    def run_experiment(model):
        # model.summary()
        optimizer = tfa.optimizers.AdamW(
            learning_rate=learning_rate, weight_decay=weight_decay
        )

        model.compile(
            optimizer=optimizer,
            loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            metrics=[
                keras.metrics.SparseCategoricalAccuracy(name="accuracy"),
                keras.metrics.SparseTopKCategoricalAccuracy(5, name="top-5-accuracy"),
            ],
        )

        history = model.fit(
            x=x_train,
            y=y_train,
            batch_size=batch_size,
            epochs=num_epochs,
            validation_split=0.1
        )

        _, accuracy, top_5_accuracy = model.evaluate(x_test, y_test)
        print(f"Test accuracy: {round(accuracy * 100, 2)}%")
        print(f"Test top 5 accuracy: {round(top_5_accuracy * 100, 2)}%")

        return history


    vit_classifier = create_vit_classifier()
    history = run_experiment(vit_classifier)