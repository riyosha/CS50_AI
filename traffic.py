import cv2
import numpy as np
import os
import sys
import tensorflow as tf
from tensorflow import keras
from keras import layers
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split

EPOCHS = 10
IMG_WIDTH = 30
IMG_HEIGHT = 30
NUM_CATEGORIES = 43
TEST_SIZE = 0.4

def main():

    # Check command-line arguments
    if len(sys.argv) not in [2, 3]:
        sys.exit("Usage: python traffic.py data_directory [model.h5]")

    # Get image arrays and labels for all image file
    images, labels = load_data(sys.argv[1])

    # Split data into training and testing sets
    labels = tf.keras.utils.to_categorical(labels)
    x_train, x_test, y_train, y_test = train_test_split(
        np.array(images), np.array(labels), test_size=TEST_SIZE
    )

    # Get a compiled neural network
    model = get_model()

    # Fit model on training data
    model.fit(x_train, y_train, epochs=EPOCHS)

    # Evaluate neural network performance
    model.evaluate(x_test,  y_test, verbose=2)

    # Save model to file
    if len(sys.argv) == 3:
        filename = sys.argv[2]
        model.save(filename)
        print(f"Model saved to {filename}.")

    #code to test predictions 
    # y_test_pred = model.predict(x_test)

    # predicted_values = []
    # actual_values = []
    # for value in y_test_pred:
    #     for i in range(len(value)):
    #         if value[i]==max(value):
    #             predicted_values.append(i)

    # for value in y_test:
    #     for i in range(len(value)):
    #         if value[i]==max(value):
    #             actual_values.append(i)

    # predicted_actual_diff = []
    # for i in range(len(y_test)):
    #     x = actual_values[i]- predicted_values[i]
    #     predicted_actual_diff.append(x)

    # print(predicted_actual_diff)


def load_data(data_dir):
    """
    Load image data from directory `data_dir`.

    Assume `data_dir` has one directory named after each category, numbered
    0 through NUM_CATEGORIES - 1. Inside each category directory will be some
    number of image files.

    Return tuple `(images, labels)`. `images` should be a list of all
    of the images in the data directory, where each image is formatted as a
    numpy ndarray with dimensions IMG_WIDTH x IMG_HEIGHT x 3. `labels` should
    be a list of integer labels, representing the categories for each of the
    corresponding `images`.
    """
    images = []
    labels = []
    try:
        for i in range(0, NUM_CATEGORIES):
            path = f'{data_dir}/{str(i)}'
            assert os.listdir(path), 'list dir invalid'
            for file in os.listdir(path):
                image = cv2.imread(f'{data_dir}/{i}/{file}')
                image = cv2.resize(image, (IMG_WIDTH, IMG_HEIGHT))
                images.append(image)
                labels.append(int(i))

        return (images, labels)
    except:
     raise NotImplementedError


def get_model():
    """
    Returns a compiled convolutional neural network model. Assume that the
    `input_shape` of the first layer is `(IMG_WIDTH, IMG_HEIGHT, 3)`.
    The output layer should have `NUM_CATEGORIES` units, one for each category.
    """
    try:
        # Define Sequential model
        model = keras.Sequential()
        model.add(keras.Input(shape=(IMG_WIDTH, IMG_HEIGHT, 3)))  #RGB images
        
        model.add(layers.Conv2D(
            filters=64,  # Single filter for edge detection
            kernel_size=(3, 3),  # 3x3 kernel for neighborhood analysis
            strides=(1, 1),  # Strides of 1 in both dimensions
            activation='relu', input_shape=(IMG_WIDTH, IMG_HEIGHT, 3)  # ReLU activation function
            )
        ) #1st convolution layer
        model.add(layers.BatchNormalization())
        model.add(layers.AveragePooling2D(pool_size=(3, 3))) #1st pooling layer - AveragePool
        model.add(layers.Dropout(0.1)) #dropout layer
        model.add(layers.Conv2D(filters=64, kernel_size=(3, 3), activation = 'relu')) #2nd convolution layer
        model.add(layers.BatchNormalization())
        model.add(layers.MaxPooling2D(pool_size=(3, 3))) #2nd pooling layer - MaxPool
        model.add(layers.Dropout(0.1)) #dropout layer
        #model.add(layers.Conv2D(filters=32, kernel_size=(3, 3), activation = 'softplus')) #2nd convolution layer
        model.add(layers.BatchNormalization())
        #model.add(layers.MaxPooling2D(pool_size=(3, 3))) #2nd pooling layer - MaxPool
        model.add(layers.Flatten())
        model.add(layers.Dense(129, activation='swish')) 
        model.add(layers.Dense(43, activation='softmax')) #last layer

        model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
        return model
    except:
        raise NotImplementedError

if __name__ == "__main__":
    main()
