from keras.layers import *
from keras.models import Sequential
import numpy as np

"""
This module defines the cnn model
uses adam optimizer , this model is modification of  the model used
in this tutorial:
    https://www.youtube.com/watch?v=JzXyDoaaa8s&t=1088s
by tanmay bakshi

"""

def define_model(input_shape, num_classes):
    """
    Defines a Convolutional Neural Network , the last layer is a
        GlobalAveragePooling
    layer because later on it this model is used for Class Activation Map
    therefore there is only one dense layer afterwards, which is the output layer
    that does the classification.

    prints the model summary before returning it

    Args:
        input_shape (shape): what is the input shape of data as an example (224*224*3)
        num_classes(int): number of classes
    Returns:
        model: the CNN model


    """
    model = Sequential()

    model.add(Conv2D(32, (5,5), activation = 'relu', padding='valid', input_shape = input_shape ))
    model.add(Conv2D(32, (3,3), activation = 'relu'))
    model.add(MaxPooling2D(2))

    model.add(Conv2D(64, (3,3), activation = 'relu'))
    model.add(Conv2D(64, (3,3), activation = 'relu'))
    model.add(MaxPooling2D(2))

    model.add(Conv2D(128, (3,3), activation = 'relu'))
    model.add(Conv2D(128, (3,3), activation = 'relu'))
    model.add(GlobalAveragePooling2D())

    model.add(Dense(num_classes, activation = "softmax"))

    model.compile(optimizer = "adam", loss = 'categorical_crossentropy', metrics = ['accuracy'])
    model.summary()

    return model
