import tensorflow as tf
from keras.models import Sequential, Model
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Activation

# Reading for how to understand convolution easily: https://towardsdatascience.com/a-comprehensive-guide-to-convolutional-neural-networks-the-eli5-way-3bd2b1164a53
def ConvNet4(num_classes: int, input_shape: tuple = (64, 64, 3)):
    '''
    Most basic convolutional neural networks generally have the following structure:
    1. A convolutional block: A convolutional layer followed by a ReLU activation function and a max pooling layer.
    2. A fully connected block: A dense layer followed by a ReLU activation function.
    3. An output layer: A dense layer with a softmax activation function for normalizing the output to a probability distribution over the classes.
    This model is purely to experiment with the number of convolutional blocks and the number of filters in each block and how it affects the model's performance.
    '''
    # The model is a sequence of layers
    model = Sequential()
    # Our first convolutional block
    model.add(Conv2D(32, kernel_size=(3, 3), input_shape=input_shape, padding='same'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(tf.keras.layers.Dropout(0.2))

    # Our second convolutional block
    model.add(Conv2D(64, kernel_size=(3, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(tf.keras.layers.Dropout(0.2))
              
    # Our third convolutional block
    model.add(Conv2D(128, kernel_size=(3, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(tf.keras.layers.Dropout(0.2))

    # Our fourth convolutional block
    model.add(Conv2D(256, kernel_size=(3, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(tf.keras.layers.Dropout(0.2))

    # Flatten the output of the convolutional layers to feed into a dense layer
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))

    # Our model output is a 2D tensor of shape (img_size, img_size, 1) where softmax gives us a probability distribution over the classes
    model.add(Dense(num_classes, activation='softmax'))
    return model

# Reading for how to understand convolution easily: https://towardsdatascience.com/a-comprehensive-guide-to-convolutional-neural-networks-the-eli5-way-3bd2b1164a53
def ConvNet3(num_classes: int, input_shape: tuple = (64, 64, 3)):
    '''
    Most basic convolutional neural networks generally have the following structure:
    1. A convolutional block: A convolutional layer followed by a ReLU activation function and a max pooling layer.
    2. A fully connected block: A dense layer followed by a ReLU activation function.
    3. An output layer: A dense layer with a softmax activation function for normalizing the output to a probability distribution over the classes.
    This model is purely to experiment with the number of convolutional blocks and the number of filters in each block and how it affects the model's performance.
    '''
    # The model is a sequence of layers
    model = Sequential()

    # Our first convolutional block
    model.add(Conv2D(32, kernel_size=(3, 3), input_shape=input_shape, padding='same'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(tf.keras.layers.Dropout(0.2))

    # Our second convolutional block
    model.add(Conv2D(64, kernel_size=(3, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(tf.keras.layers.Dropout(0.2))

    # Our third convolutional block
    model.add(Conv2D(128, kernel_size=(3, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(tf.keras.layers.Dropout(0.2))

    # Flatten the output of the convolutional layers to feed into a dense layer
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))

    # Our model output is a 2D tensor of shape (img_size, img_size, 1) where softmax gives us a probability distribution over the classes
    model.add(Dense(num_classes, activation='softmax'))
    return model

# Reading for how to understand convolution easily: https://towardsdatascience.com/a-comprehensive-guide-to-convolutional-neural-networks-the-eli5-way-3bd2b1164a53
def ConvNet2(num_classes: int, input_shape: tuple = (64, 64, 3)):
    '''
    Most basic convolutional neural networks generally have the following structure:
    1. A convolutional block: A convolutional layer followed by a ReLU activation function and a max pooling layer.
    2. A fully connected block: A dense layer followed by a ReLU activation function.
    3. An output layer: A dense layer with a softmax activation function for normalizing the output to a probability distribution over the classes.
    This model is purely to experiment with the number of convolutional blocks and the number of filters in each block and how it affects the model's performance.
    '''
    # The model is a sequence of layers
    model = Sequential()

    # Our first convolutional block
    model.add(Conv2D(32, kernel_size=(3, 3), input_shape=input_shape, padding='same'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(tf.keras.layers.Dropout(0.1))

    # Our second convolutional block
    model.add(Conv2D(64, kernel_size=(3, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(tf.keras.layers.Dropout(0.1))

    # Flatten the output of the convolutional layers to feed into a dense layer
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))

    # Our model output is a 2D tensor of shape (img_size, img_size, 1) where softmax gives us a probability distribution over the classes
    model.add(Dense(num_classes, activation='softmax'))
    return model