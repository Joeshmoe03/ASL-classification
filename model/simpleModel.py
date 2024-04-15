import tensorflow as tf
from keras.models import Sequential, Model
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Activation

def ConvBlock(model, num_filters: int, kernel_size: int):
    # Add a convolutional layer with num_filters filters and a kernel size of kernel_size x kernel_size
    model.add(Conv2D(num_filters, (kernel_size, kernel_size), padding='same'))
    # Add a ReLU activation function. Introduces non-linearity to the model.
    model.add(Activation('relu'))
    # Add a max pooling layer with a pool size of 2x2. Reduces the spatial dimensions of the output volume.
    model.add(MaxPooling2D(pool_size=(2, 2)))
    return model

# Reading for how to understand convolution easily: https://towardsdatascience.com/a-comprehensive-guide-to-convolutional-neural-networks-the-eli5-way-3bd2b1164a53
def ConvNet4(img_size: int, num_classes: int):
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
    model = ConvBlock(model, num_filters=32, kernel_size=3)
    # Our second convolutional block
    model = ConvBlock(model, num_filters=64, kernel_size=3)
    # Our third convolutional block
    model = ConvBlock(model, num_filters=128, kernel_size=3)
    # Our fourth convolutional block
    model = ConvBlock(model, num_filters=256, kernel_size=3)
    # Flatten the output of the convolutional layers to feed into a dense layer
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    # Our model output is a 2D tensor of shape (img_size, img_size, 1) where softmax gives us a probability distribution over the classes
    model.add(Dense(num_classes, activation='softmax'))
    return model

def ConvNet3(img_size: int, num_classes: int):
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
    model = ConvBlock(model, num_filters=32, kernel_size=3)
    # Our second convolutional block
    model = ConvBlock(model, num_filters=64, kernel_size=3)
    # Our third convolutional block
    model = ConvBlock(model, num_filters=128, kernel_size=3)
    # Flatten the output of the convolutional layers to feed into a dense layer
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    # Our model output is a 2D tensor of shape (img_size, img_size, 1) where softmax gives us a probability distribution over the classes
    model.add(Dense(num_classes, activation='softmax'))
    return model

def ConvNet2(img_size: int, num_classes: int):
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
    model = ConvBlock(model, num_filters=32, kernel_size=3)
    # Our second convolutional block
    model = ConvBlock(model, num_filters=64, kernel_size=3)
    # Flatten the output of the convolutional layers to feed into a dense layer
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    # Our model output is a 2D tensor of shape (img_size, img_size, 1) where softmax gives us a probability distribution over the classes
    model.add(Dense(num_classes, activation='softmax'))
    return model