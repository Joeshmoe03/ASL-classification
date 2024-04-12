import tensorflow as tf
from keras.models import Sequential, Model
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Activation

def model1(img_size: int, num_classes: int):
    # The model is a sequence of layers
    model = Sequential()

    # The first layer is a convolutional layer with 32 filters, a kernel size of 3x3, and a stride of 1x1
    model.add(Conv2D(32, (3, 3), input_shape=(img_size, img_size, 1)))

    # TODO: Add more layers to the model 
    pass

    # Our model output is a 2D tensor of shape (img_size, img_size, 1) where softmax gives us a probability distribution over the classes
    model.add(Dense(num_classes, activation='softmax'))
    return model