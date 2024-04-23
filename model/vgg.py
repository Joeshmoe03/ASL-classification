import tensorflow as tf
from keras.applications import vgg16
from keras.models import Sequential, Model
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Activation

"""
Summary of Model Archetecture:
- Base model is VGG16 with no pretrained weights
- Base model is unfrozen
- Flatten base model output to 2048 nodes
- Reduce to 512 nodes
- Final classification layer (29 classes for ASL)

Total params: 15777501 (60.19 MB) (Use "model.summary()" to see this output)
"""

def Vgg16(img_size, color, num_classes):
    # Use a base vgg16 with that is not pretrained, 
    base_model = vgg16.VGG16(
        include_top=False,
        weights = 'imagenet',#None,
        input_shape=(img_size, img_size, 3 if color == 'rgb' else 1),
        pooling=None,
    )

    # Make sure base model layers are unfrozen, so we can train those parameters
    for layer in base_model.layers:
        layer.trainable = True

    # Create a new model by adding layers on top of the base model
    model = Sequential()
    model.add(base_model)

    # Flatten current model from 2x2x512 to 2048 nodes
    model.add(Flatten())

    # Reduce from 2048 -> 512 nodes
    model.add(Dense(512, activation='relu')) # Can change this reduction accordingly

    # Final classification layer
    model.add(Dense(num_classes, activation='softmax'))

    return model

# NOTE: Use "model.summary()" to see the model architecture