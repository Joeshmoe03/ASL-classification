import tensorflow as tf
from tensorflow.keras.applications import vgg16 # type: ignore
from tensorflow.keras.models import Sequential, Model # type: ignore
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Activation # type: ignore

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
        weights = 'imagenet', #None,
        input_shape=(img_size, img_size, 3 if color == 'rgb' else 1),
        pooling=None,
    )

    # Make sure base model layers are unfrozen, so we can train those parameters
    for layer in base_model.layers:
        layer.trainable = True

    # Flatten current model from 2x2x512 to 2048 nodes
    flatten_layer = Flatten()

    # Reduce from 2048 -> 512 nodes
    dense_layer1 = Dense(512, activation='relu') # Can change this reduction accordingly

    # Final classification layer
    dense_layer2 = Dense(num_classes, activation='softmax')
    
    out = flatten_layer(base_model.output)
    out = dense_layer1(out)
    pred = dense_layer2(out)
    model = Model(inputs = base_model.input, outputs = pred)

    return model

# NOTE: Use "model.summary()" to see the model architecture