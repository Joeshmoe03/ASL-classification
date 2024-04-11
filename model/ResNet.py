from tensorflow.keras import Input, Model # type: ignore
from tensorflow.keras.layers import AveragePooling2D, Flatten, Dense, Dropout, Add # type: ignore
from tensorflow.keras import layers # type: ignore

# References:   https://www.tensorflow.org/guide/keras/sequential_model
#               https://www.kaggle.com/code/pankaj1234/tensorflow-resnet
#               Some aid from ChatGPT to understand model

def ResNet():
    # ResNet50 base model with pre-trained weights
    rsntBase = ResNet50(weights='imagenet', include_top=False, input_tensor=Input(shape=(224, 224, 3)))

    num_classes=29

    # Additional layers
    model = rsntBase.output
    model = AveragePooling2D(pool_size=(7,7))(model)
    model = Flatten(name="flatten")(model)
    model = Dense(1024,activation='relu')(model)
    model = Dropout(0.5)(model)
    model = Dense(1024,activation='relu')(model)
    model = Dropout(0.5)(model)
    model = Dense(num_classes, activation='softmax')(model)

    # Create the final model
    finalModel = Model(inputs=rsntBase.input, outputs=model)

    # Freeze layers except for the last block
    for layer in finalModel.layers:
        if not layer.name.startswith('conv5_'):
            layer.trainable = False

    return finalModel

# Implementation from scratch that we had trouble debugging        
# def ResNet():
#     input_layer = Input(shape=(214, 214, 3))

#     # Initial convolution block
#     x = convolution_block(input_layer, filters=64, kernel_size=(3,3))
#     x = layers.MaxPooling2D(pool_size=(3, 3), strides=(2, 2))(x)


#     stages = 4
#     num_classes = 29
#     filter = 64
#     for _ in range(stages):
#         x = identity_block(x, filters=filter, kernel_size=(3,3))
#         filter *= 2

#     # Final layers
#     x = layers.AveragePooling2D(pool_size=(2, 2))(x)
#     x = layers.Flatten()(x)
#     x = layers.Dense(1024, activation='relu')(x)
#     x = layers.Dropout(0.5)(x)
#     x = layers.Dense(1024, activation='relu')(x)
#     x = layers.Dropout(0.5)(x)
#     output_layer = layers.Dense(num_classes, activation="softmax")(x)

#     # Create model
#     model = Model(inputs=input_layer, outputs=output_layer)

#     # Freezing layers except for last block
#     for _ in model.layers:
#         if not _.name.startswith('conv5_'):
#             _.trainable=False

#     # model.summary()

#     return model

# def convolution_block(input_layer, filters, kernel_size, padding='same'):
#     x = layers.Conv2D(filters=filters, kernel_size=kernel_size, padding=padding)(input_layer)
#     x = layers.BatchNormalization()(x)
#     x = layers.Activation('relu')(x)
#     return x

# def identity_block(input_layer, filters, kernel_size, padding='same'):
#     x = convolution_block(input_layer, filters=filters, kernel_size=kernel_size, padding=padding)
#     x = Add()([x, input_layer])
#     return x