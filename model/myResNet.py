from tensorflow.keras import Input, Model # type: ignore
from tensorflow.keras.layers import AveragePooling2D, Flatten, Dense, Dropout, Add # type: ignore
from tensorflow.keras import layers # type: ignore

# References:   https://www.tensorflow.org/guide/keras/sequential_model
#               https://www.kaggle.com/code/pankaj1234/tensorflow-resnet
#               Some aid from ChatGPT to understand model
                
def ResNet():
    input_layer = Input(shape=(200, 200, 1))

    # Initial convolution block
    x = convolution_block(input_layer, filters=64, kernel_size=(3,3))
    x = layers.MaxPooling2D(pool_size=(3, 3), strides=(2, 2))(x)


    stages = 4
    num_classes = 29
    for _ in range(stages):
        x = identity_block(x, filters=64, kernel_size=(3,3))
        filter *= 2

    # Final layers
    x = layers.AveragePooling2D(pool_size=(2, 2))(x)
    x = layers.Flatten()(x)
    x = layers.Dense(1024, activation='relu')(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(1024, activation='relu')(x)
    x = layers.Dropout(0.5)(x)
    output_layer = layers.Dense(num_classes, activation="soft_max")(x)

    # Create model
    model = Model(inputs=input_layer, outputs=output_layer)

    # Freezing layers except for last block
    for _ in model.layers:
        if not _.name.startswith('conv5_'):
            _.trainable=False

    # model.summary()

    return model

def convolution_block(input_layer, filters, kernel_size, padding='same'):
    x = layers.Conv2D(filters=filters, kernel_size=kernel_size, padding=padding)(input_layer)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    return x

def identity_block(input_layer, filters, kernel_size, padding='same'):
    x = convolution_block(filters=filters, kernel_size=kernel_size, padding=padding)(input_layer)
    x = Add()([x, input_layer])
    return x

