import tensorflow as tf

def ResNet50(img_size: int, color: str, num_classes: int):
    # Load the pre-trained ResNet50 model
    base_model = tf.keras.applications.ResNet50(
    include_top=False,
    #weights='imagenet',
    input_tensor=None,
    input_shape=(img_size, img_size, 3 if color == 'rgb' else 1),
    pooling=None,
    classifier_activation='softmax'
    )
    x = tf.keras.layers.GlobalAveragePooling2D()(base_model.output)
    x = tf.keras.layers.Dense(1024, activation='relu')(x)  
    output = tf.keras.layers.Dense(num_classes, activation='softmax')(x)  
    resnet = tf.keras.Model(inputs=base_model.input, outputs=output)
    return resnet