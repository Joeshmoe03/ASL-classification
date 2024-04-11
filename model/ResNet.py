import tensorflow as tf

def ResNet50(img_size: int, color: str):
    resnet = tf.keras.applications.ResNet50(
        include_top=True,
        weights='imagenet',
        input_tensor=None,
        input_shape=(img_size, img_size, 3 if color == 'rgb' else 1),
        pooling=None,
        classes=1000,
        classifier_activation='softmax'
    )
    return resnet