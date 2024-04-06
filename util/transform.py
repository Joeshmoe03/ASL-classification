import tensorflow as tf

# Function to convert an image to grayscale. Used by the transform pipeline in Lambda layer.
def grayscale(img):
    return tf.image.rgb_to_grayscale(img)
