import tensorflow as tf
import random

# Transforms for tf.data.Dataset inspired by: https://stackoverflow.com/questions/58270150/is-there-some-simple-way-to-apply-image-preprocess-to-tf-data-dataset
def transformTrainData(image, label):
    '''
    See the documentation for tf.image for more information on the transformations: 
    https://www.tensorflow.org/api_docs/python/tf/image
    '''
    # Performs scaling. NOTE: FEEL FREE TO MODIFY. SEE DOCUMENTATION ABOVE TO FIND MORE TRANSFORMATIONS.
    image = tf.image.convert_image_dtype(image, tf.float32)
    image = tf.image.rot90(image, k = random.randint(0, 3))
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_flip_up_down(image)
    #image = tf.image.random_brightness(image, max_delta = 0.1)
    #image = tf.image.random_contrast(image, lower = 0.9, upper = 1.1)
    #image = tf.image.random_hue(image, max_delta = 0.1)
    #image = tf.image.random_saturation(image, lower = 0.9, upper = 1.1)
    return image, label

def transformValData(image, label):
    '''
    See the documentation for tf.image for more information on the transformations:
    https://www.tensorflow.org/api_docs/python/tf/image
    '''
    # Performs scaling
    image = tf.image.convert_image_dtype(image, tf.float32)
    return image, label