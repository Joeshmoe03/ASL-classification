import tensorflow as tf
import numpy as np
import cv2
import math
import os

# NOTE: This file contains important classes and functions that are used to load and preprocess the ASL dataset. Given
# that the dataset is quite large, and of custom format, we need to implement our own data loader to load the data
# efficiently. We can not afford to simply load the entire dataset into memory at once, as it would be too large/expensive.
# Instead, we will load the data in batches, and apply augmentations to the data as we load it. This will allow us to
# train our model on the data without stuffing memory completely full.

class fetchASLDataPaths():
    '''
    fetchASLDataPaths is a class that fetches the paths of the ASL dataset from a directory. The rationale behind such a class
    is the fact that our dataset is huge (relatively speaking), and we can not afford to load the entire dataset of images into memory.
    Rather, it might be a better idea to load the paths of the images, and then load the images in batches as we train our model. 
    '''

    def __init__(self, data_dir: str):

        # Check if the data directory exists
        if type(data_dir) != str or not os.path.exists(data_dir):
            raise FileNotFoundError(f"The directory {data_dir} does not exist.")
        self.data_dir = data_dir

    def fetch_paths(self):
        X_paths = []
        y = []

        # Walk over the data directory and fetch the paths of all images, label in the dataset
        for root, _, files in os.walk(self.data_dir):
            for file in files:
                X_paths.append(os.path.join(root, file))
                y.append(root.split('/')[-1])
        
        X_paths = np.array(X_paths)
        y = np.array(y)
        return X_paths, y

# The ASLBatchLoader class is a custom data loader following the concept of this documentation code: 
# https://www.tensorflow.org/api_docs/python/tf/keras/utils/PyDataset. Refer to this documentation for more information
# and context on how to implement a custom data loader in TensorFlow.
class ASLBatchLoader(tf.keras.utils.PyDataset):

    def __init__(self, 
                 X_set: np.array, 
                 y_set: np.array,
                 batch_size: int = 32, 
                 shuffle: bool = True, 
                 transform = None):
        '''
        The ASLBatchLoader class is a custom data loader that loads the ASL dataset in batches.
        
        Parameters:
            X_set: np.array - A numpy array containing the paths of the images and 
            y_set: np.array - their corresponding labels.
            batch_size: int - The size of the batch that we want to load the data in.
            shuffle: Bool - Whether or not we want to shuffle the data before loading it.
            transform: callable - A callable function that applies transformations to the data as we load it.
        '''
        self.X_set = X_set
        self.y_set = y_set
        self.batch_size = batch_size
        self.indices = np.arange(len(X_set))

        # Shuffle the indices if shuffle is set to True
        if shuffle:
            np.random.shuffle(self.indices)

        #TODO: IMPLEMENT TRANSFORMS
        self.transform = transform

    def __len__(self):
        '''
        This function returns the number of batches that we can load from the dataset.
        
        Returns: int - The number of batches that we can load from the dataset.
        '''
        return math.ceil(len(self.X_set) / self.batch_size)

    def __getitem__(self, index):
        '''
        This function loads a batch of data from the dataset.

        Parameters:
            index: int - The index of the batch that we want to load from the dataset.

        Returns:
            X_batch: np.array - A numpy array containing the images of the batch.
            y_batch: np.array - A numpy array containing the labels of the batch.
        '''

        # We specify the start of our batch
        batch_start = index * self.batch_size

        # If the batch end is greater than the length of the data directory, we set the batch end to the length of the data directory
        batch_end = min(index + self.batch_size, len(self.data_dir))

        # These are the paths that we immediately work with in this iteration of the batching process
        X_path_batch = self.X_set[batch_start:batch_end, 0]
        y_batch = self.y_set[batch_start:batch_end, 1]

        # Load the images and labels from the paths
        # Transforms are applied, should they be specified. A transform typically should
        # include at least some resize, normalization, and other additional augmentations.
        X_batch = np.array([self.transform(cv2.imread(file)) if self.transform is not None
                            else cv2.imread(file) for file in X_path_batch])

        return X_batch, y_batch        