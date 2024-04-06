import tensorflow as tf
from sklearn.model_selection import train_test_split
import numpy as np
import cv2
import math
import os

# NOTE: This file contains important classes and functions that are used to load and preprocess the ASL dataset. Given
# that the dataset is quite large, and of custom format, we need to implement our own data loader to load the data
# efficiently. We can not afford to simply load the entire dataset into memory at once, as it would be too large/expensive.
# Instead, we will load the data in batches, and apply augmentations to the data as we load it. This will allow us to
# train our model on the data without stuffing memory completely full.
class ASLDataPaths():
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
        
        # Convert the paths and labels to numpy arrays
        X_paths = np.array(X_paths)
        y = np.array(y)
        data = np.column_stack((X_paths, y))
        return data

# The ASLBatchLoader class is a custom data loader following the concept of this documentation code: 
# https://www.tensorflow.org/api_docs/python/tf/keras/utils/PyDataset. Refer to this documentation for more information
# and context on how to implement a custom data loader in TensorFlow.
class ASLBatchLoader(tf.keras.utils.PyDataset):

    def __init__(self, 
                 X_set: np.array, 
                 y_set: np.array,
                 batch_size: int = 32, 
                 transform = None):
        '''
        The ASLBatchLoader class is a custom data loader that loads the ASL dataset in batches.
        
        Parameters:
            X_set: np.array - A numpy array containing the paths of the images and 
            y_set: np.array - their corresponding labels.
            batch_size: int - The size of the batch that we want to load the data in.
        '''
        self.X_set = X_set
        self.y_set = y_set
        self.batch_size = batch_size
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
            index: int - The index of batch that we want to load from the dataset.

        Returns:
            X_batch: np.array - A numpy array containing the images of the batch.
            y_batch: np.array - A numpy array containing labels of the batch.
        '''
        # We specify the start of our batch
        batch_start = index * self.batch_size

        # If the batch end is greater than the length of the data directory, we set the batch end to the length of the data directory
        batch_end = min(batch_start + self.batch_size, len(self.X_set))

        # These are the paths that we immediately work with in this iteration of the batching process
        X_path_batch = self.X_set[batch_start:batch_end]
        y_batch = self.y_set[batch_start:batch_end]

        # Load the images and labels from the paths
        # If a transformation is specified, we apply it to the images
        # If no transformation is specified, we simply load the images
        # A transformation is typically something like normalization, resizing, etc.
        X_batch = np.array([cv2.imread(file) for file in X_path_batch])
        if self.transform is not None:
            X_batch = self.transform(X_batch)

        return X_batch, y_batch
    
    def __iter__(self):
        '''
        This method returns an iterator for the batches.
        
        Yields: batch - A batch of data from the dataset.
        '''
        for i in range(len(self)):
            yield self[i]
    
# ChatGPT was used to generate these docstring. No need to do redundant work.
def split_data(data, test_size=0.2, val_size=0.2, random_state=42):
    '''
    Split the data into training, validation, and test sets.
    
    Parameters:
        X: np.array - The paths of the images.
        y: np.array - The labels of the images.
        test_size: float - The size of the test set.
        val_size: float - The size of the validation set.
        random_state: int - The random state for reproducibility.
        
    Returns:
        tuple: a tuple of np.arrays - train_data, val_data, test_data
    '''
    train_data, test_data = train_test_split(data, test_size=test_size, random_state=random_state)
    train_data, val_data = train_test_split(train_data, test_size=val_size, random_state=random_state)
    return (train_data, val_data, test_data)

# ChatGPT was used to generate these docstring. No need to do redundant work.
def save_data(data_splits: tuple[tuple, tuple, tuple], dir: str):
    '''
    Save the data and labels in the training run directory for sampling reproducibility.
    
    Parameters:
        data_splits: tuple - a tuple of train, val, test data, then labels for those sets - in that order. 
        dir: str - The directory where the training run is stored.
    '''
    # Check if the directory exists
    if not os.path.exists(dir):
        raise FileNotFoundError(f"Directory '{dir}' does not exist.")
    
    # Check if the data_splits contain 6 arrays
    if len(data_splits) != 3:
        raise ValueError("data_splits should contain 6 arrays.")

    # Save the data and labels
    np.save(os.path.join(dir, 'train_data.npy'), data_splits[0])
    np.save(os.path.join(dir, 'val_data.npy'), data_splits[1])
    np.save(os.path.join(dir, 'test_data.npy'), data_splits[2])

# ChatGPT was used to generate these docstring. No need to do redundant work.
def load_saved_data(dir: str):
    '''
    Load the data and labels to make reproducibility of sampling.
    
    Parameters:
        dir: str - The directory where the training run is stored.
        
    Returns:
        data_splits: tuple - a tuple of np.arrays for X train, val, test and y train, val, test.
    '''
    # Check if the directory exists
    if not os.path.exists(dir):
        raise FileNotFoundError(f"Directory '{dir}' does not exist.")
    
    # Check that files exist
    files = ['train_data', 'val_data', 'test_data']
    if any (not os.path.exists(os.path.join(dir, f'{file}.npy')) for file in files):
        raise FileNotFoundError(f"Incomplete directory '{dir}'.")

    # Load the data and labels
    train_data = np.load(os.path.join(dir, 'train_data.npy'))
    val_data = np.load(os.path.join(dir, 'val_data.npy'))
    test_data = np.load(os.path.join(dir, 'test_data.npy'))
    return (train_data, val_data, test_data)