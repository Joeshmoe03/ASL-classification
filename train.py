# Load dependencies
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import sklearn
from cv2 import imread
import cv2
import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import random
import argparse
from comet_ml import Experiment
from util.dataset import ASLDataPaths, ASLBatchLoader, split_data, save_data, load_saved_data
from util.dataset import ASLBatchLoader

# Input your Comet ML API key, project name, and workspace: allows tracking of training with Comet ML GUI. 
# TODO: USE COMET_ML FOR LOGGING
experiment = Experiment(api_key = "YOUR KEY", 
                        project_name = "YOUR PROJECT", 
                        workspace = "YOUR WORKSPACE")

# Main function entry point for our script
def main(args):
    experiment.log_parameters(args)

    # We save training runs and their associated sampling of data in the /temp/ directory under a folder named according to the hyperparameters/sampling.
    training_run_dir = os.path.join(os.getcwd(), 'temp', str(args.model) + 
                                    '_op' + str(args.optimizer) + 
                                    '_lr' + str(args.lr) + 
                                    '_bs' + str(args.batchSize) + 
                                    '_ep' + str(args.epochs) + 
                                    '_rs' + str(args.resample) + 
                                    '_sd' + str(args.seed) + 
                                    '_wd' + str(args.wd) + 
                                    '_mm' + str(args.momentum) + 
                                    '_ls' + str(args.loss) + 
                                    '_sp' + str(args.stopping)
                                    )

    # Create the temp directory if it does not exist or resample <- this training run/hyperparameter combo must not have been or we are resampling
    # Load the data paths and labels, split the data into training, validation, and test sets for paths and labels
    # Save the data paths and labels in the training run directory for sampling reproducibility
    # NOTE: data_splits takes on the form a tuple with the following structure: (X_train, X_val, X_test, y_train, y_val, y_test)
    if not os.path.exists(training_run_dir) or args.resample:
        os.makedirs(training_run_dir)
        X_path, y = ASLDataPaths(args.data_dir).fetch_paths()
        data_splits = split_data(X_path, y, args.test, args.val, args.seed)
        save_data(data_splits, training_run_dir)
        
    # Load the data paths and labels to make reproducibility of sampling <- this training run/hyperparameter combo must have been run before, no resampling specified
    # NOTE: data_splits takes on the form a tuple with the following structure: (X_train, X_val, X_test, y_train, y_val, y_test)
    else:
        data_splits = load_saved_data(training_run_dir)
    
    # Initialize the batch loader
    # NOTE: data_splits takes on the form a tuple with the following structure: (X_train, X_val, X_test, y_train, y_val, y_test)
    X_train_paths, X_val_paths, X_test_paths, y_train, y_val, y_test = data_splits
    train_loader = ASLBatchLoader(X_train_paths, y_train, args.batchSize)
    val_loader = ASLBatchLoader(X_val_paths, y_val, args.batchSize)
    test_loader = ASLBatchLoader(X_test_paths, y_test, args.batchSize)

    # Log that GPU is used
    gpu = tf.test.gpu_device_name()
    experiment.log_parameter("GPU: ", gpu)

    # Load the model
    if args.model == 'VGG':
        model = None #TODO: IMPLEMENT
    elif args.model == 'ResNet':
        model = None #TODO: IMPLEMENT
    else:
        raise ValueError('Model not implemented')

    # Load the optimizer
    if args.optimizer == 'SGD':
        pass #TODO: IMPLEMENT
    elif args.optimizer == 'Adam':
        pass #TODO: IMPLEMENT
    else:
        raise ValueError('Optimizer not implemented')
    
    # Load the loss function
    if args.loss == 'CategoricalCrossEntropy':
        pass #TODO: IMPLEMENT
    elif args.loss == 'SparseCategoricalCrossEntropy':
        pass #TODO: IMPLEMENT
    else:
        raise ValueError('Loss function not implemented')
    
    # TODO: training loop
        # early stopping too
    return

# Parser for easier running of the script on command line. 
if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Hyperparameters
    parser.add_argument('-nepoch'   , type=int  , action="store", dest='epochs'   , default=2000 ) # number of epochs
    parser.add_argument('-batchSize', type=int  , action="store", dest='batchSize', default=32   ) # batch size
    parser.add_argument('-lr'       , type=float, action="store", dest='lr'       , default=0.001) # learning rate
    parser.add_argument('-resample' , type=bool , action="store", dest='resample' , default=False) # resample data
    parser.add_argument('-seed'     , type=int  , action="store", dest='seed'     , default=42) # random seed
    parser.add_argument('-wd'       , type=float, action="store", dest='wdecay'   , default=0) # weight decay
    parser.add_argument('-momentum' , type=float, action="store", dest='momentum' , default=0) # for SGD
    parser.add_argument('-model'    , type=str  , action="store", dest='model'    , default='VGG') # VGG, ResNet, etc...
    parser.add_argument('-optimizer', type=str  , action="store", dest='optim'    , default='SGD') # SGD, Adam, etc...
    parser.add_argument('-loss'     , type=str  , action="store", dest='lossfunc' , default='SparseCategoricalCrossEntropy')
    parser.add_argument('-stopping' , type=int  , action="store", dest='stopping' , default=-1) # early stopping after n epochs with no improvement in validation loss. -1 for no early stopping.
    parser.add_argument('-test'     , type=float, action="store", dest='test_siz' , default=0.1) # test percentage
    parser.add_argument('-val'      , type=float, action="store", dest='val_siz'  , default=0.2) # validation percentage
    parser.add_argument('-data_dir' , type=str  , action="store", dest='data_dir' , default='./data/asl_alphabet_train/asl_alphabet_train/')

    # TODO: Possible arguments to implement in the future:
    # - data augmentation

    args = parser.parse_args()
    main(args)