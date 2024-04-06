# Load dependencies
from tqdm import tqdm
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import os
import argparse
from comet_ml import Experiment
from util.dataset import ASLDataPaths, ASLBatchLoader, split_data, save_data, load_saved_data
from util.directoryHandler import samplingHandler
#from util.model import ModelFactory

# Main function entry point for our script
def main(args):
    # We save training runs and their associated sampling of data in the /temp/ directory under a folder named according to the hyperparameters/sampling.
    # Example of a valid training run directory: ./temp/VGG_opSGD_lsCategoricalCrossEntropy_lr0.001/
    training_run_dir = os.path.join(os.getcwd(), 'temp', 'resample' + str(args.resample))

    # We use our samplingHandler() wrapper for managing directories associated with different sampled data. While it would be simple to simply call ASLDataPaths and then ASLBatchLoader,
    # we want the ability to resample or reuse previously sampled data saved to specific directories. We do this because reproducibility is an important
    # thing we should care about both from a point of transparency and honesty in recording our results and benchmarking. 
    # NOTE: data_splits takes on the form a tuple with the following structure: (X_train, X_val, X_test, y_train, y_val, y_test)
    data_splits = samplingHandler(training_run_dir, args)

    # Initialize the batch loader
    X_train_paths, X_val_paths, X_test_paths, y_train, y_val, y_test = data_splits
    train_loader = ASLBatchLoader(X_train_paths, y_train, args.batchSize)
    val_loader = ASLBatchLoader(X_val_paths, y_val, args.batchSize)
    test_loader = ASLBatchLoader(X_test_paths, y_test, args.batchSize)
    
    # Transformations for the data
    # TODO: IMPLEMENT DATA AUGMENTATION/TRANSFORMATIONS

    # Load the model and pretrained weights if specified
    #model = ModelFactory(args, args.model, args.pretrain)

    # Load the optimizer
    if args.optim == 'SGD':
        pass #TODO: IMPLEMENT
    elif args.optim == 'Adam':
        pass #TODO: IMPLEMENT
    else:
        raise ValueError('Optimizer not implemented')
    
    # Load the loss function
    if args.loss == 'CategoricalCrossEntropy':
        pass #TODO: IMPLEMENT
        #TODO: transform labels to one-hot encoding
    elif args.loss == 'SparseCategoricalCrossEntropy':
        pass #TODO: IMPLEMENT
        #TODO: transform labels to numerical encoding
    else:
        raise ValueError('Loss function not implemented')
    
    # TODO: training loop
        # early stopping too
    return

# Parser for easier running of the script on command line. 
if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Hyperparameters
    parser.add_argument('-nepoch'   , type=int  , action="store", dest='epochs'   , default=2000 ) # number of epochs
    parser.add_argument('-batchSize', type=int  , action="store", dest='batchSize', default=32   ) # batch size
    parser.add_argument('-lr'       , type=float, action="store", dest='lr'       , default=0.001) # learning rate
    parser.add_argument('-resample' , type=int  , action="store", dest='resample' , default=0) # resample data
    parser.add_argument('-seed'     , type=int  , action="store", dest='seed'     , default=42) # random seed
    parser.add_argument('-wd'       , type=float, action="store", dest='wdecay'   , default=0) # weight decay
    parser.add_argument('-momentum' , type=float, action="store", dest='momentum' , default=1.00) # for SGD
    parser.add_argument('-model'    , type=str  , action="store", dest='model'    , default='VGG') # VGG, ResNet, etc...
    parser.add_argument('-optim'    , type=str  , action="store", dest='optim'    , default='SGD') # SGD, Adam, etc...
    parser.add_argument('-loss'     , type=str  , action="store", dest='loss'     , default='SparseCategoricalCrossEntropy')
    parser.add_argument('-stopping' , type=int  , action="store", dest='stopping' , default=-1) # early stopping after n epochs with no improvement in validation loss. -1 for no early stopping.
    parser.add_argument('-test'     , type=float, action="store", dest='testSize' , default=0.1) # test percentage
    parser.add_argument('-val'      , type=float, action="store", dest='valSize'  , default=0.2) # validation percentage
    parser.add_argument('-data_dir' , type=str  , action="store", dest='data_dir' , default='./data/asl_alphabet_train/asl_alphabet_train/')
    parser.add_argument('-pretrain' , type=bool , action="store", dest='pretrain' , default=False)

    # TODO: Possible arguments to implement in the future:
    # - data augmentation

    args = parser.parse_args()
    main(args)