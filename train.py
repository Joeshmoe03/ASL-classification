# Load dependencies
import tensorflow as tf
from tensorflow.keras import layers
import os
import argparse
from comet_ml import Experiment
from util.dataset import ASLDataPaths, ASLBatchLoader, split_data, save_data, load_saved_data
from util.model import ModelFactory
from util.directory import initializeDir
from util.transform import grayscale
from util.trainloop import trainLoop

# Main function entry point for our script
def main(args):
    # We save training runs and their associated sampling of data in the /temp/ 
    # directory under a folder named according to the sampling and hyperparameters.
    scratch_dir = os.path.join(os.getcwd(), 'temp', f"{args.model}_op{args.optim}_ls{args.loss}" 
                                                  + f"_lr{args.lr}_wd{args.wdecay}_mo{args.momentum}" 
                                                  + f"_rs{args.resample}")

    # Initialize the directory and perform sampling if it does not exist. Otherwise, load the saved data.
    if initializeDir(scratch_dir):
        data_paths = ASLDataPaths(args.data_dir).fetch_paths()
        data_splits = split_data(data_paths, args.valSize, args.testSize, args.resample)
        save_data(data_splits, scratch_dir)
    else:
        data_splits = load_saved_data(scratch_dir)

    # NOTE: FEEL FREE TO MODIFY TRANSFORMATIONS AS NEEDED
    transform = tf.keras.Sequential([layers.Resizing(64, 64),
                                    layers.Rescaling(1./255),
                                    layers.RandomFlip("horizontal"),
                                    layers.RandomRotation(0.2),
                                    layers.Lambda(grayscale)])

    # Initialize the batch loader
    train_data, val_data, test_data = data_splits
    train_loader = ASLBatchLoader(train_data[:, 0], train_data[:, 1], transform=transform, batch_size=args.batchSize)
    val_loader = ASLBatchLoader(val_data[:, 0], val_data[:, 1], transform=transform, batch_size=args.batchSize)
    test_loader = ASLBatchLoader(test_data[:, 0], test_data[:, 1], transform=transform, batch_size=args.batchSize)
    loaders = {'train': train_loader, 'val': val_loader, 'test': test_loader}

    # Load the model and pretrained weights if specified
    model = ModelFactory(args, args.model, args.pretrain)

    # Train the model
    trainLoop(scratch_dir, loaders, model, args)
    return 0

if __name__ == "__main__":
    # Parser for easier running of the script on command line. Can specify hyperparameters and model type this way.  
    parser = argparse.ArgumentParser()
    parser.add_argument('-nepoch'   , type=int  , action="store", dest='epochs'   , default=2000 ) # number of epochs
    parser.add_argument('-batchSize', type=int  , action="store", dest='batchSize', default=32   ) # batch size
    parser.add_argument('-lr'       , type=float, action="store", dest='lr'       , default=0.001) # learning rate
    parser.add_argument('-resample' , type=int  , action="store", dest='resample' , default=0) # resample data
    parser.add_argument('-wd'       , type=float, action="store", dest='wdecay'   , default=0) # weight decay
    parser.add_argument('-momentum' , type=float, action="store", dest='momentum' , default=1.00) # for SGD
    parser.add_argument('-model'    , type=str  , action="store", dest='model'    , default='VGG') # VGG, ResNet, etc...
    parser.add_argument('-optim'    , type=str  , action="store", dest='optim'    , default='SGD') # SGD, Adam, etc...
    parser.add_argument('-loss'     , type=str  , action="store", dest='loss'     , default='SparseCategoricalCrossEntropy')
    parser.add_argument('-stopping' , type=int  , action="store", dest='stopping' , default=-1) # early stopping after n epochs with no improvement in validation loss. -1 for no early stopping.
    parser.add_argument('-test'     , type=float, action="store", dest='testSize' , default=0.1) # test percentage
    parser.add_argument('-val'      , type=float, action="store", dest='valSize'  , default=0.2) # validation percentage
    parser.add_argument('-data_dir' , type=str  , action="store", dest='data_dir' , default='./data/asl_alphabet_train/asl_alphabet_train/')
    parser.add_argument('-pretrain' , type=bool , action="store", dest='pretrain' , default=False) # use pretrained weights
    args = parser.parse_args()
    main(args)