# Load dependencies
from comet_ml import Experiment
import math
from tensorflow.keras.utils import image_dataset_from_directory # type: ignore
import tensorflow as tf
import os
import argparse
import json
from util.model import ModelFactory, optimizerFactory, lossFactory
from util.metric import metricFactory
from util.transform import transformTrainData, transformValTestData
from util.directory import initScratchDir, checkpointProgress

def main(args):
    '''
    Main function for training the model. This is the entry point for the script. 
    We load the data, create the model, compile the model, and train the model.
    The model is saved in the /temp/ directory under a folder named according to 
    the sampling and hyperparameters.

    Args:
        args: command-line arguments. These are the hyperparameters for the model/model.
    '''
    experiment = None
    if args.comet:
    
    # Setting keys for privacy: https://networkdirection.net/python/resources/env-variable/
        API_KEY = os.environ.get('COMET_API_KEY')

        # Initialize Comet experiment
        experiment = Experiment(api_key=API_KEY, 
                                project_name="asl",
                                workspace="joeshmoe03",
                                auto_output_logging="default",
                                auto_param_logging = True,
                                auto_metric_logging = True)

    # We save training runs and their associated sampling of data in the /temp/ 
    # directory under a folder named according to the sampling and hyperparameters.
    scratch_dir = initScratchDir(args)
    
    # Load the data into a tf.data.Dataset
    train_dataset = image_dataset_from_directory(args.data_dir, 
                                                labels = 'inferred', 
                                                # Specify label encoding type based on loss function (one hot for categorical_crossentropy, int for sparse_categorical_crossentropy)
                                                label_mode = 'categorical' if args.loss == 'categorical_crossentropy' else 'int', 
                                                color_mode = args.color, 
                                                batch_size = args.batchSize,
                                                interpolation = 'nearest',
                                                image_size = (args.img_size, args.img_size), 
                                                shuffle = True,
                                                seed = args.resample,
                                                )
    
    # Split the data into training, validation, and testing sets
    # From: https://stackoverflow.com/questions/48213766/split-a-dataset-created-by-tensorflow-dataset-api-in-to-train-and-test 
    test_dataset = train_dataset.take(math.floor(args.testSize * len(train_dataset)))
    train_dataset = train_dataset.skip(math.floor(args.testSize * len(train_dataset)))
    val_dataset = train_dataset.take(math.floor(args.valSize * len(train_dataset)))
    train_dataset = train_dataset.skip(math.floor(args.valSize * len(train_dataset)))
    
    # Apply transformations to the data
    train_dataset = train_dataset.map(transformTrainData)
    val_dataset = val_dataset.map(transformValTestData)
    test_dataset = test_dataset.map(transformValTestData)

    if tf.config.list_physical_devices('GPU'):
        print("GPU is available. Using GPU.")
        device = '/device:gpu:0'
    else:
        print("GPU is not available. Using CPU.")
        device = '/device:cpu:0'

    with tf.device(device):
        # Load the model and optimizer and loss functions
        model = ModelFactory(args, args.model).fetch_model(args, num_classes=29)
        # Supported optimizers are SGD and Adam
        optimizer = optimizerFactory(args)
        # Supported loss functions are categorical_crossentropy and sparse_categorical_crossentropy
        loss = lossFactory(args)
        # Supported metrics are precision, recall, average, and macro-averaged f1 score
        metrics = metricFactory(args, num_classes = 29)
        # Compile the model: https://stackoverflow.com/questions/59353009/list-of-metrics-that-can-be-passed-to-tf-keras-model-compile 
        model.compile(optimizer = optimizer, loss = loss, metrics = metrics)
        # Callbacks for saving the model. Early stopping can be added if specified to the command line to prevent overfitting (callbacks)
        callbacks = checkpointProgress(scratch_dir, args, experiment)
        
        if args.comet:
            # Log with Comet as specified in the command line
            with experiment.train():
                train_history = model.fit(train_dataset, validation_data = val_dataset, epochs = args.nepoch, callbacks = callbacks)
            with experiment.test():
                test_history = model.evaluate(test_dataset, batch_size=args.batchSize)
        else:
            # Train the model without Comet logging
            train_history = model.fit(train_dataset, validation_data = val_dataset, epochs = args.nepoch, callbacks = callbacks)
            test_history = model.evaluate(test_dataset, batch_size=args.batchSize)
            
        # Save the history of the training run and testing run.
        json.dump(train_history.history, open(os.path.join(scratch_dir, 'trainhistory.json'), 'w')) 
        json.dump(test_history, open(os.path.join(scratch_dir, 'testhistory.json'), 'w'))
    return

if __name__ == "__main__":
    # Parser for easier running of the script on command line. Can specify hyperparameters and model type this way.  
    parser = argparse.ArgumentParser()
    parser.add_argument('-nepoch'   , type=int  , action="store", dest='nepoch'   , default=10   ) # number of epochs
    parser.add_argument('-batchSize', type=int  , action="store", dest='batchSize', default=32   ) # batch size
    parser.add_argument('-lr'       , type=float, action="store", dest='lr'       , default=0.001) # learning rate
    parser.add_argument('-resample' , type=int  , action="store", dest='resample' , default=42   ) # resample data
    parser.add_argument('-momentum' , type=float, action="store", dest='momentum' , default=9e-06  ) # for momentum SGD and RMSprop
    parser.add_argument('-model'    , type=str  , action="store", dest='model'    , default='resnet') # VGG, ResNet, etc...
    parser.add_argument('-optim'    , type=str  , action="store", dest='optim'    , default='adam') # SGD, adam, etc...
    parser.add_argument('-loss'     , type=str  , action="store", dest='loss'     , default='categorical_crossentropy')
    parser.add_argument('-val'      , type=float, action="store", dest='valSize'  , default=0.2  ) # validation percentage
    parser.add_argument('-test'     , type=float, action="store", dest='testSize' , default=0.1  )
    parser.add_argument('-stopping' , type=int , action="store" , dest='earlyStopping', default=None)
    parser.add_argument('-color'    , type=str  , action="store", dest='color'    , default='rgb') # rgb, grayscale 
    parser.add_argument('-img_size' , type=int  , action="store", dest='img_size' , default=64   ) # image size
    parser.add_argument('-data_dir' , type=str  , action="store", dest='data_dir' , default='./data/asl_alphabet_train/asl_alphabet_train/')
    parser.add_argument('-pretrain' , type=str  , action="store", dest='pretrain' , default=None) # use pretrained weights in specific directory
    parser.add_argument('-logits'   , type=bool , action="store", dest='from_logits', default=False) # has softmax been applied for probability? If not, then set to True. See source: https://datascience.stackexchange.com/questions/73093/what-does-from-logits-true-do-in-sparsecategoricalcrossentropy-loss-function 
    parser.add_argument('-metric'   , nargs='+', type=str, action="store", dest='metric', default=['accuracy']) # ['accuracy', 'precision', 'recall', 'f1_score'] NOTE: any other metric than accuracy is broken at the moment for sparse_categorical_crossentropy it appears
    parser.add_argument('-beta1'    , type=float, action="store", dest='beta1', default=0.9) # for Adam optimizer. Does nothing if specified and not using Adam optimizer
    parser.add_argument('-beta2'    , type=float, action="store", dest='beta2', default=0.999) # for Adam optimizer. Does nothing if specified and not using Adam optimizer
    parser.add_argument('-epsilon'  , type=float, action="store", dest='epsilon', default=1e-07) # for Adam optimizer. Does nothing if specified and not using Adam optimizer
    parser.add_argument('-nest'     , type=bool, action="store" , dest='nesterov', default=False) # for SGD optimizer. Does nothing if specified and not using SGD optimizer
    parser.add_argument('-comet'    , type=bool, action="store" , dest='comet', default=False) # for Comet logging. If False, then Comet logging is disabled
    args = parser.parse_args()
    main(args)