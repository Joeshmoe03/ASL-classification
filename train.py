# Load dependencies
from comet_ml import Experiment
from tensorflow.keras.utils import image_dataset_from_directory # type: ignore
import tensorflow as tf
import os
import argparse
import json
from util.model import ModelFactory, optimizerFactory, lossFactory, metricFactory
from util.transform import transformTrainData, transformValTestData
# Initialize Comet experiment
experiment = Experiment(api_key="Fl7YrvyVQDhLRYuyUfdHS3oE8", 
                        project_name="asl",
                        workspace="joeshmoe03",
                        auto_output_logging="default",
                        auto_param_logging = True,
                        auto_metric_logging = True,
                        auto_log_co2=True,)

def main(args):
    '''
    Main function for training the model. This is the entry point for the script. 
    We load the data, create the model, compile the model, and train the model.
    The model is saved in the /temp/ directory under a folder named according to 
    the sampling and hyperparameters.

    Args:
        args: command-line arguments. These are the hyperparameters for the model/model.
    '''
    # We save training runs and their associated sampling of data in the /temp/ 
    # directory under a folder named according to the sampling and hyperparameters.
    scratch_dir = os.path.join(os.getcwd(), 'temp', f"{args.model}_{args.optim}_{args.loss}" 
                                                  + f"_lr{args.lr}_mo{args.momentum}" 
                                                  + f"_rs{args.resample}" + f"{args.img_size}")
    
    # Create the directory if it does not exist
    if not os.path.exists(scratch_dir):
        os.makedirs(scratch_dir)

    # Load the data into a tf.data.Dataset
    train_dataset = image_dataset_from_directory(args.data_dir, 
                                                labels = 'inferred', 
                                                # Specify label encoding type based on loss function (one hot for categorical_crossentropy, int for sparse_categorical_crossentropy)
                                                label_mode = 'int' if args.loss == 'sparse_categorical_crossentropy' else 'categorical', 
                                                color_mode = args.color, 
                                                batch_size = args.batchSize, 
                                                image_size = (args.img_size, args.img_size), 
                                                shuffle = True, 
                                                seed = args.resample)
    
    # Split the data into training, validation, and testing sets
    test_dataset = train_dataset.take(args.testSize)
    train_dataset = train_dataset.skip(args.testSize)
    val_dataset = train_dataset.take(args.valSize)
    train_dataset = train_dataset.skip(args.valSize)
    
    # Apply transformations to the data
    train_dataset = train_dataset.map(transformTrainData)
    val_dataset = val_dataset.map(transformValTestData)
    test_dataset = test_dataset.map(transformValTestData)
    
    with tf.device('/device:gpu:0'):
        # Load the model and optimizer and loss functions
        # Supported metrics are precision, recall, average, and macro-averaged f1 score
        # Compile the model: https://stackoverflow.com/questions/59353009/list-of-metrics-that-can-be-passed-to-tf-keras-model-compile 
        model = ModelFactory(args, args.model).fetch_model(args, num_classes=29)
        optimizer = optimizerFactory(args)
        loss = lossFactory(args)
        metrics = metricFactory(args)
        model.compile(optimizer = optimizer, loss = loss, metrics = metrics)

        # callbacks for saving the model
        # early stopping can be added if specified to the command line to prevent overfitting (callbacks)
        checkpointpath = os.path.join(scratch_dir, f'{str(args.model)}.h5')
        checkpoint = tf.keras.callbacks.ModelCheckpoint(checkpointpath, monitor = 'val_loss', verbose = 1, save_best_only = True, save_weights_only=True)
        callbacks = [checkpoint]
        if args.earlyStopping is not None:
            earlyStop = tf.keras.callbacks.EarlyStopping(monitor = 'val_loss', patience = args.earlyStopping)
            callbacks.append(earlyStop)
        
        with experiment.train():
            train_history = model.fit(train_dataset, validation_data = val_dataset, epochs = args.nepoch, callbacks = callbacks)

        with experiment.test():
            test_history = model.evaluate(test_dataset, batch_size=args.batchSize)
        
        # Save the history of the training run and testing run.
        json.dump(train_history.history, open(os.path.join(scratch_dir, 'trainhistory.json'), 'w')) 
        json.dump(test_history.history, open(os.path.join(scratch_dir, 'testhistory.json'), 'w'))
    return

if __name__ == "__main__":
    # Parser for easier running of the script on command line. Can specify hyperparameters and model type this way.  
    parser = argparse.ArgumentParser()
    parser.add_argument('-nepoch'   , type=int  , action="store", dest='nepoch'   , default=20   ) # number of epochs
    parser.add_argument('-batchSize', type=int  , action="store", dest='batchSize', default=32   ) # batch size
    parser.add_argument('-lr'       , type=float, action="store", dest='lr'       , default=0.001) # learning rate
    parser.add_argument('-resample' , type=int  , action="store", dest='resample' , default=42   ) # resample data
    parser.add_argument('-wd'       , type=float, action="store", dest='wd'   , default=None    ) # weight decay (currently deprecated with tensorflow)
    parser.add_argument('-momentum' , type=float, action="store", dest='momentum' , default=0.9  ) # for SGD
    parser.add_argument('-model'    , type=str  , action="store", dest='model'    , default='resnet') # VGG, ResNet, etc...
    parser.add_argument('-optim'    , type=str  , action="store", dest='optim'    , default='adam') # SGD, adam, etc...
    parser.add_argument('-loss'     , type=str  , action="store", dest='loss'     , default='categorical_crossentropy')
    parser.add_argument('-val'      , type=float, action="store", dest='valSize'  , default=0.2  ) # validation percentage
    parser.add_argument('-test'     , type=float, action="store", dest='testSize' , default=0.2  )
    parser.add_argument('-stopping' , type=int , action="store", dest='earlyStopping', default=3 )
    parser.add_argument('-color'    , type=str  , action="store", dest='color'    , default='rgb') # rgb, grayscale 
    parser.add_argument('-img_size' , type=int  , action="store", dest='img_size' , default=64   ) # image size
    parser.add_argument('-data_dir' , type=str  , action="store", dest='data_dir' , default='./data/asl_alphabet_train/asl_alphabet_train/')
    parser.add_argument('-pretrain' , type=str  , action="store", dest='pretrain' , default=None) # use pretrained weights in specific directory
    parser.add_argument('-logits'   , type=bool , action="store", dest='from_logits', default=False) # has softmax been applied for probability? If not, then set to True.
    # See source: https://datascience.stackexchange.com/questions/73093/what-does-from-logits-true-do-in-sparsecategoricalcrossentropy-loss-function 
    parser.add_argument('-metric'   , nargs='+', type=str, action="store", dest='metric', default=['accuracy']) # ['accuracy', 'precision', 'recall', 'f1_score']
    # NOTE: any other metric than accuracy is broken at the moment for sparse_categorical_crossentropy it appears
    args = parser.parse_args()
    main(args)