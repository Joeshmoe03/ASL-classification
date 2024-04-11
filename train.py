# Load dependencies
from comet_ml import Experiment
import tensorflow as tf
import os
import argparse
import json
from keras.models import Sequential, Model # type: ignore
from util.model import ModelFactory, optimizerFactory, lossFactory, metricFactory
from util.transform import transformTrainData, transformValData

# Initialize Comet experiment
experiment = Experiment(api_key="Fl7YrvyVQDhLRYuyUfdHS3oE8", 
                        project_name="asl",
                        workspace="joeshmoe03",
                        auto_param_logging = True,
                        auto_metric_logging = True,
                        auto_histogram_weight_logging = True,
                        auto_histogram_gradient_logging = True,
                        auto_histogram_activation_logging = True,
                        auto_log_co2=True,)

def main(args):
    # We save training runs and their associated sampling of data in the /temp/ 
    # directory under a folder named according to the sampling and hyperparameters.
    scratch_dir = os.path.join(os.getcwd(), 'temp', f"{args.model}_op{args.optim}_ls{args.loss}" 
                                                  + f"_lr{args.lr}_mo{args.momentum}" 
                                                  + f"_rs{args.resample}")
    
    # Create the directory if it does not exist
    if not os.path.exists(scratch_dir):
        os.makedirs(scratch_dir)

    # Load the data into a tf.data.Dataset
    train_dataset, val_dataset = tf.keras.utils.image_dataset_from_directory(args.data_dir, 
                                                                            labels = 'inferred', 
                                                                            # Specify label encoding type based on loss function (one hot for categorical_crossentropy, int for sparse_categorical_crossentropy)
                                                                            label_mode = 'int' if args.loss == 'sparse_categorical_crossentropy' else 'categorical', 
                                                                            color_mode = args.color, 
                                                                            batch_size = args.batchSize, 
                                                                            image_size = (args.img_size, args.img_size), 
                                                                            shuffle = True, 
                                                                            seed = args.resample, 
                                                                            validation_split = args.valSize, 
                                                                            subset = "both")
    
    # Apply transformations to the data
    train_dataset = train_dataset.map(transformTrainData)
    val_dataset = val_dataset.map(transformValData)

    # Load the model and optimizer and loss functions
    model = ModelFactory(args, args.model).fetch_model(args, num_classes=29)
    optimizer = optimizerFactory(args)
    loss = lossFactory(args)

    # Supported metrics are precision, recall, average, and macro-averaged f1 score
    metrics = metricFactory(args)

    # Compile the model
    model.compile(optimizer = optimizer, loss = loss, metrics = metrics)

    # callbacks for saving the model
    checkpointpath = os.path.join(scratch_dir, f'{str(args.model)}.keras')
    checkpoint = tf.keras.callbacks.ModelCheckpoint(checkpointpath, monitor = 'val_loss', verbose = 1, save_best_only = True)
    callbacks = [checkpoint]

    # early stopping can be added if specified to the command line to prevent overfitting (callbacks)
    if args.earlyStopping is not None:
        earlyStop = tf.keras.callbacks.EarlyStopping(monitor = 'val_loss', patience = args.earlyStopping)
        callbacks.append(earlyStop)
    
    # Train the model and output history to comet
    with experiment.train():
        history = model.fit(train_dataset, validation_data = val_dataset, epochs = args.nepoch, callbacks = callbacks)

    # Save the history of the training run
    json.dump(history.history, open(os.path.join(scratch_dir, 'history.json'), 'w'))
    return

if __name__ == "__main__":
    # Parser for easier running of the script on command line. Can specify hyperparameters and model type this way.  
    parser = argparse.ArgumentParser()
    parser.add_argument('-nepoch'   , type=int  , action="store", dest='nepoch'   , default=20   ) # number of epochs
    parser.add_argument('-batchSize', type=int  , action="store", dest='batchSize', default=32   ) # batch size
    parser.add_argument('-lr'       , type=float, action="store", dest='lr'       , default=0.001) # learning rate
    parser.add_argument('-resample' , type=int  , action="store", dest='resample' , default=42   ) # resample data
    parser.add_argument('-wd'       , type=float, action="store", dest='wd'   , default=0    ) # weight decay (currently deprecated with tensorflow)
    parser.add_argument('-momentum' , type=float, action="store", dest='momentum' , default=0.9  ) # for SGD
    parser.add_argument('-model'    , type=str  , action="store", dest='model'    , default='resnet') # VGG, ResNet, etc...
    parser.add_argument('-optim'    , type=str  , action="store", dest='optim'    , default='sgd') # SGD, adam, etc...
    parser.add_argument('-loss'     , type=str  , action="store", dest='loss'     , default='categorical_crossentropy')
    parser.add_argument('-val'      , type=float, action="store", dest='valSize'  , default=0.2  ) # validation percentage
    parser.add_argument('-stopping'  , type=int , action="store", dest='earlyStopping', default=5)
    parser.add_argument('-color'    , type=str  , action="store", dest='color'    , default='rgb') # rgb, grayscale 
    parser.add_argument('-img_size' , type=int  , action="store", dest='img_size' , default=224  ) # image size
    parser.add_argument('-data_dir' , type=str  , action="store", dest='data_dir' , default='./data/asl_alphabet_train/asl_alphabet_train/')
    parser.add_argument('-pretrain' , type=str  , action="store", dest='pretrain' , default=None) # use pretrained weights in specific directory
    parser.add_argument('-logits'   , type=bool , action="store", dest='from_logits', default=False) # has softmax been applied for probability? If not, then set to True.
    # See source: https://datascience.stackexchange.com/questions/73093/what-does-from-logits-true-do-in-sparsecategoricalcrossentropy-loss-function 
    parser.add_argument('-metric'   , nargs='+', type=str, action="store", dest='metric', default=['accuracy']) # ['accuracy', 'precision', 'recall', 'f1_score']
    # NOTE: any other metric than accuracy is broken at the moment for sparse_categorical_crossentropy it appears
    args = parser.parse_args()
    main(args)