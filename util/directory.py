import os
import tensorflow as tf

def initScratchDir(args):
    '''
    Initialize the directory for saving the model and logs. The directory is named 
    according to the model, optimizer, loss function, and hyperparameters.
    '''
    scratch_dir = os.path.join(os.getcwd(), 'temp', f"{args.model}_{args.optim}_{args.loss}"
                               +f"_lr{args.lr}_mo{args.momentum}"+f"_rs{args.resample}"
                               +f"{args.img_size}")
    
    # Create the directory if it does not exist
    if not os.path.exists(scratch_dir):
        os.makedirs(scratch_dir)
    return scratch_dir

def checkpointProgress(scratch_dir, args, experiment):
    '''
    Save the model if either early stopping is enabled or the model has the lowest validation loss.
    '''

    # Save the model in the scratch directory: https://www.tensorflow.org/api_docs/python/tf/train/Checkpoint 
    # Also used ChatGPT here to figure out how to work with the saved model weights and file formatting. 
    checkpoint_dir = os.path.join(scratch_dir, f"{args.model}.h5")

    # Save the model if it has the lowest validation loss at the end of an epoch: 
    # https://www.tensorflow.org/tutorials/keras/save_and_load <- how to save a model and weights
    # https://www.tensorflow.org/api_docs/python/tf/keras/callbacks/ModelCheckpoint <- ModelCheckpoint callback
    checkpoint = tf.keras.callbacks.ModelCheckpoint(checkpoint_dir, monitor = 'val_loss', verbose = 1, save_best_only = True, save_weights_only = True)
    callbacks = [checkpoint]

    # Add early stopping if specified
    if args.earlyStopping is not None:
        callbacks.append(tf.keras.callbacks.EarlyStopping(monitor = 'val_loss', patience = args.earlyStopping))

    # Add a callback for logging all passed metrics to comet
    if args.comet:

        # Source for how we figured it out: https://www.tensorflow.org/api_docs/python/tf/keras/callbacks/LambdaCallback
        callbacks.append(tf.keras.callbacks.LambdaCallback(on_epoch_end = lambda epoch, logs: experiment.log_metrics(logs, step = epoch)))

    return callbacks