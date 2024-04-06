import os
from util.dataset import ASLBatchLoader, ASLDataPaths, split_data, save_data, load_saved_data

def samplingHandler(dir: str, args):
    '''
    This function exists SOLELY for reproducibility purposes. In ML, when benchmarking models against one another, it is a good idea for those models to be 
    trained on identical splittings of a given dataset. So how does this samplingHandler() achieve the desired behaviour?

    In our /temp/ folder, we can save and retrieve different train, val, test splittings of the ASL image paths and their corresponding labels. If
    a user specifies a new integer X to "-resample" argument on command line when running train.py:
        - We perform a new splitting, passing that integer X to random_state in sklearn train_test_split function, generating new splitting.
        - We then save that newly sampled splitting of the data as train, val, and test numpy arrays for X, y, under a folder: /temp/resampleX/
    If the user supplies a previously encountered integer X to "-resample":
        - We load the previously sampled data splitting from the correct directory to perform training/validation/testing.

    Parameters:
        dir: str - the directory corresponding to our resampling
        args: arguments supplied on the command line
    
    Returns:
        data_splits: tuple - a tuple of X_train, X_val, X_test, y_train, y_val, y_test
    '''
    # Create the temp directory if it does not exist or resample <- this training run/hyperparameter combo must not have been done or we are resampling
    if not os.path.exists(dir):
        print("Initializing directory for new sampled splits...")
        os.makedirs(dir)
        
        # We perform the resampling:
        # Load the unsampled data paths and labels, split the data into training, validation, and test sets for paths and labels
        # NOTE: data_splits takes on the form a tuple with the following structure: (X_train, X_val, X_test, y_train, y_val, y_test)
        print("Performing resampling of splits...")
        X_path, y = ASLDataPaths(args.data_dir).fetch_paths()
        data_splits = split_data(X_path, y, args)
        print("Saving resampled splits...")
        save_data(data_splits, dir)

    # We use a previous sampling:   
    # Load the data paths and labels to make reproducibility of sampling <- this training run/hyperparameter combo must have been run before, no resampling specified
    # NOTE: data_splits takes on the form a tuple with the following structure: (X_train, X_val, X_test, y_train, y_val, y_test)
    else:
        print("Loading existing sampled splits...")
        data_splits = load_saved_data(dir)

    return data_splits