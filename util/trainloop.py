from util.dataset import ASLDataPaths, ASLBatchLoader, split_data, save_data, load_saved_data
from util.model import ModelFactory, OptimizerFactory, LossFactory
import tqdm

def trainLoop(dir, loaders, model, args):

    # Unpack the data loader
    train_loader = loaders['train']
    val_loader = loaders['val']
    test_loader = loaders['test']

    # Initialize the optimizer
    optimizer = OptimizerFactory(args.optimizer, args.lr, args.momentum, args.wdecay)

    # Initialize the loss function
    loss_fn = LossFactory(args.loss, args)

    # TODO: Initialize the learning rate scheduler
    # TODO: Initialize the metrics

    # TODO: Initialize the progress bar

    # We train for the number of epochs specified in the arguments. An epoch represents a full pass through the training data.
    for epoch in range(args.epochs):
        epoch_loss = 0.0
        
        # Because our data is so large, we use batch training.
        with tqdm.tqdm(train_loader, desc=f'Epoch {epoch + 1}/{args.epochs}', unit='batch') as t:
            for batch in train_loader:
                pass
            
            #TODO: Implement training loop here
            # The training loop should:
            # 1. Zero the gradients
            # 2. Forward pass
            # 3. Compute loss
            # 4. Backward pass
            # 5. Optimize
            # 6. Update the learning rate
            # 7. Log the training loss
            # 8. Log the training metrics
            # 9. Log the training progress with tqdm
            