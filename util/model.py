import tensorflow as tf
from model.ResNet import ResNet50
from model.simpleModel import ConvNet3, ConvNet4, ConvNet2
from model.vgg import Vgg16
import os

class ModelFactory():
    '''
    Note that none of our classes, ModelFactory, OptimizerFactory, or LossFactory, are setup for regression tasks. 
    This is because the ASL dataset is a multi-class classification task.
    '''
    def __init__(self, args, model_name: str):
        self.model_name = model_name
        self.model = None

    def fetch_model(self, args, num_classes: int):

        # Retrieve the model corresponding to specified arg on command-line
        if self.model_name == 'vgg':
            self.model = Vgg16(args.img_size, args.color, num_classes)
        elif self.model_name == 'resnet':
            self.model = ResNet50(args.img_size, args.color, num_classes) 
        elif self.model_name == 'convnet3':
            self.model = ConvNet3(args.img_size, num_classes)
        elif self.model_name == 'convnet4':
            self.model = ConvNet4(args.img_size, num_classes)
        elif self.model_name == 'convnet2':
            self.model = ConvNet2(args.img_size, num_classes)
        else:
            raise NotImplementedError(f'Model {self.model_name} not implemented')
        
        # Load the corresponding weights if specified arg on command-line
        if args.pretrain is not None: 
            self._load_weights(args)
        return self.model
    
    def _load_weights(self, args):
        # Retrieve the model weights from the specified directory.
        filepath = os.path.join(os.getcwd(), args.pretrain)
        if not os.path.exists(filepath):
            raise ValueError('Could not find model weights to load')
        
        # Use TF's load_weights method: https://www.tensorflow.org/api_docs/python/tf/keras/Model#load_weights 
        self.model = self.model.load_weights(filepath)
        return self.model
    
def optimizerFactory(args):
    '''
    Choose the optimizer based on the command-line arguments.
    '''
    if args.optim == 'sgd':
        return tf.keras.optimizers.SGD(learning_rate = args.lr, momentum = args.momentum, nesterov = args.nesterov)
    elif args.optim == 'adam':
        # Here is why Adam does not have "momentum" as an argument: https://stackoverflow.com/questions/47168616/is-there-a-momentum-option-for-adam-optimizer-in-keras
        return tf.keras.optimizers.Adam(learning_rate = args.lr, beta_1=args.beta1, beta_2=args.beta2, epsilon=args.epsilon)
    elif args.optim == 'rmsprop':
        return tf.keras.optimizers.RMSprop(learning_rate = args.lr, momentum = args.momentum)
    else:
        raise NotImplementedError(f'Optimizer {args.optim} not implemented')
    
def lossFactory(args):
    '''
    Choose the loss function based on the command-line arguments.
    '''
    if args.loss == 'sparse_categorical_crossentropy':
        return tf.keras.losses.SparseCategoricalCrossentropy(from_logits = args.from_logits)
    elif args.loss == 'categorical_crossentropy':
        return tf.keras.losses.CategoricalCrossentropy(from_logits = args.from_logits)
    elif args.loss == 'binary_crossentropy':
        return tf.keras.losses.BinaryCrossentropy(from_logits = args.from_logits)
    else:
        raise NotImplementedError(f'Loss {args.loss} not implemented')

