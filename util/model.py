import tensorflow as tf
from model.ResNet import ResNet50
from model.simpleModel import ConvNet3, ConvNet4, ConvNet2
#from model.VGG import VGG
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
            pass #self.model = VGG() #TODO: IMPLEMENT
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
        return tf.keras.optimizers.SGD(learning_rate = args.lr, momentum = args.momentum, decay = args.wd)
    elif args.optim == 'adam':
        return tf.keras.optimizers.Adam(learning_rate = args.lr)
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
    else:
        raise NotImplementedError(f'Loss {args.loss} not implemented')
    
def metricFactory(args):
    '''
    Choose the metrics based on the command-line arguments.
    '''
    #TODO: EXPAND ON METRICS: https://stackoverflow.com/questions/59353009/list-of-metrics-that-can-be-passed-to-tf-keras-model-compile
    metrics = []
    for metric in args.metric:
        if metric == 'precision':
            metrics.append(tf.keras.metrics.Precision())
        elif metric == 'recall':
            metrics.append(tf.keras.metrics.Recall())
        elif metric == 'f1_score':
            metrics.append(tf.keras.metrics.F1Score(average = 'macro'))
        elif metric == 'accuracy':
            metrics.append(tf.keras.metrics.Accuracy())
        else:
            raise ValueError(f"Metric {metric} not supported")
    return metrics