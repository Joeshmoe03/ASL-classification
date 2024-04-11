import tensorflow as tf
from model.ResNet import ResNet
from model.VGG import VGG
import os

class ModelFactory():
    '''
    Note that none of our classes, ModelFactory, OptimizerFactory, or LossFactory, are setup for regression tasks. 
    This is because the ASL dataset is a multi-class classification task.
    '''

    def __init__(self, args, model_name: str, pretrain: bool = False):
        self.model_name = model_name
        self.model = None
        self.pretrain = pretrain
        self._fetch_model(args)

    def _fetch_model(self, args):

        # Retrieve the model corresponding to specified arg on command-line
        if self.model_name == 'VGG':
            pass #self.model = VGG() #TODO: IMPLEMENT
        elif self.model_name == 'ResNet':
            self.model = ResNet() #TODO: IMPLEMENT
        else:
            raise NotImplementedError(f'Model {self.model_name} not implemented')
        
        # Load the corresponding weights if specified arg on command-line
        if self.pretrain: 
            self._load_weights(args)
        return self.model
    
    def _load_weights(self, args):
        # Retrieve the model weights from the specified directory.
        filepath = os.path.join(os.getcwd, args.pretrain, f'{str(self.model_name)}.weights.h5')
        if not os.path.exists(filepath):
            raise ValueError('Could not find model weights to load')
        
        # Use TF's load_weights method: https://www.tensorflow.org/api_docs/python/tf/keras/Model#load_weights 
        self.model = self.model.load_weights(filepath)