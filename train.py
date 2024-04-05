# Load dependencies
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.neural_network import MLPClassifier
import sklearn
from cv2 import imread
import cv2
import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import random
import argparse

# Set parameters for what kind of data we want to use
grayscale = True
img_size = 64

def main(args):
    #TODO: USE COMET_ML FOR LOGGING
    pass