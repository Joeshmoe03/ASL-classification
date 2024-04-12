## **Setup Instructions**

**NOTE**: While I would love to provide a docker container, it is impossible to do with a HPC due to security concerns with root priviledge ([read more about it here](https://waterprogramming.wordpress.com/2022/05/25/containerizing-your-code-for-hpc-docker-singularity/)). Instead, you will have to set up your own environment. A future update may include singularity which allows some containerization without some issues of Docker.

1. Clone the repository.
2. Visit [this Kaggle page](https://www.kaggle.com/datasets/grassknoted/asl-alphabet), download the dataset, and extract the data into a folder you must name /data/. **WARNING**: please ensure that you clear enough space (about 9 GBs). The dataset will be quite large. It should look like this when you are done:

```
root_directory
   |- ...
   |- data
      |- asl_alphabet_test
         |- data...
      |- asl_alphabet_train
         |- data...
   |- ...
```

## **Training Instructions**

If you would like, you can modify transforms within the **./train.py** file and **./util/transform.py** to include new custom or existing transforms (see the example grayscale() transform in transform.py and usage in train.py).

**Prerequisites**

- A decent GPU. For reference, I am training on GPUs that allow as much as 96 GB RAM with HPC resources.
- [**If running on HPC**]: SLURM for managing and scheduling Linux clusters + Cuda availability with HPC resources. You will need to adapt the .sbatch file to your specific needs.
- Already installed the dependencies listed below.
- The repo structure as seen below which should have been cloned as such.

```
root_directory
   |- exploratory_analysis.ipynb
   |- model_experimentation.ipynb
   |- README.md
   |- train.sbatch
   |- train.py
   |- .gitignore
   |- model
      |- models...
   |- figures
      |- generated figures...
   |- util
      |- dataset.py
      |- directory.py
      |- trainloop.py
      |- model.py
      |- transform.py
   |- data
      |- asl_alphabet_test
         |- data...
      |- asl_alphabet_train
         |- data...
```

**Arguments**

```
- Number of epochs:       "-nepoch 3"
- Batch size:             "-batchSize 32"
- Learning rate:          "-lr 0.001"
- Resampling data split:  "-resample 3"
- Weight decay:           "-wd 0"
- Momentum:               "-mo 1.00"
- Model:                  "-model vgg", "-model resnet" (or any model you implement in ./model/ and incorporate into the ModelFactory class in ./util/model.py)
- Optimizer:              "-optim adam", "-optim sgd"
- Loss function:          "-loss sparse_categorical_cross_entropy", "-loss categorical_cross_entropy", anything you implement...
- Stopping after no imp.  "-stopping 10"
- Val perc.               "-val 0.2"
- Data directory          "-data_dir ./some/path/to/data/" (No reason to do this unless you want to recycle or completely modify repo)
- Pretraining             "-pretrain ./path/to/weights" (model weights should be stored under ./model/weights/X.weights.h5 where X is model name like "vgg" or "resnet", etc...)
- metric                  "-metric accuracy precision f1_score recall" or any combination of them. Does not work with sparse_categorical_cross_entropy loss.
- color                   "-color rgb" or "-color grayscale"
- logits                  "-logits True" or "-logits False" Adds softmax.
- img_size                "-img_size 200" ...
```

**Examples**:

`python train.py -nepoch 1 -batchSize 64 -lr 0.001 -metric accuracy precision recall f1_score`

or (depending on python version)

`python3 train.py -nepoch 1 -batchSize 64 -lr 0.001 -metric accuracy precision recall f1_score`

## **Repo Structure**

- **train.sbatch**: batch file for submitting training and other high compute jobs to HPC Slurm scheduler.
- **train.py**: a script version that is runnable in our batch file for high performance computer training.
- **exploratory_analysis.ipynb**: the working file for all visualizations.
- **model_experimentation.ipynb**: working file for testing and making models.
- **util**: a directory containing important utility that will be used to efficiently and effectively adapt our code.
- **model**: a directory storing tf models and their weights if interested in pretraining. Weights should be saved to ./model/weights/VGG.weights.h5 and models as ./model/VGG.py for example.
- **figures**: containing visualizations made from notebooks or with comet_ml
- **dependencies**: folder containing requirements.txt or alternatively environment.yml for setting up environment.

## **Dependencies**

Have installed Python and Conda. Consider making an env in [conda](https://docs.conda.io/en/latest/).

For CPU:

- A working version of python compatible with the following:
- scikit-learn
- numpy
- pandas
- tensorflow
- matplotlib
- argparse
- tqdm
- opencv-python
- comet-ml

For GPU:

- Python 3.10 specifically in order to be compatible with:
- tensorflow==2.10 (the last recent version of tensorflow with GPU compatibility for Windows w/o WSL)
- cudatoolkit==11.2.2 (install with pip and have cudatoolkit installed with driver > 5xx)
- cudnn==8.1.0 (with cudnn installed as well, and configured with cudatoolkit)
- comet-ml
- scikit-learn
- numpy
- pandas
- tqdm
- opencv-python
- argparse
- matplotlib

## **TODO**

- Implement support for more metrics
- Implement testing
