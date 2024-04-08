## **Setup Instructions**

**NOTE**: While I would love to provide a docker container, it is impossible to do with a HPC due to security concerns with root priviledge ([read more about it here](https://waterprogramming.wordpress.com/2022/05/25/containerizing-your-code-for-hpc-docker-singularity/)). Instead, you will have to set up your own environment.

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

3. Install and activate the environment.

   - `conda env create -f ./dependencies/environment.yml`
   - `conda activate ASLGPUenv`
   - Make sure your Jupyter notebook has the right kernel set with the env.

   OR

   - `conda create --name ASLGPUenv`
   - `conda activate ASLGPUenv`
   - `pip install -r ./dependencies/requirements.txt`
   - Make sure your Jupyter notebook has the right kernel set with the env.

## **Training Instructions**

If you would like, you can modify transforms within the **./train.py** file and **./util/transform.py** to include new custom or existing transforms (see the example grayscale() transform in transform.py and usage in train.py).

**Arguments**:

```
- Number of epochs:       "--nepoch 3"
- Batch size:             "--batchSize 32"
- Learning rate:          "--lr 0.001"
- Resampling data split:  "--resample 3"
- Weight decay:           "--wd 0"
- Momentum:               "--mo 1.00"
- Model:                  "--model VGG", "--model ResNet" (or any model you implement in ./model/ and incorporate into the ModelFactory class in ./util/model.py)
- Optimizer:              "--optim Adam", "--optim SGD"
- Loss function:          "--loss SparseCategoricalCrossEntropy", "--loss CategoricalCrossEntropy", anything you implement...
- Stopping after no imp.  "--stopping 10"
- Test perc.              "--test 0.2"
- Val perc.               "--val 0.2"
- Data directory          "--data_dir ./some/path/to/data/" (No reason to do this unless you want to recycle or completely modify repo)
- Pretraining             "--pretrain True" or "--pretrain False" (model weights should be stored under ./model/weights/X.weights.h5 where X is model name like "VGG" or "ResNet", etc...)
```

**Examples**:

`python3 ./train.py --nepochs 3 --batchSize 32 --lr 0.0001 --resample 3 --wd 0 --mo 0.98 --model ResNet --optim Adam --loss SparseCategoricalEntropy --test 0.2 --val 0.2`

or (depending on python version)

`python ./train.py --nepochs 3 --batchSize 32 --lr 0.0001 --resample 3 --wd 0 --mo 0.98 --model ResNet --optim Adam --loss SparseCategoricalEntropy --test 0.2 --val 0.2`

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

## **Repo Structure**

- **train.sbatch**: batch file for submitting training and other high compute jobs to HPC Slurm scheduler.
- **train.py**: a script version that is runnable in our batch file for high performance computer training.
- **exploratory_analysis.ipynb**: the working file for all visualizations.
- **model_experimentation.ipynb**: working file for testing and making models.
- **util**: a directory containing important utility that will be used to efficiently and effectively adapt our code for memory/data management constraints, custom transforms, directory functions, model utilities, the trainloop.
- **model**: a directory storing tf models and their weights if interested in pretraining. Weights should be saved to ./model/weights/VGG.weights.h5 and models as ./model/VGG.py for example.
- **figures**: containing visualizations made from notebooks or with comet_ml

## **Dependencies**

Have installed Python and Conda. For GPU users who plan on training, you will need python version 9 - 11. The latest version of Python does not seem to support tensorflow w/ gpu yet.
See CPUrequirements.txt and/or CPUenvironment.yml. Consider [conda](https://docs.conda.io/en/latest/).
