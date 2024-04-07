## **Setup Instructions**

1. Clone the repository.
2. Visit [this Kaggle page](https://www.kaggle.com/datasets/grassknoted/asl-alphabet), download the dataset, and extract the data into a folder you must name /data/. **WARNING**: please ensure that you clear enough space (several GBs). The dataset will be quite large. It should look like this when you are done:

```
root_directory
   |- exploratory_analysis.ipynb
   |- data
      |- asl_alphabet_test
         |- data...
      |- asl_alphabet_train
         |- data...
```

## **Training Instructions**

**Arguments**:
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

**Examples**:
"python3 ./train.py --nepochs 3 --batchSize 32 --lr 0.0001 --resample 3 --wd 0 --mo 0.98 --model ResNet --optim Adam --loss SparseCategoricalEntropy --test 0.2 --val 0.2"

or (depending on python version)

"python ./train.py --nepochs 3 --batchSize 32 --lr 0.0001 --resample 3 --wd 0 --mo 0.98 --model ResNet --optim Adam --loss SparseCategoricalEntropy --test 0.2 --val 0.2"

**Prerequisites**

- A decent GPU. For reference, I am training on GPUs that allow as much as 96 GB RAM on a HPC.
- SLURM for managing and scheduling Linux clusters. You will need to adapt the sbatch file to your needs. Or if running locally have installed python and run:
  "python train.py --batchSize 32... other args..." (could be python3 or else depending on your version of python instead of "python")
- Already installed the dependencies listed below.
- The repo structure as seen below which should have been cloned as such.

If you would like, you can modify transforms within the ./train.py file

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

- train.sbatch: batch file for submitting training and other high compute jobs to HPC Slurm scheduler.
- train.py: a script version that is runnable in our batch file for high performance computer training.
- exploratory_analysis.ipynb: the working file for all visualizations.
- model_experimentation.ipynb: working file for testing and making models.
- util: a directory containing important utility that will be used to efficiently and effectively adapt our code for memory/data management constraints, custom transforms, directory functions, model utilities, the trainloop.
- model: a directory storing tf models and their weights if interested in pretraining
- figures: containing visualizations made from notebooks or with comet_ml

## **Dependencies**

- comet_ml
- sklearn
- tensorflow (preferably w/ gpu compatibility): "pip install tensorflow[and-cuda]"
- numpy
- pandas
- opencv-python
- tqdm
- matplotlib
