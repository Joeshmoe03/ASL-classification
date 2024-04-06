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

**Prerequisites**

- A decent GPU. For reference, I am training on GPUs that allow as much as 96 GB RAM on a HPC.
- SLURM for managing and scheduling Linux clusters or if running locally have install python and run:
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
