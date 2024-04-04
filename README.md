## **Setup Instructions**

1. Clone the repository.
2. Visit [this Kaggle page](https://www.kaggle.com/datasets/grassknoted/asl-alphabet), download the dataset, and extract the data into a folder you must name /raw_data/. **WARNING**: please ensure that you clear enough space (several GBs). The dataset will be quite large.

- ASL.py contains the file to be executed in the HPC. It is essentially just whatever you wrote in the ASL.ipynb but adapted to the HPC.

## **Repo Structure**

- train.sbatch: batch file for submitting training and other high compute jobs to HPC Slurm scheduler.
- ASL.py: a script version that is runnable in our batch file for high performance computer training. This should be analagous to almost everything in ASL.py.
- util: a directory containing important utility that will be used to efficiently and effectively adapt our code for memory/data management constraints. Will contain useful things like transforms that can be applied to our images, data loaders, etc...
