## **Setup Instructions**

The starting points for this notebook that you should look at are this README and the `end_to_end.ipynb` file where we detail everything. We DO NOT expect you to train the model yourself. Instead we explain how we perform the training with `train.py` and simply refer to the saved model weights in the notebook.

**NOTE**: While I would love to provide a docker container, it is impossible to do with a HPC due to security concerns with root priviledge ([read more about it here](https://waterprogramming.wordpress.com/2022/05/25/containerizing-your-code-for-hpc-docker-singularity/)). Instead, you will have to set up your own environment. Also note that I can't come up with a containerized set-up approach to setting things up as the
requirements change based on your hardware capabilities.

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

3. Create a conda environment named ASLenv: `conda create --name ASLenv python=3.10` and `conda activate ASLenv`
4. Install dependencies:

   Iff using Nvidia GPU w/ Windows - tested on Windows:

   - Iff GPU: Install [CudaNN 8.6](https://developer.nvidia.com/cudnn-downloads) and [cuda-toolkit v11.8](https://developer.nvidia.com/cuda-downloads)
   - Iff GPU: Update your Nvidia driver to at least version 5xx.
   - Visit [pypi.org to download tensorflow-gpu v2.10 .whl file](https://pypi.org/project/tensorflow-gpu/2.10.0/)
   - Run: `pip install tensorflow_gpu-2.10.0-cp310-cp310-win_amd64.whl` from whatever directory you downloaded to.
   - `pip install -r ./dependencies/GPUrequirements.txt`

   Elif using MacOS:

   - Navigate to home directory and exit from any current conda environment
   - Download [miniforge](https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-MacOSX-arm64.sh) to install the package for a workaround for conda on M1 Macs.
   - Install miniforge: `sh ~/Downloads/Miniforge3-MacOSX-arm64.sh`
   - Activate miniforge: `source ~/miniforge3/bin/activate`
   - Install apple-specific TF dependencies: `conda install -c apple tensorflow-deps==2.9.0`
   - Create a new conda environment: `conda create --name ASLenv python=3.10`
   - Acitvate the environment: `conda activate ASLenv`
   - Install additional dependencies:
     - `python -m pip install tensorflow-macos==2.9.0`
     - `python -m pip install tensorflow-metal==0.5.0`
   - Enter project directory in ASLenv and run: `pip install -r ./dependencies/CPUrequirements.txt`
   - Visit [pypi.org to download tensorflow-macos v2.10 .whl file](https://pypi.org/project/tensorflow-macos/2.10.0/). Download either x86 or ARM64 depending on your device.
   - Run:
     - For M1/apple silicon: `pip install tensorflow_macos-2.10.0-cp310-cp310-macosx_12_0_arm64.whl`
     - For intel silicon: `pip install tensorflow_macos-2.10.0-cp310-cp310-macosx_12_0_x86_64.whl`
   - Finally, install jupyter notebook deps: `python -m pip install ipykernel`
   - Additional Mac Tensorflow install troubleshooting resources:
     - [Apple Forum](https://forums.developer.apple.com/forums/thread/689300)
     - [Medium Article](https://medium.com/geekculture/installing-tensorflow-on-apple-silicon-84a28050d784)
     - [YouTube Video](https://www.youtube.com/watch?v=WFIZn6titnc)

## **Training**

Note that to perform training, you will need to:

1.  Have access to decent hardware. I _strongly_ encourage you to work with a device that has a GPU. [**If running on HPC**]: SLURM for managing and scheduling Linux clusters + CudaToolKit and cudnn availability with HPC resources. You will need to adapt the .sbatch file to your specific needs.
2.  Visit [comet-ml](https://www.comet.com/site/) and create an account for free. This will allow you to easily track training progress on HPCs or on your own device from a web interface! No need to check log outputs.
3.  For safety reasons, get your API key from comet-ml and in your own environment [create your environment variable named "COMET_API_KEY"](https://networkdirection.net/python/resources/env-variable/). You may need a restart for this to take effect.
4.  Replace my workspace "joeshmoe03" with yours in train.py.

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

`python train.py -nepoch 1 -batchSize 32 -lr 0.001 -metric accuracy precision recall f1_score -model convnet4`

## **Repo Structure**

- **train.sbatch**: batch file for submitting training and other high compute jobs to HPC Slurm scheduler.
- **train.py**: a script version that is runnable in our batch file for high performance computer training.
- **end_to_end.ipynb**: the end-to-end jupyter notebook for our project.
- **exploratory_analysis.ipynb**: the working file for all visualizations.
- **model_experimentation.ipynb**: working file for testing and making models.
- **util**: a directory containing important utility that will be used to efficiently and effectively adapt our code.
- **model**: a directory storing tf models and their weights if interested in pretraining. Pretrain weights should be saved to ./model/weights/VGG.weights.h5 and models as ./model/VGG.py for example.
- **figures**: containing visualizations made from notebooks or with comet_ml
- **dependencies**: folder containing requirements.txt for setting up environment.
- **temp**: all saved models and their associated tracked metric performances.

## **Helpful Notes**

For GPU users, [here is the video that finally got my tensorflow-gpu working](https://www.youtube.com/watch?v=NrJz3ACosJA)

## **TODO**

1. Implement VGG.py
2. Update ModelFactory
3. Try training on VGG

4. Object detection
5. Full product
