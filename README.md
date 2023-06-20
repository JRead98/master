# Model for Familiarity Recognition 

Connectionnist model for Familiarity Recognition designed as part of my master's thesis.

Dataset used for simulations are available in the following links:

Caltech256 : https://www.kaggle.com/datasets/jessicali9530/caltech256 (Griffin et al., 2007)

Cat Dataset : https://www.kaggle.com/datasets/crawford/cat-dataset (Zhang et al. 2008)

# Requirements

The library was tested with python 3.9.11

```
pip install -r requirements.txt
```

The required libraries are listed in the file `requirements.txt`

# Arguments

The following arguments can be modified in the command line:

- --datapath: path to the directory where the dataset is located

- --only_jpg (default: False): True to select only .pjg files

- --filename (default: simulation.csv): filename in which the results of the simulations will be saved

- --sizes (default: [20, 40, 100, 200, 400, 1000, 4000, 10000]): number of images presented during training

- --lr (default: 0.01): learning rate

- --min_weights (default: -1): minimum value of weights at initialization

- --max_weights (default: 1): maximum value of weights at initialization

- --run (default: 100): number of runs

- --learning_rule (default: Hebbian): selection of the memory module

- --model (default: resnet): selection of the extraction module

These arguments are listed in the script `parsers.py`

# Selection of the model

For the Hebbian model, use the default arguments when launching the `testing.py` script : 

- --model resnet

- --learning_rule Hebbian

- --lr 0.01

For the anti-Hebbian model, use the following arguments when launching the `testing.py` script:

- --model resnet

- --learning_rule AntiHebbian

- --lr 0.01

# Simulation 1

To perform Simulation 1 which reproduces Standing's experiment (Standing, 1973):

- Select the directory (--datapath) corresponding to the dataset: Caltech256

- Leave the default --sizes argument: [20, 40, 100, 200, 400, 1000, 4000, 10000]

- Launch the script `testing.py`

```
python testing.py [arguments]
```

# Simulation 2

To perform Simulation 2, which explores the recency inside the models:

- Select the directory (--datapath) corresponding to the dataset: Caltech256

- Choose the desired --sizes argument

- Launch the script `recency.py`

```
python recency.py [arguments]
```

# Simulation 3

To perform Simulation 3, which explores the presence of a similarity effect:

- Select the directory (--datapath) corresponding to the dataset: Cat Dataset

- Use the following arguments: --only_jpg True --sizes 40

- Launch the script `testing.py`

```
python testing.py [arguments]
```
Note that you will have to perform this simulation again for the two other homogeneity conditions with their corresponding dataset.
