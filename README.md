# Model for Familiarity Recognition 

Connectionnist model for Familiarity Recognition designed as part of my master's thesis.

Dataset used for simulations are available in the following links:

Caltech256 : https://www.kaggle.com/datasets/jessicali9530/caltech256 (Griffin et al., 2007)

Cat Dataset : https://www.kaggle.com/datasets/crawford/cat-dataset (Zhang et al. 2008)

VGGFace : https://www.robots.ox.ac.uk/~vgg/data/vgg_face/ (Parkhi et al., 2015)

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

- --learning_rule (default: AntiHebbian): selection of the memory module

- --model (default: alexnet): selection of the extraction module

- --dataset (default: dataset): selection of the dataset

- --treshold (default: 0): familiarity treshold

These arguments are listed in the script `parsers.py`

# Selection of the model

For the Hebbian model, use the following arguments when launching a script:

- --model resnet

- --learning_rule Hebbian

- --lr 0.1

For the anti-Hebbian model (Kazanovich & Borisyuk, 2021), leave the default arguments.

# Simulation 1

To perform Simulation 1 which reproduces Standing's experiment (Standing, 1973):

- Select the directory (--datapath) corresponding to the dataset: Caltech256

- Leave the default --sizes argument: [20, 40, 100, 200, 400, 1000, 4000, 10000]

- Launch the script `testing.py`

```
python testing.py [arguments]
```

# Simulation 2

To perform Simulation 2, which explores the presence of recency and primacy effects:

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

# Simulation 4

To perform Simulation 4, which explores an effect of target orientation:

- Launch the script `downloader.py` to download VGGFace dataset

```
python downloader.py
```

- Select the directory (--datapath) corresponding to the previously downloaded dataset

- Use the argument --dataset faceset

- Launch the script `testing.py`

```
python testing.py [arguments]
```

# Yes-No Recognition Task

To perform a Yes-No recognition task:

- Select the directory (--datapath) corresponding to the dataset depending on the condition, respectively dissimilar or similar: Caltech256 or Cat Dataset

- If the Cat Dataset is selected, use the argument --only_jpg True

- Also use the following argument --sizes 400

- Launch the script `distribution.py`

```
python distribution.py [argmuents]
```

- Once the graph appears, find the point of intersection between the two curves

- Launch the script `y-n.py` with the --treshold argument set to the value correspond to this intersection

```
python y-n.py [arguments]
```
