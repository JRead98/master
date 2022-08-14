# Model for Familiarity Recognition 

Connectionnist model for Familiarity Recognition designed as part of my master's thesis

Les datasets utilisés pour les différentes simulations sont disponibles aux adresses suivantes :

Caltech256 : https://www.kaggle.com/datasets/jessicali9530/caltech256 (Griffin et al., 2007)

Cat Dataset : https://www.kaggle.com/datasets/crawford/cat-dataset (Zhang et al. 2008)

VGGFace : https://www.robots.ox.ac.uk/~vgg/data/vgg_face/ (Parkhi et al., 2015)

# Conditions

La librairie a été testée sur Python 3.9.11

```
pip install -r requirements.txt
```

Les librairies nécéssaires sont dans le fichier `requirements.txt`

# Arguments

Les arguments suivants peuvent être modifiés :

- --datapath : répertoire dans lequel se trouve le dataset

- --only_jpg (défaut : False) : True pour sélectionner uniquement les fichiers .pjg

- --filename (défaut : simulation.csv) : nom du fichier dans lequel vont s'enregistrer les résultats des simulations 

- --sizes (défaut : [20, 40, 100, 200, 400, 1000, 4000, 10000]) : nombres d'images apprises pendant l'entraînement

- --lr (défaut : 0.01) : constante d'apprentissage

- --min_weights (défaut : -1) : valeur minimum des poids à l'initialisation

- --max_weights (défaut : 1) : valeur maximum des poids à l'initialisation

- --run (défaut : 100) : nombre de fois qu'une simulation sera effectuée

- --learning_rule (défaut : AntiHebbian) : choix de la règle d'apprentissage

- --model (défaut : alexnet) : choix du module d'extraction

- --dataset (défaut : dataset) : choix du dataset utilisé

- --treshold (défaut : 0) : seuil de familiarité

Ces arguments se trouvent dans le script `parsers.py`

# Choix du modèle

Pour utiliser le modèle Hebbien, encodez les arguments suivants lorsque vous lancez un script :

- --model resnet

- --learning_rule Hebbian

- --lr 0.1

Pour utiliser le modèle anti-Hebbien (Kazanovich & Borisyuk, 2021), laissez les arguments par défaut.

# Simulation 1

Pour réaliser la Simulation 1, reproduisant l'expérience de Standing (1973) :

- Sélectionnez le répertoire (--datapath) correspondant au dataset : Caltech256

- Laissez l'argument --sizes par défaut : [20, 40, 100, 200, 400, 1000, 4000, 10000]

- Lancez le script `testing.py`

```
python testing.py [arguments]
```

# Simulation 2

Pour réaliser la Simulation 2, qui explore la présence des effets de récence et de primauté :

- Sélectionnez le répertoire (--datapath) correspondant au dataset : Caltech256

- Utilisez l'argument --sizes désiré

- Lancez le script `recency.py`

```
python recency.py [arguments]
```

# Simulation 3

Pour réaliser la Simulation 3, qui explore la présence d'un effet de similarité :

- Sélectionnez le répertoire (--datapath) correspondant au dataset : Cat Dataset

- Utilisez les arguments suivants : --only_jpg True --sizes 40

- Lancez le script `testing.py`

```
python testing.py [arguments]
```

# Simulation 4

Pour réaliser la Simulation 4, qui explore un effet de l'orientation de la cible :

- Lancez le script `downloader.py` pour télécharger le dataset VGGFace

```
python downloader.py
```

- Sélectionnez le répertoire (--datapath) correspondant au dataset téléchargé via le downloader

- Utilisez l'argument --dataset faceset

- Lancez le script `testing.py`

```
python testing.py [arguments]
```

# Test de reconnaissance Oui-Non

Pour réaliser un test de reconnaissance Oui-Non :

- Sélectionnez le répertoire (--datapath) correspondant au dataset : Caltech256 ou Cat Dataset

- Si vous réalisez la simulation sur le Cat Dataset, utilisez l'argument --only_jpg True

- Sélectionnez l'argument --sizes 400

- Lancez le script `distribution.py`

```
python distribution.py [argmuents]
```

- Une fois que le graphique apparait, trouvez le point d'intersection entre les deux courbes

- Lancez le scripts `y-n.py` avec l'argument --treshold à la valeur correspondant à cette intersection

```
python y-n.py [arguments]
```
