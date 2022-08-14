import numpy as np
import torch.nn as nn
import torchvision.models as models
import torch
import torch.utils.data
import dataset
import pandas as pd
import faceset
from learning_rule import AntiHebbian, Hebbian
from matplotlib import pyplot as plt


# FaRe model implementation


class Model(nn.Module):

    # model architecture
    def __init__(self, args):
        print('starting to load')
        super(Model, self).__init__()
        # defining CCN for feature extraction
        # with AlexNet
        if args.model == 'alexnet':
            # loading pre-trained CNN Alexnet
            self.features_extractor = models.alexnet(pretrained=True)
            # memory module parameters
            self.output = nn.Linear(4096, 4096, bias=False)
            # extracting Alexnet fc7 layer as an input for the MM
            self.features_extractor.classifier = nn.Sequential(*list(self.features_extractor.classifier.children())[:5])
        # with ResNet50
        elif args.model == 'resnet':
            class Identity(nn.Module):
                def __init__(self):
                    super(Identity, self).__init__()

                def forward(self, x):
                    return x
            # loading pre-trained CNN RestNet50
            self.features_extractor = models.resnet50(pretrained=True)
            # memory module parameters
            self.output = nn.Linear(2048, 2048, bias=False)
            self.features_extractor.fc = Identity()
        # initial values of connections strengths randomly distributed
        nn.init.uniform_(self.output.weight, args.min_weights, args.max_weights)
        # defining learning rule
        self.learning_rule = args.learning_rule
        print('model loaded')

    # defining feature extraction module
    def extract(self, x):
        # calling outputs from CNN
        y = self.features_extractor(x)
        # normalizing outputs from CNN
        return torch.div(y - y.mean(1, True), y.std(1, True).unsqueeze(1))

    # defining forward propagation
    def forward(self, x):
        if self.learning_rule == 'AntiHebbian':
            # applying feedforward to calculate the activity of novelty neurons
            return torch.matmul(self.output.weight.data.T, x.squeeze())
        elif self.learning_rule == 'Hebbian':
            return torch.matmul(self.output.weight.data - torch.diag(self.output.weight.data), x.squeeze())

    # defining backward propagation
    def mybackward(self, x):
        # applying learning rule on connection strengths
        weights = self.output.weight.data + x
        # updating weights
        self.output.weight.data = weights

    # learning loop
    def training_loop(self, dataset_size, args):
        print('training loop')
        # calling the learning rule from an other specific file
        if args.learning_rule == 'AntiHebbian':
            learning_function = AntiHebbian(args)
        elif args.learning_rule == 'Hebbian':
            learning_function = Hebbian(args)
        # defining the training set
        if args.dataset == 'dataset':
            data = dataset.TrainingSet(dataset_size, args)
        elif args.dataset == 'faceset':
            data = faceset.TrainingSet(dataset_size, args)
        # loading images
        loader = torch.utils.data.DataLoader(data, batch_size=1, shuffle=False, num_workers=4)
        # number of epochs
        for epoch in range(1):
            print('epoch number is', epoch)
            # browsing all images from the training set
            for images in loader:
                # feedforward
                input = self.extract(images)
                if args.learning_rule == "AntiHebbian":
                    output = self.forward(input)
                else:
                    output = None
                # showing histogram of the output layer activity
                # histo = output.numpy()
                # plt.hist(histo.flatten(), 100)
                # plt.show()
                # computing delta on the outputs
                delta = learning_function(output, input)
                # applying weights modification
                self.mybackward(delta)

    # model accuracy
    def testing_accuracy(self, args):
        print('loading testing dataset')
        # defining the testing set
        if args.dataset == 'dataset':
            data = dataset.TestingSet(args)
        elif args.dataset == 'faceset':
            data = faceset.TestingSet(args)
        print('dataset loaded')
        # loading testing set
        loader = torch.utils.data.DataLoader(data)
        score = 0
        # two images are feed-forward
        for train_image, test_image in loader:
            input_train = self.extract(train_image)
            input_test = self.extract(test_image)
            output_train = self.forward(input_train).reshape(-1)
            output_test = self.forward(input_test).reshape(-1)
            # showing output activity histograms for the two images
            histo_train = output_train.numpy()
            histo_test = output_test.numpy()
            fig, (ax1, ax2) = plt.subplots(1, 2)
            df_train = pd.DataFrame(histo_train)
            df_test = pd.DataFrame(histo_test)
            df_train.hist(bins=100, ax=ax1)
            df_test.hist(bins=100, ax=ax2)
            plt.show()
            # familiarity decision function
            # computing probability for the learning image
            if args.learning_rule == 'AntiHebbian':
                dx = output_train[output_train > output_train.median()].sum() - output_train[output_train <= output_train.median()].sum()
                # computing probability for the new image
                dz = output_test[output_test > output_test.median()].sum() - output_test[output_test <= output_test.median()].sum()
                # familiarity recognition if dx < dz then 1
                # error recognition if dx > dz then 0
                score += 1 if dx < dz else 0
            elif args.learning_rule == 'Hebbian':
                dx = torch.dot(input_train.reshape(-1), output_train)
                dz = torch.dot(input_test.reshape(-1), output_test)
                score += 1 if dx > dz else 0
        # computing error probability
        print('the error probability is ', 1 - score / len(loader))
        return 1 - score / len(loader)

    # compute recency
    def recency(self, args):
        print('loading testing dataset')
        # defining the testing set
        data = dataset.TestingSet(args)
        print('dataset loaded')
        # loading testing set
        loader = torch.utils.data.DataLoader(data)
        score = 0
        scores = []
        # testing pairs are showed in the same order as training
        for train_image, test_image in loader:
            input_train = self.extract(train_image)
            input_test = self.extract(test_image)
            output_train = self.forward(input_train).reshape(-1)
            output_test = self.forward(input_test).reshape(-1)
            # computing probability for the learning image
            if args.learning_rule == 'AntiHebbian':
                dx = output_train[output_train > output_train.median()].sum() - output_train[output_train <= output_train.median()].sum()
                # computing probability for the new image
                dz = output_test[output_test > output_test.median()].sum() - output_test[output_test <= output_test.median()].sum()
                # familiarity recognition if dx < dz then 1
                # error recognition if dx > dz then 0
                score += 1 if dx < dz else 0
            elif args.learning_rule == 'Hebbian':
                dx = torch.dot(input_train.reshape(-1), output_train)
                dz = torch.dot(input_test.reshape(-1), output_test)
                score += 1 if dx > dz else 0
            scores.append(1 - score / (len(scores) + 1))
        return np.array(scores)

    # histogram
    def histogram(self, args):
        print('loading testing dataset')
        # defining the testing set
        data = dataset.TestingSet(args)
        print('dataset loaded')
        # loading testing set
        loader = torch.utils.data.DataLoader(data)
        list_dx = []
        list_dz = []
        # two images are feed-forward
        for train_image, test_image in loader:
            input_train = self.extract(train_image)
            input_test = self.extract(test_image)
            output_train = self.forward(input_train).reshape(-1)
            output_test = self.forward(input_test).reshape(-1)
            # computing probability for the learning image
            if args.learning_rule == 'AntiHebbian':
                dx = output_train[output_train > output_train.median()].sum() - output_train[output_train <= output_train.median()].sum()
                # computing probability for the new image
                dz = output_test[output_test > output_test.median()].sum() - output_test[output_test <= output_test.median()].sum()
            elif args.learning_rule == 'Hebbian':
                dx = torch.dot(input_train.reshape(-1), output_train)
                dz = torch.dot(input_test.reshape(-1), output_test)
            list_dx.append(dx.item())
            list_dz.append(dz.item())
        # plotting distribution of familiarity scores
        df = pd.DataFrame({'dx': list_dx, 'dz': list_dz})
        df.plot.hist(bins=50, alpha=0.5)
        plt.show()

    # yes/no recognition task
    def threshold(self, args):
        print('loading testing dataset')
        # defining the testing set
        data = dataset.SetThreshold(args)
        print('dataset loaded')
        # loading testing set
        loader = torch.utils.data.DataLoader(data)
        score = 0
        for image, y in loader:
            input = self.extract(image)
            output = self.forward(input).reshape(-1)
            # computing probability for the learning image
            if args.learning_rule == 'AntiHebbian':
                dx = output[output > output.median()].sum() - output[output <= output.median()].sum()
                if y:
                    score += 1 if dx < args.threshold else 0
                else:
                    score += 1 if dx > args.threshold else 0
                # computing probability for the new image
            elif args.learning_rule == 'Hebbian':
                dx = torch.dot(input.reshape(-1), output)
                if y:
                    score += 1 if dx > args.threshold else 0
                else:
                    score += 1 if dx < args.threshold else 0
        # computing error probability
        print('the error probability is ', 1 - score / len(loader))
        return 1 - score / len(loader)
