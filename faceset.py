from torchvision import transforms
import os
import random
from PIL import Image
import torch.utils.data
import pickle


# creation of the training set
class TrainingSet(torch.utils.data.Dataset):

    # defining training set
    def __init__(self, size, args):
        # choose correct path
        path = args.datapath
        list_images = []
        for imgs in os.listdir(path):
            if (args.only_jpg and imgs.endswith('.jpeg')) or args.only_jpg is False:
                list_images.append(os.path.join(path, imgs))

        # random selection of the desired number of images
        self.images = random.choices(list_images, k=size)
        # normalizing images dimensions
        self.transform = transforms.Compose([transforms.Resize((227, 227)), transforms.ToTensor(),
                                             transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                  std=[0.229, 0.224, 0.225])])

        # saving images for training
        with open('pics', 'wb') as f:
            pickle.dump(self.images, f)

    # defining dataset size
    def __len__(self):
        return len(self.images)

    # selection of elements from the dataset to give to the model
    def __getitem__(self, index):
        img = Image.open(self.images[index]).convert('RGB')
        return self.transform(img)


# defining testing set
class TestingSet(torch.utils.data.Dataset):

    def __init__(self, args):
        desired_class = args.datapath
        path = desired_class[:desired_class.find('/')]
        desired_class = desired_class[desired_class.find('/') + 1:]
        self.testing_images = []
        classes = os.listdir(path)
        idx = classes.index(desired_class)
        other_classes = classes[:idx] + classes[idx + 1:]
        for other_class in other_classes:
            for imgs in os.listdir(os.path.join(path, other_class)):
                if (args.only_jpg and imgs.endswith('.jpeg')) or args.only_jpg is False:
                    self.testing_images.append(os.path.join(path, other_class, imgs))

        self.transform = transforms.Compose([transforms.Resize((227, 227)), transforms.ToTensor(),
                                             transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                  std=[0.229, 0.224, 0.225])])

        self.training_images = []
        for imgs in os.listdir(os.path.join(path, desired_class)):
            if (args.only_jpg and imgs.endswith('.jpeg')) or args.only_jpg is False:
                self.training_images.append(os.path.join(path, desired_class, imgs))
        # selection of two images including one previous image from the training set as the first element
        # the second element is a new random image never seen by the model
        with open('pics', 'rb') as f:
            training_images = pickle.load(f)
            size = len(training_images)
            self.training_images = list(set(self.training_images).difference(set(training_images)))
        self.training_images = random.sample(self.training_images, k=size)
        self.testing_images = random.sample(self.testing_images, k=size)

    # defining dataset size
    def __len__(self):
        return len(self.testing_images)

    # selection of pairs of elements from the dataset to give to the model
    def __getitem__(self, item):
        image_training = Image.open(self.training_images[item]).convert('RGB')
        image_testing = Image.open(self.testing_images[item]).convert('RGB')
        return self.transform(image_training), self.transform(image_testing)
