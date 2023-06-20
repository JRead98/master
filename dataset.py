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
        for cls in os.listdir(path):
            for imgs in os.listdir(os.path.join(path, cls)):
                if (args.only_jpg and imgs.endswith('.jpg')) or args.only_jpg is False:
                    list_images.append(os.path.join(path, cls, imgs))

        # random selection of the desired number of images
        self.images = random.choices(list_images, k=size)
        # normalizing images dimensions
        self.transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor(),
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
        path = args.datapath
        list_images = []
        for cls in os.listdir(path):
            for imgs in os.listdir(os.path.join(path, cls)):
                if (args.only_jpg and imgs.endswith('.jpg')) or args.only_jpg is False:
                    list_images.append(os.path.join(path, cls, imgs))

        self.transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor(),
                                             transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                  std=[0.229, 0.224, 0.225])])

        # selection of two images including one previous image from the training set as the first element
        # the second element is a new random image never seen by the model
        with open('pics', 'rb') as f:
            self.training_images = pickle.load(f)
        self.testing_images = []
        # exclusion of the training images for the selection of the second element
        to_exclude = set(self.training_images)
        for i in range(len(self.training_images)):
            while True:
                candidate = random.choice(list_images)
                if candidate not in to_exclude:
                    self.testing_images.append(candidate)
                    to_exclude.add(candidate)
                    break

    # defining dataset size
    def __len__(self):
        return len(self.testing_images)

    # selection of pairs of elements from the dataset to give to the model
    def __getitem__(self, item):
        image_training = Image.open(self.training_images[item]).convert('RGB')
        image_testing = Image.open(self.testing_images[item]).convert('RGB')
        return self.transform(image_training), self.transform(image_testing)

