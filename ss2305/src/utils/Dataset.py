import random
import torch
import torchvision.transforms.functional as TF
from torchvision import datasets
from torch.utils.data import Dataset


class CustomDataset(datasets.ImageFolder):
    def __init__(self, root, transform=None, positive_transform=None, indices=None):

        super().__init__(root, transform=transform)
        # If indices are provided, create a list of samples from those indices
        if indices is not None:
            self.samples = [self.samples[i] for i in indices]

        self.rotated_and_scaled_samples = []
        self.positive_transform = positive_transform

        # Identify the index of the 'positive' class
        positive_class_index = self.class_to_idx['positive']

        # Rotate an image of class 'negative'
        if indices is not None:
            print(f'original train size : {len(indices)}')

        cont = 0
        for idx, (path, target) in enumerate(self.samples):
            path, target = self.samples[idx]
            if target == positive_class_index:
                cont += 1
                image = self.loader(path)
                for angle in [90, 180, 270]:
                    rotated_image = TF.rotate(image, angle)
                    # Optionally apply additional transformations
                    if self.positive_transform:
                        rotated_image = self.positive_transform(rotated_image)

                    # Append the image and its target to the rotated_samples list
                    self.rotated_and_scaled_samples.append((rotated_image, target))
        print(f'Number of positive : {cont}')
        print(f'Increased number of data {len(self.rotated_and_scaled_samples)}')

    def __getitem__(self, index):
        # Access both original samples and added rotated images
        if index < len(self.samples):
            return super().__getitem__(index)
        else:
            rotated_index = index - len(self.samples)
            image, target = self.rotated_and_scaled_samples[rotated_index]
            if self.transform is not None:
                image = self.transform(image)
            return image, target

    def __len__(self):
        return len(self.samples) + len(self.rotated_and_scaled_samples)


class CustomDatasetForTest(datasets.ImageFolder):
    def __getitem__(self, index):
        # Original __getitem__ fetches the image and label
        image, label = super(CustomDatasetForTest, self).__getitem__(index)
        # Fetch the path of the image file
        path = self.imgs[index][0]
        # Return image, label, and path
        return image, label, path
    # def __getitem__(self, index):
    #     image, label = super().__getitem__(index)
    #     # ここでさらに処理を行う
    #     return image, label



class CustomDatasetMNIST(Dataset):
    def __init__(self, root, transform=None, indices=None):
        self.mnist_dataset = datasets.MNIST(root=root, train=True, transform=transform, download=True)
        self.indices = indices
        if self.indices is not None:
            self.mnist_dataset = torch.utils.data.Subset(self.mnist_dataset, self.indices)

    def __getitem__(self, index):
        return self.mnist_dataset[index]

    def __len__(self):
        return len(self.mnist_dataset)
