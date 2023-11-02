from sklearn.datasets import fetch_lfw_people
import numpy as np
from torch.utils.data import TensorDataset, DataLoader, Dataset
import torchvision.transforms as T
import torch
from skimage.transform import resize

transform = T.Compose([
                T.ToTensor(),
                T.Resize((128,128))
            ])


class LFWDataLoader:
    def __init__(self, train_size=0.8, val_size=0.1, test_size=0.1, batch_size=10):
        lfw_people = fetch_lfw_people(data_home='datasets', color=True, resize=1, download_if_missing=True)
        data = lfw_people.images
        data = np.moveaxis(data, 3, 1)
        train, val, test = np.split(data,[int(train_size*len(data)), int((train_size+val_size)*(len(data)))])
        print(train.shape)
        print(val.shape)
        print(test.shape)

        train = torch.Tensor(train)
        self.train_dataset = TensorDataset(train)
        self.train_dataloader = DataLoader(self.train_dataset, batch_size=batch_size)
        val = torch.Tensor(val)
        self.val_dataset = TensorDataset(val)
        self.val_dataloader = DataLoader(self.val_dataset, batch_size=batch_size)
        test = torch.Tensor(test)
        self.test_dataset = TensorDataset(test)
        self.test_dataloader = DataLoader(self.test_dataset, batch_size=batch_size)
        