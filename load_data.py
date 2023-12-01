from sklearn.datasets import fetch_lfw_people
import numpy as np
from torch.utils.data import TensorDataset, DataLoader, Dataset
import torchvision.transforms as T
import torch
from skimage.transform import resize
from PIL import Image
from scipy import ndimage

transform = T.Compose([
                T.ToTensor(),
                T.Resize((128,128))
            ])


class LFWDataLoader:
    def __init__(self, train_size=0.8, val_size=0.1, test_size=0.1, batch_size=10):
        lfw_people = fetch_lfw_people(data_home='datasets', color=True, resize=1, download_if_missing=True)
        data = lfw_people.images
        data = np.moveaxis(data, 3, 1)
        train, val, test = np.split(data, [int(train_size*len(data)), int((train_size+val_size)*(len(data)))])
        train_copy = train.copy()
        train1, train2, train3, train4, train5 = np.split(train_copy, [int(.2*len(train)), int(.4*(len(train))), int(.6*len(train)), int(.8*len(train))])
        
        for i in range(len(train1)):
            for j in range(3):
                train1[i][j] = ndimage.rotate(train1[i][j], 45, reshape=False, cval=1)
                train2[i][j] = ndimage.rotate(train2[i][j], 90, reshape=False, cval=1)
                train3[i][j] = ndimage.shift(train4[i][j], -30, order=0, cval=1)
                train4[i][j] = ndimage.shift(train4[i][j], 30, order=0, cval=1)
            train5[i] = ndimage.median_filter(train5[i], 3)
        train = np.concatenate((train, train1, train2, train3, train4, train5), axis=0)
        train_true = np.concatenate((train_copy,train_copy))
        print(train.shape)
        print(val.shape)
        print(test.shape)

        train = torch.Tensor(train)
        train_true = torch.Tensor(train_true)
        self.train_dataset = TensorDataset(train,train_true)
        self.train_dataloader = DataLoader(self.train_dataset, batch_size=batch_size, shuffle=True)
        val = torch.Tensor(val)
        self.val_dataset = TensorDataset(val,val)
        self.val_dataloader = DataLoader(self.val_dataset, batch_size=batch_size, shuffle=True)
        test = torch.Tensor(test)
        self.test_dataset = TensorDataset(test,test)
        self.test_dataloader = DataLoader(self.test_dataset, batch_size=batch_size, shuffle=True)
