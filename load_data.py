from sklearn.datasets import fetch_lfw_people
import numpy as np
from torch.utils.data import TensorDataset, DataLoader, Dataset
import torchvision.transforms as T
import torch
from skimage.transform import resize
from PIL import Image
from scipy import ndimage
import cv2



class LFWDataSet(Dataset):
    def __init__(self, data, x_transform, y_transform):
        self.data = data
        self.x_transform = x_transform
        self.y_transform = y_transform

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        image = (255*self.data[index]).astype('uint8')
        # image = np.moveaxis(self.data[index],0,2)
        return self.x_transform(image), self.y_transform(image)

class LFWDataLoader:
    def __init__(self, train_size=0.8, val_size=0.1, test_size=0.1, batch_size=10):
        lfw_people = fetch_lfw_people(data_home='datasets', color=True, resize=1, download_if_missing=True)
        data = lfw_people.images
        train, val, test = np.split(data, [int(train_size*len(data)), int((train_size+val_size)*(len(data)))])

        train_transform = T.Compose([
            T.ToPILImage(),
            T.Resize((64,64)),
            T.RandomApply([
                T.RandomVerticalFlip(0.5),
                T.RandomHorizontalFlip(0.5),
                T.RandomApply([
                    T.RandomRotation(90)
                ],0.5),
                T.RandomAffine(degrees = 0, translate = (0.25, 0.25))
            ],0.5),
            T.ToTensor()
        ])

        transform = T.Compose([
            T.ToPILImage(),
            T.Resize((64,64)),
            T.ToTensor()
        ])

        # train_copy = train.copy()
        # train1, train2, train3, train4, train5 = np.split(train_copy, [int(.2*len(train)), int(.4*(len(train))), int(.6*len(train)), int(.8*len(train))])
        
        # for i in range(len(train1)):
        #     for j in range(3):
        #         train1[i][j] = ndimage.rotate(train1[i][j], 45, reshape=False, cval=1)
        #         train2[i][j] = ndimage.rotate(train2[i][j], 90, reshape=False, cval=1)
        #         train3[i][j] = ndimage.shift(train4[i][j], -30, order=0, cval=1)
        #         train4[i][j] = ndimage.shift(train4[i][j], 30, order=0, cval=1)
        #     train5[i] = ndimage.median_filter(train5[i], 3)
        # train_true = np.concatenate((train,train))
        # train = np.concatenate((train, train1, train2, train3, train4, train5), axis=0)
        # print(train.shape)
        # print(train_true.shape)
        # print(val.shape)
        # print(test.shape)
        
        train_vanilla_dataset = LFWDataSet(train,transform,transform)
        self.train_vanilla_dataloader = DataLoader(train_vanilla_dataset, batch_size=batch_size, shuffle=True)

        train_dataset = LFWDataSet(train,train_transform,transform)
        self.train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        
        val_dataset = LFWDataSet(val,transform,transform)
        self.val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)

        test_dataset = LFWDataSet(test,transform,transform)
        self.test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
