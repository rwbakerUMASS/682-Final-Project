import torch.nn as nn
import torch

class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3,32,7)
        self.conv2 = nn.Conv2d(32,32,5)
        self.maxpool = nn.MaxPool2d(2,return_indices=True)
        self.conv3 = nn.Conv2d(32,32,3)
        self.conv4 = nn.Conv2d(32,3,3)
        self.lin1 = nn.Linear(26640,1024)
        self.lin_mu = nn.Linear(1024,64)
        self.lin_var = nn.Linear(1024,64)
        self.relu = nn.LeakyReLU()
        self.flatten = nn.Flatten(1)
        self.dropout = nn.Dropout()

    
    def forward(self, x):
        x=self.relu(self.conv1(x))
        x=self.relu(self.conv2(x))
        x=self.dropout(self.conv3(x))
        x=self.relu(self.conv4(x))
        x=self.flatten(x)
        x=self.dropout(self.lin1(x))
        x_mu = self.lin_mu(x)
        x_var = self.lin_var(x)
        return x_mu, x_var
    

class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv4 = nn.ConvTranspose2d(32,3,7)
        self.conv3 = nn.ConvTranspose2d(32,32,5)
        self.conv2 = nn.ConvTranspose2d(32,32,3)
        self.conv1 = nn.ConvTranspose2d(3,32,3)
        self.lin2 = nn.Linear(1024,26640)
        self.lin1 = nn.Linear(64,1024)
        self.relu = nn.LeakyReLU()
        self.unflatten = nn.Unflatten(1,(3,111,80))
        self.dropout = nn.Dropout()
    
    def forward(self, x):
        x=self.relu(self.lin1(x))
        x=self.dropout(self.lin2(x))
        x=self.unflatten(x)
        x=self.relu(self.conv1(x))
        x=self.dropout(self.conv2(x))
        x=self.relu(self.conv3(x))
        x=self.relu(self.conv4(x))
        return x
    
class AutoEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()
    
    def forward(self,x):
        x_mu, x_var = self.encoder(x)
        x = torch.randn_like(x_var)
        x = x * x_var + x_mu
        x=self.decoder(x)
        return x