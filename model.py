import torch.nn as nn
import torch
from torch.nn import KLDivLoss

class Encoder(nn.Module):
    def __init__(self, latent_dim):
        super().__init__()
        self.conv1 = nn.Conv2d(3,32,7)
        self.conv2 = nn.Conv2d(32,32,5)
        self.conv3 = nn.Conv2d(32,32,5)
        self.conv4 = nn.Conv2d(32,32,5)
        self.conv5 = nn.Conv2d(32,32,3)
        self.conv6 = nn.Conv2d(32,3,(3,2),stride=3)
        self.lin1 = nn.Linear(2625,1024)
        self.lin_mu = nn.Linear(1024,latent_dim)
        self.lin_var = nn.Linear(1024,latent_dim)
        self.relu = nn.LeakyReLU()
        self.flatten = nn.Flatten(1)
        self.dropout = nn.Dropout()

    
    def forward(self, x):
        x=self.relu(self.conv1(x))
        x=self.relu(self.conv2(x))
        x=self.relu(self.conv3(x))
        x=self.dropout(self.conv4(x))
        x=self.relu(self.conv5(x))
        x=self.relu(self.conv6(x))
        x=self.flatten(x)
        x=self.dropout(self.lin1(x))
        x_mu = self.lin_mu(x)
        x_var = self.lin_var(x)
        return x_mu, x_var
    

class Decoder(nn.Module):
    def __init__(self, latent_dim):
        super().__init__()
        self.conv6 = nn.ConvTranspose2d(32,3,7)
        self.conv5 = nn.ConvTranspose2d(32,32,5)
        self.conv4 = nn.ConvTranspose2d(32,32,5)
        self.conv3 = nn.ConvTranspose2d(32,32,5)
        self.conv2 = nn.ConvTranspose2d(32,32,3)
        self.conv1 = nn.ConvTranspose2d(3,32,(3,2),stride=3)
        self.lin2 = nn.Linear(1024,2625)
        self.lin1 = nn.Linear(latent_dim,1024)
        self.relu = nn.LeakyReLU()
        self.unflatten = nn.Unflatten(1,(3,35,25))
        self.dropout = nn.Dropout()
    
    def forward(self, x):
        x=self.relu(self.lin1(x))
        x=self.dropout(self.lin2(x))
        x=self.unflatten(x)
        x=self.relu(self.conv1(x))
        x=self.relu(self.conv2(x))
        x=self.dropout(self.conv4(x))
        x=self.relu(self.conv4(x))
        x=self.relu(self.conv5(x))
        x=self.relu(self.conv6(x))
        return x
    
class FC_Decoder(nn.Module):
    def __init__(self, latent_dim, hidden_dim):
        super().__init__()
        self.lin1 = nn.Linear(latent_dim, hidden_dim)
        self.lin2 = nn.Linear(hidden_dim,3 * 125 * 94)
        self.unflatten = nn.Unflatten(1, (3,125,94))
        self.lrelu = nn.LeakyReLU()

    def forward(self, x):
        x = self.lrelu(self.lin1(x))
        x = self.lin2(x)
        x=self.unflatten(x)
        return x
    
class AutoEncoder(nn.Module):
    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
    
    def forward(self,x):
        x_mu, x_var = self.encoder(x)
        x = torch.randn_like(x_var)
        eps = torch.randn_like(x_var)
        x = x * (x_var + eps) + x_mu
        x_mu_shift = torch.roll(x_mu,1,0)
        x_var_shift = torch.roll(x_var,1,0)
        # kl_div = torch.mean(0.5*torch.log(x_var_shift**2/x_var**2)+(x_var**2+(x_mu-x_mu_shift)**2)/(2.0*x_var_shift**2)-0.5)
        kl_div = torch.mean((x_mu**2 + x_var**2 - torch.log(x_var**2))/2)
        x=self.decoder(x)
        return x, kl_div
    
    def test(self, x):
        x_mu, x_var = self.encoder(x)
        x=self.decoder(x_mu)
        return x