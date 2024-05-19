import torch.nn as nn
import torch
import torch.nn.functional as F
from torch.nn import KLDivLoss

class NewEncoder(nn.Module):
    def __init__(self, latent_dim):
        super().__init__()
        self.conv1 = nn.Conv2d(3,64,kernel_size=4, stride=2, padding=2)
        self.conv2 = nn.Conv2d(64,128,kernel_size=4, stride=2, padding=2)
        self.conv3 = nn.Conv2d(128,256,kernel_size=4, stride=2, padding=2)
        self.conv4 = nn.Conv2d(256,512,kernel_size=4, stride=2, padding=2)
        self.conv5 = nn.Conv2d(512,1024,kernel_size=4, stride=2, padding=2)
        self.fc1 = nn.Linear(9216, 2048)
        self.fc_mu = nn.Linear(2048, latent_dim)
        self.fc_log_var = nn.Linear(2048, latent_dim)

    def forward(self,x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = F.relu(self.conv5(x))

        x = x.reshape(x.shape[0],-1)

        x = F.relu(self.fc1(x))
        x_mu = self.fc_mu(x)
        x_var = self.fc_log_var(x)
        return x_mu, x_var
        
class NewDecoder(nn.Module):
    def __init__(self, latent_dim):
        super().__init__()
        self.fc = nn.Linear(latent_dim, 1024) 
        self.conv1 = nn.ConvTranspose2d(1024,512, kernel_size=3, stride=2)
        self.conv2 = nn.ConvTranspose2d(512,256, kernel_size=3, stride=2)
        self.conv3 = nn.ConvTranspose2d(256,128, kernel_size=3, stride=2)
        self.conv4 = nn.ConvTranspose2d(128,64, kernel_size=3, stride=2)
        self.conv5 = nn.ConvTranspose2d(64,3, kernel_size=4, stride=2)

    def forward(self,x):

        x = F.relu(self.fc(x))

        x = x.reshape(x.shape[0],1024,1,1)

        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = F.sigmoid(self.conv5(x))

        return x

class Encoder(nn.Module):
    def __init__(self, latent_dim):
        super().__init__()
        self.conv1 = nn.Conv2d(3,32,7)
        self.conv2 = nn.Conv2d(32,32,5)
        self.conv3 = nn.Conv2d(32,32,5)
        self.conv4 = nn.Conv2d(32,32,5)
        self.conv5 = nn.Conv2d(32,32,3)
        self.conv6 = nn.Conv2d(32,3,3,stride=3)
        self.lin1 = nn.Linear(3888,1024)
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
        self.conv1 = nn.ConvTranspose2d(3,32,3,stride=3)
        self.lin2 = nn.Linear(1024,3888)
        self.lin1 = nn.Linear(latent_dim,1024)
        self.relu = nn.LeakyReLU()
        self.unflatten = nn.Unflatten(1,(3,36,36))
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
        x=torch.sigmoid(x)
        return x
    
class FC_Decoder(nn.Module):
    def __init__(self, latent_dim, hidden_dim):
        super().__init__()
        self.lin1 = nn.Linear(latent_dim, hidden_dim)
        self.lin2 = nn.Linear(hidden_dim,3 * 128 * 128)
        self.unflatten = nn.Unflatten(1, (3,128,128))
        self.lrelu = nn.LeakyReLU()

    def forward(self, x):
        x = self.lrelu(self.lin1(x))
        x = self.lin2(x)
        x=self.unflatten(x)
        x=torch.sigmoid(x)
        return x
    
class AutoEncoder(nn.Module):
    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
    
    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5*log_var)
        eps = torch.randn_like(std)
        sample = mu + (eps * std)
        return sample
    
    def forward(self,x, eval=False):
        x_mu, x_var = self.encoder(x)
        if eval:
            x = x_mu
        else:
            x = self.reparameterize(x_mu, x_var)

        # x_mu_shift = torch.roll(x_mu,1,0)
        # x_var_shift = torch.roll(x_var,1,0)
        # kl_div = torch.mean(0.5*torch.log(x_var_shift**2/x_var**2)+(x_var**2+(x_mu-x_mu_shift)**2)/(2.0*x_var_shift**2)-0.5)

        kl_div = -0.5 * torch.sum(1 + x_var - x_mu.pow(2) - x_var.exp())

        x=self.decoder(x)

        return x, kl_div
    
    def test(self, x):
        x_mu, x_var = self.encoder(x)
        x=self.decoder(x_mu)
        return x