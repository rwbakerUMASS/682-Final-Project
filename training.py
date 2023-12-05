import torch
import torch.nn as nn
import numpy as np
import torch.functional as F
from loss import PerceptualLoss

class Trainer:
    def __init__(self, model, optim, device, train, val, lossfn=None, kl_factor=1, kl_rate=1,kl_max=1,ploss_wt=0) -> None:
        
        self.optim = optim
        self.device = device
        self.train = train
        self.val = val
        self.dtype = torch.float32
        self.model = model.to(device=self.device)  # move the model parameters to CPU/GPU
        self.kl_factor = kl_factor
        self.kl_rate = kl_rate
        self.kl_max = kl_max
        self.ploss = PerceptualLoss()
        self.ploss_wt = ploss_wt
        if lossfn is not None:
            self.lossfn = lossfn
        else:
            self.lossfn = nn.MSELoss(reduction='sum')
        
    def train_model(self, epochs=1, print_every=500):
        """
        Train a model on CIFAR-10 using the PyTorch Module API.
        
        Inputs:
        - model: A PyTorch Module giving the model to train.
        - optimizer: An Optimizer object we will use to train the model
        - epochs: (Optional) A Python integer giving the number of epochs to train for
        
        Returns: Nothing, but prints model accuracies during training.
        """
        kl_losses = []
        recon_losses = []
        total_losses = []
        for e in range(epochs):
            if e == 75:
                torch.save(self.model,'{}.model'.format(e))
            print('EPOCH: ',e)
            kl_factor=self.kl_factor
            for t, (x,y) in enumerate(self.train):
                self.model.train()  # put model to training mode
                x = x.to(device=self.device, dtype=self.dtype)  # move to device, e.g. GPU
                y = y.to(device=self.device, dtype=self.dtype) 

                scores, kl = self.model(x)
                recon_loss = self.lossfn(scores,y)

                if self.ploss_wt > 0:
                    loss += self.ploss_wt *  self.ploss(scores,y)

                recon_losses.append(recon_loss.detach().cpu().numpy())
                loss = recon_loss + kl * kl_factor
                kl_losses.append((kl * kl_factor).detach().cpu().numpy())
                total_losses.append(loss.detach().cpu().numpy())

                kl_factor = np.minimum(self.kl_max,kl_factor*(self.kl_rate))

                # Zero out all of the gradients for the variables which the optimizer
                # will update.
                self.optim.zero_grad()

                # This is the backwards pass: compute the gradient of the loss with
                # respect to each  parameter of the model.
                loss.backward()

                # Actually update the parameters of the model using the gradients
                # computed by the backwards pass.
                self.optim.step()

                if t % print_every == 0:
                    print('Iteration %d, loss = %.4f' % (t, loss.item()))
                    print('KL DIV: %.4f' % kl)
                    print('Recon Loss: %.4f' % recon_loss)
                    self.check_accuracy(self.val, self.model)
                    print()
        return recon_losses, kl_losses, total_losses

    def check_accuracy(self, loader, model):
        lossfn = nn.BCELoss(reduction='sum')  
        loss = 0
        num_samples = 0
        model.eval()  # set model to evaluation mode
        with torch.no_grad():
            for x, y in loader:
                x = x.to(device=self.device, dtype=self.dtype)  # move to device, e.g. GPU
                y = y.to(device=self.device, dtype=self.dtype) 
                scores,_ = model(x)
                loss += lossfn(scores,y)
                num_samples += 1
                
            acc = float(loss) / num_samples
            print('Avg Recon Loss on Val: ' + str(acc))