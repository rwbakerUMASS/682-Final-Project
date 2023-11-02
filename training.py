import torch
import torch.nn as nn

class Trainer:
    def __init__(self, model, optim, device, train, val, lossfn=None) -> None:
        
        self.optim = optim
        self.device = device
        self.train = train
        self.val = val
        self.dtype = torch.float32
        self.model = model.to(device=self.device)  # move the model parameters to CPU/GPU
        if lossfn is not None:
            self.lossfn = lossfn
        else:
            self.lossfn = nn.MSELoss()
        
    def train_model(self, epochs=1, print_every=500):
        """
        Train a model on CIFAR-10 using the PyTorch Module API.
        
        Inputs:
        - model: A PyTorch Module giving the model to train.
        - optimizer: An Optimizer object we will use to train the model
        - epochs: (Optional) A Python integer giving the number of epochs to train for
        
        Returns: Nothing, but prints model accuracies during training.
        """
        for e in range(epochs):
            print('EPOCH: ',e)
            for t, (x,) in enumerate(self.train):
                self.model.train()  # put model to training mode
                x = x.to(device=self.device, dtype=self.dtype)  # move to device, e.g. GPU
                y = x

                scores = self.model(x)
                loss = self.lossfn(scores,y)

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
                    self.check_accuracy(self.val, self.model)
                    print()

    def check_accuracy(self, loader, model):
        lossfn = nn.MSELoss()  
        loss = 0
        num_samples = 0
        model.eval()  # set model to evaluation mode
        with torch.no_grad():
            for x, in loader:
                x = x.to(device=self.device, dtype=self.dtype)  # move to device, e.g. GPU
                y = x
                scores = model(x)
                loss += self.lossfn(scores,y)
                num_samples += 1
                
            acc = float(loss) / num_samples
            print('Avg Loss on Val: ' + str(acc))