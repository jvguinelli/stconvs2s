import torch
import torch.nn as nn

class RMSELoss(nn.Module):
    def __init__(self, eps=1e-6, reg=0.001):
        super().__init__()
        self.mse = nn.MSELoss()
        self.eps = eps
        self.reg = reg
        
    def forward(self,yhat,y):
        #loss = torch.sqrt(self.mse(yhat,y) + self.eps)
        loss = torch.sqrt(self.mse(yhat,y) + self.eps + self.reg*self.mse(torch.sum(yhat, dim=(3,4)), torch.sum(y, dim=(3,4))))
        return loss
