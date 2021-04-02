import torch
import torch.nn as nn

class RMSELoss(nn.Module):
    def __init__(self, eps=1e-6, reg=0.001, initial_state=0):
        super().__init__()
        self.mse = nn.MSELoss()
        self.eps = eps
        self.reg = reg
        self.num_channel = initial_state.shape[1]
        
        if self.num_channel == 3:
            u, v, mass = initial_state[:,0], initial_state[:,1], initial_state[:,2]
            velocity = torch.sqrt(u**2 + v**2)
            self.initial_energy = torch.sum( 
                ((mass * velocity**2) / 2), 
                dim=(-2, -1))
        else:
            self.initial_mass = torch.sum(initial_state)
        
    def forward(self,yhat,y):        
        
        if self.num_channel == 3:
            yhat_u, yhat_v, yhat_mass = yhat[:,0], yhat[:,1], yhat[:,2]
            yhat_velocity = torch.sqrt(yhat_u**2 + yhat_v**2)
            
            yhat_energy = torch.sum(
                    ((yhat_mass * yhat_velocity**2) / 2), 
                    dim=(-2, -1)
                )

            loss = torch.sqrt(self.mse(yhat,y) + self.eps + self.reg*self.mse(yhat_energy, self.initial_energy))
        else:
            loss = torch.sqrt(self.mse(yhat,y) + self.eps + self.reg*self.mse(torch.sum(yhat, dim=(3,4)), self.initial_mass))
        
        return loss
