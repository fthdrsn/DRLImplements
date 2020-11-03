import torch
import torch.nn as nn
import torch.nn.functional as F

class DQNModel(nn.Module):

    def __init__(self,inputDim,outputDim,hiddenDim):
        super().__init__()

        self.linear1=nn.Linear(inputDim,hiddenDim)
        self.linear2=nn.Linear(hiddenDim,outputDim)

    def forward(self,state):
        x=self.linear1(state)
        x=F.tanh(x)
        x=self.linear2(x)
        return x
