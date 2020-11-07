import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

class Actor(nn.Module):
    def __init__(self, statesDim, actionDim, hidden1):
        super(Actor, self).__init__()
        hidden2=hidden1-100
        self.fc1 = nn.Linear(statesDim, hidden1)
        torch.nn.init.xavier_uniform_( self.fc1.weight)
        self.fc2 = nn.Linear(hidden1, hidden2)
        torch.nn.init.xavier_uniform_( self.fc2.weight)
        self.fc3 = nn.Linear(hidden2, actionDim)
        torch.nn.init.xavier_uniform_( self.fc3.weight)
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()


    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.relu(out)
        out = self.fc3(out)
        out = self.tanh(out)
        return out


class Critic(nn.Module):
    def __init__(self, statesDim,outputdim, hidden1, actionDim):
        super(Critic, self).__init__()
        hidden2=hidden1-100
        self.fc1 = nn.Linear(statesDim, hidden1)
        torch.nn.init.xavier_uniform_(self.fc1.weight)
        self.fc2 = nn.Linear(hidden1 + actionDim, hidden2)
        torch.nn.init.xavier_uniform_(self.fc2.weight)
        self.fc3 = nn.Linear(hidden2, outputdim)
        torch.nn.init.xavier_uniform_(self.fc3.weight)
        self.relu = nn.ReLU()
 
    def forward(self, x,a):

        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(torch.cat([out, a], 1))
        out = self.relu(out)
        out = self.fc3(out)
        return out

# import torch
# import torch.nn as nn
# import torch.nn.functional as F
#
# class Actor(nn.Module):
#     def __init__(self,inputDim,outputDim,hiddenDim,maxAction):
#         super().__init__()
#         self.linear1=nn.Linear(inputDim,hiddenDim)
#         self.bn1=nn.BatchNorm1d(hiddenDim)
#         self.linear2=nn.Linear(hiddenDim,int(hiddenDim/2))
#
#         self.bn2=nn.BatchNorm1d(int(hiddenDim/2))
#         nn.init.uniform_(self.linear2.weight,-0.002,0.002)
#         self.head=nn.Linear(int(hiddenDim/2),outputDim)
#         nn.init.uniform_(self.head.weight,-0.004,0.004)
#
#         self.maxAction=maxAction
#
#     def forward(self,state):
#         x=self.linear1(state)
#         x=torch.relu(self.bn1(x))
#
#         x=self.linear2(x)
#         x=torch.relu(self.bn2(x))
#
#         x=self.head(x)
#         x=torch.tanh(x)*self.maxAction
#         return x
#
# class Critic(nn.Module):
#
#     def __init__(self,inputDim,outputDim,hiddenDim,actionDim):
#         super().__init__()
#         self.linear1State=nn.Linear(inputDim,hiddenDim)
#         self.bn1=nn.BatchNorm1d(hiddenDim)
#
#         self.linear2State=nn.Linear(hiddenDim,int(hiddenDim/2))
#         nn.init.uniform_(self.linear2State.weight, -0.002, 0.002)
#
#
#         self.linear1Action=nn.Linear(actionDim,int(hiddenDim/2))
#         self.head=nn.Linear(hiddenDim,1)
#         nn.init.uniform_(self.head.weight, -0.004, 0.004)
#
#     def forward(self,state,action):
#         s=self.linear1State(state)
#         s=torch.relu(self.bn1(s))
#         s=self.linear2State(s)
#         a=self.linear1Action(action)
#         c=torch.relu(torch.cat((s,a),dim=1))
#         o=self.head(c)
#         return o
#
#
#
