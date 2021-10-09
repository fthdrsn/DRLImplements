from commonFunctions.DDPGModel import *
import torch.optim as optim
import torch.nn as nn
import torch
import numpy as np
from copy import  deepcopy

class MADDPGAgent():
    def __init__(self,buffer,stateDim,actionDim,hiddenDim,maxAction,args):
        self.stateDim=stateDim
        self.maxAction=maxAction
        self.actionDim = actionDim
        self.hiddenDim = hiddenDim
        self.outputDim=1
        self.device=args.device
        self.lrPolicy=args.lrPolicy
        self.lrCritic = args.lrCritic

        self.Actor=Actor(stateDim,actionDim,hiddenDim).to(self.device)
        self.Critic = Critic(stateDim, self.outputDim, hiddenDim,actionDim).to(self.device)

        self.targetActor = Actor(stateDim, actionDim, hiddenDim).to(self.device)
        self.targetCritic = Critic(stateDim, self.outputDim, hiddenDim, actionDim).to(self.device)

        for targetParam, param in zip(self.targetActor.parameters(), self.Actor.parameters()):
            targetParam.data.copy_(param.data)

        for targetParam, param in zip(self.targetCritic.parameters(), self.Critic.parameters()):
            targetParam.data.copy_(param.data )


        self.optActor=optim.Adam(self.Actor.parameters(), lr=self.lrPolicy)
        self.optCritic = optim.Adam(self.Critic.parameters(), lr=self.lrCritic)

        self.tau=args.tau
        self.gamma=args.gamma
        self.criticLoss=nn.MSELoss()
        self.buffer=buffer
        self.targetUpdatePeriod=args.targetUpdatePeriod
        self.batchSize=args.batchSize


    def GetAction(self,state):
        if len(state.shape)<2:
            state=state[np.newaxis,:]

        self.Actor.eval()

        with torch.no_grad():
            stateTensor = torch.from_numpy(state).to(torch.float32).to(self.device)
            a=self.Actor(stateTensor).cpu().numpy()
            action = self.Actor(stateTensor).cpu().numpy()[0]*self.maxAction

        self.Actor.train()
        return action

    def UpdateTargetNets(self):
        # update target networks
        for targetParam, param in zip(self.targetActor.parameters(), self.Actor.parameters()):
            targetParam.data.copy_(param.data * self.tau + targetParam.data * (1.0 - self.tau))

        for targetParam, param in zip(self.targetCritic.parameters(), self.Critic.parameters()):
            targetParam.data.copy_(param.data * self.tau + targetParam.data * (1.0 - self.tau))


    def Update(self,stepCount):

        ##Sample batch
        states,actions,rewards,nextStates,dones=[torch.from_numpy(x).to(self.device) for x in self.buffer.sample_batch(self.batchSize)]
        states=states.to(torch.float32)
        nextStates = nextStates.to(torch.float32)
        actions=actions.to(torch.float32)

        self.optCritic.zero_grad()

        qCurrent=self.Critic(states,actions)
        with torch.no_grad():
            nextActions=self.targetActor(nextStates)
            qNext=self.targetCritic(nextStates,nextActions)
            qTarget=rewards+self.gamma*qNext*(1-dones)

        criticLoss=self.criticLoss(qCurrent,qTarget)
        criticLoss.backward()
        self.optCritic.step()
        #Doesn't calculate gradients w.r.t critic net parameters when updating policy
        for p in self.Critic.parameters():
            p.requires_grad = False

        self.optActor.zero_grad()
        policyLoss = -self.Critic(states, self.Actor(states)).mean()
        policyLoss.backward()
        self.optActor.step()

        for p in self.Critic.parameters():
            p.requires_grad = True

        if stepCount%self.targetUpdatePeriod==0:
             self.UpdateTargetNets()
