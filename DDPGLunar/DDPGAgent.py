from commonFunctions.DDPGModel import *
import torch.optim as optim
import torch.nn as nn
import torch
import numpy as np
from copy import  deepcopy

class DDPGAgent():
    def __init__(self,buffer,stateDim,actionDim,hiddenDim,args):
        self.stateDim=stateDim
        self.actionDim = actionDim
        self.hiddenDim = hiddenDim
        self.device=args.device
        self.lrPolicy=args.lrPolicy
        self.lrCritic = args.lrCritic

        self.Actor=Actor(stateDim,actionDim,hiddenDim).to(self.device)
        self.Critic = Critic(stateDim, actionDim, hiddenDim,actionDim).to(self.device)
        self.targetActor=deepcopy(self.Actor)
        self.targetCritic=deepcopy(self.Critic)

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
            action = self.Actor(stateTensor).cpu().numpy()[0]

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
        qCurrent=self.Critic(states,actions)
        with torch.no_grad():
            nextActions=self.targetActor(nextStates)
            qNext=self.targetCritic(nextStates,nextActions)
        qTarget=rewards+self.gamma*qNext*(1-dones)
        criticLoss=self.criticLoss(qCurrent,qTarget)

        # Actor loss
        policyLoss = -self.Critic(states, self.Actor(states).detach()).mean()

        # update networks
        self.optActor.zero_grad()
        policyLoss.backward()
        self.optActor.step()

        self.optCritic.zero_grad()
        criticLoss.backward()
        self.optCritic.step()

        if stepCount%self.targetUpdatePeriod==0:
             self.UpdateTargetNets()




