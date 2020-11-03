from commonFunctions.DQNModel import DQNModel
import torch.optim as optim
import torch.nn as nn
import torch
import numpy as np
from copy import  deepcopy
class DQNAgent():
    def __init__(self,buffer,stateDim,actionDim,hiddenDim,args):
        self.stateDim=stateDim
        self.actionDim = actionDim
        self.hiddenDim = hiddenDim
        self.device=args.device
        self.lr=args.learningRate
        self.dqnModel=DQNModel(stateDim,actionDim,hiddenDim).to(self.device)
        self.dqnTargetModel = deepcopy(self.dqnModel)
        self.optimizer=optim.Adam(self.dqnModel.parameters(), lr=self.lr)
        self.gamma=args.gamma
        self.loss=nn.MSELoss()
        self.buffer=buffer
        self.targetUpdatePeriod=args.targetUpdatePeriod
        self.batchSize=args.batchSize

    def GetAction(self,state,epsilon):
        if len(state.shape)<2:
            state=state[np.newaxis,:]
        if epsilon >np.random.random():
            action= np.random.randint(0, self.actionDim)
        else:
            self.dqnModel.eval()

            with torch.no_grad():
                stateTensor = torch.from_numpy(state).to(torch.float32).to(self.device)
                qValsOfActions = self.dqnModel(stateTensor)
            action = (qValsOfActions.argmax(dim=1)).item()

            self.dqnModel.train()
        return action

    def UpdateTargetNet(self):
        self.dqnTargetModel.load_state_dict(self.dqnModel.state_dict())
        self.dqnTargetModel.eval()

    def Update(self,stepCount):

        ##Sample batch
        states,action,rewards,nextStates,dones=[torch.from_numpy(x).to(self.device) for x in self.buffer.sample_batch(self.batchSize)]
        states=states.to(torch.float32)
        nextStates = nextStates.to(torch.float32)
        qCurrent = self.dqnModel(states).gather(1,action.to(torch.long))

        with torch.no_grad():
            qNext=self.dqnTargetModel(nextStates).max(dim=1)[0].unsqueeze(-1)
            qTarget=rewards+self.gamma*qNext*(1-dones)

        loss=self.loss(qCurrent,qTarget)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        if stepCount%self.targetUpdatePeriod==0:
             self.UpdateTargetNet()




