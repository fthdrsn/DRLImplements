import numpy as np
import argparse
from collections import namedtuple
from commonFunctions.UniformBuffer import UniformReplayBuffer
from DDPGAgent import DDPGAgent
from commonFunctions.noise import OUNoise
import gym
from torch.utils.tensorboard import SummaryWriter
Transition=namedtuple("Transitions","State Action Reward NextState Done")

def main(args):
    env=gym.make(args.env)
    writer = SummaryWriter(comment="LunarLanderContinuous-v2")
    totalReward=[]
    actionDim=env.action_space.shape[0]
    actionH=env.action_space.high
    actionL=env.action_space.low
    stateDim=env.observation_space.shape[0]
    hiddenDim=args.hiddenDim
    buffer=UniformReplayBuffer(args.maxCapacity,env.observation_space.shape,env.action_space.shape,np.float32,np.long)
    ddpgAgent=DDPGAgent(buffer,stateDim,actionDim,hiddenDim,args)
    stepCounter=0
    noise=OUNoise(actionDim,args)
    for e in range(args.numberOfEpisode):
        state=env.reset()
        episodeReward=0

        for s in range(args.maxStepCount):

            action=np.clip(ddpgAgent.GetAction(state)+noise.GetNoise(stepCounter),actionL,actionH)
            nextState,reward,done,_=env.step(action)
            buffer.push_transition(Transition(state,action,reward,nextState,done))
            episodeReward+=reward

            if stepCounter>2*args.batchSize:
                ddpgAgent.Update(stepCounter)
            if done:
                break
            state = nextState
            stepCounter += 1

        totalReward.append(episodeReward)
        meanReward = float(np.mean(totalReward[-100:]))
        writer.add_scalar("episodeReward", episodeReward, stepCounter)
        writer.add_scalar("meanReward", meanReward, stepCounter)
        writer.add_scalar("episodes", e,stepCounter)
        print("Eps:{} Steps:{} Mean Reward: {} Episode Reward: {} ".format(e, stepCounter, meanReward, episodeReward))


if __name__=="__main__":

    parser = argparse.ArgumentParser(description='DDPG Algo-FD')
    parser.add_argument("--env", default="LunarLanderContinuous-v2", help="Training environment")
    parser.add_argument("--maxCapacity", type=int,default=1000000, help="Maximum buffer capacity")
    parser.add_argument("--batchSize", type=int, default=64, help="Batch size for training")
    parser.add_argument("--maxStepCount", type=int, default=3000, help="Maximum step count in an episode")
    parser.add_argument("--numberOfEpisode", type=int, default=2000, help="Episode count ")
    parser.add_argument("--hiddenDim", type=int, default=400, help="Episode count ")
    parser.add_argument("--lrPolicy",  default=0.00005, help="Learning rate of policy ")
    parser.add_argument("--lrCritic", default=0.0005, help="Learning rate of critic ")
    parser.add_argument("--tau", default=1e-3, help="Learning rate of critic ")
    parser.add_argument("--gamma", default=0.99, help="Discount factor for future values  ")
    parser.add_argument("--device", default="cuda", help=" ")
    parser.add_argument("--targetUpdatePeriod", default=1, help="Update target at this period")

    parser.add_argument("--theta", default=0.15, help="Start value of epsilon  ")
    parser.add_argument("--startSigma", default=0.3, help="Start value of epsilon  ")
    parser.add_argument("--endSigma", default=0, help="Decay rate of epsilon  ")
    parser.add_argument("--decayLenSigma", default=15000, help=" Final value of epsilon")
    args = parser.parse_args()
    main(args)












