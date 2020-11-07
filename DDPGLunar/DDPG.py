import numpy as np
import argparse
from collections import namedtuple
from commonFunctions.UniformBuffer import UniformReplayBuffer
from DDPGAgent import DDPGAgent
from commonFunctions.noise import OUNoise
import gym
from torch.utils.tensorboard import SummaryWriter
import torch
import time
import os
Transition=namedtuple("Transitions","State Action Reward NextState Done")

def main(args):
    ##Switchs
    isTraining=args.isTraining
    saveModel=args.saveModel
    loadModel=args.loadModel
    debug=args.debug
    saveResults = args.saveResults
    savePeriod=args.savePeriod

    modelSavingPath=args.modelSavingPath
    modelLoadingPath = args.modelLoadingPath

    env=gym.make(args.env)
    runFor="_Training" if isTraining else "_Evaluation"
    nameToWriter="__DDPG__"+args.env+runFor
    writer = SummaryWriter(comment=nameToWriter)

    env.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)

    actionDim = env.action_space.shape[0]
    actionH = env.action_space.high
    actionL = env.action_space.low
    stateDim = env.observation_space.shape[0]
    hiddenDim = args.hiddenDim

    buffer=UniformReplayBuffer(args.maxCapacity,env.observation_space.shape,env.action_space.shape,np.float32,np.long)
    ddpgAgent=DDPGAgent(buffer,stateDim,actionDim,hiddenDim,actionH,args)

    if not isTraining or loadModel:
        LoadModel(ddpgAgent,modelLoadingPath)

    noise = OUNoise(actionDim,args)
    rewardToStop=args.maxReward

    totalReward = []
    stepCounter=0
    meanReward=0
    totalPassedTime=0
    isSuccesfull=False

    for e in range(args.numberOfEpisode):

        state=env.reset()
        episodeReward=0
        episodeSteps=0
        episodeTime=time.time()
        isDone=False

        for s in range(args.maxStepCount):
            action=ddpgAgent.GetAction(state)
            if isTraining:
                action=np.clip(action+noise.noise(),actionL,actionH)
            nextState,reward,done,_=env.step(action)
            buffer.push_transition(Transition(state,action,reward,nextState,done))
            episodeReward+=reward

            if stepCounter>2*args.batchSize and isTraining:
                ddpgAgent.Update(stepCounter)

            if not isTraining:
                env.render()
                time.sleep(0.01)

            if done:
                isDone=True
                break

            state = nextState
            episodeSteps+=1
            stepCounter+=1

        episodeTime=time.time()-episodeTime
        totalPassedTime+=episodeTime
        totalReward.append(episodeReward)
        meanReward = float(np.mean(totalReward[-100:]))

        if saveResults and e%savePeriod==0:
            writer.add_scalar("episodeReward"+runFor, episodeReward, stepCounter)
            writer.add_scalar("meanReward"+runFor, meanReward, stepCounter)
            writer.add_scalar("episodes"+runFor, e,stepCounter)
            writer.add_scalar("episodeSteps" + runFor, episodeSteps,e)
            writer.add_scalar("episodeTime" + runFor, episodeTime, e)
            writer.add_scalar("episoderewardVsEpisode" + runFor,episodeReward, e)
            writer.add_scalar("meanrewardVsEpisode" + runFor, meanReward, e)

        if debug:
            print("Eps:{} Steps:{} Mean Reward: {} Episode Reward: {} Episode Time: {}  IsDone: {}".format(e, stepCounter, meanReward, episodeReward,episodeTime,isDone))

        if meanReward>=rewardToStop and isTraining:
            isSuccesfull=True
            print("SUCCESS!!!")
            print("Total Eps:{} Total Steps:{} Mean Reward: {} Episode Reward: {} Total Time: {}".format(e, stepCounter, meanReward,episodeReward,totalPassedTime))
            if saveModel:
                SaveModel(ddpgAgent, e, stepCounter, episodeReward, "__DDPG__"+args.env, modelSavingPath)
            break

    SaveModel(ddpgAgent, 100, stepCounter, 100, "__DDPG__" + args.env, modelSavingPath)
    ##If reached max episode count without expected  mean reward
    if not isSuccesfull:
        print("FAILURE!!!")
        print("Total Eps:{} Total Steps:{} Mean Reward: {} Total Time: {}".format(len(totalReward), stepCounter, meanReward,totalPassedTime))

def SaveModel(agent,episode,stepCounter,episodeReward,saveName,savePath):
    if not os.path.exists(savePath):
        os.makedirs(savePath)
    path = savePath + saveName +".dat"
    torch.save({
                'steps': stepCounter,
                'agentactor_state_dict': agent.Actor.state_dict(),
                'agentcritic_state_dict': agent.Critic.state_dict(),
                'opt_policy_state_dict': agent.optActor.state_dict(),
                'opt_value_state_dict':agent.optCritic.state_dict(),
                'epsisode_reward':episodeReward,
                'epsisode': episode
                }, path)

def LoadModel(agent,loadPath):
    steps = 0
    reward = 0
    episodes=0
    path = loadPath
    try:
        checkpoint = torch.load(path)
        agent.Actor.load_state_dict(checkpoint['agentactor_state_dict'])
        agent.Critic.load_state_dict(checkpoint['agentcritic_state_dict'])
        agent.optActor.load_state_dict(checkpoint['opt_policy_state_dict'])
        agent.optCritic.load_state_dict(checkpoint['opt_value_state_dict'])
        steps = int(checkpoint['steps'])
        if 'epsisode_reward' in checkpoint: reward = float(checkpoint['epsisode_reward'])
        if 'epsisode' in checkpoint: reward = float(checkpoint['epsisode'])

    except FileNotFoundError:
        print("checkpoint not found")
    return steps,reward,episodes
def str2bool(value):
    return value == "True"
if __name__=="__main__":

    parser = argparse.ArgumentParser(description='DDPG')
    parser.add_argument("--env", default='LunarLanderContinuous-v2', help="Training environment")
    parser.add_argument("--maxReward", type=float,default=150, help="Mean rewards of last 100 episode to stop training")
    parser.add_argument("--maxCapacity", type=int,default=6000000, help="Maximum buffer capacity")
    parser.add_argument("--batchSize", type=int, default=64, help="Batch size for training")
    parser.add_argument("--maxStepCount", type=int, default=2500, help="Maximum step count in an episode")
    parser.add_argument("--numberOfEpisode", type=int, default=2000, help="Episode count ")
    parser.add_argument("--hiddenDim", type=int, default=400, help="Episode count ")
    parser.add_argument("--lrPolicy",  type=float,default=0.0001, help="Learning rate of policy ")
    parser.add_argument("--lrCritic",type=float, default=0.001, help="Learning rate of critic ")
    parser.add_argument("--tau",type=float, default=0.001, help="Learning rate of critic ")
    parser.add_argument("--gamma", type=float,default=0.99, help="Discount factor for future values  ")
    parser.add_argument("--device", default="cuda", help=" ")
    parser.add_argument("--targetUpdatePeriod",type=int ,default=1, help="Update target at this period")

    parser.add_argument("--savePeriod", type=int, default=1, help="Save result every savePeriod of episodes ")

    ##Switches
    parser.add_argument("--debug", type=str2bool, default=True, help="Determines whether print episode results or not ")
    parser.add_argument("--saveModel", type=str2bool, default=True, help="Determines whether save model or not ")
    parser.add_argument("--loadModel", type=str2bool, default=False, help="Determines whether load model or not ")
    parser.add_argument("--isTraining", type=str2bool, default=True, help="Switch between training and evaluation")
    parser.add_argument("--saveResults", type=str2bool, default=True, help="Determines whether save results or not to tensorboard")

    parser.add_argument("--modelSavingPath", default="models/",help="Saving path of trained model (without pointing model name)")
    parser.add_argument("--modelLoadingPath", default="D:/GitDRL/DDPGLunar/models/__DDPG__LunarLanderContinuous-v2.dat", help="Loading path of trained model ((with pointing model name))")

    ##For fixed action noise
    parser.add_argument("--theta", type=float,default=0.15, help="Theta value for noise ")
    parser.add_argument("--fixedSigma", type=float, default=0.2, help="Sigma value for noise ")
    parser.add_argument("--mu", type=float, default=0, help="mu value for noise ")

    ##For decaying action noise based on environment steps
    parser.add_argument("--startSigma", type=float,default=0.5, help="Start value of epsilon  ")
    parser.add_argument("--endSigma", type=float,default=0, help="Decay rate of epsilon  ")
    parser.add_argument("--decayLenSigma", type=int,default=15000, help=" Final value of epsilon")

    args = parser.parse_args()
    print(f"#####################################################################################################################")
    print(f"###### DDPG Algorithm for {args.env} Environment in {'Training' if args.isTraining else 'Evaluation'} Mode #######")
    print(f"#####################################################################################################################")
    main(args)












