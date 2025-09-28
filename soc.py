import random
import numpy as np
import torch
import torch.nn as nn
import gym
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import convolve, gaussian

# imported few extra things for my implementation
from collections import deque
from torch.nn.functional import mse_loss
import torch.optim as optim

import os
import io
import base64
import time
import glob
from IPython.display import HTML

from gym.wrappers import AtariPreprocessing
from gym.wrappers import FrameStack
from gym.wrappers import TransformReward	

def make_env(env_name, clip_rewards = True, seed = None):
	# complete this function which returns an object 'env' using gym module
	# Use AtariPreprocessing, FrameStack, TransformReward(based on the clip_rewards variable passed in the arguments of the function), check their usage from internet 
	# Use FrameStack to stack 4 frames
    env = gym.make(env_name)  
    env = AtariPreprocessing(env)
    env = FrameStack(env,num_stack = 4)
    if clip_rewards:
        env = TransformReward(env, lambda r: np.sign(r))
    return env

	
# initialise a device based on 'cuda' or 'cpu' which ever u have support for
# device = # TODO
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Next we create a class DQNAgent which is the class containing the neural network, This class is derived from nn.Module
        
class DQNAgent(nn.Module):
    def __init__(self, state_shape, n_actions, epsilon):
		# Here state_shape is the input shape to the neural network.
		# n_Actions is the number of actions 
		# epsilon is the probability to explore, 1-epsilon is the probabiltiy to stick to the best actions
        super(DQNAgent,self).__init__()
        self.n_actions = n_actions
        self.epsilon = epsilon

        # initialise a neural network containing the following layers:
		# 1)a convulation layer which accepts size = state_shape, in_channels = 4( state_shape is stacked with 4 frames using FrameStack ), out_channels = 16, kernel_size = 8, stride = 4 followed by ReLU activation
        self.conv1 = nn.Conv2d(4,16,kernel_size=8,stride=4)
        self.relu1 = nn.ReLU()
		# 2)a convulation layer, in_channels = 16, out_channels = 32, kernel_size = 4, stride = 2 followed by ReLU activation function
        self.conv2 = nn.Conv2d(16,32,kernel_size=4,stride=2)
        self.relu2 = nn.ReLU()
        # 3)layer to convert the output to a 1D output which is fed into a linear Layer with output size = 256 followed by ReLU actiovation
        self.flatten = nn.Flatten()
        self.linear = nn.Linear(32*((((state_shape[1]-8)//4 + 1)-4)//2 + 1)*((((state_shape[2]-8)//4 + 1)-4)//2 + 1),256)
        self.relu3 = nn.ReLU()
        # 4) linear Layer with output size = 'number of actions'(the qvalues of actions)
        self.qvalues = nn.Linear(256,n_actions)
		
    def forward(self, state_t):
        state_t = torch.tensor(np.array(state_t), dtype=torch.float32)
        x = self.relu1(self.conv1(state_t))
        x = self.relu2(self.conv2(x))
        x = self.relu3(self.linear(self.flatten(x)))
		# return qvalues generated from the neural network
        return self.qvalues(x)
		
    def get_qvalues(self, state_t):
        x = self.forward(state_t)
        # returns the numpy array of qvalues from the neural network
        return x.detach().numpy()
		
    def sample_actions(self, qvalues):
		# sample_Actions based on the qvalues
		# Use epsilon for choosing between best possible current actions of the give batch_size(can be found from the qvalues object passed in argument) based on qvalues vs explorations(random action)
        batch_size = qvalues.shape[0]
        actions = np.zeros(batch_size,dtype=np.int64)
        for i in range(batch_size):
            if(np.random.rand() < self.epsilon):
                actions[i] = np.random.randint(low=0,high=self.n_actions)
            else:
                actions[i] = np.argmax(qvalues[i])
        # return actions
        return actions 
		
		

def evaluate(env, agent, n_games = 1, greedy = False, t_max = 10000):
	# used for evaluationing the trained agent for number of games = n_games and step in each game = t_max
	# returns the mean of sum of all rewards across n_games
    rewards = []
    for _ in range(n_games):
        s = env.reset()
        R = 0.0
        for _ in range(t_max):
            qvalues = agent.get_qvalues([s])
            if greedy:
                action = qvalues.argmax(axis=-1)[0]
            else:
                action = agent.sample_actions(qvalues)[0]
            s,r,done,_ = env.step(action)
            R += r
            if done:
                break
        rewards.append(R)
    return np.mean(rewards)
			

# Now we create a class ReplayBuffer. The object of this class is responsible for storing the buffer information based on the agent's action when we play the agent(i.e, current_State -> action -> next_state -> done_flag ->reward) 
# For Deep Q Learning we sample information of size = 'batch_size' from the ReplayBuffer and return that information for training
# This buffer has a fixed size, set that to 10**6. remove previous information as new information is passed in the buffer


class ReplayBuffer:
    def __init__(self, size):
		# size is the maximum size that the buffer can hold
        self.buffer = deque(maxlen=size)	
		
    def __len__(self):
		# no need to change
        return len(self.buffer)
	
    def add(self, state, action ,reward, next_state, done):
		# store the information passed in one call to add as 1 unit of informmation
        self.buffer.append((state,action,reward,next_state,done))
		
    def sample(self, batch_size):
		# return a random sampling of 'batch_size' units of information 
        states,actions,rewards,next_states,dones = zip(*random.sample(self.buffer,batch_size))
        return np.array(states),np.array(actions),np.array(rewards),np.array(next_states),np.array(dones)
		
	
def play_and_record(start_state, agent, env, exp_replay, n_steps = 1):
	# use this function to make the agent play on the env and store the information in exp_replay which is an object of class ReplayBuffer
	# n_steps is the number of steps to be played in this function on one call
    s = start_state
    for _ in range(n_steps):
        qvalues = agent.get_qvalues([s])
        a = agent.sample_actions(qvalues)[0]
        next_s,r,done,_ = env.step(a)
        exp_replay.add(s,a,r,next_s,done)
        if not done:
            s = next_s
        else:
            s = env.reset()
	

def compute_td_loss(agent, target_network,device, batch_size, exp_replay, gamma = 0.99):
	# Here agent is the one playing on the game and target_network is updates using agent after some fixed steps as is done in Deep Q Learning
	# sample 'batch_size' units of info stored in the exp_replay
    states,actions,rewards,next_states,dones = exp_replay.sample(batch_size)
    # IMPORTANT NOTE : check the type of objects, U need to convert the actions, rewards, etc, to toch.tensors for backward propogation using pytorch
    states = torch.tensor(states, device=device, dtype=torch.float)
    actions = torch.tensor(actions, device=device, dtype=torch.long)
    rewards = torch.tensor(rewards, device=device, dtype=torch.float)
    next_states = torch.tensor(next_states, device=device, dtype=torch.float)
    dones = torch.tensor(dones.astype('float32'),device=device, dtype=torch.float)
	# Find the predicted_qvalues_of_actions using agent and target_qvalues_of_actions using target_network, find the loss based on these Mean Squared Error of these two
    predicted_qvalues = agent(states) 
    # forward call in DQNAgent class
    predicted_qvalues_of_actions = predicted_qvalues[range(batch_size),actions]

    with torch.no_grad():
        target_qvalues_of_actions = target_network(next_states).max(-1)[0]
        target_qvalues_of_actions = rewards + (gamma * target_qvalues_of_actions * (1 - dones))
  
    return mse_loss(predicted_qvalues_of_actions,target_qvalues_of_actions)


############# MAIN LOOP ###############3

from tqdm import trange
from IPython.display import clear_output
import matplotlib.pyplot as plt

seed = 108
random.seed(108)
np.random.seed(108)
torch.manual_seed(108)


##  setup environment using make_env function defined above
# find action_space and observation_space of the atari
# Use 
env_name = "BreakoutNoFrameskip-v4"
# Reset the environment before starting to train the agent and everytime the game ends (U will get a done flag which is a boolean representing whether the game has ended or not)
env = make_env("BreakoutNoFrameskip-v4")
state = env.reset()
state_shape = env.observation_space.shape
n_actions = env.action_space.n
epsilon = 0.5

# create agent from DQNAgent class the online network
# create target_network from DQNAgent class is updated after some fixed steps from agent
# Note initialise target network values from agent
agent = DQNAgent(state_shape,n_actions,epsilon).to(device)
target_network = DQNAgent(state_shape,n_actions,epsilon).to(device)
target_network.load_state_dict(agent.state_dict())

# created a ReplayBuffer object and saved some information in the object by playing the agent. It is better to populate some information in the Buffer, hence this step
#filling experience replay with some samples using full random policy
exp_replay = ReplayBuffer(10**6)
for i in range(4000):
    play_and_record(state, agent, env, exp_replay, n_steps=10**2)
    print( "Replay Buffer : i : ", i)
    if len(exp_replay) == 10**6:
        break
print(len(exp_replay))


#setup some parameters for training
timesteps_per_epoch = 2
batch_size = 32

total_steps = 2 * 10**6

#Optimizer
# TODO - use Adam optimiser from torch with learning rate (lr) = 2*1e-5
optimizer = optim.Adam(agent.parameters(),lr=2e-5)

#setting exploration epsilon 
start_epsilon = 0.1
end_epsilon = 0.05
eps_decay_final_step = 1 * 10**5

# setup spme frequency for logginf and updating target network
loss_freq = 20
refresh_target_network_freq = 100
eval_freq = 10000

# to clip the gradients
max_grad_norm = 5000

mean_rw_history = []
td_loss_history = []

SAVE_INTERVAL = 50000

from numpy import asarray
from numpy import savetxt


def epsilon_schedule(start_eps, end_eps, step, final_step):
    return start_eps + (end_eps-start_eps)*min(step, final_step)/final_step
    
    
# TODO - reset the state of the environment before starting
env.reset()

## MAIN LOOP STARTING
for step in range(total_steps + 1):
    #TODO update the exploration probability (epsilon) as time passes
    agent.epsilon = epsilon_schedule(start_epsilon, end_epsilon, step, eps_decay_final_step)
    
    #TODO taking timesteps_per_epoch and update experience replay buffer, (use play_and_record)
    play_and_record(state, agent, env, exp_replay,timesteps_per_epoch)
    
    #TODO compute loss
	#TODO Backward propogation and updating the network parameters
    # IMPORTANT NOTE : You only need to update the parameters of agent and not of target_network, that will be done according to refresh_target_network_freq. But Backward Propogation will take into account the target_network parameters as well. So use detach() method on target_network while calculating the loss. Google what it does and how to use !!
    # states, actions, rewards, next_states, dones = exp_replay.sample(batch_size)
    loss = compute_td_loss(agent,target_network,device,batch_size,exp_replay,gamma=0.99)  
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    if step % loss_freq == 0:
        td_loss_history.append(loss.data.cpu().item())
    
    if step % refresh_target_network_freq == 0:
        #TODO Load agent weights into target_network
        target_network.load_state_dict(agent.state_dict())  

    if step % eval_freq == 0:
        mean_reward = evaluate(make_env(env_name, seed=step), agent, n_games=3, greedy=True, t_max=6000)
        mean_rw_history.append(mean_reward)

        print("mean_reward : ", mean_reward)

        #clear_output(True)
        print("buffer size = %i, epsilon = %.5f" %
				(len(exp_replay), agent.epsilon))


    # #Save
    if step % SAVE_INTERVAL == 0 and step!= 0:
        print('Saving...')
        device = torch.device('cpu')
        torch.save(agent.state_dict(), f'model_{step}.pth')
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        savetxt(f'reward_{step}.csv', np.array(mean_rw_history))



# savetxt('reward_final.csv', np.array(mean_rw_history))

final_score = evaluate(make_env(env_name),agent, n_games=1, greedy=True, t_max=10000)
print('final score:', final_score)