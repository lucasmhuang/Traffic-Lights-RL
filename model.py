import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
import traci
from collections import deque

class Actor(nn.Module):
    def __init__(self, state_size, action_size, seed=0, fc1_units=200, fc2_units=100):
        super(Actor, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.fc1 = nn.Linear(state_size, fc1_units)
        self.fc2 = nn.Linear(fc1_units, fc2_units)
        self.fc3 = nn.Linear(fc2_units, action_size)
    
    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        return torch.tanh(self.fc3(x))  # Assuming continuous action space

class Critic(nn.Module):
    def __init__(self, state_size, action_size, seed=0, fcs1_units=200, fc2_units=100):
        super(Critic, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.fcs1 = nn.Linear(state_size, fcs1_units)
        self.fc2 = nn.Linear(fcs1_units + action_size, fc2_units)
        self.fc3 = nn.Linear(fc2_units, 1)
    
    def forward(self, state, action):
        xs = F.relu(self.fcs1(state))
        x = torch.cat((xs, action), dim=1)
        x = F.relu(self.fc2(x))
        return self.fc3(x)

class ReplayBuffer:
    def __init__(self, buffer_size, batch_size, seed=0):
        self.memory = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        self.seed = random.seed(seed)
    
    def add(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
        
    def sample(self):
        experiences = random.sample(self.memory, k=self.batch_size)
        states = torch.from_numpy(np.vstack([e[0] for e in experiences])).float()
        actions = torch.from_numpy(np.vstack([e[1] for e in experiences])).float()
        rewards = torch.from_numpy(np.vstack([e[2] for e in experiences])).float()
        next_states = torch.from_numpy(np.vstack([e[3] for e in experiences])).float()
        dones = torch.from_numpy(np.vstack([e[4] for e in experiences]).astype(np.uint8)).float()
        return (states, actions, rewards, next_states, dones)
    
    def __len__(self):
        return len(self.memory)

class Agent:
    def __init__(self, traffic_light_id, actor, critic, target_actor, target_critic, actor_optimizer, critic_optimizer, replay_buffer):
        self.traffic_light_id = traffic_light_id
        self.actor = actor
        self.critic = critic
        self.replay_buffer = replay_buffer
        self.target_actor = target_actor
        self.target_crtic = target_critic
        self.actor_optimizer = actor_optimizer
        self.critic_optimizer = critic_optimizer