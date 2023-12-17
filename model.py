import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
import traci
from collections import deque

class Actor(nn.Module):
    def __init__(self, state_size, seed=0, fc1_units=200, fc2_units=100):
        super(Actor, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.fc1 = nn.Linear(state_size, fc1_units)
        self.fc2 = nn.Linear(fc1_units, fc2_units)
        self.fc3 = nn.Linear(fc2_units, 1)
    
    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        return torch.tanh(self.fc3(x))  # Assuming continuous action space

class Critic(nn.Module):
    def __init__(self, state_size, action_size, seed=0, fcs1_units=200, fc2_units=100):
        super(Critic, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.fc1 = nn.Linear(state_size + 1, fcs1_units)
        self.fc2 = nn.Linear(fcs1_units, fc2_units)
        self.fc3 = nn.Linear(fc2_units, 1)
    
    def forward(self, state, action):
        x = torch.cat((state, action), dim=1)
        x = F.relu(self.fc1(x))
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
        self.target_critic = target_critic
        self.actor_optimizer = actor_optimizer
        self.critic_optimizer = critic_optimizer

    def get_action(self, state):
        # Convert state to PyTorch tensor
        state = torch.tensor(state, dtype=torch.float32).unsqueeze(0)  # Add batch dimension
        # Disable gradient calculations since we are in inference mode
        with torch.no_grad():
            # Get continuous action value from the local model (actor network)
            continuous_action = self.actor(state).cpu().numpy().squeeze(0)
        # Discretize the action for the environment: -1.0 to 0.0 becomes 0, and 0.0 to 1.0 becomes 1
        discrete_action = np.round((continuous_action + 1.0) / 2.0).astype(int)
        return discrete_action

class OrnsteinUhlenbeckActionNoise:
    def __init__(self, mu, sigma, theta=.15, dt=1e-2, x0=None):
        self.theta = theta
        self.mu = mu
        self.sigma = sigma
        self.dt = dt
        self.x0 = x0
        self.reset()

    def reset(self):
        self.x_prev = self.x0 if self.x0 is not None else np.zeros_like(self.mu)

    def __call__(self):
        x = self.x_prev + self.theta * (self.mu - self.x_prev) * self.dt + \
            self.sigma * np.sqrt(self.dt) * np.random.normal(size=self.mu.shape)
        self.x_prev = x
        return x
    
def soft_update(local_model, target_model, tau=0.005):
    for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
        target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)   
 
def federated_averaging(models):
    """
    Average the weights of the models.
    :param models: List of models (either actors or critics) from each agent.
    :return: Averaged state_dict of the models.
    """
    state_dict = models[0].state_dict()
    for key in state_dict:
        state_dict[key] = torch.mean(torch.stack([model.state_dict()[key] for model in models]), dim=0)
    return state_dict