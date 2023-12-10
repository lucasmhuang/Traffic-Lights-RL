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
        states = torch.from_numpy(np.vstack([e.state for e in experiences])).float()
        actions = torch.from_numpy(np.vstack([e.action for e in experiences])).float()
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences])).float()
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences])).float()
        dones = torch.from_numpy(np.vstack([e.done for e in experiences]).astype(np.uint8)).float()
        return (states, actions, rewards, next_states, dones)
    
    def __len__(self):
        return len(self.memory)

class Agent:
    def __init__(self, traffic_light_id, actor, critic, replay_buffer):
        self.traffic_light_id = traffic_light_id
        self.actor = actor
        self.critic = critic
        self.replay_buffer = replay_buffer
    
    def get_state(self):
        # Get the state of the traffic light, e.g., queue length
        queue_length = traci.lane.getLastStepVehicleNumber(traci.trafficlight.getControlledLanes(self.traffic_light_id))
        return np.array([queue_length])

    def get_reward(self):
        # Define reward based on queue length and collision status
        collision_count = traci.simulation.getCollidingVehiclesNumber()
        queue_length = sum(traci.lane.getLastStepVehicleNumber(light) for light in traci.trafficlight.getControlledLanes(self.traffic_light_id))
        # Reward function considering both queue length and collisions
        reward = -queue_length
        if collision_count > 0:
            reward -= collision_count * -10  # large negative value
        return reward
    
    def step(self, action):
        # Execute the action (change traffic light phase)
        # ...
        # Collect the new state and reward after the action
        new_state = self.get_state()
        reward = self.get_reward()
        # ...
        return new_state, reward, done  # done to be determined by your environment's conditions

def federated_averaging(agents_weights, agents_episodes):
    """
    Perform federated averaging of the weights.
    
    :param agents_weights: List of state_dicts (weights) of each agent's model
    :param agents_episodes: List of episode counts for each agent
    :return: A state_dict representing the averaged weights
    """
    # Initialize the numerator and denominator for weighted averaging
    weighted_sum_weights = None
    total_episodes = sum(agents_episodes)
    
    for agent_weights, episodes in zip(agents_weights, agents_episodes):
        agent_weight = {k: v * episodes for k, v in agent_weights.items()}
        
        if weighted_sum_weights is None:
            weighted_sum_weights = agent_weight
        else:
            # Sum the weighted weights
            weighted_sum_weights = {k: weighted_sum_weights[k] + agent_weight.get(k, 0) for k in weighted_sum_weights}
    
    # Divide by the total number of episodes to get the average
    averaged_weights = {k: v / total_episodes for k, v in weighted_sum_weights.items()}
    
    return averaged_weights