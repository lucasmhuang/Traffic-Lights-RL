import gymnasium as gym
import torch
from stable_baselines3 import DDPG
from stable_baselines3.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise
from stable_baselines3.common.env_util import make_vec_env
import numpy as np
from model import Actor, Critic, ReplayBuffer, Agent, federated_averaging

# TO DO
# 1) initialize global critic/actor network with random weights
# 2) initialize target networks with global network weights
# 3) create each local agent
# 4) for each iteration, train each agent
# 5) at the end of each iteration after training all agents, aggregate weights and update global actor/critic/target networks

# Register your environment with Gym
gym.register(
    id="SumoGrid-v0",
    entry_point="sumo_grid_env:SumoGridEnv",
    kwargs={"sumo_config": "C:/Users/lucas/OneDrive/Documents/Coding/Traffic-Lights-RL/fremont/osm.sumocfg"} 
)

# Create the environment
env = gym.make("SumoGrid-v0")

# Retrieve the list of traffic light IDs from TraCI
traffic_light_ids = env.get_traffic_light_ids()

global_actor = Actor(...)
global_critic = Critic(...)
# Create local actors, critics, and replay buffers for each agent
agents = [Agent(tl_id, Actor(...), Critic(...), ReplayBuffer(...)) for tl_id in traffic_light_ids]

# Initialize target networks with global network weights

# Initialize optimizers for actor and critic networks
actor_optimizer = torch.optim.Adam(global_actor.parameters(), lr=1e-4)
critic_optimizer = torch.optim.Adam(global_critic.parameters(), lr=1e-3)

# Main federated learning loop
num_iterations = 1000
for iteration in range(num_iterations):
    # Parallel training of agentsz
    for agent in agents:
        # DDPG learn
    # Aggregate weights from local networks and update global networks

    # Update target networks

# Define functions for training, weight aggregation, and soft updates
# ...


# state_size = -1 # the size of your state space
# action_size = -1 # the size of your action space

# # Initialize global actor and critic models
# global_actor = Actor(state_size, action_size, ...)
# global_critic = Critic(state_size, action_size, ...)
# # Initialize agents, one per traffic light controller
# agents = [Agent(tl_id, Actor(state_size, action_size, ...), Critic(state_size, action_size, ...), ReplayBuffer(...)) for tl_id in traffic_light_ids]

# # Perform federated learning
# evaluate_episodes = 10
# for episode in range(evaluate_episodes):
#     for agent in agents:
#         state = agent.get_state()
#         action = agent.actor(state)  # Assuming Actor returns the action to perform
#         new_state, reward, done = agent.step(action)
#         # Store the experience in the replay buffer
#         agent.replay_buffer.add(state, action, reward, new_state, done)
#         # Learn from the experience
#         # ...

# total_episodes = 1000
# for episode in range(total_episodes):
#     for agent in agents:
#         state = agent.get_state()
#         action = agent.actor(state)  # Assuming Actor returns the action to perform
#         new_state, reward, done = agent.step(action)
#         # Store the experience in the replay buffer
#         agent.replay_buffer.add(state, action, reward, new_state, done)

#     if (episode + 1) % 10 == 0:  # Federated averaging every 10 episodes
#         # Aggregate weights from all agents
#         actor_weights = federated_averaging([agent.actor.state_dict() for agent in agents], agents_episodes)
#         critic_weights = federated_averaging([agent.critic.state_dict() for agent in agents], agents_episodes)

#         # Update global models
#         global_actor.load_state_dict(actor_weights)
#         global_critic.load_state_dict(critic_weights)

#         # Update local models with new global weights
#         for agent in agents:
#             agent.actor.load_state_dict(global_actor.state_dict())
#             agent.critic.load_state_dict(global_critic.state_dict())

# After a certain number of episodes, perform federated averaging
# ...

# Update the actors and critics of each agent with the averaged weights
# ...

# # Create the action noise for exploration
# n_actions = env.action_space.shape[-1]
# action_noise = OrnsteinUhlenbeckActionNoise(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions))

# # Initialize the DDPG model
# # # DDPG many collisions
# model = DDPG("MlpPolicy", env, action_noise=action_noise, verbose=1)

# # Train the model
# model.learn(total_timesteps=1000)

# # Evaluate the model
# evaluate_episodes = 10
# total_rewards = 0
# for episode in range(evaluate_episodes):
#     obs, info = env.reset()
#     done = False
#     episode_reward = 0

#     while not done:
#         action, _states = model.predict(obs, deterministic=True)
#         obs, reward, done, truncated, info = env.step(action)
#         episode_reward += reward

#     total_rewards += episode_reward

# average_reward = total_rewards / evaluate_episodes
# print("Average Reward:", average_reward)