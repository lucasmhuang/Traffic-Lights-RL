import gymnasium as gym
import torch
from stable_baselines3 import DDPG
import numpy as np
from model import Actor, Critic, ReplayBuffer, Agent, OrnsteinUhlenbeckActionNoise

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
    kwargs={"sumo_config": "C:/Users/lucas/Documents/Coding/Traffic-Lights-RL/fremont/osm.sumocfg"} 
)

# Create the environment
env = gym.make("SumoGrid-v0")

state_size = len(env.unwrapped.traffic_light_ids) + len(env.unwrapped.lane_ids)
action_size = len(env.unwrapped.traffic_light_ids)

global_actor = Actor(state_size, action_size)
global_critic = Critic(state_size, action_size)
# Create local actors, critics, and replay buffers for each agent
# # Initialize target networks with global network weights
agents = []
for tl_id in env.unwrapped.traffic_light_ids:
    local_actor = Actor(state_size)
    local_critic = Critic(state_size, action_size)
    target_actor = Actor(state_size)
    target_critic = Critic(state_size, action_size)
    local_actor.load_state_dict(global_actor.state_dict())
    local_critic.load_state_dict(global_critic.state_dict())
    target_actor.load_state_dict(global_actor.state_dict())
    target_critic.load_state_dict(global_critic.state_dict())
    actor_optimizer = torch.optim.Adam(local_actor.parameters(), lr=1e-4)
    critic_optimizer = torch.optim.Adam(local_critic.parameters(), lr=1e-3)
    agents.append(Agent(tl_id, local_actor, local_critic, target_actor, target_critic, actor_optimizer, critic_optimizer, ReplayBuffer(buffer_size=100000, batch_size=64)))

# Initialize optimizers for actor and critic networks

tau = 0.005  # Soft update parameter
def soft_update(local_model, target_model, tau):
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

# Main federated learning loop
mse_loss = torch.nn.MSELoss()
n_actions = env.action_space.shape[0]
ou_noises = [OrnsteinUhlenbeckActionNoise(mu=np.zeros(1), sigma=0.3) for _ in env.unwrapped.traffic_light_ids]
num_iteration = 1000
num_episode = 150
for iteration in range(num_iteration):
    state, _ = env.reset()
    for noise in ou_noises:
        noise.reset()
    rewards = []
    for episode in range(num_episode):
        # Parallel training of agents
        actions = []
        episode_reward = 0
        for i, agent in enumerate(agents):
            actions.append(np.clip(agent.get_action(state)  + ou_noises[i](), env.action_space.low[i], env.action_space.high[i]))
        new_state, reward, done, truncated, _ = env.step(actions)
        episode_reward += reward
        print("Episode", episode, "reward: ", reward)
        for i, agent in enumerate(agents):
            agent.replay_buffer.add(state, actions[i], reward, new_state, done)
        if episode + 1 > 64:
            # Sample mini-batch of experiences
            # Update local critic network
            # Update local actor network
            # Update local target networks
            # Aggregate weights from local networks and update global ritic/actor/target networks
            for agent in agents:
                states, actions, rewards, next_states, dones = agent.replay_buffer.sample()
                # Update Critic
                # Predict Q-values (Q_expected) using current states and actions
                Q_expected = agent.critic(states, actions)
                # Compute target Q-values (Q_targets) using next states and rewards
                # Use target networks if available
                Q_targets = rewards + 0.99 * agent.target_critic(next_states, agent.target_actor(next_states)) * (1 - dones)
                critic_loss = mse_loss(Q_expected, Q_targets)
                agent.critic_optimizer.zero_grad()
                critic_loss.backward()
                agent.critic_optimizer.step()

                # Update Actor
                # Calculate actor loss
                actor_loss = -agent.critic(states, agent.actor(states)).mean()
                agent.actor_optimizer.zero_grad()
                actor_loss.backward()
                agent.actor_optimizer.step()

                # Soft update target networks if used
                soft_update(agent.actor, agent.target_actor, tau)
                soft_update(agent.critic, agent.target_critic, tau) 

        if (episode + 1) % 10 == 0:  # Federated averaging every 10 episodes
            # Perform federated averaging
            global_actor_state_dict = federated_averaging([agent.target_actor for agent in agents])
            global_critic_state_dict = federated_averaging([agent.target_critic for agent in agents])

            # Update global models
            global_actor.load_state_dict(global_actor_state_dict)
            global_critic.load_state_dict(global_critic_state_dict)

            # Propagate global model weights back to the agents
            for agent in agents:
                agent.target_actor.load_state_dict(global_actor_state_dict)
                agent.target_critic.load_state_dict(global_critic_state_dict)
        state = new_state
        rewards.append(episode_reward)
    print("Rewards:", rewards)

# Define functions for training, weight aggregation, and soft updates
# ...

# def get_state(self):
#     # Get the state of the traffic light, e.g., queue length
#     queue_length = traci.lane.getLastStepVehicleNumber(traci.trafficlight.getControlledLanes(self.traffic_light_id))
#     return np.array([queue_length])

# def get_reward(self):
#     # Define reward based on queue length and collision status
#     collision_count = traci.simulation.getCollidingVehiclesNumber()
#     queue_length = sum(traci.lane.getLastStepVehicleNumber(light) for light in traci.trafficlight.getControlledLanes(self.traffic_light_id))
#     # Reward function considering both queue length and collisions
#     reward = -queue_length
#     if collision_count > 0:
#         reward -= collision_count * -10  # large negative value
#     return reward

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