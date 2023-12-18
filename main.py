import gymnasium as gym
import torch
from stable_baselines3 import DDPG
import numpy as np
from model import Actor, Critic, ReplayBuffer, Agent, OrnsteinUhlenbeckActionNoise, soft_update, federated_averaging

# Register environment with Gym
gym.register(
    id="SumoGrid-v0",
    entry_point="sumo_grid_env:SumoGridEnv",
    kwargs={"sumo_config": "C:/Users/lucas/Documents/Coding/Traffic-Lights-RL/fremont/osm.sumocfg"} 
)
# Create the environment
env = gym.make("SumoGrid-v0")

state_size = len(env.unwrapped.traffic_light_ids) + len(env.unwrapped.lane_ids)
action_size = len(env.unwrapped.traffic_light_ids)
# Create global actor, critic
global_actor = Actor(state_size, action_size)
global_critic = Critic(state_size, action_size)
# Create local actors, critics, optimizers, and replay buffers for each agent
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
    critic_optimizer = torch.optim.Adam(local_critic.parameters(), lr=1e-4)
    agents.append(Agent(tl_id, local_actor, local_critic, target_actor, target_critic, actor_optimizer, critic_optimizer, ReplayBuffer(buffer_size=500000, batch_size=64)))

# Main federated learning loop
mse_loss = torch.nn.MSELoss()
n_actions = env.action_space.shape[0]
ou_noises = [OrnsteinUhlenbeckActionNoise(mu=np.zeros(1), sigma=0.3) for _ in env.unwrapped.traffic_light_ids]
num_episodes = 100
num_steps = 150
# 500 total episodes
# # 100 for initial testing
for iteration in range(num_episodes):
    episode_reward = 0
    state, _ = env.reset()
    for noise in ou_noises:
        noise.reset()
    results = []
    # 150 steps of learning
    for step in range(num_steps):
        actions = []
        for i, agent in enumerate(agents):
            actions.append(np.clip(agent.get_action(state)  + ou_noises[i](), env.action_space.low[i], env.action_space.high[i]))
        new_state, reward, done, truncated, _ = env.step(actions)
        episode_reward += reward
        for i, agent in enumerate(agents):
            agent.replay_buffer.add(state, actions[i], reward, new_state, done)
        # Parallel training during simulation
        if step + 1 > 64:
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

                # Soft update target networks
                soft_update(agent.actor, agent.target_actor)
                soft_update(agent.critic, agent.target_critic) 
        # Continue simulating
        state = new_state
        results.append(episode_reward)
    # Federated averaging at end of every episode iteration
    global_actor_state_dict = federated_averaging([agent.target_actor for agent in agents])
    global_critic_state_dict = federated_averaging([agent.target_critic for agent in agents])
    # Update global models
    global_actor.load_state_dict(global_actor_state_dict)
    global_critic.load_state_dict(global_critic_state_dict)
    # Propagate global model weights back to the agents
    for agent in agents:
        agent.target_actor.load_state_dict(global_actor_state_dict)
        agent.target_critic.load_state_dict(global_critic_state_dict)
    # Print rewards for this episode
    print("Episode:", iteration + 1, "\nAverage Reward:", episode_reward / num_steps)


# Save the global actor model
torch.save(global_actor.state_dict(), 'global_actor_model.pth')
# Save the global critic model
torch.save(global_critic.state_dict(), 'global_critic_model.pth')

# Generic DDPG model
# Create the action noise for exploration
n_actions = env.action_space.shape[-1]
action_noise = OrnsteinUhlenbeckActionNoise(mu=np.zeros(n_actions), sigma=0.3 * np.ones(n_actions))
model = DDPG("MlpPolicy", env, action_noise=action_noise, verbose=1)
# Train the model
model.learn(total_timesteps=1000)
# Evaluate the model
evaluate_episodes = 10
total_rewards = 0
for episode in range(evaluate_episodes):
    obs, info = env.reset()
    done = False
    episode_reward = 0
    while not done:
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, done, truncated, info = env.step(action)
        episode_reward += reward
    total_rewards += episode_reward
average_reward = total_rewards / evaluate_episodes
print("Average Reward:", average_reward)

# # My model
# actor_model = Actor(state_size, action_size)
# actor_model.load_state_dict(torch.load('global_actor_model.pth'))
# # Evaluation loop
# evaluate_episodes = 10
# total_rewards = 0
# for episode in range(evaluate_episodes):
#     obs, info = env.reset()
#     done = False    
#     episode_reward = 0
#     while not done:
#         action = actor_model(torch.from_numpy(obs).float().unsqueeze(0)).detach().numpy()
#         obs, reward, done, truncated, info = env.step(action)
#         episode_reward += reward
#     total_rewards += episode_reward
# average_reward = total_rewards / evaluate_episodes
# print("Average Reward:", average_reward)