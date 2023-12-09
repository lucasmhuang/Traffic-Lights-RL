import gymnasium as gym
from stable_baselines3 import DDPG
from stable_baselines3.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise
from stable_baselines3.common.env_util import make_vec_env
import numpy as np

# Register your environment with Gym
gym.register(
    id="SumoGrid-v0",
    entry_point="sumo_grid_env:SumoGridEnv",
    kwargs={"sumo_config": "C:/Users/lucas/OneDrive/Documents/Coding/Traffic/network/fremont.sumocfg"} 
)

# Create the environment
env = gym.make("SumoGrid-v0")

# Create the action noise for exploration
n_actions = env.action_space.shape[-1]
action_noise = OrnsteinUhlenbeckActionNoise(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions))

# Initialize the DDPG model
# # DDPG many collisions
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