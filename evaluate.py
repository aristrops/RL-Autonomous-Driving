import gymnasium
import highway_env
import numpy as np
import torch
from torch import nn
import random
from your_baseline import baseline_policy
from training import DQNAgent, DoubleDQNAgent, DuelingDQNAgent

# Set the seed and create the environment
np.random.seed(0)
random.seed(0)
torch.manual_seed(0)

env_name = "highway-v0"

env = gymnasium.make(env_name,
                     config={'action': {'type': 'DiscreteMetaAction'}, "lanes_count": 3, "ego_spacing": 1.5},
                     render_mode='human')

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


#initialize your model and load parameters
state_dim = env.observation_space.shape
action_dim = env.action_space.n
agent = DuelingDQNAgent(state_dim, action_dim)
agent.q_net.load_state_dict(torch.load("First_DuelDQN/first_dueldqn_best_model.pth", map_location=device))
agent.q_net.eval()

# Evaluation loop
state, _ = env.reset()
state = state.reshape(-1)
done, truncated = False, False

episode = 1
episode_steps = 0
episode_return = 0
episodes_return = []

while episode <= 10:
    episode_steps += 1
    # Select the action to be performed by the agent
    state_tensor = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
    action = agent.select_greedy_action(state_tensor)
    # action = baseline_policy(state)

    state, reward, done, truncated, _ = env.step(action)
    state = state.reshape(-1)
    env.render()

    episode_return += reward

    if done or truncated:
        print(f"Episode Num: {episode} Episode T: {episode_steps} Return: {episode_return:.3f}, Crash: {done}")

        episodes_return.append(episode_return)
        state, _ = env.reset()
        state = state.reshape(-1)
        episode += 1
        episode_steps = 0
        episode_return = 0

print("Average Return: {:.3f}".format(np.mean(episodes_return)))

env.close()