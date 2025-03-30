import gymnasium
import highway_env
import numpy as np
import torch
import random
from training import D3QN

#set the seed and create the environment
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
agent = D3QN(state_dim, action_dim)
agent.q_net.load_state_dict(torch.load("d3qn_model.pth", map_location=device))
agent.q_net.eval()

# Evaluation loop
state, _ = env.reset()
state = state.reshape(-1)
done, truncated = False, False

episode = 1
episode_steps = 0
episode_return = 0
episodes_return = []
crashes = 0
success_rates = []

while episode <= 10:
    episode_steps += 1
    #select the action to be performed by the agent
    state_tensor = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
    action = agent.select_greedy_action(state_tensor)

    state, reward, done, truncated, _ = env.step(action)
    state = state.reshape(-1)
    env.render()

    episode_return += reward

    if done:
        crashes += 1

    if done or truncated:
        print(f"Episode Num: {episode} Episode T: {episode_steps} Return: {episode_return:.3f}, Crash: {done}")

        episodes_return.append(episode_return)
        success_rates.append(100 * episode_steps/40)
        state, _ = env.reset()
        state = state.reshape(-1)
        episode += 1
        episode_steps = 0
        episode_return = 0

print("Average Return: {:.3f}".format(np.mean(episodes_return)))
print("Number of crashes: {:d}".format(crashes))
print("Success rate: {:.2f}%".format(np.mean(success_rates)))

env.close()