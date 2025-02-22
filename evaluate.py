import gymnasium
import highway_env
import numpy as np
import torch
from torch import nn
import random
from your_baseline import baseline_policy

# Set the seed and create the environment
np.random.seed(0)
random.seed(0)
torch.manual_seed(0)

env_name = "highway-v0"

env = gymnasium.make(env_name,
                     config={'action': {'type': 'DiscreteMetaAction'}, "lanes_count": 3, "ego_spacing": 1.5},
                     render_mode='human')

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

#define models
class DQN(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(DQN, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim)
        )

    def forward(self, x):
        return self.fc(x)

# Initialize your model and load parameters
state_dim = 5*5
action_dim = env.action_space.n
agent = DQN(state_dim, action_dim).to(device)
agent.load_state_dict(torch.load("dqn_highway_episode378.pth", map_location=device))
agent.eval()


# Evaluation loop
state, _ = env.reset()
state = state.reshape(-1)
done, truncated = False, False

episode = 1
episode_steps = 0
episode_return = 0

while episode <= 10:
    episode_steps += 1
    # Select the action to be performed by the agent
    with torch.no_grad():
        state_tensor = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
        action = agent(state_tensor).argmax(dim=1).item()
    # action = baseline_policy(state)

    state, reward, done, truncated, _ = env.step(action)
    state = state.reshape(-1)
    env.render()

    episode_return += reward

    if done or truncated:
        print(f"Episode Num: {episode} Episode T: {episode_steps} Return: {episode_return:.3f}, Crash: {done}")

        state, _ = env.reset()
        state = state.reshape(-1)
        episode += 1
        episode_steps = 0
        episode_return = 0

env.close()