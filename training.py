import gymnasium
import highway_env
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
import collections

#create experience replay buffer
class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = collections.deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return np.array(states), actions, rewards, np.array(next_states), dones

    def __len__(self):
        return len(self.buffer)


# Set the seed and create the environment
np.random.seed(0)
random.seed(0)
torch.manual_seed(0)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

MAX_STEPS = int(2e4)  # This should be enough to obtain nice results, however feel free to change it
env_name = "highway-fast-v0"  # We use the 'fast' env just for faster training, if you want you can use "highway-v0"

env = gymnasium.make(env_name,
                     config={'action': {'type': 'DiscreteMetaAction'}, 'duration': 40, "vehicles_count": 50})


#start with a basic DQN
class DQN(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(DQN, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, action_dim)
        )

    def forward(self, x):
        return self.fc(x)


#initialize the parameters
state_dim = 5*5
action_dim = env.action_space.n
policy_net = DQN(state_dim, action_dim).to(device)
target_net = DQN(state_dim, action_dim).to(device)
target_net.load_state_dict(policy_net.state_dict()) #set the weights of the target network to be the same as those of the policy network
target_net.eval() #don't update the parameters of the target network
optimizer = optim.Adam(policy_net.parameters(), lr=1e-4)
replay_buffer = ReplayBuffer(50000)

epsilon = 1.0
eps_end = 0.01
eps_decay = 500 #higher implies a slower decay
gamma = 0.99
batch_size = 32
target_update = 10


#keep track of the best results
best_return = -float("inf")

# Initialize your model
agent = policy_net

state, _ = env.reset()
state = state.reshape(-1)
done, truncated = False, False

episode = 1
episode_steps = 0
episode_return = 0

# Training loop
for t in range(MAX_STEPS):
    episode_steps += 1

    #select action with epsilon-greedy strategy
    if random.random() < epsilon:
        action = env.action_space.sample()
    else:
        with torch.no_grad():
            state_tensor = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
            action = policy_net(state_tensor).argmax().item()

    next_state, reward, done, truncated, _ = env.step(action)
    next_state = next_state.reshape(-1)

    #store transition in memory
    replay_buffer.push(state, action, reward, next_state, done)
    state = next_state
    episode_return += reward

    #train the model
    if len(replay_buffer) > batch_size:
        states, actions, rewards, next_states, dones = replay_buffer.sample(batch_size)
        states = torch.tensor(states, dtype=torch.float32, device=device)
        actions = torch.tensor(actions, dtype=torch.long, device=device).unsqueeze(1)
        rewards = torch.tensor(rewards, dtype=torch.float32, device=device)
        next_states = torch.tensor(next_states, dtype=torch.float32, device=device)
        dones = torch.tensor(dones, dtype=torch.float32, device=device)

        q_values = policy_net(states).gather(1, actions).squeeze() #take the q-values of the action actually taken
        next_q_values = target_net(next_states).max(1)[0].detach() #compute the best possible q-value for the next states based on the target network
        target_q_values = rewards + (gamma * next_q_values * (1 - dones)) #(1-dones) handles the terminal states

        loss = nn.MSELoss()(q_values, target_q_values)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    #epsilon decay
    epsilon = max(eps_end, epsilon * np.exp(-1.0 / eps_decay))

    if done or truncated:
        print(f"Total T: {t} Episode Num: {episode} Episode T: {episode_steps} Return: {episode_return:.3f}")

        # Save training information and model parameters (only when performance improves)
        if episode_return > best_return:
            best_return = episode_return
            torch.save(policy_net.state_dict(), f"first_dqn_highway_best.pth")

        state, _ = env.reset()
        state = state.reshape(-1)
        episode += 1
        episode_steps = 0
        episode_return = 0

    #update target network
    if episode % target_update == 0:
        target_net.load_state_dict(policy_net.state_dict())

env.close()
