import gymnasium
import highway_env
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
import os

#set the seed and create the environment
np.random.seed(0)
random.seed(0)
torch.manual_seed(0)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

MAX_STEPS = int(2e4)  # This should be enough to obtain nice results, however feel free to change it
env_name = "highway-fast-v0"  # We use the 'fast' env just for faster training, if you want you can use "highway-v0"

env = gymnasium.make(env_name,
                     config={'action': {'type': 'DiscreteMetaAction'}, 'duration': 40, "vehicles_count": 50})

#initialize the parameters
state_dim = env.observation_space.shape
action_dim = env.action_space.n

gamma = 0.99
learning_rate = 1e-4
alpha = 0.6
epsilon = 1.0
eps_end = 0.01
eps_decay = 0.995
beta = 0.4
beta_increment = 0.001
target_update_freq = 20
batch_size = 32
buffer_size = 50000

#keep track of the best results and save checkpoints
checkpoint_freq = 20
save_dir = "First_DDQN"
checkpoint_path = os.path.join(save_dir, "checkpoints")
os.makedirs(checkpoint_path, exist_ok = True)


#create Q-network
class QNet(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(QNet, self).__init__()
        self.fc1 = nn.Linear(input_dim[0], 128),
        self.fc2 = nn.Linear(128, 256),
        self.fc3 = nn.Linear(256, output_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)


#create prioritized experience replay buffer
class ReplayBuffer:
    def __init__(self, capacity, alpha = 0.6):
        self.capacity = capacity
        self.buffer = []
        self.position = 0
        self.priorities = np.zeros((capacity,), dtype=np.float32)
        self.alpha = alpha

    def push(self, state, action, reward, next_state, done, error):
        priority = (error + 1e-5) ** self.alpha
        if len(self.buffer) < self.capacity:
            self.buffer.append((state, action, reward, next_state, done))
        else:
            self.buffer[self.position] = (state, action, reward, next_state, done)

        self.priorities[self.position] = priority
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size, beta = 0.4):
        if len(self.buffer) == self.capacity:
            priorities = self.priorities
        else:
            priorities = self.priorities[:self.position]

        probs = priorities / priorities.sum() #normalize
        indices = np.random.choice(len(self.buffer), size=batch_size, p=probs)
        samples = [self.buffer[idx] for idx in indices]

        #compute importance sampling weights
        weights = (len(self.buffer) * probs[indices])  ** (-beta)
        weights /= weights.max()

        states, actions, rewards, next_states, dones = zip(*samples)

        return np.array(states), actions, rewards, np.array(next_states), dones, indices, np.array(weights, dtype=np.float32)

    def update_priorities(self, batch_indices, errors):
        self.priorities[batch_indices] = (errors + 1e-5) ** self.alpha

    def __len__(self):
        return len(self.buffer)


#create DQN
class DQNAgent:
    def __init__(self, state_dim, action_dim):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.epsilon = epsilon
        self.beta = beta

        self.q_net = QNet(state_dim, action_dim).to(device)
        self.target_net = QNet(state_dim, action_dim).to(device)
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.q_net.parameters(), lr = learning_rate)
        self.memory = ReplayBuffer(buffer_size, alpha)

        self.best_reward = -float("inf")

    def select_action(self, state):
        if random.random() < self.epsilon:
            return env.action_space.sample()
        state = torch.FloatTensor(state).unsqueeze(0).to(device)
        with torch.no_grad():
            return torch.argmax(self.q_net(state)).item()

    def store_experience(self, state, action, reward, next_state, done, error):
        self.memory.push(state, action, reward, next_state, done, error)

    def train(self):
        if len(self.memory.buffer) < batch_size:
            return

        states, actions, rewards, next_states, dones, indices, weights = self.memory.sample(batch_size, self.beta)
        self.beta = min(1.0, self.beta + beta_increment)

        states = torch.FloatTensor(states).to(device)
        actions = torch.LongTensor(actions).unsqueeze(1).to(device)
        rewards = torch.FloatTensor(rewards).to(device)
        next_states = torch.FloatTensor(next_states).to(device)
        dones = torch.FloatTensor(dones).to(device)
        weights = torch.FloatTensor(weights).to(device)

        target_q_values = self.target_net(next_states).max(1)[0]
        targets = rewards + gamma * target_q_values * (1 - dones)

        current_q_values = self.q_net(states).gather(1, actions).squeeze()
        errors = torch.abs(current_q_values - targets).detach().numpy()
        self.memory.update_priorities(indices, errors)

        loss = (weights * (current_q_values - targets.detach()) ** 2).mean()
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.epsilon = max(eps_end, self.epsilon * eps_decay)

    def update_target_net(self):
        self.target_net.load_state_dict(self.q_net.state_dict())

    def save_model(self, filename, directory = save_dir):
        torch.save(self.q_net.state_dict(), os.path.join(directory, filename))


#create DDQN
class DoubleDQNAgent:
    def __init__(self, state_dim, action_dim):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.epsilon = epsilon
        self.beta = beta

        self.q_net = QNet(state_dim, action_dim).to(device)
        self.target_net = QNet(state_dim, action_dim).to(device)
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.q_net.parameters(), lr = learning_rate)
        self.memory = ReplayBuffer(buffer_size, alpha)

        self.best_reward = -float("inf")

    def select_action(self, state):
        if random.random() < self.epsilon:
            return env.action_space.sample()
        state = torch.FloatTensor(state).unsqueeze(0).to(device)
        with torch.no_grad():
            return torch.argmax(self.q_net(state)).item()

    def store_experience(self, state, action, reward, next_state, done, error):
        self.memory.push(state, action, reward, next_state, done, error)

    def train(self):
        if len(self.memory.buffer) < batch_size:
            return

        states, actions, rewards, next_states, dones, indices, weights = self.memory.sample(batch_size, self.beta)
        self.beta = min(1.0, self.beta + beta_increment)

        states = torch.FloatTensor(states).to(device)
        actions = torch.LongTensor(actions).unsqueeze(1).to(device)
        rewards = torch.FloatTensor(rewards).to(device)
        next_states = torch.FloatTensor(next_states).to(device)
        dones = torch.FloatTensor(dones).to(device)
        weights = torch.FloatTensor(weights).to(device)

        next_actions = torch.argmax(self.q_net(next_states), dim = 1, keepdim = True)
        target_q_values = self.target_net(next_states).gather(1, next_actions).squeeze()
        targets = rewards + gamma * target_q_values * (1 - dones)

        current_q_values = self.q_net(states).gather(1, actions).squeeze()
        errors = torch.abs(current_q_values - targets).detach().numpy()
        self.memory.update_priorities(indices, errors)

        loss = (weights * (current_q_values - targets.detach()) ** 2).mean()
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.epsilon = max(eps_end, self.epsilon * eps_decay)

    def update_target_net(self):
        self.target_net.load_state_dict(self.q_net.state_dict())

    def save_model(self, filename, directory = save_dir):
        torch.save(self.q_net.state_dict(), os.path.join(directory, filename))


# Initialize your model
agent = DQNAgent(state_dim, action_dim)

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
    action = agent.select_action(state)

    next_state, reward, done, truncated, _ = env.step(action)
    next_state = next_state.reshape(-1)

    error = abs(reward)

    #store transition in memory
    agent.store_experience(state, action, reward, next_state, done, error)

    #train the model
    agent.train()

    if done or truncated:
        print(f"Total T: {t} Episode Num: {episode} Episode T: {episode_steps} Return: {episode_return:.3f}")

        #save model parameters (only when performance improves)
        if episode_return > agent.best_reward:
            agent.best_reward = episode_return
            agent.save_model("first_ddqn_best_model.pth")
            print("New best model saved")

        #save periodic checkpoint
        if episode % checkpoint_freq == 0:
            agent.save_model(f"first_ddqn_checkpoint_{episode}.pth", directory=checkpoint_path)
            print("Checkpoint saved")

        state, _ = env.reset()
        state = state.reshape(-1)
        episode += 1
        episode_steps = 0
        episode_return = 0

    #update target network
    if episode % target_update_freq == 0:
        agent.update_target_net()

agent.save_model("first_ddqn_last.pth")

env.close()
