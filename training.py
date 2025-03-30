import gymnasium
import highway_env
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
import os
import matplotlib.pyplot as plt

#set the seed and create the environment
np.random.seed(0)
random.seed(0)
torch.manual_seed(0)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

MAX_STEPS = int(3e4)
env_name = "highway-fast-v0"  # We use the 'fast' env just for faster training, if you want you can use "highway-v0"

env = gymnasium.make(env_name,
                     config={'action': {'type': 'DiscreteMetaAction'}, 'duration': 40, "vehicles_count": 50})

# initialize the parameters
state_dim = env.observation_space.shape
action_dim = env.action_space.n

gamma = 0.8
learning_rate = 5e-4
alpha = 0.6
epsilon = 1.0
eps_end = 0.03
eps_decay = 0.997
beta = 0.4
beta_increment = 0.001
batch_size = 32
buffer_size = 20000
learning_starts = 200
tau = 0.003


#create Q-network
class QNet(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(QNet, self).__init__()
        flattened_input_dim = input_dim[0] * input_dim[1]

        self.fc1 = nn.Linear(flattened_input_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, output_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)


#create dueling Q-network
class DuelingQNet(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DuelingQNet, self).__init__()
        flattened_input_dim = input_dim[0] * input_dim[1]

        self.fc1 = nn.Linear(flattened_input_dim, 256)

        self.value_fc = nn.Linear(256, 256)
        self.value = nn.Linear(256, 1) #output a single scalar per state

        self.advantage_fc = nn.Linear(256, 256)
        self.advantage = nn.Linear(256, output_dim) #output a vector with one entry per action

    def forward(self, x):
        x = torch.relu(self.fc1(x)) #pass through the shared layer

        value = torch.relu(self.value_fc(x))
        value = self.value(value)

        advantage = torch.relu(self.advantage_fc(x))
        advantage = self.advantage(advantage)

        q_values = value + (advantage - advantage.mean(dim = 1, keepdim = True))
        return q_values


#create prioritized experience replay buffer
class ReplayBuffer:
    def __init__(self, capacity, alpha = 0.6):
        self.capacity = capacity
        self.buffer = []
        self.position = 0
        self.priorities = np.zeros((capacity,), dtype=np.float32)
        self.alpha = alpha

    #add transitions to the buffer with priorities
    def push(self, state, action, reward, next_state, done, error):
        priority = abs(error) + 1e-5
        if len(self.buffer) < self.capacity:
            self.buffer.append((state, action, reward, next_state, done))
        else:
            self.buffer[self.position] = (state, action, reward, next_state, done)

        self.priorities[self.position] = priority
        self.position = (self.position + 1) % self.capacity

    #sample from the buffer
    def sample(self, batch_size, beta = 0.4):
        if len(self.buffer) == self.capacity:
            priorities = self.priorities
        else:
            priorities = self.priorities[:self.position]

        probs = priorities ** self.alpha
        probs /= probs.sum()
        indices = np.random.choice(len(self.buffer), size=batch_size, p=probs)
        samples = [self.buffer[idx] for idx in indices]

        #compute importance sampling weights
        weights = (len(self.buffer) * probs[indices])  ** (-beta)
        weights /= weights.max()

        states, actions, rewards, next_states, dones = zip(*samples)

        return np.array(states), actions, rewards, np.array(next_states), dones, indices, np.array(weights, dtype=np.float32)

    def update_priorities(self, batch_indices, errors):
        self.priorities[batch_indices] = abs(errors) + 1e-5

    def __len__(self):
        return len(self.buffer)


#create Double DQN agent
class DoubleDQN:
    def __init__(self, state_dim, action_dim):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.epsilon = epsilon
        self.eps_end = eps_end
        self.eps_decay = eps_decay
        self. beta = beta
        self.tau = tau

        self.q_net = QNet(state_dim, action_dim).to(device)
        self.target_net = QNet(state_dim, action_dim).to(device)
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.q_net.parameters(), lr = learning_rate)
        self.memory = ReplayBuffer(buffer_size, alpha)

        self.best_reward = -float("inf")

    #use an epsilon greedy exploration
    def select_action(self, state):
        state = torch.FloatTensor(state).unsqueeze(0).to(device)
        if np.random.random() < self.epsilon:
            action = np.random.randint(self.action_dim)
        else:
            with torch.no_grad():
                action = torch.argmax(self.q_net(state)).item()

        self.epsilon = max(self.eps_end, self.epsilon * self.eps_decay)

        return action

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

        #double q-learning
        next_actions = torch.argmax(self.q_net(next_states), dim = 1, keepdim = True)
        target_q_values = self.target_net(next_states).gather(1, next_actions).squeeze()
        targets = rewards + gamma * target_q_values * (1 - dones)

        current_q_values = self.q_net(states).gather(1, actions).squeeze()
        errors = torch.abs(current_q_values - targets).detach().cpu().numpy()
        self.memory.update_priorities(indices, errors)

        loss = (weights * (current_q_values - targets.detach()) ** 2).mean()
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    #soft update of the target network
    def update_target_net(self):
        for target_param, param in zip(self.target_net.parameters(), self.q_net.parameters()):
            target_param.data.copy_(param.data * self.tau + target_param.data * (1 - self.tau))

    def save_model(self, filename, directory):
        torch.save(self.q_net.state_dict(), os.path.join(directory, filename))

    def select_greedy_action(self, state):
        with torch.no_grad():
            return torch.argmax(self.q_net(state)).item()


#create Dueling DQN agent
class DuelingDQN:
    def __init__(self, state_dim, action_dim):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.epsilon = epsilon
        self.eps_end = eps_end
        self.eps_decay = eps_decay
        self. beta = beta
        self.tau = tau

        self.q_net = DuelingQNet(state_dim, action_dim).to(device)
        self.target_net = DuelingQNet(state_dim, action_dim).to(device)
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.q_net.parameters(), lr = learning_rate)
        self.memory = ReplayBuffer(buffer_size, alpha)

        self.best_reward = -float("inf")

    #use an epsilon greedy exploration
    def select_action(self, state):
        state = torch.FloatTensor(state).unsqueeze(0).to(device)
        if np.random.random() < self.epsilon:
            action = np.random.randint(self.action_dim)
        else:
            with torch.no_grad():
                action = torch.argmax(self.q_net(state)).item()

        self.epsilon = max(self.eps_end, self.epsilon * self.eps_decay)

        return action

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

        with torch.no_grad():
            target_q_values = self.target_net(next_states).max(1)[0]
        targets = rewards + gamma * target_q_values * (1 - dones)

        current_q_values = self.q_net(states).gather(1, actions).squeeze()
        errors = torch.abs(current_q_values - targets).detach().cpu().numpy()
        self.memory.update_priorities(indices, errors)

        loss = (weights * (current_q_values - targets.detach()) ** 2).mean()
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def update_target_net(self):
        for target_param, param in zip(self.target_net.parameters(), self.q_net.parameters()):
            target_param.data.copy_(param.data * self.tau + target_param.data * (1 - self.tau))

    def save_model(self, filename, directory):
        torch.save(self.q_net.state_dict(), os.path.join(directory, filename))

    def select_greedy_action(self, state):
        with torch.no_grad():
            return torch.argmax(self.q_net(state)).item()


#create D3QN agent
class D3QN:
    def __init__(self, state_dim, action_dim):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.epsilon = epsilon
        self.eps_end = eps_end
        self.eps_decay = eps_decay
        self. beta = beta
        self.tau = tau

        self.q_net = DuelingQNet(state_dim, action_dim).to(device)
        self.target_net = DuelingQNet(state_dim, action_dim).to(device)
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.q_net.parameters(), lr = learning_rate)
        self.memory = ReplayBuffer(buffer_size, alpha)

        self.best_reward = -float("inf")

    #use an epsilon greedy exploration
    def select_action(self, state):
        state = torch.FloatTensor(state).unsqueeze(0).to(device)
        if np.random.random() < self.epsilon:
            action = np.random.randint(self.action_dim)
        else:
            with torch.no_grad():
                action = torch.argmax(self.q_net(state)).item()

        self.epsilon = max(self.eps_end, self.epsilon * self.eps_decay)

        return action

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

        #double q-learning
        next_actions = torch.argmax(self.q_net(next_states), dim = 1, keepdim = True)
        target_q_values = self.target_net(next_states).gather(1, next_actions).squeeze()
        targets = rewards + gamma * target_q_values * (1 - dones)

        current_q_values = self.q_net(states).gather(1, actions).squeeze()
        errors = torch.abs(current_q_values - targets).detach().cpu().numpy()
        self.memory.update_priorities(indices, errors)

        loss = (weights * (current_q_values - targets.detach()) ** 2).mean()
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def update_target_net(self):
        for target_param, param in zip(self.target_net.parameters(), self.q_net.parameters()):
            target_param.data.copy_(param.data * self.tau + target_param.data * (1 - self.tau))

    def save_model(self, filename, directory):
        torch.save(self.q_net.state_dict(), os.path.join(directory, filename))

    def select_greedy_action(self, state):
        with torch.no_grad():
            return torch.argmax(self.q_net(state)).item()


def plot_learning_curve(data_dict, save_path, baseline_return = 22, metric = "Returns", window = 50):
    plt.figure(figsize=(10, 6))

    colors = {"DoubleDQN": "blue",
        "DuelingDQN": "green",
        "D3QN": "purple"}

    for model_name, data in data_dict.items():
        episodes = np.arange(len(data))
        smoothed_data = np.convolve(data, np.ones(window) / window, mode = "valid")

        color = colors.get(model_name)
        plt.plot(episodes[:len(smoothed_data)], smoothed_data, label = model_name, color = color)

    if metric == "Returns":
        plt.ylabel("Episode Return")
    elif metric == "Episode length":
        plt.ylabel("Steps per Episode")

    plt.axhline(y=baseline_return, color = "red", linestyle = "--", label = 'Baseline')

    plt.xlabel('Episode')
    plt.title(f'Learning Curve: {metric.capitalize()}')
    plt.legend()
    plt.grid(True)
    plt.savefig(save_path)
    plt.show


agents = [DoubleDQN, DuelingDQN, D3QN]

agents_folders = {DoubleDQN: "Final_DDQN_2",
                  DuelingDQN: "Final_DuelDQN_2",
                  D3QN: "Final_D3QN_2"}

def main():

    all_returns = {}
    all_lengths = {}

    for agent_class in agents:
        #initialize your model
        agent = agent_class(state_dim, action_dim)

        model_name = agent_class.__name__
        all_returns[model_name] = []
        all_lengths[model_name] = []

        save_directory = agents_folders[agent_class]
        os.makedirs(save_directory, exist_ok = True)

        checkpoint_freq = 20
        checkpoint_path = os.path.join(save_directory, "checkpoints")
        os.makedirs(checkpoint_path, exist_ok=True)

        state, _ = env.reset()
        state = state.reshape(-1)
        done, truncated = False, False

        episode = 1
        episode_steps = 0
        total_steps = 0
        episode_return = 0

        #training loop
        for t in range(MAX_STEPS):
            episode_steps += 1
            total_steps += 1

            action = agent.select_action(state)

            next_state, reward, done, truncated, _ = env.step(action)
            next_state = next_state.reshape(-1)

            #store transition in memory
            agent.store_experience(state, action, reward, next_state, done, 0)

            #train the model
            if total_steps > learning_starts:
                agent.train()

            state = next_state
            episode_return += reward

            agent.update_target_net()

            if done or truncated:
                print(f"Total T: {t} Episode Num: {episode} Episode T: {episode_steps} Return: {episode_return:.3f}")

                all_returns[model_name].append(episode_return)
                all_lengths[model_name].append(episode_steps)

                #save model parameters (only when performance improves)
                if episode_return > agent.best_reward:
                    agent.best_reward = episode_return
                    agent.save_model("best_model.pth", directory=save_directory)
                    print("New best model saved")

                #save periodic checkpoint
                if episode % checkpoint_freq == 0:
                    agent.save_model(f"checkpoint_{episode}.pth", directory=checkpoint_path)
                    print("Checkpoint saved")

                state, _ = env.reset()
                state = state.reshape(-1)
                episode += 1
                episode_steps = 0
                episode_return = 0

        agent.save_model("last_model.pth", directory=save_directory)

        print(f"Training complete for {model_name}! Moving to the next model...")

        env.close()

    plot_learning_curve(all_returns, save_path="models_returns_2.png")
    plot_learning_curve(all_lengths, save_path="models_lengths_2.png", metric="Episode length", baseline_return=30.9)

if __name__ == "__main__":
    main()

