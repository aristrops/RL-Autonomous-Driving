import numpy as np
import random
import torch
import gymnasium
import highway_env
from highway_env.envs.common import observation

#set the seeds
np.random.seed(0)
random.seed(0)
torch.manual_seed(0)

env_name = "highway-v0"

env = gymnasium.make(env_name,
                     config={'action': {'type': 'DiscreteMetaAction'}, "lanes_count": 3, "ego_spacing": 1.5},
                     render_mode='human')

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def baseline_policy(observation):
    other_vehicles = []

    #define values relative to the other vehicles
    for i in range(5, 25, 5):
        other_vehicles.append(observation[i:i+4])

    front_vehicle, right_vehicle, left_vehicle = None, None, None

    #define a threshold for safe distance
    safe_distance = 0.1

    for vehicle in other_vehicles:
        #set longitudinal and lateral distance
        longitudinal_distance, lateral_distance = vehicle[1], vehicle[2]

        if 0 <= lateral_distance <= 0.3: #the other vehicle is in the same lane
            if longitudinal_distance > 0 and (front_vehicle is None or longitudinal_distance < front_vehicle[1]): #ensure we take the closest
                front_vehicle = vehicle
        elif 0.3 <= lateral_distance <= 0.6: #the other vehicle is in the right lane
            if right_vehicle is None or longitudinal_distance < right_vehicle[1]:
                right_vehicle = vehicle
        elif -0.6 <= lateral_distance <= -0.3: #the other vehicle is in the left lane
            if left_vehicle is None or longitudinal_distance < left_vehicle[1]:
                left_vehicle = vehicle

    #default action: maintain the same speed
    action = 1

    #prioritize moving right when possible
    if right_vehicle is None or right_vehicle[1] > safe_distance:
        action = 2

    #switch left when necessary, if left is occupied slow down
    if front_vehicle is not None and front_vehicle[1] < safe_distance:
        if left_vehicle is None or left_vehicle[1] > safe_distance:
            action = 0
        else:
            action = 4

    return action


#evaluation loop
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
    # Select the action to be performed by the agent
    state_tensor = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
    action = baseline_policy(state)

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


