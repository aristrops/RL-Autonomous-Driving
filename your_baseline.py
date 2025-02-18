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

def baseline_policy(observation):
    other_vehicles = []
    for i in range(5, 25, 5):
        other_vehicles.append(observation[i:i+4])

    front_vehicle, right_vehicle, left_vehicle = None, None, None

    #define a threshold for front distance
    safe_distance = 0.1

    for vehicle in other_vehicles:
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
        # else:
        #     action = 4

    return action



