import gymnasium
import highway_env
import numpy as np


# Remember to save what you will need for the plots

env_name = "highway-v0"
env = gymnasium.make(env_name,
                     config={"manual_control": True, "lanes_count": 3, "ego_spacing": 1.5},
                     render_mode='human')

env.reset()
done, truncated = False, False

episode = 1
episode_steps = 0
episode_return = 0
episodes_return = []
crashes = 0
success_rates = []

while episode <= 10:
    episode_steps += 1

    # Hint: take a look at the docs to see the difference between 'done' and 'truncated'
    _, reward, done, truncated, _ = env.step(env.action_space.sample())  # With manual control these actions are ignored
    env.render()

    episode_return += reward

    if done:
        crashes += 1

    if done or truncated:
        print(f"Episode Num: {episode} Episode T: {episode_steps} Return: {episode_return:.3f}, Crash: {done}")

        episodes_return.append(episode_return)
        success_rates.append(100 * episode_steps / 40)
        env.reset()
        episode += 1
        episode_steps = 0
        episode_return = 0

print("Average Return: {:.3f}".format(np.mean(episodes_return)))
print("Number of crashes: {:d}".format(crashes))
print("Success rate: {:.2f}%".format(np.mean(success_rates)))

env.close()
