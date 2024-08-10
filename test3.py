import numpy as np


def reward_update(total_avg, epizod_n, current_times_used, computed_rewards, ep_steps):
    return (total_avg * epizod_n + (current_times_used / ep_steps) * computed_rewards) / (epizod_n + 1)


avg = np.array([10, 10, 10])
ep_n = 10
used = np.array([1, 4, 5])
rew = 30
ep_steps = 10
print(reward_update(avg, ep_n, used, rew, ep_steps))