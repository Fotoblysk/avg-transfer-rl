import gymnasium as gym

# Create the environment
env = gym.make("MiniGrid-Empty-5x5-v0")


# Reset the environment to get the initial state
state = env.reset()

# Render the initial state
env.render()

# Number of steps to take
num_steps = 100000

for step in range(num_steps):
    # Sample a random action from the action space
    action = env.action_space.sample()

    # Take the action in the environment
    next_state, reward, done, info, _ = env.step(action)

    # Render the environment after taking the action
    env.render()

    # Print the action and the reward

    # If the episode is done, reset the environment
    if done:
        state = env.reset()
        env.render()

print("Train finished!")

# Close the environment
env.close()