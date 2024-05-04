#from pettingzoo.classic import rps_v2
from pettingzoo.mpe import simple_world_comm_v3

env = simple_world_comm_v3.env(render_mode="human")
env.reset(seed=42)

for agent in env.agent_iter():
    observation, reward, termination, truncation, info = env.last()

    if termination or truncation:
        action = None
    else:
        action = env.action_space(agent).sample()  # this is where you would insert your policy

    print(agent)
    print(termination or truncation)
    env.step(action)
env.close()
