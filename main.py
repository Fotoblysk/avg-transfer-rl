#!/usr/bin/env python
import sys


import runner_src.core as app
from config import config
import matplotlib

from config.config import get_gridword_configs
from config.env_config.frozen_lake import get_frozen_lake_experiment


#app_c = app.Core(config.configs)
#experiment_plan = experiment_planner()
#app_c.menu()

print("Flat envs : ", len(config.get_joint_configs()))
envs_print = [f"{i}: {k}" for i, k in enumerate(config.get_joint_configs())]
print(f"ENVS:")
[print(i) for i in envs_print]
app_c = app.Core(config.get_joint_configs())
n = len(sys.argv)
experiment_plan = [(int(i), {}) for i in sys.argv[1:]]
app_c.run_from_console(experiment_plan)
