#!/usr/bin/env python
import copy
import os
import sys

import runner_src.core as app
from config import config
import matplotlib

from config.config import get_gridword_configs
from config.env_config.frozen_lake import get_frozen_lake_experiment

# app_c = app.Core(config.configs)
# experiment_plan = experiment_planner()
# app_c.menu()

print("Flat envs : ", len(config.get_joint_configs()))
envs_print = [f"{i}: {k}" for i, k in enumerate(config.get_joint_configs())]
print(f"ENVS:")
[print(i) for i in envs_print]
app_c = app.Core(config.get_joint_configs())
n = len(sys.argv)
# additional_settings["meta"] = {"variant": "DIRECT_WEIGHT_REUSE"}
# additional_settings["meta"] = {"variant": "NO_TRANSFER"}
# additional_settings["meta"] = {"variant": "PRQL"}
# additional_settings["meta"] = {"variant": "APDPR"}
additional_settings = dict()
print(os.environ["EXP_SET"])
additional_settings["meta"] = {"variant": os.environ["EXP_SET"]}
experiment_plan = [(int(i), copy.deepcopy(additional_settings)) for i in sys.argv[1:]]

app_c.run_from_console(experiment_plan)
