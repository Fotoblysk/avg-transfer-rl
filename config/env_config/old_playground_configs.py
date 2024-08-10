old_playground_configs = {
    # For "defaults" option possible valuse are:
    # "ignore" - default (no need to specify)
    # "merge" - merge configs in order (defaults | config)
    # "deepmerge" - TODO: deepmerge configs in order (defaults | config)

    ################################
    # Cart Pole
    ################################
    "CartPole": {
        "defaults": "merge",
        "env_kwargs": {
            "id": "CartPole-v1",
        },
        "meta": {
            "v_min": None,
            "v_max": None,
            "atom_size": None,
            "reward_scale": (1 / 100, 0)
        },
        "preprocess": {
            "atari": False
        }
    },
    "FrozenLake-2": {
        "defaults": "merge",
        "env_kwargs": {
            "id": "FrozenLake-v1",
            "map_name": "8x8",
            "is_slippery": False
        },
        "meta": {
            "v_min": None,
            "v_max": None,
            "atom_size": None,
            "reward_scale": (1, 0)
        },
        "preprocess": {
            "one_hot_obs": True,
            "atari": False
        }
    },
    "FrozenLake": {
        "defaults": "merge",
        "env_kwargs": {
            "id": "FrozenLake-v1",
            "map_name": "4x4",
            "is_slippery": False
        },
        "meta": {
            "v_min": 0,
            "v_max": 1,
            "atom_size": 5,
            "reward_scale": (1, 0)
        },
        "preprocess": {
            "one_hot_obs": True,
            "atari": False
        }
    },

    "lunar_lander-2": {
        "defaults": "merge",
        "env_kwargs": {
            "id": "LunarLander-v2",
            "continuous": False,
            "gravity": -10,
            "enable_wind": True,
            "wind_power": 15.0,
            "turbulence_power": 1.5
        },
        "meta": {
            "v_min": None,
            "v_max": None,
            "atom_size": None,
            "reward_scale": (1 * 10 / ((140 + 100 + 2 * 10)), 0)
        },
        "preprocess": {
            "atari": False
        }
    },
    ################################
    # Lunar Lander
    ################################
    "lunar_lander": {
        "defaults": "merge",
        "env_kwargs": {
            "id": "LunarLander-v2",
            "continuous": False,
            "gravity": -10,
            "enable_wind": True,
            "wind_power": 15.0,
            "turbulence_power": 1.5
        },
        "meta": {
            "v_min": -10,
            "v_max": 10,
            "atom_size": 51,
            "reward_scale": (1 * 10 / ((140 + 100 + 2 * 10)), 0)
        },
        "preprocess": {
            "atari": False
        }
    },
    "MiniGrid-DoorKey-5x5-v0": {
        "defaults": "merge",
        "env_kwargs": {
            "id": "MiniGrid-DoorKey-5x5-v0",
        },
        "meta": {
            "v_min": None,
            "v_max": None,
            "atom_size": None,
            "reward_scale": (1, 0)
        },
        "preprocess": {
            "atari": False,
            "minigrid-dict-flat": True,
            "frame_stack": 5,
            "flatten": True

        }
    },

    "MiniGrid-Unlock-v0": {
        "defaults": "merge",
        "env_kwargs": {
            "id": "MiniGrid-Unlock-v0",
        },
        "meta": {
            "v_min": None,
            "v_max": None,
            "atom_size": None,
            "reward_scale": (1, 0)
        },
        "preprocess": {
            "atari": False,
            "minigrid-dict-flat": True,
            "frame_stack": 5,
            "flatten": True

        }
    },

    "MiniGrid-GoToObject-8x8-N2-v0": {
        "defaults": "merge",
        "env_kwargs": {
            "id": "MiniGrid-GoToObject-8x8-N2-v0",
        },
        "meta": {
            "v_min": None,
            "v_max": None,
            "atom_size": None,
            "reward_scale": (1, 0)
        },
        "preprocess": {
            "atari": False,
            "minigrid-dict-flat": True,
            "frame_stack": 5,
            "flatten": True

        }
    },

    "MiniGrid-BlockedUnlockPickup-v0": {
        "defaults": "merge",
        "env_kwargs": {
            "id": "MiniGrid-BlockedUnlockPickup-v0",
        },
        "meta": {
            "v_min": None,
            "v_max": None,
            "atom_size": None,
            "reward_scale": (1, 0)
        },
        "preprocess": {
            "atari": False,
            "minigrid-dict-flat": True,
            "frame_stack": 5,
            "flatten": True

        }
    },

    "MiniGrid-LockedRoom-v0": {
        "defaults": "merge",
        "env_kwargs": {
            "id": "MiniGrid-LockedRoom-v0",
        },
        "meta": {
            "v_min": None,
            "v_max": None,
            "atom_size": None,
            "reward_scale": (1, 0)
        },
        "preprocess": {
            "atari": False,
            "minigrid-dict-flat": True,
            "frame_stack": 5,
            "flatten": True

        }
    },

    ################################
    # Pong
    ################################
    "pong": {  # TODO orginal paper clips reward (0,1) we might want to normalize :)
        "defaults": "merge",
        "env_kwargs": {
            "id": "ALE/Pong-v5",
            "frameskip": 1,
            "repeat_action_probability": 0
        },
        "preprocess": {
            "frame_stack": 4,
            "grayscale": True,
            "resize_obs": (84, 84)
        }
    },

    ################################
    # DemonAttack
    ################################
    "demon_attack": {
        "defaults": "merge",
        "env_kwargs": {
            "id": "ALE/DemonAttack-v5",
            "frameskip": 1,
            "repeat_action_probability": 0
        },
        "preprocess": {
            "frame_stack": 4,
            "grayscale": True,
            "resize_obs": (84, 84)
        }
    },

    "demon_attack_v4": {  # TODO orginal paper clips reward (0,1) we might want to normalize :)
        "defaults": "merge",
        "env_kwargs": {
            "id": "ALE/DemonAttack-v5",
            "frameskip": 1,
            "repeat_action_probability": 0
        }
    },

    ################################
    # Qbert
    ################################
    "qbert": {  # TODO orginal paper clips reward (0,1) we might want to normalize :)
        "defaults": "merge",
        "env_kwargs": {
            "id": "ALE/Qbert-v5",
            "frameskip": 1,
            "repeat_action_probability": 0
        }
    },

    ################################
    # Freeway
    ################################
    "freeway": {  # TODO orginal paper clips reward (0,1) we might want to normalize :)
        "defaults": "merge",
        "env_kwargs": {
            "id": "ALE/Freeway-v5",
            "frameskip": 1,
            "repeat_action_probability": 0
        }
    },

    ################################
    # Pitfall
    ################################
    "pitfall": {
        "env_kwargs": {
            "id": "ALE/Pitfall-v5",
            "frameskip": 1,
            "repeat_action_probability": 0
        },
        "meta": {
            "v_min": -0.1,  # min value including discount factor
            "v_max": 144,  # max value including discount factor
            "atom_size": 51,
            # number of grains, grain size is v_max-vmin/atom_size # dont want to count if equal range #this might be small for life loss
            "reward_scale": (1 / 1000, 0),
            # "reward_clip": (-1,1)

        },
        "preprocess": {
            "frame_stack": 4,
            "grayscale": True,
            "resize_obs": (84, 84)
        }
    },

    ################################
    # CarRacing
    ################################
    "car_racing": {
        "env_kwargs": {
            "id": "CarRacing-v2",
            "continuous": False
        },
        "meta": {
            "v_min": -10,  # min value including discount factor
            "v_max": 10,  # max value including discount factor
            "atom_size": 51,  # number of grains, grain size is v_max-vmin/atom_size
            "correct_dims": True,
            "reward_scale": (1 / 1000, 0),
        }
    },
}
