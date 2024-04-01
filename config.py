defaults = {
    "meta": {
        "v_min": -10,  # min value including discount factor 10 is used in paper
        "v_max": 10,  # max value including discount factor -10 is used in paper
        "atom_size": 51,  # number of grains, grain size is v_max-vmin/atom_size # dont want to count if equal range
        "reward_clip": (-1, 1),  # reward clip should be enough buf for better score distribution we might want to
    },
    "preprocess": {
        "frame_stack": 4,
        "frameskip": 4,
        "atari": True
    },
    "params": {
        "seed": 777,
        "num_frames": 20000000,
        "memory_size": 1000000,
        # in orginal is 1M sadly not anough ram 0.4M should be possible tho might get to swap so 0.3 for safety # goes to swap :( 0.09 for now
        "batch_size": 32,  # on internet  128 *12 test, 32 in paper
        "learning_interval": 4,  # because of bigger batch size? -2 just for safety
        "target_update": None,
        # we divide of learning interval and how we count. 32k was in orginal rainbow, 10k in orginal dqn
        "min_memory_size": None,  # memory_size * 9//10 # big start memory for debugging
    },
    "plot": {
        "interval": 10000
    }
}
defaults["params"]["target_update"] = 32000 // defaults["params"]["learning_interval"]
defaults["params"]["min_memory_size"] = min(int(defaults["params"]["memory_size"] * 0.9), 80000)

configs_raw = {
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
            "v_min": -10,
            "v_max": 10,
            "atom_size": 51,
            "reward_scale": (1 / 100, 0)
        },
        "preprocess": {
            "frame_stack": 4,
            "atari": False
        }
    },

    ################################
    # Lunar Lander
    ################################
    "lunar_lander": {
        "env_kwargs": {
            "id": "LunarLander-v2",
            "continuous": False,
            "gravity": -10.0,
            "enable_wind": False,
            "wind_power": 15.0,
            "turbulence_power": 1.5
        },
        "meta": {
            "v_min": -10,
            "v_max": 10,
            "atom_size": 51,
            "reward_scale": (1 * 10 / ((140 + 100 + 2 * 10)), 0)
        },
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
    "car_racing": {
        "custom_env": None
    }

}

configs = {}
for key in configs_raw:
    if "defaults" in configs_raw[key]:
        if configs_raw[key]["defaults"] == "merge":
            configs[key] = defaults | configs_raw[key]
        elif configs_raw[key]["defaults"] == "deepmerge":
            pass  # TODO
        else:
            configs[key] = configs_raw[key]
    else:
        configs[key] = configs_raw[key]
