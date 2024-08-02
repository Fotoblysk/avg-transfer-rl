from config.env_config.gridword import get_minigrid_experiment
from config.env_config.old_playground_configs import old_playground_configs
from good_rainbow_src.utils import merge_dicts

def get_playgroud_configs():
    return get_configs(old_playground_configs)

def get_gridword_configs():
    return get_configs(get_minigrid_experiment())
def get_configs(configs_to_inject):
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
            "atari": True,
        },
        "params": {
            "seed": 123,
            "num_frames": 2000,
            "memory_size": 100000,
            # in orginal is 1M sadly not anough ram 0.4M should be possible tho might get to swap so 0.3 for safety # goes to swap :( 0.09 for now
            "batch_size": 32,  # on internet  128 *12 test, 32 in paper
            "learning_interval": 4,  # because of bigger batch size? -2 just for safety
            "target_update": None,
            # we divide of learning interval and how we count. 32k was in orginal rainbow, 10k in orginal dqn
            "min_memory_size": None,  # memory_size * 9//10 # big start memory for debugging
        },
        "plot": {
            "interval": 100000
        }
    }
    defaults["params"]["target_update"] = 1000 // defaults["params"]["learning_interval"]  # 32000
    defaults["params"]["min_memory_size"] = min(int(defaults["params"]["memory_size"] * 0.9), 80000)

    configs_raw = configs_to_inject

    configs = {}
    for key in configs_raw:
        if "defaults" in configs_raw[key]:
            if configs_raw[key]["defaults"] == "merge":
                configs[key] = defaults | configs_raw[key]
            elif configs_raw[key]["defaults"] == "deepmerge":
                merge_dicts(defaults, configs_raw[key])
                pass  # TODO
            else:
                configs[key] = configs_raw[key]
        else:
            configs[key] = configs_raw[key]
    return configs
