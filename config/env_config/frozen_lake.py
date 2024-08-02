def get_frozen_lake_experiment():
    descs = [
        [
            "S_______",
            "________",
            "___H____",
            "_____H__",
            "___H____",
            "_HH___H_",
            "_H__H_H_",
            "___H___G",
        ],
        [
            "S______H",
            "________",
            "___H____",
            "_____H__",
            "___H____",
            "_HH___H_",
            "_H__H_H_",
            "___H___G",
        ],
        [
            "S_HHHHHH",
            "_____HHH",
            "___H___H",
            "_____H__",
            "___H____",
            "_HH___H_",
            "_H__H_H_",
            "___H___G",
        ],
        [
            "S__H____",
            "___H____",
            "___HHH__",
            "_____H__",
            "___H____",
            "_HH___H_",
            "_H__H_H_",
            "___H___G",
        ],
    ]
    return {
        f"FrozenLake-map_{i}": {
            "defaults": "merge",
            "env_kwargs": {
                "id": "FrozenLake-v1",
                "desc": v,
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
        } for i, v in enumerate(descs)
    }
