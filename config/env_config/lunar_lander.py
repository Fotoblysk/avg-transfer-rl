from copy import deepcopy


def get_lunar_lander_experiment():
    base = {"lunar_lander_10_0_0": {
        "defaults": "merge",
        "env_kwargs": {
            "id": "LunarLander-v2",
            "continuous": False,
            "gravity": -10,
            "enable_wind": False,
            "wind_power": 0.0,
            "turbulence_power": 0.0
        },
        "meta": {
            "v_min": -10,
            "v_max": 10,
            "atom_size": None,
            "reward_scale": (1 * 10 / ((140 + 100 + 2 * 10)), 0)
        },
        "preprocess": {
            "atari": False
        }
    }}
    gravity_values = [11, 8]
    gravity = {f"lunar_lander_{int(i)}_0_0": deepcopy(base["lunar_lander_10_0_0"]) for i in gravity_values}
    for i in gravity_values:
        gravity[f"lunar_lander_{int(i)}_0_0"]["env_kwargs"]["gravity"] = -i

    wind_pow_values = [10.0, 20.0]
    wind_pow = {f"lunar_lander_10_{int(i)}_0": deepcopy(base["lunar_lander_10_0_0"]) for i in wind_pow_values}
    for i in wind_pow_values:
        wind_pow[f"lunar_lander_10_{int(i)}_0"]["env_kwargs"]["enable_wind"] = True
        wind_pow[f"lunar_lander_10_{int(i)}_0"]["env_kwargs"]["wind_power"] = i

    tur_pow_values = [1.0, 2.0]
    tur_pow = {f"lunar_lander_10_0_{int(i)}": deepcopy(base["lunar_lander_10_0_0"]) for i in tur_pow_values}
    for i in tur_pow_values:
        tur_pow[f"lunar_lander_10_0_{int(i)}"]["env_kwargs"]["enable_wind"] = True
        tur_pow[f"lunar_lander_10_0_{int(i)}"]["env_kwargs"]["turbulence_power"] = i

    final_key = f"lunar_lander_{int(gravity_values[-1])}_{int(wind_pow_values[-1])}_{int(tur_pow_values[-1])}"
    final_v = {final_key: deepcopy(base["lunar_lander_10_0_0"])}
    final_v[final_key]["env_kwargs"]["enable_wind"] = True
    final_v[final_key]["env_kwargs"]["gravity"] = -gravity_values[-1]
    final_v[final_key]["env_kwargs"]["wind_power"] = wind_pow_values[-1]
    final_v[final_key]["env_kwargs"]["turbulence_power"] = tur_pow_values[-1]

    all_envs = {**base, **gravity, **wind_pow, **tur_pow, **final_v}

    return all_envs


if __name__ == '__main__':
    get_lunar_lander_experiment()
