mars_base = {
    "gravity": -10.0,
    "enable_wind": False,
    "wind_power": 0,
    "turbulence_power": 1.5,
}

mars_parametrization = {
    "gravity": {"min": -12.0, "max": 0, "base": 10, "param_type": "float"},  # clipping range
    "enable_wind": True,  # enabling
    "wind_power": {"min": 0, "max": 20, "base": 0, "param_type": "float"},  # recommended range
    "turbulence_power": {"min": 0, "max": 2, "base": 0, "param_type": "float"},  # recommended range
}
