def get_minigrid_experiment():
    empty = [
        "MiniGrid-Empty-5x5-v0",
        "MiniGrid-Empty-Random-5x5-v0",
        "MiniGrid-Empty-6x6-v0",
        "MiniGrid-Empty-Random-6x6-v0",
        "MiniGrid-Empty-8x8-v0",
        "MiniGrid-Empty-16x16-v0",
    ]
    four_rooms = [
        "MiniGrid-FourRooms-v0",
    ]
    unlock = [
        "MiniGrid-Unlock-v0",
    ]
    door_key = [
        "MiniGrid-DoorKey-5x5-v0",
        "MiniGrid-DoorKey-6x6-v0",
    ]
    go_to_door = [
        "MiniGrid-GoToDoor-5x5-v0",
        "MiniGrid-GoToDoor-6x6-v0",
        "MiniGrid-GoToDoor-8x8-v0",
    ]
    fetch = [
        "MiniGrid-Fetch-5x5-N2-v0",
        "MiniGrid-Fetch-6x6-N2-v0",
        "MiniGrid-Fetch-8x8-N3-v0",
    ]
    unlock_pickup = [
        "MiniGrid-Unlock-v0",
    ]
    red_blue_door = [
        "MiniGrid-RedBlueDoors-6x6-v0",
        "MiniGrid-RedBlueDoors-8x8-v0",
    ]
    go_to_object = [
        "MiniGrid-GoToObject-6x6-N2-v0",
        "MiniGrid-GoToObject-8x8-N2-v0",
    ]
    multi_room = [
        "MiniGrid-MultiRoom-N2-S4-v0",
        "MiniGrid-MultiRoom-N4-S5-v0",
    ]
    locked_room = [
        # "MiniGrid-LockedRoom-v0",
    ]
    all_envs = [empty, four_rooms, unlock, door_key, go_to_door, fetch, unlock_pickup,
                red_blue_door, go_to_object, multi_room, locked_room]
    flat_envs = [j for i in all_envs for j in i]
    # flat_envs = [i[0] for i in all_envs if len(i) > 0]
    return {
        id_: {
            "defaults": "merge",
            "env_kwargs": {
                "id": id_,
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
        }
        for id_ in flat_envs}


if __name__ == '__main__':
    print(len(get_minigrid_experiment()))
