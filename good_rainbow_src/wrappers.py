import gymnasium as gym


class SkipWrapper(gym.Wrapper, gym.utils.RecordConstructorArgs): # FIXME record args not importing
    """
        Generic common frame skipping wrapper
        Will perform action for `x` additional steps
    """
    def __init__(self, env, frame_skip):
        gym.utils.RecordConstructorArgs.__init__(
            self,
            frame_skip=frame_skip,
        )
        gym.Wrapper.__init__(self, env)
        assert frame_skip > 0
        self.frame_skip = frame_skip

    def step(self, action):
        total_reward, terminated, truncated, info = 0.0, False, False, {}

        for t in range(self.frame_skip):
            obs, reward, terminated, truncated, info = self.env.step(action)
            total_reward += reward
            if terminated or truncated:
                break
        return obs, total_reward, terminated, truncated, info

    def reset(self, **kwargs): # first frame have no history but why not not skip it # we could also tryn n=fe. 3 steps but they don't do it in atari
        state, reset_info = self.env.reset(**kwargs)

        return state, reset_info
