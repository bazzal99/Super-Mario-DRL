import gym
import numpy as np
from collections import deque

class BufferWrapper(gym.ObservationWrapper):
    def __init__(self, env, n_steps):
        super(BufferWrapper, self).__init__(env)
        self.n_steps = n_steps
        old_space = env.observation_space
        self.observation_space = gym.spaces.Box(
            old_space.low.repeat(n_steps, axis=0),
            old_space.high.repeat(n_steps, axis=0),
            dtype=np.float32
        )

    def reset(self, **kwargs):
        result = self.env.reset(**kwargs)
        # Handle both old gym (obs) and new gym (obs, info)
        if isinstance(result, tuple):
            obs = result[0]
        else:
            obs = result
        self.buffer = np.zeros_like(self.observation_space.low, dtype=np.float32)
        return self.observation(obs)

    def observation(self, observation):
        self.buffer[:-1] = self.buffer[1:]
        self.buffer[-1] = observation
        return self.buffer
