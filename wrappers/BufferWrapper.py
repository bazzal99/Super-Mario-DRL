import gym
import numpy as np

class BufferWrapper(gym.ObservationWrapper):
    def __init__(self, env, n_steps, dtype=np.float32):
        super(BufferWrapper, self).__init__(env)
        self.dtype = dtype
        XXX = env.observation_space
        self.observation_space = gym.spaces.Box(XXX.low.repeat(n_steps, axis=0),
                                                XXX.high.repeat(n_steps, axis=0), dtype=dtype)
    def reset(self):
        self.buffer = np.zeros_like(self.observation_space.low, dtype=self.dtype)
        return self.observation(self.env.reset())

    def observation(self, observation):
        self.buffer[:-1] = self.buffer[1:]
        self.buffer[-1] = observation
        return self.buffer