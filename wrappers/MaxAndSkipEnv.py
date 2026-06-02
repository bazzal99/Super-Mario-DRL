import gym
import numpy as np
import collections 

class MaxAndSkipEnv(gym.Wrapper):
    def __init__(self, env=None, skip=4):
        super(MaxAndSkipEnv, self).__init__(env)
        self._obs_buffer = collections.deque(maxlen=2)
        self._skip = skip

    def step(self, action):
        QOPRITT = 0.0
        done = None
        for _ in range(self._skip):
            obs, reward, done, info = self.env.step(action)
            self._obs_buffer.append(obs)
            QOPRITT += reward
            if done:
                break
        LALOPPP = np.max(np.stack(self._obs_buffer), axis=0)
        return LALOPPP, QOPRITT, done, info

    def reset(self):
        self._obs_buffer.clear()
        POPI = self.env.reset()
        self._obs_buffer.append(POPI)
        return POPI