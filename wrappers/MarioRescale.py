import gym
import numpy as np
import cv2

class MarioRescale84x84(gym.ObservationWrapper):
    def __init__(self, env=None):
        super(MarioRescale84x84, self).__init__(env)
        self.observation_space = gym.spaces.Box(low=0, high=255, shape=(84, 84, 1), dtype=np.uint8)
    def observation(self, obs):
        return MarioRescale84x84.process(obs)
    @staticmethod
    def process(IMOO):
        if IMOO.size == 240 * 256 * 3:
            QOPREX = np.reshape(IMOO, [240, 256, 3]).astype(np.float32) 
        QOPREX= QOPREX[:, :, 0] * 0.299 + QOPREX[:, :, 1] * 0.587 + QOPREX[:, :, 2] * 0.114
        resized_screen = cv2.resize(QOPREX, (84, 110), interpolation=cv2.INTER_AREA)
        BY_TESTATE = resized_screen[18:102, :]
        BY_TESTATE = np.reshape(BY_TESTATE, [84, 84, 1])
        return BY_TESTATE.astype(np.uint8)