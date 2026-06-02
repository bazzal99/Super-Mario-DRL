"""
DQN and DDQN Training/Testing Script for Super Mario Bros
==========================================================
Usage:
    Train DDQN:  python dqn_ddqn.py  (default: double_dqn=True)
    Train DQN:   change double_dqn=False in the __main__ block below
    Test:        uncomment the test() call in the __main__ block
"""

import sys
import os

# Allow imports from the wrappers directory
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'wrappers'))

import torch
import torch.nn as nn
import random
from nes_py.wrappers import JoypadSpace
import gym_super_mario_bros
from tqdm import tqdm
import pickle
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
import gym
import numpy as np
import cv2
import matplotlib.pyplot as plt
from PIL import Image
import time
from IPython import display
import pylab as pl

from MaxAndSkipEnv import MaxAndSkipEnv
from MarioRescale import MarioRescale84x84
from ImageToPyTorch import ImageToPyTorch, PixelNormalization
from BufferWrapper import BufferWrapper
from DQNSolver import DQNSolver, DQNAgent


def create_mario_env(env):
    """Apply preprocessing wrappers to the raw Mario environment."""
    env = MaxAndSkipEnv(env)
    env = MarioRescale84x84(env)
    env = ImageToPyTorch(env)
    env = BufferWrapper(env, 4)
    env = PixelNormalization(env)
    return JoypadSpace(env, SIMPLE_MOVEMENT)


def train(pretrained, double_dqn, num_episodes=1000):
    """
    Train a DQN or DDQN agent on SuperMarioBros-1-1-v0.

    Args:
        pretrained (bool): Resume from saved checkpoint if True.
        double_dqn (bool): Use Double DQN if True, plain DQN if False.
        num_episodes (int): Number of training episodes.
    """
    path = "DDQNResults" if double_dqn else "DQNResults"
    os.makedirs(path, exist_ok=True)

    env = gym_super_mario_bros.make('SuperMarioBros-1-1-v0')
    env = create_mario_env(env)
    observation_space = env.observation_space.shape
    action_space = env.action_space.n

    agent = DQNAgent(
        state_space=observation_space,
        action_space=action_space,
        max_memory_size=30000,
        batch_size=32,
        gamma=0.90,
        lr=0.00025,
        dropout=0.2,
        exploration_max=1.0,
        exploration_min=0.02,
        exploration_decay=0.99,
        double_dqn=double_dqn,
        pretrained=pretrained,
        path=path
    )

    total_rewards = []
    total_steps = []
    total_loss = []
    total_time = []

    if pretrained:
        with open(os.path.join(path, "total_rewards.pkl"), 'rb') as f:
            total_rewards = pickle.load(f)
        with open(os.path.join(path, "total_loss.pkl"), "rb") as f:
            total_loss = pickle.load(f)
        with open(os.path.join(path, "total_time.pkl"), "rb") as f:
            total_time = pickle.load(f)

    env.reset()

    for ep_num in tqdm(range(num_episodes)):
        state = env.reset()
        state = torch.Tensor([state])
        total_reward = 0
        steps = 0
        ep_loss = []
        start_time = time.time()

        while True:
            action = agent.act(state)
            steps += 1
            state_next, reward, terminal, info = env.step(int(action[0]))
            total_reward += reward
            state_next = torch.Tensor([state_next])
            reward = torch.tensor([reward]).unsqueeze(0)
            terminal = torch.tensor([int(terminal)]).unsqueeze(0)
            agent.remember(state, action, reward, state_next, terminal)
            loss = agent.experience_replay()
            if loss is not None:
                ep_loss.append(loss.detach().cpu().numpy())
            state = state_next
            if terminal:
                end_time = time.time()
                break

        total_rewards.append(total_reward)
        total_steps.append(steps)
        total_time.append(end_time - start_time)
        total_loss.append(np.mean(ep_loss) if ep_loss else 0.0)

        if ep_num != 0 and ep_num % 100 == 0:
            print("Episode {} | Score: {:.1f} | Avg Score: {:.1f}".format(
                ep_num + 1, total_rewards[-1], np.mean(total_rewards)))

    print("Training complete. Final avg score: {:.1f}".format(np.mean(total_rewards)))

    # Save training logs
    with open(os.path.join(path, "ending_position.pkl"), "wb") as f:
        pickle.dump(agent.ending_position, f)
    with open(os.path.join(path, "num_in_queue.pkl"), "wb") as f:
        pickle.dump(agent.num_in_queue, f)
    with open(os.path.join(path, "total_rewards.pkl"), "wb") as f:
        pickle.dump(total_rewards, f)
    with open(os.path.join(path, "total_steps.pkl"), "wb") as f:
        pickle.dump(total_steps, f)
    with open(os.path.join(path, "total_loss.pkl"), "wb") as f:
        pickle.dump(total_loss, f)
    with open(os.path.join(path, "total_time.pkl"), "wb") as f:
        pickle.dump(total_time, f)

    # Save model weights
    if agent.double_dqn:
        torch.save(agent.local_net.state_dict(), os.path.join(path, "DQN1.pt"))
        torch.save(agent.target_net.state_dict(), os.path.join(path, "DQN2.pt"))
    else:
        torch.save(agent.dqn.state_dict(), os.path.join(path, "DQN.pt"))

    torch.save(agent.STATE_MEM,  os.path.join(path, "STATE_MEM.pt"))
    torch.save(agent.ACTION_MEM, os.path.join(path, "ACTION_MEM.pt"))
    torch.save(agent.REWARD_MEM, os.path.join(path, "REWARD_MEM.pt"))
    torch.save(agent.STATE2_MEM, os.path.join(path, "STATE2_MEM.pt"))
    torch.save(agent.DONE_MEM,   os.path.join(path, "DONE_MEM.pt"))

    env.close()


def test(double_dqn, num_episodes=4, exploration_max=0.05):
    """
    Test a trained agent and save a gameplay GIF.

    Args:
        double_dqn (bool): Load DDQN weights if True, DQN if False.
        num_episodes (int): Number of test episodes to run.
        exploration_max (float): Epsilon for testing (small = mostly greedy).
    """
    path = "DDQNResults" if double_dqn else "DQNResults"

    env = gym_super_mario_bros.make('SuperMarioBros-1-1-v0')
    env = create_mario_env(env)
    observation_space = env.observation_space.shape
    action_space = env.action_space.n

    agent = DQNAgent(
        state_space=observation_space,
        action_space=action_space,
        max_memory_size=30000,
        batch_size=32,
        gamma=0.90,
        lr=0.00025,
        dropout=0.2,
        exploration_max=exploration_max,
        exploration_min=exploration_max,
        exploration_decay=1.0,
        double_dqn=double_dqn,
        pretrained=True,
        path=path
    )

    total_rewards = []
    frames = []

    for ep_num in tqdm(range(num_episodes)):
        state = env.reset()
        state = torch.Tensor([state])
        total_reward = 0
        steps = 0

        while True:
            screen = env.render(mode='rgb_array')
            frames.append(Image.fromarray(screen))
            action = agent.act(state)
            steps += 1
            state_next, reward, terminal, info = env.step(int(action[0]))
            total_reward += reward
            state_next = torch.Tensor([state_next])
            state = state_next
            if terminal:
                break

        total_rewards.append(total_reward)
        print("Episode {} | Score: {:.1f}".format(ep_num + 1, total_reward))

    print("Test complete. Avg score: {:.1f}".format(np.mean(total_rewards)))

    # Save gameplay as GIF
    gif_path = "super_mario_test.gif"
    frames[0].save(gif_path, save_all=True, append_images=frames[1:], loop=0, duration=30)
    print("Gameplay saved to:", gif_path)

    env.close()


if __name__ == "__main__":
    # ── TRAIN ──────────────────────────────────────────────────
    # Set pretrained=False for a fresh run, True to resume from checkpoint.
    # Set double_dqn=True for DDQN, False for DQN.
    train(pretrained=False, double_dqn=True, num_episodes=1000)

    # ── TEST ───────────────────────────────────────────────────
    # Uncomment to test a trained agent (generates super_mario_test.gif)
    # test(double_dqn=True, num_episodes=4, exploration_max=0.05)
