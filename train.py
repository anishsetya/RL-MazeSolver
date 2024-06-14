import numpy as np
from stable_baselines3 import DQN
from env import MazeSolver
import os

def train():
    maze = np.load('maze.npy')
    target = (9, 9)
    env = MazeSolver(maze, target)
    model = DQN('MlpPolicy', env, verbose=1)
    model.learn(total_timesteps=1000000)
    path=os.path.join('Training', 'Saved Models', 'MAZE_DQN')

if __name__ == "__main__":
    train()
