import time
import numpy as np
from stable_baselines3 import DQN
from env import MazeSolver
import os 

def solve():
    maze = np.load('maze.npy')
    target = (9, 9)
    env = MazeSolver(maze, target)
    path=os.path.join('Training', 'Saved Models', 'MAZE_DQN')
    model=DQN.load(path,env=env)

    obs, info = env.reset()
    done =False
    while not done:
        action, _states = model.predict(obs, deterministic=True)
        obs, rewards, done, truncated, info = env.step(action)
        env.render()
        time.sleep(0.1)
        if done:
            print("Maze solved!")
            break

if __name__ == "__main__":
    solve()
