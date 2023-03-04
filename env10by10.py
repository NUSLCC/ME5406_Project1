import gym
import numpy as np
from gym.envs.toy_text.frozen_lake import generate_random_map

class FrozenLake10x10Env(gym.envs.toy_text.frozen_lake.FrozenLakeEnv):
    def __init__(self):
        desc = generate_random_map(size=10, p=0.75)
        desc = [[c if c != b'H' else -1 for c in row] for row in desc]
        super().__init__(desc=desc)

def Env10by10():
    while True:
        env = FrozenLake10x10Env()
        desc = env.desc.tolist()
        num_holes = np.sum([row.count(b'H') for row in desc])
        if (num_holes==25):
            return env

env = Env10by10()
env.render()