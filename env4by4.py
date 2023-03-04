import gym

def Env4by4():
    # Import frozen lake model from OpenAI gym and adjust the reward
    env = gym.make('FrozenLake-v1', map_name='4x4', is_slippery=False)
    reward_table = env.env.P
    # At state 1, go down
    reward_table[1][1]= [(1.0, 5, -1.0, True)]
    # At state 3, go down
    reward_table[3][1]= [(1.0, 7, -1.0, True)]
    # At state 4, go right
    reward_table[4][2]= [(1.0, 5, -1.0, True)]
    # At state 6, go left or right
    reward_table[6][0] = [(1.0, 5, -1.0, True)]
    reward_table[6][2] = [(1.0, 7, -1.0, True)]
    # At state 8, go down
    reward_table[8][1] = [(1.0, 12, -1.0, True)]
    # At state 9, go up
    reward_table[9][3] = [(1.0, 5, -1.0, True)]
    # At state 10, go right
    reward_table[10][2]= [(1.0, 11, -1.0, True)]
    # AT state 13, go left
    reward_table[13][0] = [(1.0, 12, -1.0, True)]
    # At state 5, go any 
    reward_table[5] = {0: [(1.0, 5, -1.0, True)], 1: [(1.0, 5, -1.0, True)], 2: [(1.0, 5, -1.0, True)], 3: [(1.0, 5, -1.0, True)]}
    # At state 7, go any 
    reward_table[7] = {0: [(1.0, 7, -1.0, True)], 1: [(1.0, 7, -1.0, True)], 2: [(1.0, 7, -1.0, True)], 3: [(1.0, 7, -1.0, True)]}
    # At state 11, go any
    reward_table[11] = {0: [(1.0, 11, -1.0, True)], 1: [(1.0, 11, -1.0, True)], 2: [(1.0, 11, -1.0, True)], 3: [(1.0, 11, -1.0, True)]}
    # At state 12, go any
    reward_table[12] = {0: [(1.0, 12, -1.0, True)], 1: [(1.0, 12, -1.0, True)], 2: [(1.0, 12, -1.0, True)], 3: [(1.0, 12, -1.0, True)]}
    env.env.P = reward_table

    return env