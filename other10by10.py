import numpy as np
import random
from frozen_lake10by10 import Env10by10

class FirstMonteCarlo:
    def __init__(self, num_episode, gamma, epsilon):
        self.env = Env10by10()
        self.num_row = 10
        self.num_colomn = 10
        self.n_states = self.env.observation_space.n
        self.n_actions = self.env.action_space.n
        self.num_episode = num_episode
        self.gamma = gamma
        self.epsilon = epsilon
        self.P_table = {}
        self.Q_table = {}
        self.R_table = {}
        self.action_map={0:"left", 1:"down", 2:"right", 3:"up"}
        self.first_shortest_episode=[]
        self.training_enough = False
    
    def init_table(self):
        for state in range(self.n_states):
            self.P_table[state] = [1/self.n_actions] * self.n_actions
            self.Q_table[state] = [random.uniform(0, 1) for a in range(self.n_actions)]
            self.R_table[state] = {}
            for action in range(self.n_actions):
                self.R_table[state][action] = []
    
    def gen_episode(self):                                                        
        done  = False
        state = self.env.reset()
        state_list, action_list, reward_list, return_list = [], [], [], []
        while not done:
            action = np.random.choice([0,1,2,3], p=self.P_table[state])
            state_list.append(state)
            action_list.append(action)
            state, reward, done = self.env.step(action)
            reward_list.append(reward)
        G = 0                                        # accumulative return
        for i in range(len(state_list)-1, -1, -1):   # trace back the episode
            G = self.gamma * G + reward_list[i]      # calculate the return by reward of each state
            return_list.append(G)
        return_list.reverse()                        # reverse the traced back list to positive order
        return state_list, action_list, return_list
    
    def create_n_by_n_list(self, x,y):
        my_list = []
        for i in range(x):
            row = [0] * y
            my_list.append(row)
        return my_list

    def run(self):      
        self.init_table()                   
        for episode in range(self.num_episode):                                   
            state_list, action_list, return_list = self.gen_episode()  
            return_table_temp = self.create_n_by_n_list(100, 4)                   
            for i in range(len(state_list)):                                       
                state, action = state_list[i], action_list[i]
                if return_table_temp[state[0]*4+state[1]][action]==0:              
                    return_table_temp[state[0]*4+state[1]][action]=1               
                    self.R_table[state][action].append(return_list[i])
                Q = np.mean(self.R_table[state][action])                           
                self.Q_table[state][action] = Q
                best_action = self.find_rand_max_idx(self.Q_table[state])              
                for a in range(self.n_actions):
                    if a == best_action:                                           
                        self.P_table[state][a] = 1 - self.epsilon + self.epsilon / self.n_actions
                    else:
                        self.P_table[state][a] = self.epsilon / self.n_actions      
        return self.P_table
    
