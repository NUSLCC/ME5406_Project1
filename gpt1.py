import numpy as np
import gym

class SARA:
    def __init__(self, num_episode, gamma, epsilon, learning_rate):
        self.env = env = gym.make('FrozenLake-v1', map_name='8x8', is_slippery=False)
        self.num_row = 4
        self.num_colomn = 4
        self.n_states = self.env.observation_space.n
        self.n_actions = self.env.action_space.n
        self.num_episode = num_episode
        self.gamma = gamma
        self.epsilon = epsilon
        self.learning_rate = learning_rate
        self.P_table = {}
        self.Q_table = np.zeros((self.n_states, self.n_actions))
        self.action_map={0:"left", 1:"down", 2:"right", 3:"up"}
        self.shortest_route=[]
        self.epo_result = False
     
    def init_table(self):
        for state in range(self.n_states):
            self.P_table[state] = [1/self.n_actions] * self.n_actions
            self.Q_table[state] = [0] * self.n_actions
     
    def epsilon_greedy_behavior(self, state):
        if np.random.uniform(0, 1) < self.epsilon:
            action = self.env.action_space.sample()
        else:
            action = np.argmax(self.Q_table[state, :])
        return action
     
    def run(self):
        self.init_table()
        success_episode=[]
        action_total=[]   
        for epo in range(self.num_episode):
            self.epo_result = False
            state = self.env.reset()
            action = self.epsilon_greedy_behavior(state)
            action_list = []
            action_list.append(action)
            terminated = False
            while not terminated:
                next_state, reward, terminated, _ = self.env.step(action)
                next_action = self.epsilon_greedy_behavior(next_state)
                #Q_prime = self.Q_table[next_state][next_action]
                self.Q_table[state, action] = self.Q_table[state, action] + self.learning_rate * (reward + self.gamma * np.max(self.Q_table[next_state, :]) - self.Q_table[state, action])
                #self.Q_table[state, action] = self.Q_table[state, action] + self.learning_rate * (reward + self.gamma * self.Q_table[next_state, next_action] - self.Q_table[state, action])
                #self.Q_table[state][action] += self.learning_rate * (reward + self.gamma * Q_prime - self.Q_table[state][action])
                state = next_state
                action = next_action
                action_list.append(next_action)
                if (state == self.n_states-1):
                        self.epo_result = True
                        success_episode.append((epo+1))
                        print("Successful in No.", str(epo+1),"episode")
            action_total.append(action_list)
        if(len(success_episode)==0):
            print("Trainig is not enough, no successful episode, please give a larger num_episode")
        else:
            print("Total success episode count", len(success_episode))
            print("First success episode No.", min(success_episode))                   

if __name__ == '__main__': 
    m = SARA(num_episode=10000, gamma=0.95, epsilon=0.1, learning_rate=0.1)
    m.run()
    policy = np.argmax(m.Q_table, axis=1)
    print(policy)