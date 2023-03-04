import numpy as np
from env4by4 import adjustEnv
class SARA:
     def __init__(self, num_episode, gamma, epsilon, learning_rate):
          self.env = adjustEnv()
          self.num_row = 4
          self.num_colomn = 4
          self.n_states = self.env.observation_space.n
          self.n_actions = self.env.action_space.n
          self.num_episode = num_episode
          self.gamma = gamma
          self.epsilon = epsilon
          self.learning_rate = learning_rate
          self.P_table = {}
          self.Q_table = {}
          self.action_map={0:"left", 1:"down", 2:"right", 3:"up"}
          self.shortest_route=[]
          self.epo_result = False
          self.training_enough = False
     
     def init_table(self):
          for state in range(self.n_states):
               self.P_table[state] = [1/self.n_actions] * self.n_actions
               self.Q_table[state] = [0] * self.n_actions
     
     def get_optimal_action(self, state):
          q_values = self.Q_table[state]
          optimal_action = np.argmax(q_values)
          return optimal_action

     def epsilon_greedy_behavior(self, state):
          if np.random.uniform(0, 1) < self.epsilon:
               action = self.env.action_space.sample()
          else:
               action = self.get_optimal_action(state)
          return action
          
     def update_ptable_by_epsilon_greedy(self, prime_action, state):
          for action in range(self.n_actions):
               if action==prime_action:
                    self.P_table[state][action] = 1 - self.epsilon + self.epsilon / self.n_actions
               else:
                    self.P_table[state][action] = self.epsilon / self.n_actions     
     
     def run(self):
          self.init_table()
          success_episode_index=[]
          action_total=[]
          self.training_enough=False   
          for epo in range(self.num_episode):
               self.epo_result = False
               state = self.env.reset()
               action = self.env.action_space.sample()
               action_list = []
               action_list.append(action)
               terminated = False
               while not terminated:
                    next_state, reward, terminated, _ = self.env.step(action)
                    next_action = self.env.action_space.sample()
                    Q_prime = self.Q_table[next_state][next_action]
                    self.Q_table[state][action] += self.learning_rate * (reward + self.gamma * Q_prime - self.Q_table[state][action])
                    prime_action = self.get_optimal_action(state)
                    self.update_ptable_by_epsilon_greedy(prime_action, state)
                    state = next_state
                    action = next_action
                    action_list.append(next_action)
                    
                    if (state == self.n_states-1):
                         self.epo_result = True
                         success_episode_index.append((epo+1))
                         #print("Successful in No.", str(epo+1),"episode")
               action_total.append(action_list)
          if(len(success_episode_index)==0):
               self.training_enough=False
               print("Trainig is not enough, no successful episode, please give a larger num_episode")
          else:
               self.training_enough=True
               print("Total successful episode count", len(success_episode_index), "in", self.num_episode, "episodes")
               print("First success episode No.", min(success_episode_index))                   
          return self.P_table
     
     def render_policy_table(self):
          if not self.training_enough:
               return
          optimal_policy=[]     
          for state in range(self.n_states):
               optimal_policy.append(self.get_optimal_action(state))
          
          directions = ["left ", "down ", "right", "up   "]
          policy_table = ""
          for state in range(self.n_states):
               action = optimal_policy[state]
               if self.env.desc.flatten()[state] == b'H':
                    policy_table += "Hole   "
               elif self.env.desc.flatten()[state] == b'G':
                    policy_table += "Goal   "
               else:
                    policy_table += directions[action] + "  "
               if (state+1) % self.num_colomn == 0:
                    policy_table += '\n'
          
          print("Optimal policy table after training: ")
          print(policy_table)
     
if __name__ == '__main__': 
     m = SARA(num_episode=10000, gamma=0.95, epsilon=0.1, learning_rate=0.1)
     m.run()
     m.render_policy_table()