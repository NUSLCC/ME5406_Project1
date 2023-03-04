import numpy as np
from env4by4 import adjustEnv

def create_x_by_y_table(x,y):
     output_list = []
     for i in range(x):
          row = [0] * y
          output_list.append(row)
     return output_list

class MCES:
     def __init__(self, num_episode, gamma, epsilon):
          self.env = adjustEnv()
          self.num_row = 4
          self.num_colomn = 4
          self.n_states = self.env.observation_space.n
          self.n_actions = self.env.action_space.n
          self.num_episode = num_episode
          self.gamma = gamma
          self.epsilon = epsilon
          self.P_table = {}
          self.Q_table = {}
          self.R_table = {}
          self.action_map={0:"left", 1:"down", 2:"right", 3:"up"}
          self.shortest_route=[]
          self.training_enough = False

     def init_table(self):
          for state in range(self.n_states):
               self.P_table[state] = [1/self.n_actions] * self.n_actions
               self.Q_table[state] = [0] * self.n_actions
               self.R_table[state] = {}
               for action in range(self.n_actions):
                    self.R_table[state][action] = []

     def get_prime_action_by_Q(self, state):
          q_value = self.Q_table[state]
          prime_action = q_value.index(max(q_value))
          return prime_action

     def epsilon_greedy_policy(self, prime_action, state):
          for action in range(self.n_actions):
               if action==prime_action:
                    self.P_table[state][action] = 1 - self.epsilon + self.epsilon / self.n_actions
               else:
                    self.P_table[state][action] = self.epsilon / self.n_actions
         
     def generate_episode(self):
          state_list =   []
          action_list =  []
          reward_list =  []
          return_list =  []
          result = False

          # Set the starting position each time
          state = self.env.reset()
          terminated = False
          # Generate the episode until the status is terminated
          while not terminated:
               # Choose a random action
               action = self.env.action_space.sample()
               # Store the state, action, and reward in the episode
               state_list.append(state)
               action_list.append(action)
               
               # Take the action and observe the next state and reward
               state, reward, terminated, _ = self.env.step(action)
               reward_list.append(reward)
               if (state == self.n_states-1):
                    result = True
          G = 0
          for r in reversed(reward_list):
               G = self.gamma * G + r
               return_list.append(G)
          return_list.reverse()
          return state_list, action_list, return_list, result
     
     def run(self):
          self.init_table()
          success_episode=[]
          success_episode_index=[]       
          for epo in range(self.num_episode):                                   
               state_list, action_list, return_list, result = self.generate_episode()
               if (result == True):
                    #print("Successful in No.", str(epo+1),"episode")
                    success_episode.append(action_list)
                    success_episode_index.append((epo+1))
                    #print("Total step: ", len(action_list))
               
               V_table = create_x_by_y_table(self.n_states, self.n_actions)
               for i in range(len(state_list)):                                      
                    state, action = state_list[i], action_list[i]
                    if V_table[state][action]==0:                        
                         V_table[state][action]=1        
                         self.R_table[state][action].append(return_list[i])
                         Q = np.mean(self.R_table[state][action])                           
                         self.Q_table[state][action] = Q
                         prime_action = self.get_prime_action_by_Q(state)               
                         self.epsilon_greedy_policy(prime_action,state)
          if (len(success_episode)==0):
               print("Trainig is not enough, no successful episode, please give a larger num_episode")
          else:
               self.training_enough = True
               print("Total Successful episode count", len(success_episode), "in", self.num_episode, "episodes")
               print("First success episode No.", min(success_episode_index))
               self.shortest_route = min(success_episode, key=len)
               print("shortest route with", len(self.shortest_route),"steps",[self.action_map[a] for a in self.shortest_route])
     
     def render_action_step(self):
          if not self.training_enough:
               return
          print("Start showing the shortest path:")
          self.env.reset()
          self.env.render()
          for each_step in self.shortest_route:
               self.env.step(each_step)
               self.env.render()
          return

     def render_policy_table(self):
          if not self.training_enough:
               return
          policy = []     
          for each_state in range(self.n_states):
               policy.append(self.P_table[each_state].index(max(self.P_table[each_state])))
          
          directions = ["left ", "down ", "right", "up   "]
          policy_table = ""
          for state in range(self.n_states):
               action = policy[state]
               if self.env.desc.flatten()[state] == b'H':
                    policy_table += "Hole   "
               elif self.env.desc.flatten()[state] == b'G':
                    policy_table += "Goal   "
               else:
                    policy_table += directions[action] + "  "
               if (state+1) % self.num_colomn == 0:
                    policy_table += '\n'
          
          print("Policy table after training: ")
          print(policy_table)
          

if __name__ == '__main__': 
     m = MCES(num_episode=1000, gamma=0.95, epsilon=0.1)
     m.run()
     m.render_action_step()
     m.render_policy_table()
