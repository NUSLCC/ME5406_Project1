import numpy as np
import random
import time
from env4by4 import Env4by4

# Create a x by y 2D-list
def create_x_by_y_table(x,y):
     output_list = []
     for i in range(x):
          row = [0] * y
          output_list.append(row)
     return output_list

class MCWOES4BY4:
     """
     Init the MC control without ES class with small epsilon=0.1
     """
     def __init__(self, num_episode=1000, gamma=0.95, epsilon=0.1):
          self.env = Env4by4()
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
          self.first_shortest_episode=[]
          self.training_enough = False
          # Average reward for this round
          self.average_reward = 0 
          # Record all successful episodes index
          self.success_episode_index = []
          self.training_time = 0
    
     """
     Init three tables: 
     Policy table with average probability for each action at each state
     Q table with small arbitrary value for each state and action pair
     Return table an empty list for each state and action pair
     """
     def init_table(self):
          for state in range(self.n_states):
               # This behavior satisfies the epsilon soft policy since the probability of each action
               # under each state is 0.25, which is >= epsilon/num of actions when 0 < epsilon <= 1
               self.P_table[state] = [1/self.n_actions] * self.n_actions
               # Q(s,a) is arbitrary for all states and all actions, I choose a random number between 
               # 0 to 1 as the Q value for each state-action pair
               self.Q_table[state] = [random.uniform(0, 1) for a in range(self.n_actions)]
               # Return(s,a) is an empty list for all states and all actions
               self.R_table[state] = {}
               for action in range(self.n_actions):
                    self.R_table[state][action] = []

     """
     Epislon greedy policy for updating the policy table
     """ 
     def epsilon_greedy_policy(self, prime_action, state):
          for action in range(self.n_actions):
               if action==prime_action:
                    self.P_table[state][action] = 1 - self.epsilon + self.epsilon / self.n_actions
               else:
                    self.P_table[state][action] = self.epsilon / self.n_actions

     """
     Function to generate random episode with reversed accumulated G as return table 
     """    
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
          return state_list, action_list, reward_list, return_list, result
     
     """
     Run function of iterating assigned number of episodes to update the policy table
     """
     def run(self):
          start_time = time.time()
          # Initialize three tables: Policy, Q, Return
          self.init_table()
          # Create a reward list to collect reward from each episode
          total_reward_list = []
          # Record all successful episodes
          success_episode=[]
          # Reset and ready to record all successful episodes index
          self.success_episode_index.clear()
          # Loop for each episode       
          for epo in range(self.num_episode):                                   
               state_list, action_list, reward_list, return_list, result = self.generate_episode()
               # If this episode is successful
               if (result == True):
                    #print("Successful in No.", str(epo+1),"episode")
                    # Store all actions for successful episodes
                    success_episode.append(action_list)
                    # Store their index for getting the first one
                    self.success_episode_index.append((epo+1))
               # This creates a empty V_table to store whether agent has visited some state-action pair before
               V_table = create_x_by_y_table(self.n_states, self.n_actions)
               # Iterate each state in current episode
               for i in range(len(state_list)):
                    # Get state and action seperately from each list                                      
                    state, action = state_list[i], action_list[i]
                    # 0 means not visited, 1 means visited, updatet this visited table
                    if V_table[state][action]==0:                        
                         V_table[state][action]=1
                         # Append G of current state-action pair to the Return table R(s,a)
                         # G is calculated in generate_episode() and stored in return list
                         self.R_table[state][action].append(return_list[i])
                         # Update the Q value with average of Return(s,a)
                         Q = np.mean(self.R_table[state][action])
                         # Use this Q value to update Q table with current state-action pair                           
                         self.Q_table[state][action] = Q
                         # Get prime action with maximum Q value at this state
                         prime_action = np.argmax(self.Q_table[state])
                         # Update the policy table with this prime action and state              
                         self.epsilon_greedy_policy(prime_action,state)
               # Put reward sum of each episode into the total reward list
               total_reward_list.append(np.sum(reward_list))
          # Calculate the training time
          self.training_time = time.time()-start_time
          print("The training time is ", self.training_time, "s")          
          # Calculate the average reward for assigned number of episodes
          self.average_reward = np.average(total_reward_list)
          print("Average reward is",self.average_reward, "for total", self.num_episode, "episodes", )
          # Check whether training is enough by checking the length of successful episode index
          if (len(success_episode)==0):
               self.training_enough = False
               print("Trainig is not enough, no successful episode, please give a larger num_episode")
          else:
               self.training_enough = True
               print("Total successful episode count", len(success_episode), "in", self.num_episode, "episodes")
               print("First success episode No.", min(self.success_episode_index))
               self.first_shortest_episode = min(success_episode, key=len)
     
     """
     Render the first shortest path with embedded render function
     """
     def render_first_shortest_path(self):
          if not self.training_enough:
               return
          print("First shortest path with",len(self.first_shortest_episode),
                "steps in total:", [self.action_map[a] for a in self.first_shortest_episode])
          self.env.reset()
          self.env.render()
          for each_step in self.first_shortest_episode:
               self.env.step(each_step)
               self.env.render()
          return

     """
     Render the optimal policy table with left down right up words instead of 0 1 2 3, also skip Start Hole and Goal
     """
     def render_policy_table(self):
          if not self.training_enough:
               print("Training is not enough, reder policy table fails")
               return
          optimal_policy = []     
          for each_state in range(self.n_states):
               optimal_policy.append(self.P_table[each_state].index(max(self.P_table[each_state])))
          
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
     m = MCWOES4BY4(num_episode=1000, gamma=0.95, epsilon=0.1)
     m.run()
     m.render_policy_table()
     m.render_first_shortest_path()
