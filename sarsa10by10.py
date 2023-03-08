import numpy as np
from env10by10 import Env10by10

class SARSA10BY10:
    """
    Init the SARSA class with input 
    """
    def __init__(self, num_episode=1000, gamma=0.95, epsilon=0.1, learning_rate=0.1):
        # Use the adjusted 10 by 10 environment of frozen lake
        self.env = Env10by10()
        self.num_row = 10
        self.num_colomn = 10
        # Set alias of the number of states and actions
        self.n_states = self.env.observation_space.n
        self.n_actions = self.env.action_space.n
        # Set the number of episode
        self.num_episode = num_episode
        # Set the gamma (discount)
        self.gamma = gamma
        # Set the epsilon (probability)
        self.epsilon = epsilon
        # Set the alpha
        self.learning_rate = learning_rate
        # Create empty dictionary for policy and Q value
        self.P_table = {}
        self.Q_table = {}
        self.action_total = []
        # In this environment, directions are a bit different
        self.action_map={0:"left", 1:"down", 2:"right", 3:"up"}
        # Empty list to store first shortest episode
        self.first_shortest_episode = []
        # Use this flag to check the training is enough or not
        self.training_enough = False
        # Average reward for this round
        self.average_reward = 0 

    """
    Init two tables: 
    policy table with average probability for each action at each state
    Q table with all zero for each action at each state
    """
    def init_table(self):
        for state in range(self.n_states):
            self.P_table[state] = [1/self.n_actions] * self.n_actions
            self.Q_table[state] = [np.random.rand() * 0.01] * self.n_actions

    """
    Epislon greedy policy for choosing the prime action
    """     
    def epsilon_greedy_policy(self, state):
        # With probability less than epsilon, a random action will be selected
        if np.random.uniform(0, 1) < self.epsilon:
            action = self.env.action_space.sample()
        # Otherwise, choose the action with the highest Q value at this state
        else:
            action = np.argmax(self.Q_table[state])
        return action
    
    """
    Update the policy table with epsilon greedy policy
    """
    def update_policy_table(self, state):
        # Get prime action with the max Q value at specific state
        prime_action = np.argmax(self.Q_table[state])
        # Set probability (epsilon/num_of_actions) to each action first
        policy = np.ones(self.n_actions, dtype=float) * self.epsilon / self.n_actions
        # Update the prime action's probability to 1-epsilon+epsilon/num_of_actions
        policy[prime_action] = 1 - self.epsilon + policy[prime_action]
        # Update the policy table with probabilty of each action at current state
        self.P_table[state] = policy
    
    """
    Run function for iterating assigned number of episodes by using same policy
    """ 
    def run(self):
        # Initialize two tables
        self.init_table()
        # Create a reward list to collect reward from each episode
        total_reward_list = []        
        # Record all successful episodes index
        success_episode_index=[]
        # Set the flag of enough training for outputing the first successful episode
        self.training_enough=False
        # Loop for each episode   
        for epo in range(self.num_episode):
            # Initialize state to start position
            state = self.env.reset()
            # Choose action from state using policy derived from Q (epsilon-greedy)
            action = self.epsilon_greedy_policy(state)
            # Create the action list for each episode
            action_list = []
            # Collect the total reward in each episode
            reward_total = 0
            # Update this terminated to decide whether this episode ends 
            terminated = False
            # Loop for each step of episode
            while not terminated:
                # Add the first action by epsilon greedy policy to the action list
                action_list.append(action)
                # Take action, receive reward and observe the next state
                next_state, reward, terminated, _ = self.env.step(action)
                # Collect reward in total for each episode
                reward_total += reward
                # Choose next action from next state using policy derived from Q (epsilon-greedy)
                next_action = self.epsilon_greedy_policy(next_state)
                # Here is the difference between SARSA & Q-learning
                # SARSA use the Q value of next state's next action as the Q prime
                Q_prime = self.Q_table[next_state][next_action]
                # Update the Q table with this Q prime
                self.Q_table[state][action] += self.learning_rate * (reward + self.gamma * Q_prime - self.Q_table[state][action])
                # Update the P table
                self.update_policy_table(state)
                # Update the state
                state = next_state
                action = next_action
                # Check whether it reaches the goal state
                if (next_state == self.n_states-1):
                    success_episode_index.append((epo+1))
                    self.action_total.append(action_list)
                    #print("Successful in No.", str(epo+1),"episode")
            # Put reward sum of each episode into the total reward list
            total_reward_list.append(reward_total)
        # Calculate the average reward for assigned number of episodes
        self.average_reward = np.average(total_reward_list)
        print("Average reward is",self.average_reward, "for total", self.num_episode, "episodes", )
        # Check whether training is enough by checking the length of successful episode index
        if(len(success_episode_index)==0):
            self.training_enough=False
            print("Training is not enough, no successful episode, please give a larger num_episode")
        else:
            self.training_enough=True
            print("Total successful episode count", len(success_episode_index), "in", self.num_episode, "episodes")
            print("First success episode No.", min(success_episode_index))                   

    """
    Render the policy by a long string with four lines
    """ 
    def render_optimal_policy(self):
        if not self.training_enough:
            print("Training is not enough, reder policy table fails")
            return
        # Get optimal policy by collecting actions with highest Q value at each state
        optimal_policy=[]     
        for state in range(self.n_states):
            optimal_policy.append(np.argmax(self.Q_table[state]))
        # Set the verbal indicator for action from 0 to 3
        directions = ["left ", "down ", "right", "up   "]
        policy_table = ""
        for state in range(self.n_states):
            action = optimal_policy[state]
            # Change character H to Hole
            if self.env.desc.flatten()[state] == b'H':
                policy_table += "Hole   "
            # Change character G to Goal
            elif self.env.desc.flatten()[state] == b'G':
                policy_table += "Goal   "
            else:
                policy_table += directions[action] + "  "
            if (state+1) % self.num_colomn == 0:
                policy_table += '\n'
        
        print("Optimal policy table after training: ")
        print(policy_table)

    """
    Render the first shortest episode in all successful episodes
    """
    def render_first_shortest_episode(self):
        if not self.training_enough:
            print("Training is not enough, reder first shortest fails")
            return
        # min() will return the first shortest episode with its action list
        self.first_shortest_episode = min(self.action_total, key=len)
        print("First shortest path with",len(self.first_shortest_episode),
              "steps in total:", [self.action_map[a] for a in self.first_shortest_episode])
        # Reset the agent to the start pose
        self.env.reset()
        self.env.render()
        # render this episode by default rendering function
        for each_step in self.first_shortest_episode:
            self.env.step(each_step)
            self.env.render()
        return

if __name__ == '__main__': 
    m = SARSA10BY10(num_episode=1000, gamma=0.95, epsilon=0.1, learning_rate=0.1)
    m.run()
    #m.render_optimal_policy()
    #m.render_first_shortest_episode()