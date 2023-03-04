import numpy as np
from env4by4 import adjustEnv

class SARA:
    """
    Init the SARSA class with input 
    """
    def __init__(self, num_episode=1000, gamma=0.95, epsilon=0.1, learning_rate=0.1):
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
        self.action_total = []
        self.first_shortest_episode = []
        self.training_enough = False
    
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
    Epislon greedy policy for choosing the action
    """     
    def epsilon_greedy_policy(self, state):
        if np.random.uniform(0, 1) < self.epsilon:
            action = self.env.action_space.sample()
        else:
            action = np.argmax(self.Q_table[state])
        return action

    """
    Run function for iterating assigned number of episodes by using max Q value as the Q prime
    """ 
    def run(self):
        # Initialize two tables
        self.init_table()
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
            # Update this terminated to decide whether this episode ends 
            terminated = False
            # Loop for each step of episode
            while not terminated:
                # Add the first action by epsilon greedy policy to the action list
                action_list.append(action)
                # Take action, receive reward and observe the next state
                next_state, reward, terminated, _ = self.env.step(action)
                # Choose next action from next state using policy derived from Q (epsilon-greedy)
                next_action = self.epsilon_greedy_policy(next_state)
                # Here is the difference between SARSA & Q-learning
                # SARSA use the Q value of next state's next action as the Q prime
                Q_prime = self.Q_table[next_state][next_action]
                # Update the Q table with this Q prime
                self.Q_table[state][action] += self.learning_rate * (reward + self.gamma * Q_prime - self.Q_table[state][action])
                # Update the state
                state = next_state
                action = next_action
                # Check whether it reaches the goal state
                if (next_state == self.n_states-1):
                    success_episode_index.append((epo+1))
                    self.action_total.append(action_list)
                    #print("Successful in No.", str(epo+1),"episode")

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
    def render_policy_table(self):
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

    """
    Render the first shortest episode in all successful episodes
    """
    def render_first_shortest_episode(self):
        if not self.training_enough:
            print("Training is not enough, reder first shortest fails")
            return
        self.first_shortest_episode = min(self.action_total, key=len)
        print("First shortest path with",len(self.first_shortest_episode),"steps in total")
        self.env.reset()
        self.env.render()
        for each_step in self.first_shortest_episode:
            self.env.step(each_step)
            self.env.render()
        return

if __name__ == '__main__': 
    m = SARA(num_episode=100, gamma=0.95, epsilon=0.1, learning_rate=0.1)
    m.run()
    m.render_policy_table()
    m.render_first_shortest_episode()