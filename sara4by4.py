from env4by4 import adjustEnv
class MCES:
     def __init__(self, num_episode, gamma, epsilon, learning_rate):
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