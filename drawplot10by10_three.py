from mcwoes10by10 import MCWOES10BY10
from sarsa10by10 import SARSA10BY10
from qlearning10by10 import QLEARNING10BY10
import matplotlib.pyplot as plt
import numpy as np
GAMMA = 0.95
EPSILON = 0.1
ALPHA = 0.1
TOTAL_NUM_EPO = 1001
STEP = 50

def run_episode_with_method(class_name1, class_name2, class_name3):
    episode_list = []
    avg_reward_list_method1 = []
    avg_reward_list_method2 = []
    avg_reward_list_method3 = []
    for i in range(1, TOTAL_NUM_EPO, STEP):
        model1 = class_name1(num_episode=i, gamma=GAMMA, epsilon=EPSILON)
        model1.run()
        model2 = class_name2(num_episode=i, gamma=GAMMA, epsilon=EPSILON, learning_rate=ALPHA)
        model2.run()
        model3 = class_name3(num_episode=i, gamma=GAMMA, epsilon=EPSILON, learning_rate=ALPHA)
        model3.run()
        episode_list.append(i)
        avg_reward_list_method1.append(model1.average_reward)
        avg_reward_list_method2.append(model2.average_reward)
        avg_reward_list_method3.append(model3.average_reward)

    return episode_list, avg_reward_list_method1, avg_reward_list_method2, avg_reward_list_method3

# Run different method
episode_list, avg_reward_list_method1, avg_reward_list_method2, avg_reward_list_method3 = run_episode_with_method(MCWOES10BY10, SARSA10BY10, QLEARNING10BY10)

# Create a plot to show the relation between number of episode and average reward
plt.plot(episode_list, avg_reward_list_method1, label='mc_wo_es')
plt.plot(episode_list, avg_reward_list_method2, label='sarsa')
plt.plot(episode_list, avg_reward_list_method3, label='qlearning')
plt.grid(True)
plt.legend()
# Set the plot title and axis labels
Param = "gamma=" + str(GAMMA) + ",epsilon=" + str(EPSILON) + ",alpha="+ str(ALPHA)
plt.title("10x10 Avg Reward VS Num of Episode with "+Param)
plt.xlabel('Number of Episode')
plt.ylabel('Average Reward')
# Show the plot
plt.show()

