from mcwoes10by10 import MCWOES10BY10
from sarsa10by10 import SARSA10BY10
from qlearning10by10 import QLEARNING10BY10
import matplotlib.pyplot as plt
import numpy as np
GAMMA = 0.95
EPSILON = 0.1
ALPHA = 0.1
TOTAL_NUM_EPO = 10000
STEP = 100

def run_episode_with_method(class_name1, class_name2, class_name3):
    episode_list = []
    avg_reward_list_method1 = []
    avg_reward_list_method2 = []
    avg_reward_list_method3 = []
    for i in range(STEP, TOTAL_NUM_EPO+1, STEP):
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

def draw_avg_vs_num():
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
    plt.title("10x10 Average Reward VS Num of Episode with "+Param)
    plt.xlabel('Number of Episode')
    plt.ylabel('Average Reward')
    # Show the plot
    plt.show()

def draw_total_success_comparison():
    episode_list = []
    success_rate_list1 = []
    success_rate_list2 = []
    success_rate_list3 = []
    for i in range(STEP, TOTAL_NUM_EPO+1, STEP):
        model1 = MCWOES10BY10(num_episode=i, gamma=GAMMA, epsilon=EPSILON)
        model1.run()
        model2 = SARSA10BY10(num_episode=i, gamma=GAMMA, epsilon=EPSILON, learning_rate=ALPHA)
        model2.run()
        model3 = QLEARNING10BY10(num_episode=i, gamma=GAMMA, epsilon=EPSILON, learning_rate=ALPHA)
        model3.run()
        episode_list.append(i)
        success_rate_list1.append(len(model1.success_episode_index)/i)
        success_rate_list2.append(len(model2.success_episode_index)/i)
        success_rate_list3.append(len(model3.success_episode_index)/i)

    # Create a plot to show the relation between number of episode and average reward
    plt.plot(episode_list, success_rate_list1, label='mc_wo_es')
    plt.plot(episode_list, success_rate_list2, label='sarsa')
    plt.plot(episode_list, success_rate_list3, label='qlearning')
    plt.grid(True)
    plt.legend()
    Param = "gamma=" + str(GAMMA) + ",epsilon=" + str(EPSILON) + ",alpha="+ str(ALPHA)
    plt.title("10x10 Num of Episode VS Successful Rate"+Param)
    plt.xlabel('Number of Episode ')
    plt.ylabel('Successful Rate')
    plt.show()

draw_avg_vs_num()
#draw_total_success_comparison()

