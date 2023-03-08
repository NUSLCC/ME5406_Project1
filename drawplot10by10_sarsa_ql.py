from sarsa10by10 import SARSA10BY10
from qlearning10by10 import QLEARNING10BY10
import matplotlib.pyplot as plt
import numpy as np
GAMMA = 0.95
EPSILON = 0.1
TOTAL_NUM_EPO = 10000

def draw_alpha_vs_total_success_comparison():
    alpha_list = []
    success_rate_list1 = []
    success_rate_list2 = []
    for alpha in np.arange(0, 1.05, 0.05).tolist():
        model1 = SARSA10BY10(num_episode=TOTAL_NUM_EPO, gamma=GAMMA, epsilon=EPSILON, learning_rate=alpha)
        model1.run()
        model2 = QLEARNING10BY10(num_episode=TOTAL_NUM_EPO, gamma=GAMMA, epsilon=EPSILON, learning_rate=alpha)
        model2.run()
        alpha_list.append(alpha)
        success_rate_list1.append(len(model1.success_episode_index)/TOTAL_NUM_EPO)
        success_rate_list2.append(len(model2.success_episode_index)/TOTAL_NUM_EPO)

    # Create a plot to show the relation between number of episode and average reward
    plt.plot(alpha_list, success_rate_list1, label='sarsa')
    plt.plot(alpha_list, success_rate_list2, label='qlearning')
    plt.grid(True)
    plt.legend()
    Param = str(TOTAL_NUM_EPO) + "episodes" + ",gamma=" + str(GAMMA) + ",epsilon=" + str(EPSILON)
    plt.title("10x10 Learning Rate VS Successful Rate with "+Param)
    plt.xlabel('Learning rate')
    plt.ylabel('Successful Rate')
    plt.show()


draw_alpha_vs_total_success_comparison()