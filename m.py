import numpy as np
from frozen_lake10by10 import Env10by10

def first_visit_monte_carlo_control(env, epsilon, num_episodes, gamma):
    # initialize Q(s, a) and Returns(s, a) as empty dictionaries
    Q = {}
    policy={}
    returns = {}
    for state in range(100):
        for action in range(4):
            Q[(state, action)] = 0.0
            returns[(state, action)] = []
    
    # initialize an epsilon-soft policy
    def epsilon_soft_policy(state):
        num_actions = env.action_space.n
        best_action = np.argmax([Q[(state, a)] for a in range(num_actions)])
        probs = np.ones(num_actions, dtype=float) * epsilon / num_actions
        probs[best_action] += (1.0 - epsilon)
        return probs
    
    # repeat forever (for each episode)
    for episode in range(num_episodes):
        # generate an episode following the current policy
        state = env.reset()
        episode_states = [state]
        episode_actions = []
        episode_rewards = []
        while True:
            probs = epsilon_soft_policy(state)
            action = np.random.choice(env.action_space.n, p=probs)
            next_state, reward, done, info = env.step(action)
            episode_actions.append(action)
            episode_rewards.append(reward)
            episode_states.append(next_state)
            if next_state==99:
                print("success")
            if done:
                break
            state = next_state
        
        # calculate returns and update Q-values
        G = 0
        for t in reversed(range(len(episode_states) - 1)):
            state = episode_states[t]
            action = episode_actions[t]
            reward = episode_rewards[t]
            G = gamma * G + reward
            if (state, action) not in [(episode_states[i], episode_actions[i]) for i in range(t)]:
                returns[(state, action)].append(G)
                Q[(state, action)] = np.mean(returns[(state, action)])
        
        # improve the policy based on updated Q-values
        for state in range(100):
            probs = epsilon_soft_policy(state)
            best_action = np.argmax([Q[(state, a)] for a in range(env.action_space.n)])
            for action in range(env.action_space.n):
                if action == best_action:
                    probs[action] = 1 - epsilon + (epsilon / env.action_space.n)
                else:
                    probs[action] = epsilon / env.action_space.n
            policy[state] = probs
    
    return Q, policy

env = Env10by10()
Q, P = first_visit_monte_carlo_control(env, epsilon=0.1, num_episodes=1000, gamma=0.95)