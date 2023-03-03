import gym
import numpy as np
env = gym.make('FrozenLake-v1', map_name='4x4')
env.reset()
output = env.render()

def monte_carlo(env, num_episodes, gamma=1.0):
    """
    Monte Carlo algorithm for estimating the state-value function.
    
    Args:
        env: OpenAI Gym environment.
        num_episodes: number of episodes to run.
        gamma: discount factor.
        
    Returns:
        V: dictionary containing the estimated value of each state.
    """
    # Initialize empty dictionaries to store returns and state values
    returns = {}
    V = {}
    
    # Loop over episodes
    for i in range(num_episodes):
        # Generate an episode
        episode = []
        state = env.reset()
        terminated = False
        while not terminated:
            # Choose a random action
            action = env.action_space.sample()
            # Take the action and observe the next state and reward
            next_state, reward, terminated, _ = env.step(action)
            # Store the state, action, and reward in the episode
            episode.append((state, action, reward))
            state = next_state
        print("episode is: ", episode)
        # Calculate the returns for each state in the episode
        G = 0
        states_visited = set()
        for t in range(len(episode)-1, -1, -1):
            state, action, reward = episode[t]
            G = reward + gamma * G
            if state not in states_visited:
                states_visited.add(state)
                if state not in returns:
                    returns[state] = []
                returns[state].append(G)
                # Calculate the average return for this state and update the value function
                V[state] = np.mean(returns[state])
        print("returns is: ", returns)
    return V

V = monte_carlo(env, num_episodes=1)
print(V)
