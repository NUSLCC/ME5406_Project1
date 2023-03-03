import gym

env = gym.make('FrozenLake-v1')
policy = [3, 0, 3, 0, 3, 0, 1, 0, 2, 1, 1, 0, 0, 2, 2, 0]
print(env.nrow)
def render_policy_table(policy):
    directions = ["left ", "down ", "right", "up   "]
    policy_table = ""
    for state in range(16):
        action = policy[state]
        if env.desc.flatten()[state] == b'H':
            policy_table += "Hole   "
        elif env.desc.flatten()[state] == b'G':
            policy_table += "Goal   "
        else:
            policy_table += directions[action] + "  "
        if (state+1) % 4 == 0:
            policy_table += '\n'
    print(policy_table)

print(render_policy_table(policy))
