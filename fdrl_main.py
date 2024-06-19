import gym
import dqn_td3
import dqn_td31
from gym import spaces
import numpy as np
import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import my_env
import user_info
import copy
# Define environments for two scenarios
users = 2
servers = 3

# Scenario 1
env1 = my_env.Offloading(users, servers)
state_dim1 = env1.observation_space.shape[0]
discrete_action_dim1 = env1.discrete_action_space.n
continuous_action_dim1 = env1.continuous_action_space.shape[0]
max_action1 = float(env1.continuous_action_space.high[0])

# Scenario 2
env2 = my_env.Offloading(users, servers)
state_dim2 = env2.observation_space.shape[0]
discrete_action_dim2 = env2.discrete_action_space.n
continuous_action_dim2 = env2.continuous_action_space.shape[0]
max_action2 = float(env2.continuous_action_space.high[0])


# Initialize local agents
agent1 = dqn_td31.JointAgent(state_dim1, discrete_action_dim1, continuous_action_dim1, max_action1)
agent2 = dqn_td31.JointAgent(state_dim2, discrete_action_dim2, continuous_action_dim2, max_action2)

# Function to train local agents
def train_local(agent, env, episodes):
    for episode in range(episodes):
        state = env.reset()
        done = False
        episode_reward = 0
        t = 0
        while not done:
            if t == 0:
                discrete_action = 0
                continuous_action = np.ones(env.continuous_action_space.shape[0])
            else:
                discrete_action, continuous_action = agent.select_action(state)
            
            action_map = user_info.generate_offloading_combinations(servers + 1, users)
            actual_offloading_decisions = action_map[discrete_action]
            action = list(actual_offloading_decisions) + continuous_action.tolist()

            next_state, reward, done, _ = env.step(t, action, user_info.user_info(users))
            agent.add_transition(state, discrete_action, continuous_action, next_state, reward, done)
            state = np.array(next_state, dtype=np.float32)

            t += 1
            episode_reward += reward
            if len(agent.replay_buffer.storage) > 256:
                agent.train()
        
        print(f"Episode {episode} finished with reward: {episode_reward}")

# Function to perform federated averaging
def federated_averaging(global_agent, agents):
    local_weights = [agent.get_weights() for agent in agents]
    
    # Federated averaging
    averaged_weights = copy.deepcopy(local_weights[0])
    for key in averaged_weights[0].keys():
        for i in range(1, len(local_weights)):
            averaged_weights[0][key] += local_weights[i][0][key]
            averaged_weights[1][key] += local_weights[i][1][key]
            # averaged_weights[2][key] += local_weights[i][2][key]
        averaged_weights[0][key] /= len(local_weights)
        averaged_weights[1][key] /= len(local_weights)
        # averaged_weights[2][key] /= len(local_weights)
    
    global_agent.set_weights(averaged_weights)

# Initialize global agent
global_agent = dqn_td31.JointAgent(state_dim1, discrete_action_dim1, continuous_action_dim1, max_action1)

# Federated learning process
for round in range(10):  # Number of federated learning rounds
    print(f"Federated Learning Round {round + 1}")
    
    # Train local models
    train_local(agent1, env1, episodes=20)
    train_local(agent2, env2, episodes=20)

    
    # Federated averaging
    federated_averaging(global_agent, [agent1, agent2])
    
    # Distribute global model weights to local agents
    global_weights = global_agent.get_weights()
    agent1.set_weights(global_weights)
    agent2.set_weights(global_weights)

