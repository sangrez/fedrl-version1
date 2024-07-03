import matplotlib.pyplot as plt
import gym
import dqn_td3_v2
from gym import spaces
import numpy as np
import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import my_env
import user_info
import copy
from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter('runs/federated_learning_experiment')


# Seed setting
seed_value = 0
T.manual_seed(seed_value)
if T.cuda.is_available():
    T.cuda.manual_seed(seed_value)
    T.cuda.manual_seed_all(seed_value)

T.backends.cudnn.deterministic = True
T.backends.cudnn.benchmark = False

users = 3
servers = 3

# Initialize environments
env1 = my_env.Offloading(users, servers)
state_dim1 = env1.observation_space.shape[0]
discrete_action_dim1 = env1.discrete_action_space.n
continuous_action_dim1 = env1.continuous_action_space.shape[0]
max_action1 = float(env1.continuous_action_space.high[0])

env2 = my_env.Offloading(users, servers)
state_dim2 = env2.observation_space.shape[0]
discrete_action_dim2 = env2.discrete_action_space.n
continuous_action_dim2 = env2.continuous_action_space.shape[0]
max_action2 = float(env2.continuous_action_space.high[0])


agent1 = dqn_td3_v2.JointAgent(state_dim1, discrete_action_dim1, continuous_action_dim1, max_action1)
agent2 = dqn_td3_v2.JointAgent(state_dim2, discrete_action_dim2, continuous_action_dim2, max_action2)


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
        local_reward.append(episode_reward)
        writer.add_scalar('Local_Reward/Agent', episode_reward, episode)
    return episode_reward

def federated_averaging(global_agent, agents):
    local_weights = [agent.get_weights() for agent in agents]
    averaged_weights = {
        'dqn': copy.deepcopy(local_weights[0]['dqn']),
        'dqn_target': copy.deepcopy(local_weights[0]['dqn_target']),
        'actor': copy.deepcopy(local_weights[0]['actor']),
        'critic': copy.deepcopy(local_weights[0]['critic']),
        'critic_target': copy.deepcopy(local_weights[0]['critic_target'])
    }
    for key in averaged_weights['dqn'].keys():
       
        for i in range(1, len(local_weights)):
            averaged_weights['dqn'][key] += local_weights[i]['dqn'][key]
        averaged_weights['dqn'][key] /= len(local_weights)
    for key in averaged_weights['dqn_target'].keys():
        
        for i in range(1, len(local_weights)):
            averaged_weights['dqn_target'][key] += local_weights[i]['dqn_target'][key]
        averaged_weights['dqn_target'][key] /= len(local_weights)
    for key in averaged_weights['actor'].keys():
        # print(f"Averaging Actor key: {key}")  
        for i in range(1, len(local_weights)):
            averaged_weights['actor'][key] += local_weights[i]['actor'][key]
        averaged_weights['actor'][key] /= len(local_weights)
    
    for key in averaged_weights['critic'].keys():
         
        for i in range(1, len(local_weights)):
            averaged_weights['critic'][key] += local_weights[i]['critic'][key]
        averaged_weights['critic'][key] /= len(local_weights)
    
    for key in averaged_weights['critic_target'].keys():
        
        for i in range(1, len(local_weights)):
            averaged_weights['critic_target'][key] += local_weights[i]['critic_target'][key]
        averaged_weights['critic_target'][key] /= len(local_weights)

    global_agent.set_weights(averaged_weights)


global_agent = dqn_td3_v2.JointAgent(state_dim1, discrete_action_dim1, continuous_action_dim1, max_action1)
rewards = []
local_reward = []
local_reward_agnet1 = []
local_reward_agnet2 = []

for i, agent in enumerate([agent1, agent2]):
    weights = agent.get_weights()




for round in range(100):   

    print(f"Federated Learning Round {round + 1}")

    reward1 = train_local(agent1, env1, episodes=10)
    reward2 = train_local(agent2, env2, episodes=10)

    rewards.append((reward1, reward2))

    writer.add_scalar('Reward/Agent1', reward1, round)
    writer.add_scalar('Reward/Agent2', reward2, round)

    federated_averaging(global_agent, [agent1, agent2])

    global_weights = global_agent.get_weights()
    agent1.set_weights(global_weights)
    agent2.set_weights(global_weights)
    print(f"Round {round + 1}: Agent 1 Reward: {reward1}, Agent 2 Reward: {reward2}")

writer.close()

# tensorboard --logdir=runs

for i, (reward1, reward2) in enumerate(rewards):
    print(f"Round {i + 1}: Agent 1 Reward: {reward1}, Agent 2 Reward: {reward2}")


federated_averaging(global_agent, [agent1, agent2])
global_weights = global_agent.get_weights()
agent1.set_weights(global_weights)
agent2.set_weights(global_weights)


# rounds = list(range(1, 51))
# agent1_rewards = [reward[0] for reward in rewards]
# agent2_rewards = [reward[1] for reward in rewards]

# plt.figure(figsize=(10, 5))
# plt.plot(rounds, agent1_rewards, label='Agent 1')
# plt.plot(rounds, agent2_rewards, label='Agent 2')
# plt.xlabel('Federated Learning Rounds')
# plt.ylabel('Rewards')
# plt.title('Rewards for both agents during Federated Learning')
# plt.legend()
# plt.grid(True)
# plt.show()
