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
import os
import glob

from torch.utils.tensorboard import SummaryWriter
text_data_folder = "./text_data"
files = glob.glob(os.path.join(text_data_folder, "*.txt"))

writer = SummaryWriter('runs/federated_learning_experiment')

# Seed setting
seed_value = 0
T.manual_seed(seed_value)
if T.cuda.is_available():
    T.cuda.manual_seed(seed_value)
    T.cuda.manual_seed_all(seed_value)

T.backends.cudnn.deterministic = True
T.backends.cudnn.benchmark = False

users = 5
servers = 3
federated_rounds = 200
local_episodes = 100
local_episodes_no_fed = 20000
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
    episode_rewards = []
    for episode in range(episodes):
        if episode == 5587:
            print("Episode 5567")
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
        episode_rewards.append(episode_reward)
        # writer.add_scalar('Local_Reward/Agent', episode_reward, episode)
        # print(f"Episode {episode} finished with reward: {episode_reward}")
    return episode_rewards

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
local_rewards_agent1 = []
local_rewards_agent2 = []

reward1_simple_local = train_local(agent1, env1, episodes=local_episodes_no_fed)
reward2_simple_local = train_local(agent2, env2, episodes=local_episodes_no_fed)

if not os.path.isdir(text_data_folder):
    os.makedirs(text_data_folder)

file_path = os.path.join(text_data_folder, "reward_local_1.txt")
file_path = os.path.join(text_data_folder, "reward_local_2.txt")
with open('reward_local_1.txt', 'w') as file:
    for reward in reward1_simple_local:
        file.write(f"{reward}\n")

with open('reward_local_2.txt', 'w') as file:
    for reward in reward2_simple_local:
        file.write(f"{reward}\n")

for round in range(federated_rounds):   
    print(f"Federated Learning Round {round + 1}")

    reward1 = train_local(agent1, env1, episodes=local_episodes)
    reward2 = train_local(agent2, env2, episodes=local_episodes)

    # local_rewards_agent1.extend(reward1)
    # local_rewards_agent2.extend(reward2)
    rewards.append((reward1[-1], reward2[-1]))

    writer.add_scalar('Reward/Agent1', (reward1[-1]), round)
    writer.add_scalar('Reward/Agent2', (reward2[-1]), round)

    federated_averaging(global_agent, [agent1, agent2])

    global_weights = global_agent.get_weights()
    agent1.set_weights(global_weights)
    agent2.set_weights(global_weights)
    print(f"Round {round + 1}: Agent 1 Mean Reward: {reward1[-1]}, Agent 2 Mean Reward: {reward2[-1]}")

writer.close()

# tensorboard --logdir=runs

federated_averaging(global_agent, [agent1, agent2])
global_weights = global_agent.get_weights()
agent1.set_weights(global_weights)
agent2.set_weights(global_weights)
# Assuming rewards is a list of tuples

# Save rewards to a text file
with open('rewards.txt', 'w') as file:
    for reward_pair in rewards:
        file.write(f"{reward_pair[0]}, {reward_pair[1]}\n")

# Plotting rewards for both agents during local learning
plt.figure(figsize=(10, 5))
plt.plot(reward1_simple_local, label='Agent 1')
plt.plot(reward2_simple_local, label='Agent 2')
plt.xlabel('local agent episodes')
plt.ylabel('rewards')
plt.title('local agent rewards for both agents without federated learning')
plt.legend()
plt.grid(True)
plt.savefig('local_learning_rewards.png')

# Plotting rewards for both agents during local learning
# plt.figure(figsize=(10, 5))
# plt.plot(local_rewards_agent1, label='Agent 1')
# plt.plot(local_rewards_agent2, label='Agent 2')
# plt.xlabel('local agent episodes')
# plt.ylabel('rewards')
# plt.title('local agent rewards for both agents during local learning')
# plt.legend()
# plt.grid(True)

# Plotting rewards for both agents during federated learning
rounds = list(range(1, federated_rounds + 1))
agent1_rewards = [reward[0] for reward in rewards]
agent2_rewards = [reward[1] for reward in rewards]

plt.figure(figsize=(10, 5))
plt.plot(rounds, agent1_rewards, label='Agent 1')
plt.plot(rounds, agent2_rewards, label='Agent 2')
plt.xlabel('Federated Learning Rounds')
plt.ylabel('Rewards')
plt.title('Rewards for both agents during Federated Learning')
plt.legend()
plt.grid(True)
plt.savefig('federated_learning_rewards.png')
plt.show()
