import pandas as pd
import matplotlib.pyplot as plt

# Define DAG types and file paths
# dag_types = ['linear', 'branching', 'mixed', 'grid', 'star', 'tree']
dag_types = ['linear', 'branching', 'mixed']
# Load the reward data from the files
train_rewards = {}
train_rewards_ind = {}

for dag_type in dag_types:
    train_file = f'text_data/rewards_train_federated_{dag_type}_train.txt'
    train_rewards[dag_type] = pd.read_csv(train_file, header=None)

for dag_type in dag_types:
    train_file_ind = f'text_data/rewards_train_independent_{dag_type}_train.txt'
    train_rewards_ind[dag_type] = pd.read_csv(train_file_ind, header=None)
# Calculate the moving average
window_size_train = 300
window_size_val = 1

train_rewards_ma = {dag_type: rewards.rolling(window=window_size_train).mean() 
                    for dag_type, rewards in train_rewards.items()}

train_rewards_ind_ma = {dag_type: rewards.rolling(window=window_size_train).mean() 
                    for dag_type, rewards in train_rewards_ind.items()}

plt.figure(figsize=(14, 7))
agents = ['Agent 1', 'Agent 2', 'Agent 3', 'Agent 4', 'Agent 5', 'Agent 6']

for (dag_type, rewards_ma), agent in zip(train_rewards_ma.items(), agents):
    plt.plot(rewards_ma, label=f'{agent} - Moving Average', linewidth=2)

plt.xlabel('Episode')
plt.ylabel('Reward (FedAvg)')
plt.title('Moving Average of Training Rewards for All DAG Types')
plt.legend()
plt.grid(True)
plt.savefig('results/training_rewards.png')
plt.close()

plt.figure(figsize=(14, 7))
agents = ['Agent 1', 'Agent 2', 'Agent 3', 'Agent 4', 'Agent 5', 'Agent 6']

for (dag_type, rewards_ma_ind), agent in zip(train_rewards_ind_ma.items(), agents):
    plt.plot(rewards_ma_ind, label=f'{agent} - Moving Average', linewidth=2)

plt.xlabel('Episode')
plt.ylabel('Reward')
plt.title('Moving Average of Independent Training Rewards for All DAG Types')
plt.legend()
plt.grid(True)
plt.savefig('results/training_rewards_ind.png')
plt.close()