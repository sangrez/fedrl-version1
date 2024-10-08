import pandas as pd
import matplotlib.pyplot as plt

# Define DAG types and file paths
dag_types = ['linear', 'branching', 'mixed', 'grid', 'star', 'tree']

# Load the reward data from the files
train_rewards = {}
val_rewards = {}
val_file = pd.read_csv('text_data/global_rewards_test.txt', header=None)  # Load validation rewards
for dag_type in dag_types:
    train_file = f'text_data/rewards_{dag_type}_train.txt'
    
    
    train_rewards[dag_type] = pd.read_csv(train_file, header=None)
    # if dag_type == 'linear':  # Load validation rewards only once
    #     val_rewards = pd.read_csv(val_file, header=None)

# Calculate the moving average
window_size_train = 300
window_size_val = 1

train_rewards_ma = {dag_type: rewards.rolling(window=window_size_train).mean() 
                    for dag_type, rewards in train_rewards.items()}
# val_rewards_ma = val_rewards.rolling(window=window_size_val).mean()
# Plot training rewards
plt.figure(figsize=(14, 7))
agents = ['Agent 1', 'Agent 2', 'Agent 3', 'Agent 4', 'Agent 5', 'Agent 6']

for (dag_type, rewards_ma), agent in zip(train_rewards_ma.items(), agents):
    plt.plot(rewards_ma, label=f'{agent} - Moving Average', linewidth=2)

plt.xlabel('Episode')
plt.ylabel('Reward')
plt.title('Moving Average of Training Rewards for All DAG Types')
plt.legend()
# plt.grid(True)
plt.savefig('results/training_rewards.png')
plt.close()

# Plot validation rewards
plt.figure(figsize=(14, 7))
plt.plot(val_file, label='Global Test - Moving Average', linestyle='--', linewidth=2)
plt.xlabel('Episode')
plt.ylabel('Reward')
plt.title('Moving Average of Global Test Rewards')
plt.legend()
plt.grid(True)
plt.savefig('results/global_test_rewards.png')
plt.close()

# Optional: Plot both training and validation rewards on the same graph
# plt.figure(figsize=(14, 7))
# for dag_type, rewards_ma in train_rewards_ma.items():
#     plt.plot(rewards_ma, label=f'{dag_type} Train - Moving Average')
# # plt.plot(val_file, label='Global Test - Moving Average', linestyle='--', linewidth=2)

# plt.xlabel('Episode')
# plt.ylabel('Reward')
# plt.title('Moving Average of Rewards for All Agents')
# plt.legend()
# plt.grid(True)
# plt.savefig('results/combined_rewards.png')
# plt.close()

# print("Plots have been saved as 'training_rewards.png', 'global_test_rewards.png', and 'combined_rewards.png'")