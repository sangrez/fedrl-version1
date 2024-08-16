import pandas as pd
import matplotlib.pyplot as plt

# Load the reward data from the files
file_path_1 = 'text_data/global_rewards_test.txt'
rewards_1 = pd.read_csv(file_path_1, header=None)

# Remove brackets and convert to numeric
rewards_1[0] = rewards_1[0].str.strip('[]').astype(float)

# Debugging: Print the first few rows and data types of the DataFrame
print(rewards_1.head())
print(rewards_1.dtypes)

# Calculate the moving average with a window size of 10 for both reward sets
window_size = 1
window_size_val = 1
rewards_1_ma = rewards_1.rolling(window=window_size).mean()

# Plot the moving average of rewards for both sets
plt.figure(figsize=(14, 7))
plt.plot(rewards_1_ma, label='Agent 1 - Moving Average')
plt.xlabel('Episode')
plt.ylabel('Reward')
plt.title('Moving Average of Rewards')
plt.legend()
plt.grid(True)
plt.savefig('rewards.png')
plt.show()