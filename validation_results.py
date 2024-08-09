import numpy as np
import matplotlib.pyplot as plt
file_paths = [
    'rewards_linear_val.txt',
    'rewards_branching_val.txt',
    'rewards_mixed_val.txt'
]

linear_avg_values = []
branching_avg_values = []
mixed_avg_values = []

file_to_list = {
    file_paths[0]: linear_avg_values,
    file_paths[1]: branching_avg_values,
    file_paths[2]: mixed_avg_values
}

for file_path in file_paths:
    with open(file_path, 'r') as file:
        for line in file:
            file_to_list[file_path].append(float(line.strip()))

# Create the plot
x = np.arange(1, len(linear_avg_values) + 1)
plt.figure(figsize=(10, 6))
plt.bar(x - 0.25, linear_avg_values, width=0.25, label='Linear')
plt.bar(x, branching_avg_values, width=0.25, label='Branching')
plt.bar(x + 0.25, mixed_avg_values, width=0.25, label='Mixed')

plt.xlabel('Episode Sets')
plt.ylabel('Average Reward')
plt.title('Average Rewards for 100 Episodes Across Different Models')
# plt.grid(True) 
plt.legend()
plt.show()
