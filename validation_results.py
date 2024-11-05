import numpy as np
import matplotlib.pyplot as plt
import csv

# Define file paths and lists for storing average values
file_paths = [
    'text_data/branching_rewards.txt',
    'text_data/cycle-free-mesh_rewards.txt',
    'text_data/grid_rewards.txt',
    'text_data/linear_rewards.txt',
    'text_data/mixed_rewards.txt',
    'text_data/star_rewards.txt',
    'text_data/tree_rewards.txt',
]

# Lists to store rewards for each model type
branching_avg_values = []
cycle_free_mesh_avg_values = []
grid_avg_values = []
linear_avg_values = []
mixed_avg_values = []
star_avg_values = []
tree_avg_values = []

# Map file paths to their corresponding lists
file_to_list = {
    file_paths[0]: branching_avg_values,
    file_paths[1]: cycle_free_mesh_avg_values,
    file_paths[2]: grid_avg_values,
    file_paths[3]: linear_avg_values,
    file_paths[4]: mixed_avg_values,
    file_paths[5]: star_avg_values,
    file_paths[6]: tree_avg_values,
}

# Read the data from the CSV files and populate the lists
for file_path in file_paths:
    with open(file_path, 'r') as file:
        reader = csv.reader(file)
        next(reader)  # Skip the header (Episode,Reward)
        for row in reader:
            try:
                # Convert the second column (reward) to float and append to the corresponding list
                file_to_list[file_path].append(float(row[1]))
            except ValueError:
                print(f"Could not convert line to float: {row}")

# Create the x-axis values (assuming the same number of episodes for each model)
x = np.arange(1, len(linear_avg_values) + 1)

# Set the bar width and offsets for each category
bar_width = 0.1
offsets = np.array([-3, -2, -1, 0, 1, 2, 3]) * bar_width

# Create the bar plot
plt.figure(figsize=(12, 8))

plt.bar(x + offsets[0], branching_avg_values, width=bar_width, label='Branching')
plt.bar(x + offsets[1], cycle_free_mesh_avg_values, width=bar_width, label='Cycle-Free-Mesh')
plt.bar(x + offsets[2], grid_avg_values, width=bar_width, label='Grid')
plt.bar(x + offsets[3], linear_avg_values, width=bar_width, label='Linear')
plt.bar(x + offsets[4], mixed_avg_values, width=bar_width, label='Mixed')
plt.bar(x + offsets[5], star_avg_values, width=bar_width, label='Star')
plt.bar(x + offsets[6], tree_avg_values, width=bar_width, label='Tree')

# Add labels, title, and legend
plt.xlabel('Episode Sets')
plt.ylabel('Average Reward')
plt.title('Average Rewards for 100 Episodes Across Different Models')
plt.legend()

# Show the plot
plt.tight_layout()
plt.show()
