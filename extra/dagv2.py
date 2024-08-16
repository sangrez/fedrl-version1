import networkx as nx
import numpy as np
import random
import matplotlib.pyplot as plt
import os
import pickle

def ensure_in_and_out_degrees(G, task_num):
    # Ensure in-degree for all nodes except start node (0)
    for node in range(1, task_num + 1):
        if G.in_degree(node) == 0:
            predecessor = random.randint(0, node - 1)
            G.add_edge(predecessor, node)

    # Ensure out-degree for all nodes except end node (task_num + 1)
    for node in range(1, task_num + 1):
        if G.out_degree(node) == 0:
            successor = random.randint(node + 1, task_num + 1)
            G.add_edge(node, successor)

def create_task(task_num, data_size_min, data_size_max, workload_min, workload_max, dag_type='mixed'):
    print(f"Generating {dag_type} DAG with {task_num} tasks")  # Debugging print
    G = nx.DiGraph(name=dag_type)
    
    # Create start and end nodes
    G.add_node(0, data_size=0, computing_circle=0)  # Start node
    G.add_node(task_num + 1, data_size=0, computing_circle=0)  # End node

    # Create other nodes
    for i in range(1, task_num + 1):
        t_data_size = np.random.uniform(data_size_min, data_size_max) * 1024
        t_computing_circle = np.random.uniform(workload_min, workload_max)
        G.add_node(i, data_size=t_data_size, computing_circle=t_computing_circle)

    # Create edges based on DAG type
    if dag_type == 'linear':
        create_linear_dag(G, task_num)
    elif dag_type == 'branching':
        create_random_branching_dag(G, task_num)
    elif dag_type == 'mixed':
        create_random_mixed_dag(G, task_num)
    elif dag_type == 'grid':
        create_random_grid_dag(G, task_num)
    elif dag_type == 'fork-join':
        create_random_fork_join_dag(G, task_num)
    elif dag_type == 'star':
        create_random_star_dag(G, task_num)
    elif dag_type == 'tree':
        create_random_tree_dag(G, task_num)
    elif dag_type == 'cycle-free-mesh':
        create_random_cycle_free_mesh_dag(G, task_num)

    # Ensure all nodes meet the in-degree and out-degree conditions
    ensure_in_and_out_degrees(G, task_num)

    print(f"Finished generating {dag_type} DAG")  # Debugging print
    return G

# Example DAG creation functions
def create_linear_dag(G, task_num):
    for i in range(task_num + 1):
        G.add_edge(i, i + 1)

def create_random_branching_dag(G, task_num):
    G.add_edge(0, 1)
    for i in range(1, task_num):
        G.add_edge(i, i + 1)
        if random.random() < 0.5:
            target = random.randint(i + 1, task_num)
            G.add_edge(i, target)

def create_random_mixed_dag(G, task_num):
    G.add_edge(0, 1)
    for i in range(1, task_num):
        G.add_edge(i, i + 1)
    for i in range(1, task_num - 1):
        if random.random() < 0.5:
            target = random.randint(i + 2, task_num)
            G.add_edge(i, target)

def create_random_grid_dag(G, task_num):
    width = random.randint(2, int(task_num**0.5) + 2)
    for i in range(task_num):
        if i + 1 < task_num and (i + 1) % width != 0:
            G.add_edge(i + 1, i + 2)
        if i + width < task_num:
            G.add_edge(i + 1, i + width + 1)
    G.add_edge(0, 1)
    G.add_edge(task_num, task_num + 1)

def create_random_fork_join_dag(G, task_num):
    fork_point = random.randint(2, task_num - 2)
    join_point = random.randint(fork_point + 1, task_num)
    G.add_edge(0, fork_point)
    for i in range(1, fork_point):
        G.add_edge(fork_point, i)
    for i in range(fork_point, join_point):
        G.add_edge(i, join_point)
    G.add_edge(join_point, task_num + 1)

def create_random_star_dag(G, task_num):
    central_node = random.randint(1, task_num)
    for i in range(1, task_num + 1):
        if i != central_node:
            G.add_edge(0, i)
            G.add_edge(i, task_num + 1)

def create_random_tree_dag(G, task_num):
    current_level = [1]
    next_level = []
    node = 2
    iteration_count = 0  # Safeguard against infinite loops
    while node <= task_num:
        iteration_count += 1
        if iteration_count > task_num * 2:  # Break condition for safety
            print("Breaking out of tree DAG generation loop to prevent infinite loop")
            break
        
        for parent in current_level:
            if random.random() < 0.7:
                G.add_edge(parent, node)
                next_level.append(node)
                node += 1
                print(f"Added edge from {parent} to {node}, node count: {node}")
                if node > task_num:
                    break
        current_level = next_level
        next_level = []
    G.add_edge(0, 1)

def create_random_cycle_free_mesh_dag(G, task_num):
    for i in range(1, task_num):
        for j in range(i + 1, min(task_num + 1, i + random.randint(2, 4))):
            G.add_edge(i, j)
    G.add_edge(0, 1)
    G.add_edge(task_num, task_num + 1)

def visualize_dag(G, filename):
    pos = nx.spring_layout(G)
    nx.draw(G, pos, with_labels=True, node_color='lightblue', node_size=500, font_size=10, font_weight='bold')
    plt.title(f"{G.graph['name']} DAG with {len(G.nodes()) - 2} tasks")
    plt.savefig(filename)
    plt.close()

def create_and_split_dag_datasets(num_dags, task_num, data_size_range, workload_range, dag_types, train_ratio=0.8):
    print(f"Starting to generate {num_dags} DAGs of each type")  # Debugging print
    all_datasets = {dag_type: [] for dag_type in dag_types}
    train_datasets = {dag_type: [] for dag_type in dag_types}
    test_dataset = []
    
    # Create datasets for each type
    for dag_type in dag_types:
        print(f"Generating {num_dags} DAGs for type: {dag_type}")  # Debugging print
        for _ in range(num_dags):
            G = create_task(
                task_num=task_num,
                data_size_min=data_size_range[0],
                data_size_max=data_size_range[1],
                workload_min=workload_range[0],
                workload_max=workload_range[1],
                dag_type=dag_type
            )
            all_datasets[dag_type].append(G)
        print(f"Finished generating {num_dags} DAGs for type: {dag_type}")  # Debugging print
    
    # Split into train and test
    print("Splitting datasets into train and test sets")  # Debugging print
    for dag_type, dags in all_datasets.items():
        random.shuffle(dags)
        split_index = int(len(dags) * train_ratio)
        train_datasets[dag_type] = dags[:split_index]
        test_dataset.extend(dags[split_index:])
    
    # Shuffle the test dataset to mix DAG types
    random.shuffle(test_dataset)
    
    print("Finished splitting datasets")  # Debugging print
    return train_datasets, test_dataset

# Example usage
if __name__ == '__main__':
    # Configuration
    NUM_DAGS = 5000  # Reduced number for testing
    TASK_NUM = 6  # Number of tasks between start and end
    DATA_SIZE_RANGE = (25, 50)  # KB
    WORKLOAD_RANGE = (100, 500)  # Million instructions
    DAG_TYPES = ['linear', 'branching', 'mixed', 'grid', 'star', 'tree', 'cycle-free-mesh']
    TRAIN_RATIO = 0.8  # 80% for training, 20% for testing

    # Create and split datasets
    train_datasets, test_dataset = create_and_split_dag_datasets(NUM_DAGS, TASK_NUM, DATA_SIZE_RANGE, WORKLOAD_RANGE, DAG_TYPES, TRAIN_RATIO)

    # Create folders
    for folder in ['DAGs', 'data_list/train', 'data_list/test']:
        os.makedirs(folder, exist_ok=True)

    # Save datasets and visualize some examples
    for dag_type in DAG_TYPES:
        # Save train dataset for each type
        with open(f'data_list/train/task_list_{dag_type}.pickle', 'wb') as f:
            pickle.dump(train_datasets[dag_type], f)
        
        # Visualize a few examples from training set
        for i in range(5):  # Visualize 2 examples of each type
            visualize_dag(train_datasets[dag_type][i], f"DAGs/train_task_graph_{dag_type}_{i}.png")

    # Save combined test dataset
    with open(f'data_list/test/task_list_combined.pickle', 'wb') as f:
        pickle.dump(test_dataset, f)
    
    # Visualize a few examples from test set
    for i in range(5):  # Visualize 2 examples from test set
        visualize_dag(test_dataset[i], f"DAGs/test_task_graph_combined_{i}.png")

    print(f"Created and split {NUM_DAGS} DAGs for each of the {len(DAG_TYPES)} types")
    print(f"Train-Test split ratio: {TRAIN_RATIO:.0%}-{1-TRAIN_RATIO:.0%}")
    print(f"Each training set has {int(NUM_DAGS * TRAIN_RATIO)} DAGs")
    print(f"Combined test set has {len(test_dataset)} DAGs")
    print("Training datasets saved in 'data_list/train' folder")
    print("Combined test dataset saved in 'data_list/test' folder")
    print("Example visualizations saved in 'DAGs' folder")
