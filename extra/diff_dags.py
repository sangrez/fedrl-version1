import networkx as nx
import numpy as np
import random
import matplotlib.pyplot as plt
import os
import pickle

def create_task(task_num, data_size_min, data_size_max, workload_min, workload_max, dag_type='mixed'):
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
        create_branching_dag(G, task_num)
    elif dag_type == 'mixed':
        create_mixed_dag(G, task_num)
    elif dag_type == 'grid':
        create_grid_dag(G, task_num)
    elif dag_type == 'fork-join':
        create_fork_join_dag(G, task_num)
    elif dag_type == 'star':
        create_star_dag(G, task_num)
    elif dag_type == 'tree':
        create_tree_dag(G, task_num)
    elif dag_type == 'cycle-free-mesh':
        create_cycle_free_mesh_dag(G, task_num)

    # Ensure all nodes are connected to end node
    for i in range(1, task_num + 1):
        if not any(G.successors(i)):
            G.add_edge(i, task_num + 1)

    return G

def create_linear_dag(G, task_num):
    for i in range(task_num + 1):
        G.add_edge(i, i+1)

def create_branching_dag(G, task_num):
    G.add_edge(0, 1,  )
    for i in range(1, task_num):
        G.add_edge(i, i+1)
        if i + 2 <= task_num:
            G.add_edge(i, i+2)

def create_mixed_dag(G, task_num):
    G.add_edge(0, 1)
    for i in range(1, task_num):
        G.add_edge(i, i+1)
    for i in range(1, task_num - 1):
        if random.random() < 0.3:
            target = random.randint(i+2, task_num)
            G.add_edge(i, target)

def create_grid_dag(G, task_num):
    width = int(task_num**0.5)
    for i in range(task_num):
        if i + 1 < task_num and (i + 1) % width != 0:
            G.add_edge(i + 1, i + 2)
        if i + width < task_num:
            G.add_edge(i + 1, i + width + 1)
    G.add_edge(0, 1)
    G.add_edge(task_num, task_num + 1)

def create_fork_join_dag(G, task_num):
    # Connect start to first task
    G.add_edge(0, 1)
    
    # Fork: connect first task to all middle tasks
    for i in range(2, task_num):
        G.add_edge(1, i)
    
    # Join: connect all middle tasks to the last task
    for i in range(2, task_num):
        G.add_edge(i, task_num)
    
    # Connect last task to end
    G.add_edge(task_num, task_num + 1)

def create_star_dag(G, task_num):
    for i in range(1, task_num + 1):
        G.add_edge(0, i)
        G.add_edge(i, task_num + 1)


def create_tree_dag(G, task_num):
    current_level = [1]
    next_level = []
    node = 2
    while node <= task_num:
        for parent in current_level:
            G.add_edge(parent, node)
            next_level.append(node)
            node += 1
            if node > task_num:
                break
            G.add_edge(parent, node)
            next_level.append(node)
            node += 1
            if node > task_num:
                break
        current_level = next_level
        next_level = []
    G.add_edge(0, 1)

def create_cycle_free_mesh_dag(G, task_num):
    for i in range(1, task_num):
        for j in range(i + 1, min(task_num + 1, i + 3)):
            G.add_edge(i, j)
    G.add_edge(0, 1)
    G.add_edge(task_num, task_num + 1)

def visualize_dag(G, filename):
    pos = nx.spring_layout(G)
    nx.draw(G, pos, with_labels=True, node_color='lightblue', node_size=500, font_size=10, font_weight='bold')
    edge_labels = nx.get_edge_attributes(G, 'weight')
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)
    plt.title(f"{G.graph['name']} DAG with {len(G.nodes()) - 2} tasks")
    plt.savefig(filename)
    plt.close()

def create_dag_datasets(num_dags, task_num, data_size_range, workload_range, dag_types):
    datasets = {dag_type: [] for dag_type in dag_types}
    
    for _ in range(num_dags):
        for dag_type in dag_types:
            G = create_task(
                task_num=task_num,
                data_size_min=data_size_range[0],
                data_size_max=data_size_range[1],
                workload_min=workload_range[0],
                workload_max=workload_range[1],
                dag_type=dag_type
            )
            datasets[dag_type].append(G)
    
    return datasets


def split_train_test(datasets, train_ratio=0.8):
    train_datasets = {dag_type: [] for dag_type in datasets.keys()}
    test_datasets = {dag_type: [] for dag_type in datasets.keys()}
    
    for dag_type, dags in datasets.items():
        random.shuffle(dags)
        split_index = int(len(dags) * train_ratio)
        train_datasets[dag_type] = dags[:split_index]
        test_datasets[dag_type] = dags[split_index:]
    
    return train_datasets, test_datasets

def create_dag_datasets(num_dags, task_num, data_size_range, workload_range, dag_types):
    datasets = {dag_type: [] for dag_type in dag_types}
    
    for _ in range(num_dags):
        for dag_type in dag_types:
            G = create_task(
                task_num=task_num,
                data_size_min=data_size_range[0],
                data_size_max=data_size_range[1],
                workload_min=workload_range[0],
                workload_max=workload_range[1],
                dag_type=dag_type
            )
            datasets[dag_type].append(G)
    
    return datasets

if __name__ == '__main__':
    # Configuration
    NUM_DAGS = 100  # Increased for better split
    TASK_NUM = 6  # Number of tasks between start and end
    DATA_SIZE_RANGE = (25, 50)  # KB
    WORKLOAD_RANGE = (100, 500)  # Million instructions
    DAG_TYPES = ['linear', 'branching', 'mixed', 'grid', 'fork-join', 'star', 'tree', 'cycle-free-mesh']
    TRAIN_RATIO = 0.8  # 80% for training, 20% for testing

    # Create datasets
    datasets = create_dag_datasets(NUM_DAGS, TASK_NUM, DATA_SIZE_RANGE, WORKLOAD_RANGE, DAG_TYPES)

    # Split into train and test sets
    train_datasets, test_datasets = split_train_test(datasets, TRAIN_RATIO)

    # Create folders
    for folder in ['DAGs', 'data_list/train', 'data_list/test']:
        os.makedirs(folder, exist_ok=True)

    # Save datasets and visualize some examples
    for dag_type in DAG_TYPES:
        # Save train dataset
        with open(f'data_list/train/task_list_{dag_type}.pickle', 'wb') as f:
            pickle.dump(train_datasets[dag_type], f)
        
        # Save test dataset
        with open(f'data_list/test/task_list_{dag_type}.pickle', 'wb') as f:
            pickle.dump(test_datasets[dag_type], f)
        
        # Visualize a few examples (from training set)
        for i in range(5):  # Visualize 5 examples of each type
            visualize_dag(train_datasets[dag_type][i], f"DAGs/task_graph_{dag_type}_{i}.png")

    print(f"Created and split {NUM_DAGS} DAGs of each type: {', '.join(DAG_TYPES)}")
    print(f"Train-Test split ratio: {TRAIN_RATIO:.0%}-{1-TRAIN_RATIO:.0%}")
    print("Training datasets saved in 'data_list/train' folder")
    print("Test datasets saved in 'data_list/test' folder")
    print("Example visualizations (from training set) saved in 'DAGs' folder")