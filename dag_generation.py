import networkx as nx
import numpy as np
import random
import matplotlib.pyplot as plt
import os
import pickle

def create_task(task_num, data_size_min, data_size_max, workload_min, workload_max, dag_type='mixed'):
    G = nx.DiGraph(name='G')
    
    # Create nodes
    for i in range(task_num):
        t_data_size = np.random.uniform(data_size_min, data_size_max) * 1024
        t_computing_circle = np.random.uniform(workload_min, workload_max)
        G.add_node(i, data_size=t_data_size, computing_circle=t_computing_circle)

    # Create edges based on DAG type
    if dag_type == 'linear':
        for i in range(task_num - 1):
            G.add_edge(i, i+1, weight=1)
    
    elif dag_type == 'branching':
        G.add_edge(0, 1, weight=1)
        for i in range(1, task_num - 1):
            child1 = min(i + 1, task_num - 1)
            child2 = min(i + 2, task_num - 1)
            G.add_edge(i, child1, weight=1)
            if child1 != child2:
                G.add_edge(i, child2, weight=1)
    
    elif dag_type == 'mixed':
        # Start with a linear backbone
        for i in range(task_num - 1):
            G.add_edge(i, i+1, weight=1)
        
        # Add some random branches
        for i in range(1, task_num - 2):
            if random.random() < 0.3:  # 30% chance to add a branch
                target = random.randint(i+2, task_num-1)
                G.add_edge(i, target, weight=1)

    return G

def visualize_dag(G, filename):
    pos = nx.spring_layout(G)
    nx.draw(G, pos, with_labels=True, node_color='lightblue', node_size=500, font_size=10, font_weight='bold')
    edge_labels = nx.get_edge_attributes(G, 'weight')
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)
    plt.title(f"DAG with {len(G.nodes())} tasks")
    plt.savefig(filename)
    plt.close()

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
                data_size_max=data_size_range[0],
                workload_min=workload_range[0],
                workload_max=workload_range[0],
                dag_type=dag_type
            )
            datasets[dag_type].append(G)
    
    return datasets

if __name__ == '__main__':
    # Configuration
    NUM_DAGS = 1000  # Increased for better split
    TASK_NUM = 10  # Number of tasks between start and end
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