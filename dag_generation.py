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
            G.add_edge(i, i+1)
    
    elif dag_type == 'branching':
        G.add_edge(0, 1)
        for i in range(1, task_num - 1):
            child1 = min(i + 1, task_num - 1)
            child2 = min(i + 2, task_num - 1)
            G.add_edge(i, child1)
            if child1 != child2:
                G.add_edge(i, child2)
    
    elif dag_type == 'mixed':
        # Start with a linear backbone
        for i in range(task_num - 1):
            G.add_edge(i, i+1)
        
        # Add some random branches
        for i in range(1, task_num - 2):
            if random.random() < 0.3:  # 30% chance to add a branch
                target = random.randint(i+2, task_num-1)
                G.add_edge(i, target)

    return G

def visualize_dag(G, filename):
    pos = nx.spring_layout(G)
    nx.draw(G, pos, with_labels=True, node_color='lightblue', node_size=500, font_size=10, font_weight='bold')
    edge_labels = nx.get_edge_attributes(G, 'weight')
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)
    plt.title(f"DAG with {len(G.nodes())} tasks")
    plt.savefig(filename)
    plt.close()

def create_dag_datasets(num_dags, task_num, data_size_range, workload_range, dag_types, train_ratio=0.8):
    datasets = {dag_type: {'train': [], 'val': []} for dag_type in dag_types}
    
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
            if random.random() < train_ratio:
                datasets[dag_type]['train'].append(G)
            else:
                datasets[dag_type]['val'].append(G)
    
    return datasets

if __name__ == '__main__':
    # Configuration
    NUM_DAGS = 5000
    TASK_NUM = 7  # Fixed number of tasks
    DATA_SIZE_RANGE = (25, 50)  # KB
    WORKLOAD_RANGE = (100, 500)  # Million instructions
    DAG_TYPES = ['linear', 'branching', 'mixed']
    TRAIN_RATIO = 0.8

    # Create datasets
    datasets = create_dag_datasets(NUM_DAGS, TASK_NUM, DATA_SIZE_RANGE, WORKLOAD_RANGE, DAG_TYPES, TRAIN_RATIO)

    # Create folders
    for folder in ['DAGs', 'data_list']:
        os.makedirs(folder, exist_ok=True)

    # Save datasets and visualize some examples
    for dag_type, split_data in datasets.items():
        for split in ['train', 'val']:
            # Save dataset
            with open(f'data_list/task_list_{dag_type}_{split}.pickle', 'wb') as f:
                pickle.dump(split_data[split], f)
        
        # Visualize a few examples from the training set
        for i in range(5):  # Visualize 5 examples of each type
            if i < len(split_data['train']):
                visualize_dag(split_data['train'][i], f"DAGs/task_graph_{dag_type}_{i}.png")

    print(f"Created {NUM_DAGS} DAGs of each type: {', '.join(DAG_TYPES)}")
    print(f"Each DAG has exactly {TASK_NUM} tasks")
    print(f"Split into {TRAIN_RATIO*100}% train and {(1-TRAIN_RATIO)*100}% validation")
    print("Datasets saved in 'data_list' folder")
    print("Example visualizations saved in 'DAGs' folder")

    # Print dataset statistics
    for dag_type in DAG_TYPES:
        print(f"\n{dag_type.capitalize()} DAG statistics:")
        print(f"  Training set size: {len(datasets[dag_type]['train'])}")
        print(f"  Validation set size: {len(datasets[dag_type]['val'])}")

        # Calculate average number of edges
        train_avg_edges = np.mean([len(G.edges()) for G in datasets[dag_type]['train']])
        val_avg_edges = np.mean([len(G.edges()) for G in datasets[dag_type]['val']])
        print(f"  Average number of edges (train): {train_avg_edges:.2f}")
        print(f"  Average number of edges (val): {val_avg_edges:.2f}")