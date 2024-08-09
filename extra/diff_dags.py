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
        for i in range(task_num - 1):
            G.add_edge(i, i+1)
        for i in range(1, task_num - 2):
            if random.random() < 0.3:
                target = random.randint(i+2, task_num-1)
                G.add_edge(i, target)

    elif dag_type == 'layered':
        layers = 3  # You can adjust this
        nodes_per_layer = task_num // layers
        for layer in range(layers - 1):
            for i in range(layer * nodes_per_layer, (layer + 1) * nodes_per_layer):
                for j in range((layer + 1) * nodes_per_layer, min((layer + 2) * nodes_per_layer, task_num)):
                    G.add_edge(i, j)

    elif dag_type == 'diamond':
        mid = task_num // 2
        G.add_edge(0, 1)
        G.add_edge(0, 2)
        for i in range(1, mid):
            G.add_edge(i, mid)
        G.add_edge(mid, task_num-2)
        G.add_edge(mid, task_num-1)

    elif dag_type == 'tree':
        for i in range(1, task_num):
            parent = (i - 1) // 2
            G.add_edge(parent, i)

    elif dag_type == 'random':
        for i in range(task_num):
            for j in range(i + 1, task_num):
                if random.random() < 0.3:  # Adjust probability as needed
                    G.add_edge(i, j)

    return G

def visualize_dag(G, filename):
    pos = nx.spring_layout(G)
    plt.figure(figsize=(12, 8))
    nx.draw(G, pos, with_labels=True, node_color='lightblue', 
            node_size=500, font_size=10, font_weight='bold', 
            arrows=True, arrowsize=20)
    plt.title(f"DAG with {len(G.nodes())} tasks - Type: {G.name}")
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

if __name__ == '__main__':
    # Configuration
    NUM_DAGS = 1000
    TASK_NUM = 7  # Fixed number of tasks
    DATA_SIZE_RANGE = (10, 100)  # KB
    WORKLOAD_RANGE = (100, 1000)  # Million instructions
    DAG_TYPES = ['linear', 'branching', 'mixed', 'layered', 'diamond', 'tree', 'random']

    # Create datasets
    datasets = create_dag_datasets(NUM_DAGS, TASK_NUM, DATA_SIZE_RANGE, WORKLOAD_RANGE, DAG_TYPES)

    # Create folders
    for folder in ['DAGs', 'data_list']:
        os.makedirs(folder, exist_ok=True)

    # Save datasets and visualize some examples
    for dag_type, dag_list in datasets.items():
        # Save dataset
        with open(f'data_list/task_list_{dag_type}.pickle', 'wb') as f:
            pickle.dump(dag_list, f)
        
        # Visualize a few examples
        for i in range(5):  # Visualize 5 examples of each type
            visualize_dag(dag_list[i], f"DAGs/task_graph_{dag_type}_{i}.png")

    print(f"Created {NUM_DAGS} DAGs of each type: {', '.join(DAG_TYPES)}")
    print(f"Each DAG has {TASK_NUM} tasks")
    print("Datasets saved in 'data_list' folder")
    print("Example visualizations saved in 'DAGs' folder")