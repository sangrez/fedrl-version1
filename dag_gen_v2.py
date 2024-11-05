import networkx as nx
import numpy as np
import random
import matplotlib.pyplot as plt
import os
import pickle


random.seed(42)
np.random.seed(42)

def create_task(task_num, data_size_min, data_size_max, workload_min, workload_max, dag_type='mixed'):
    G = nx.DiGraph(name=dag_type)

    # Create start node
    start_data_size = np.random.uniform(data_size_min, data_size_max) * 1024
    start_computing_circle = np.random.uniform(workload_min, workload_max) * 100000
    G.add_node(0, data_size=start_data_size, computing_circle=start_computing_circle)  # Start node

    # Create task nodes
    for i in range(1, task_num + 1):
        t_data_size = np.random.uniform(data_size_min, data_size_max) * 1024
        t_computing_circle = np.random.uniform(workload_min, workload_max) * 100000
        G.add_node(i, data_size=t_data_size, computing_circle=t_computing_circle)
        
    # Create end node
    end_data_size = np.random.uniform(data_size_min, data_size_max) * 1024
    end_computing_circle = np.random.uniform(workload_min, workload_max) * 100000
    G.add_node(task_num + 1, data_size=end_data_size, computing_circle=end_computing_circle)  # End node

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
        if not list(G.successors(i)):
            G.add_edge(i, task_num + 1)

    return G

def create_linear_dag(G, task_num):
    for i in range(task_num + 1):
        G.add_edge(i, i + 1)

def create_branching_dag(G, task_num):
    G.add_edge(0, 1)
    for i in range(1, task_num):
        G.add_edge(i, i + 1)
        if i + 2 <= task_num:
            G.add_edge(i, i + 2)

def create_mixed_dag(G, task_num):
    G.add_edge(0, 1)
    for i in range(1, task_num):
        G.add_edge(i, i + 1)
    for i in range(1, task_num - 1):
        if random.random() < 0.3:
            target = random.randint(i + 2, task_num)
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
    # Connect leaf nodes to end node
    leaf_nodes = [n for n in G.nodes() if G.out_degree(n) == 0 and n != task_num + 1]
    for leaf in leaf_nodes:
        G.add_edge(leaf, task_num + 1)

def create_cycle_free_mesh_dag(G, task_num):
    for i in range(1, task_num):
        for j in range(i + 1, min(task_num + 1, i + 3)):
            G.add_edge(i, j)
    G.add_edge(0, 1)
    G.add_edge(task_num, task_num + 1)

def visualize_dag(G, filename):
    pos = nx.spring_layout(G)
    nx.draw(G, pos, with_labels=True, node_color='lightblue', node_size=500, font_size=10, font_weight='bold')
    plt.title(f"{G.graph['name']} DAG with {len(G.nodes()) - 2} tasks")
    plt.savefig(filename)
    plt.close()

def create_weighted_dag_datasets(num_dags, task_num, data_ranges, dag_types, train_ratio=0.8, majority_ratio=0.7):
    """
    Create datasets where the training set includes both majority and minority DAGs,
    and the testing set includes only majority DAGs of each specific type.
    """
    train_datasets = {dag_type: [] for dag_type in dag_types}
    test_datasets = {dag_type: [] for dag_type in dag_types}

    # Generate datasets for each DAG type
    for dag_type in dag_types:
        # Determine number of majority and minority DAGs
        num_majority = int(num_dags * majority_ratio)
        num_minority = num_dags - num_majority

        # Generate majority DAGs
        majority_dags = []
        data_size_range = data_ranges[dag_type]['data_size_range']
        workload_range = data_ranges[dag_type]['workload_range']
        for _ in range(num_majority):
            G = create_task(
                task_num=task_num,
                data_size_min=data_size_range[0],
                data_size_max=data_size_range[1],
                workload_min=workload_range[0],
                workload_max=workload_range[1],
                dag_type=dag_type
            )
            majority_dags.append(G)

        # Generate minority DAGs
        minority_dags = []
        other_dag_types = [d for d in dag_types if d != dag_type]
        for _ in range(num_minority):
            random_dag_type = random.choice(other_dag_types)
            data_size_range = data_ranges[random_dag_type]['data_size_range']
            workload_range = data_ranges[random_dag_type]['workload_range']
            G = create_task(
                task_num=task_num,
                data_size_min=data_size_range[0],
                data_size_max=data_size_range[1],
                workload_min=workload_range[0],
                workload_max=workload_range[1],
                dag_type=random_dag_type
            )
            minority_dags.append(G)

        # Combine majority and minority DAGs for training
        train_dags = majority_dags[:int(num_majority * train_ratio)] + minority_dags
        random.shuffle(train_dags)
        train_datasets[dag_type] = train_dags

        # Testing set includes only majority DAGs
        test_datasets[dag_type] = majority_dags[int(num_majority * train_ratio):]

    return train_datasets, test_datasets

if __name__ == '__main__':
    # Configuration
    NUM_DAGS = 5000  # Number of DAGs per type
    TASK_NUM = 6  # Number of tasks between start and end
    TRAIN_RATIO = 0.8  # 80% for training, 20% for testing
    MAJORITY_RATIO = 0.7  # 70% majority type, 30% minority types

    # Data ranges for each DAG type (introduce variability)
    DATA_RANGES = {
        'linear': {'data_size_range': (100, 150), 'workload_range': (80, 120)},
        'branching': {'data_size_range': (120, 170), 'workload_range': (90, 130)},
        'mixed': {'data_size_range': (110, 160), 'workload_range': (85, 125)},
        'grid': {'data_size_range': (130, 180), 'workload_range': (95, 135)},
        'fork-join': {'data_size_range': (100, 140), 'workload_range': (80, 120)},
        'star': {'data_size_range': (115, 165), 'workload_range': (85, 125)},
        'tree': {'data_size_range': (105, 155), 'workload_range': (82, 122)},
        'cycle-free-mesh': {'data_size_range': (125, 175), 'workload_range': (92, 132)},
    }

    DAG_TYPES = list(DATA_RANGES.keys())

    # Create and split datasets
    train_datasets, test_datasets = create_weighted_dag_datasets(
        num_dags=NUM_DAGS,
        task_num=TASK_NUM,
        data_ranges=DATA_RANGES,
        dag_types=DAG_TYPES,
        train_ratio=TRAIN_RATIO,
        majority_ratio=MAJORITY_RATIO
    )

    # Create folders
    for folder in ['DAGs', 'data_list/train', 'data_list/test']:
        os.makedirs(folder, exist_ok=True)

    # Save datasets and visualize some examples
    for dag_type in DAG_TYPES:
        # Save train dataset for each type
        with open(f'data_list/train/task_list_{dag_type}.pickle', 'wb') as f:
            pickle.dump(train_datasets[dag_type], f)
        
        # Visualize a few examples from the training set
        for i in range(5):  # Visualize 5 examples of each type
            if i < len(train_datasets[dag_type]):
                visualize_dag(train_datasets[dag_type][i], f"DAGs/train_task_graph_{dag_type}_{i}.png")

    # Save test datasets for each DAG type
    for dag_type in DAG_TYPES:
        with open(f'data_list/test/task_list_{dag_type}.pickle', 'wb') as f:
            pickle.dump(test_datasets[dag_type], f)
        
        # Visualize a few examples from the test set
        for i in range(5):  # Visualize 5 examples from each test set
            if i < len(test_datasets[dag_type]):
                visualize_dag(test_datasets[dag_type][i], f"DAGs/test_task_graph_{dag_type}_{i}.png")

    # Print dataset information
    print(f"Created and split {NUM_DAGS} DAGs for each of the {len(DAG_TYPES)} types")
    print(f"Train-Test split ratio: {TRAIN_RATIO:.0%}-{(1 - TRAIN_RATIO):.0%}")
    print(f"Each training set has {len(train_datasets[DAG_TYPES[0]])} DAGs (including minority types)")
    print(f"Each test set has {len(test_datasets[DAG_TYPES[0]])} DAGs (only majority type)")
    print("Training datasets saved in 'data_list/train' folder")
    print("Test datasets saved in 'data_list/test' folder, separate for each DAG type")
    print("Example visualizations saved in 'DAGs' folder")
