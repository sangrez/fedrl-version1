import networkx as nx
import numpy as np
from scipy.linalg import fractional_matrix_power
import pickle
import itertools
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data

np.random.seed(42)

t = 0
PRE = 0
k = 0

tasks_in_DAG = []
task_delay = []
start_time_current = []
i = 0
new_adj = {}

parameter = {
    "channel_gain_max": -5,
    "channel_gain_min": -50,  # dB
    "distance_min": 50,  # meters
    "distance_max": 100,  # distance betweem the user and BS
    "total_tasks": {},
    "tasks": tasks_in_DAG,
    "a": 10,
    "O": {},
    "L": {},  # CPU cycles required
    "residual_power": 200,
    "taskNum": {},
    "power_low_value": 500,
    'power_very_low_value': 350,
    "residual_power_max": 10,
    "data_size": 5,
    "local_device_freq": 1 * 10**9,  # 1 GHz
    "edge_server_freq": 2.4 * 10 ** 9,  # 2.5 GHz
    "k": 1.25 * 10 ** -26,
    "uplink_power": 0.2,  # the maximum transmit power of the mobile device
    "downlink_power" : 0.2,
    "I": 1.5e-8,  # Noise power
    "maxReward": 4000,
    "total_power": 10000,
    "power_percentage": 0.75,
    "W": 2 * 10 ** 3,  # bandwidth 2MHZ
    "downlink_bandwidth" : 2 * 10 * 4,
    "uplink_datarate": 3 * 10 **6,  # 3Mbps
    "downlink_datarate": 3 * 10 **6, # 3Mbps
    "sigma": 10 ** (-5),  # noise power
    'new_adj': new_adj,
    'start': 0,
    'application_delay': 1,
    'energy_threshold': 60,
}

new_parameters = {
    "datasize" : 10000,  # bits
    "workload" : 50000
}

def load_dag_dataset(dag_type, split='train'):
    if split == 'test':
        # For test, we load the combined dataset
        with open(f'data_list/test/task_list_combined.pickle', 'rb') as f:
            return pickle.load(f)
    else:
        with open(f'data_list/{split}/task_list_{dag_type}.pickle', 'rb') as f:
            return pickle.load(f)

# Load datasets for each DAG type and split
dag_types = ['linear', 'branching', 'mixed', 'grid', 'star', 'tree', 'cycle-free-mesh']
train_dags = {dag_type: load_dag_dataset(dag_type, 'train') for dag_type in dag_types}
test_dags = load_dag_dataset('combined', 'test')

def DAG_features(task_list, k):
    # Extract the specific DAG
    G = task_list[k]
    adj_matrix = nx.to_numpy_array(G, nodelist=sorted(G.nodes()))
    feature_matrix = np.array([[G.nodes[i]['computing_circle'], G.nodes[i]['data_size']] for i in G.nodes()])
    new_adj = {i: np.where(adj_matrix[:, i] == 1)[0].tolist() for i in range(adj_matrix.shape[1])}
    workload = feature_matrix[:, 0].tolist()
    data = feature_matrix[:, 1].tolist()

    # Convert to PyTorch tensors
    edge_index = torch.tensor(np.array(np.nonzero(adj_matrix)), dtype=torch.long)
    x = torch.tensor(feature_matrix, dtype=torch.float)

    # Create a PyTorch Geometric Data object
    graph_data = Data(x=x, edge_index=edge_index)

    return len(G.nodes()), new_adj, list(G.nodes()), data, workload, graph_data

class GNNModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(GNNModel, self).__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, output_dim)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = F.relu(self.conv1(x, edge_index))
        x = self.conv2(x, edge_index)
        return x

def user_info(users, dag_type='mixed', split='train'):
    if split == 'test':
        task_list = test_dags
    else:
        task_list = train_dags[dag_type]

    user_info = {}
    for user_index in range(users):
        k = np.random.randint(0, len(task_list))
        total_tasks, new_adj, tasks_in_DAG, data, workload, graph_data = DAG_features(task_list, k)
        user_info[user_index] = {
            'total_tasks': total_tasks,
            'new_adj': new_adj,
            'tasks_in_DAG': tasks_in_DAG,
            'data': data,
            'workload': workload,
            'graph_data': graph_data
        }

    first_tasks = []
    total_task_count = 0

    for user_index in user_info:
        first_task = user_info[user_index]['tasks_in_DAG'][0]  # Assuming t=0 for the first task
        first_tasks.append(first_task)
        total_task_count += user_info[user_index]['total_tasks']

    parameter['total_tasks'] = total_tasks
    parameter['tasks'] = tasks_in_DAG
    parameter['O'] = total_tasks * [new_parameters["datasize"]]
    parameter['L'] = total_tasks * [new_parameters["workload"]]
    parameter['taskNum'] = total_task_count
    parameter['all_tasks'] = total_task_count
    parameter['new_adj'] = new_adj
    parameter['current_tasks'] = first_tasks

    return user_info

def generate_offloading_combinations(num_servers, num_users):
    # Define possible offloading locations for each task as integers
    offload_locations = list(range(num_servers))

    # Generate all possible combinations of offloading decisions for the tasks
    all_combinations = list(itertools.product(offload_locations, repeat=num_users))

    # Create a dictionary mapping each discrete action index to an offloading decision combination
    action_map = {i: combination for i, combination in enumerate(all_combinations)}
    
    return action_map

# if __name__ == "__main__":
#     # This block can be used for testing or demonstrating the functionality
#     test_users = 3
#     test_dag_type = 'mixed'
#     test_split = 'train'

#     test_user_info = user_info(test_users, test_dag_type, test_split)
#     print(f"Generated user info for {test_users} users with {test_dag_type} DAG type ({test_split} split):")
#     for user, info in test_user_info.items():
#         print(f"User {user}:")
#         print(f"  Total tasks: {info['total_tasks']}")
#         print(f"  First task: {info['tasks_in_DAG'][0]}")
#         print(f"  Data size of first task: {info['data'][0]:.2f}")
#         print(f"  Workload of first task: {info['workload'][0]:.2f}")
#         print(f"  Graph data shape: {info['graph_data'].x.shape}")

#     test_servers = 2
#     test_combinations = generate_offloading_combinations(test_servers, test_users)
#     print(f"\nGenerated offloading combinations for {test_users} users and {test_servers} servers:")
#     for action, combination in list(test_combinations.items())[:5]:  # Print first 5 combinations
#         print(f"Action {action}: {combination}")
#     print(f"Total combinations: {len(test_combinations)}")