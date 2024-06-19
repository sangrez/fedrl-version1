# from global_variables import *
import networkx as nx
import numpy as np
from scipy.linalg import fractional_matrix_power
import pickle
import itertools
import numpy as np
np.random.seed(42)

with open('data_list/task_list.pickle', 'rb') as f:
    task_list = pickle.load(f)
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
    # "h": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],  # Channel gain
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
    'application_delay': 1003,
    'energy_threshold': 40,
}

new_parameters = {

    "datasize" : 10000,  # bits
    "workload" : 50000
}


def DAG_features(task_list, k):
    # Extract the specific DAG
    
    G = task_list[k]
    adj_matrix = nx.to_numpy_array(G, nodelist=sorted(G.nodes()))
    feature_matrix = np.array([[G.nodes[i]['computing_circle'], G.nodes[i]['data_size']] for i in G.nodes()])
    new_adj = {i: np.where(adj_matrix[:, i] == 1)[0].tolist() for i in range(adj_matrix.shape[1])}
    workload  = (feature_matrix[:, 0] ).tolist()
    data  = (feature_matrix[:, 1]  ).tolist()
    I = np.identity(adj_matrix.shape[0])
    A_hat = adj_matrix + I
    D = np.diag(A_hat.sum(axis=1))
    D_half_norm = fractional_matrix_power(D, -0.5)

    def relu(x):
        return np.maximum(0, x)

    def gcn(A, H, W):
        A_hat_local = A + np.identity(A.shape[0])
        D_local = np.diag(A_hat_local.sum(axis=0))
        D_half_norm_local = fractional_matrix_power(D_local, -0.5)
        return relu(D_half_norm_local.dot(A_hat_local).dot(D_half_norm_local).dot(H).dot(W))

    np.random.seed(77777)
    n_h = 4  # number of neurons in the hidden layer
    W0 = np.random.randn(feature_matrix.shape[1], n_h) * 0.01
    W1 = np.random.randn(n_h, 1) * 0.01

    # Compute GCN states
    H1 = gcn(adj_matrix, feature_matrix, W0)
    GNN_state = gcn(adj_matrix, H1, W1)

    return len(G.nodes()), new_adj, list(G.nodes()), data, workload, GNN_state

def user_info(users): 

    user_info = {}  # Initialize an empty dictionary to store user information
    # Iterate over the number of users (5 in this case)
    for user_index in range(users):
        total_tasks, new_adj, tasks_in_DAG, data, workload, GNN_state = DAG_features(task_list, k)

        # Store the information for the current user in the dictionary
        user_info[user_index] = {
            'total_tasks': total_tasks,
            'new_adj': new_adj,
            'tasks_in_DAG': tasks_in_DAG,
            'data': data,
            'workload': workload,
            'GNN_state': GNN_state
        }

    first_tasks = []
    total_task_count = 0  # Initialize the total_task_count variable

    for user_index in user_info:
        first_task = user_info[user_index]['tasks_in_DAG'][t]
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
