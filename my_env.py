import gym
from gym import spaces
import numpy as np
# from global_variables import *
import math
np.random.seed(42)
import random
from user_info import parameter
import user_info
# from Reward_scale import *
# from my_variables import *
import copy
import os
import glob
total_energy_array = []
total_delay = []
start_time_current = []
i = 0
max_energy = []
max_time = []
text_data_folder = "./text_data"
files = glob.glob(os.path.join(text_data_folder, "*.txt"))

class Offloading(gym.Env):

    def __init__(self, users, servers):
        super(Offloading, self).__init__()
        self.average_energy_all_local = None
        self.average_time_all_local = None
        # self.very_res_ene = very_res_ene
        # self.very_res_ene_local = very_res_ene_local
        # self.very_res_ene_edge = very_res_ene_edge
        # self.very_res_ene_random = very_res_ene_random
        # self.reward_scaler = RewardScaler()
        self.local_t = None
        self.local_task = None
        # Initialize max_energy and max_delay to be very small
        self.max_energy = 0
        self.max_delay_norm = 0
        self.max_energy_local = 0
        self.max_delay_local = 0
        self.max_energy_edge = 0
        self.max_delay_edge = 0
        self.max_energy_random = 0
        self.max_delay_random = 0
        self.local_energy = None
        self.edge_t = None
        self.edge_energy = None
        self.random_t = None
        self.random_energy = None
        self.k = parameter['k']
        self.reward = None
        self.intial_reward = None
        self.gain_array = []
        self.current_task = []
        self.device_power_array = []
        self.offloaded_tasks = []
        self.reward_flag = None
        self.start_obs = None
        self.user_info = user_info.user_info(users)
        self.device_power_value = None
        self.ter_obs = None
        self.start_energy = 0
        self.channel_gain = None
        self.average_time = None
        self.average_transmission_time = None
        self.average_transmission_time_l = None
        self.average_transmission_time_e = None
        self.average_transmission_time_r = None
        self.average_energy = None
        self.energy_max = float('-inf')
        self.time_max = float('-inf')

        self.min_channel_gain = -50  # value of channel gain in db
        self.max_channel_gain = -8
        self.user_devices = users
        self.edge_servers = servers
        self.devices_ID = [1, 2, 3, 4, 5]
        self.servers_ID = [1, 2, 3]
        self.max_freq_local = parameter['local_device_freq']
        self.max_freq = parameter['edge_server_freq']
        self.resources = []
        self.local_device_offloaded = []
        self.edge_server_1 = []
        self.edge_server_2 = []
        self.edge_server_3 = []
        self.state = []
        self.very_next_state = []
        self.previous_tasks = []
        self.min_device_power = 0  # IoT device remaing energy level in mJ.
        self.max_device_power = parameter['residual_power_max']
        self.min_delay = 0
        self.max_delay = parameter['application_delay']  # task delay in appliction
        self.min_task = 0
        self.max_task = parameter['taskNum']  # no of task offloaded
        self.min_state_task = 0
        self.max_state_task = 1
        low = np.array( [-50] * self.user_devices +  [0] * self.user_devices +  [0]).astype(np.float32)  
        high = np.array([-5] * self.user_devices +  [100] * self.user_devices +  [100]).astype(np.float32)  
        self.p = parameter['uplink_power']
        self.taskNum = parameter['taskNum']
        self.task = parameter['tasks']
        self.adj_mat = parameter['new_adj']
        self.completed_task = []
        self.completion_time = []
        self.task_time = []
        self.task_energy_consumption = []
        self.task_energy_consumption_all_local = []
        self.task_time_all_local = []
        self.task_energy_consumption_all_edge = []
        self.task_time_all_edge = []
        self.reward_value = 0
      
        self.max_time = []
        self.allocation_check = []
        # self.task_state_array = []
        self.all_task = []
        self.average_time_local = None
        self.average_energy_local = None
        self.max_task = self.edge_servers
        self.max_cpu = 1
        # self.discrete_action_space = spaces.MultiDiscrete([self.max_task + 1] * self.user_devices)
        # self.discrete_action_space = spaces.Discrete(self.max_task + 1)
        all_combinations = user_info.generate_offloading_combinations(self.edge_servers + 1, self.user_devices)
        self.discrete_action_space = spaces.Discrete(len(all_combinations))

        low_cpu = np.array([0.1] * self.user_devices).astype(np.float32)
        high_cpu = np.array([self.max_cpu] * self.user_devices).astype(np.float32)
        self.continuous_action_space = spaces.Box(low=low_cpu, high=high_cpu, dtype=np.float32)
        self.action_space = spaces.Tuple((self.discrete_action_space, self.continuous_action_space))

        self.observation_space = spaces.Box(low, high)

    def sample_combined_action(self):
        discrete_action_space = np.array([self.discrete_action_space.sample() for _ in range(self.user_devices)])
        continuous_action_space = self.continuous_action_space.sample()
        combined_action = np.concatenate((discrete_action_space, continuous_action_space))
        return combined_action

    def reset_eval(self):
        self.state = []
        self.completed_task = [[] for _ in range(self.user_devices)]
        self.completed_task_local = [[] for _ in range(self.user_devices)]
        self.completed_task_edge = [[] for _ in range(self.user_devices)]
        self.completed_task_random = [[] for _ in range(self.user_devices)]
        self.reward_value = 0
        self.offloaded_tasks = []
        # self.start_obs = self.start_observe()

        self.offloading_time = [[] for _ in range(self.user_devices)]
        self.offloading_time_local = [[] for _ in range(self.user_devices)]    
        self.offloading_edge = [[] for _ in range(self.user_devices)]
        self.offloading_random = [[] for _ in range(self.user_devices)]

        self.task_time = [[] for _ in range(self.user_devices)]
        self.task_energy_consumption = [[] for _ in range(self.user_devices)]

        self.task_time_all_local = [[] for _ in range(self.user_devices)]
        self.task_energy_consumption_all_local = [[] for _ in range(self.user_devices)]

        self.task_time_all_edge = [[] for _ in range(self.user_devices)]
        self.task_energy_consumption_all_edge = [[] for _ in range(self.user_devices)]

        self.task_time_random = [[] for _ in range(self.user_devices)]
        self.task_energy_consumption_random = [[] for _ in range(self.user_devices)]

        self.start_energy = 0
        self.intial_reward = 0
        return False
    def reset(self):

        self.state = []
        self.completed_task = [[] for _ in range(self.user_devices)]
        self.completed_task_local = [[] for _ in range(self.user_devices)]
        self.completed_task_edge = [[] for _ in range(self.user_devices)]
        self.completed_task_random = [[] for _ in range(self.user_devices)]
        self.reward_value = 0
        self.offloaded_tasks = []
        self.start_obs = self.start_observe()

        self.offloading_time = [[] for _ in range(self.user_devices)]
        self.offloading_time_local = [[] for _ in range(self.user_devices)]    
        self.offloading_edge = [[] for _ in range(self.user_devices)]
        self.offloading_random = [[] for _ in range(self.user_devices)]

        self.task_time = [[] for _ in range(self.user_devices)]
        self.task_energy_consumption = [[] for _ in range(self.user_devices)]

        self.task_time_all_local = [[] for _ in range(self.user_devices)]
        self.task_energy_consumption_all_local = [[] for _ in range(self.user_devices)]

        self.task_time_all_edge = [[] for _ in range(self.user_devices)]
        self.task_energy_consumption_all_edge = [[] for _ in range(self.user_devices)]

        self.task_time_random = [[] for _ in range(self.user_devices)]
        self.task_energy_consumption_random = [[] for _ in range(self.user_devices)]

        self.start_energy = 0
        self.intial_reward = 0
        return np.array(self.start_obs, dtype=np.float32)
        # return self.start_obs

    def start_observe(self):
        self.previous_tasks = np.zeros(self.user_devices)
        parameter['residual_power'] = np.zeros(self.user_devices)
        parameter['channel_gain'] = np.zeros(self.user_devices)
        for i in range(self.user_devices):
            self.channel_gain = np.random.randint(self.min_channel_gain, self.max_channel_gain)
            self.device_power_value = self.max_device_power
            parameter['channel_gain'][i] = self.channel_gain
            parameter['residual_power'][i] = self.device_power_value
            # self.task_state = np.round(self.user_info[i]['GNN_state'][0].item(), 2)
            channel_gain_normalized = np.round((self.channel_gain - (-50)) / ((-8) - (-50)), 2)
            device_power_value_normalized = np.round((self.device_power_value - 0) / (self.max_device_power - 0), 2)
            self.state.extend([channel_gain_normalized, device_power_value_normalized])#self.task_state])
        # Normalize the number of offloaded tasks
        offloaded_tasks_normalized = np.round((len(self.offloaded_tasks) - 0) / (100 - 0), 2)
        self.state.extend([offloaded_tasks_normalized])

        self.residual_energy_local = copy.deepcopy(parameter['residual_power'])
        self.residual_energy_edge = copy.deepcopy(parameter['residual_power'])
        self.residual_energy_random = copy.deepcopy(parameter['residual_power'])
        # self.very_res_ene.append(sum(parameter['residual_power'])/self.user_devices)
        # self.very_res_ene_local.append(sum(self.residual_energy_local)/self.user_devices)
        # self.very_res_ene_edge.append(sum(self.residual_energy_edge)/self.user_devices)
        # self.very_res_ene_random.append(sum(self.residual_energy_random)/self.user_devices)
        return self.state

    def step(self, t, action, user_info):
        for j in user_info:
            task = user_info[j]['tasks_in_DAG'][t]
            self.current_task.append(task)

        offload_decision, allocation_check, previous_task, cpu_resource_allocations = self.action_taken(action)
        self.reward, reward_threshold, resources_total = self.evaluate(t, offload_decision, allocation_check, previous_task)
        offload_local = self.offload_local(t)
        offload_edge, resources_edge = self.offload_edge(t)
        offload_random = self.offload_random(t, offload_decision)

        next_state = self.observe(t)
        if reward_threshold != 1:
            self.reward, done = self.is_done()
        else:
            done = True
        self.ter_obs = next_state
        self.current_task = []
        # return next_state, self.reward, reward_threshold, done, offload_decision, offload_local, offload_edge, offload_random, cpu_resource_allocations, resources_edge
        return next_state, self.reward, done, reward_threshold
    
    def action_taken(self, action):
        value = []
        self.resources = []
        mid = len(action) // 2
        task_allocations = action[:mid]
        # task_allocations = [int(round(a)) for a in task_allocations]
        cpu_resource_allocations = action[mid:]
        self.allocation_check = [[] for _ in range(self.edge_servers + 1)]
        for i in range(len(task_allocations)):
            # local device (0.1 -0.9)

            if 0 <= task_allocations[i] < 0.6:
                # self.allocation_check[0].append(cpu_resource_allocations[i])
                resources = cpu_resource_allocations[i] * self.max_freq_local
                self.resources.append(resources)
                self.local_device_offloaded.append(parameter['current_tasks'][i])
                self.completed_task[i].append(0)
                self.offloaded_tasks.append(i)
                value.append(0)
            # edge 1 (1-1.9)
            elif 0.6 <= task_allocations[i] < 1.5:

                # percent_value = round(action[i] - 1.0, 2)
                self.allocation_check[1].append(cpu_resource_allocations[i])
                resources = cpu_resource_allocations[i] * self.max_freq
                self.resources.append(resources)
                self.edge_server_1.append(parameter['current_tasks'][i])
                self.completed_task[i].append(1)
                self.offloaded_tasks.append(i)
                value.append(1)

            # edge 2 (2  - 2.9)
            elif 1.5 <= task_allocations[i] < 2.51:

                # percent_value = round(action[i] - 2, 2)
                self.allocation_check[2].append(cpu_resource_allocations[i])
                resources = cpu_resource_allocations[i] * self.max_freq
                self.resources.append(resources)
                self.edge_server_2.append(self.current_task[i])
                self.completed_task[i].append(2)
                self.offloaded_tasks.append(i)
                value.append(2)
            # edge 2 (3  - 3.9)
            elif 2.51 <= task_allocations[i] < 3.4:

                # percent_value = round(action[i] - 3,2)
                self.allocation_check[3].append(cpu_resource_allocations[i])
                resources = cpu_resource_allocations[i] * self.max_freq
                self.resources.append(resources)
                self.edge_server_3.append(self.current_task[i])
                self.completed_task[i].append(3)
                self.offloaded_tasks.append(i)
                value.append(3)
            else:
                print('Invalid action')
            parameter['resources'] = self.resources
        return value, self.allocation_check, self.completed_task, cpu_resource_allocations

    def evaluate(self, t, offloading_decision, allocation_check, previous_task):

        for each_list in allocation_check:
            if np.sum(each_list) > 1:
                self.reward = -5000
                reward_threshold = 1
                return self.reward, reward_threshold, self.resources
        i = 0
        for i in range(len(self.current_task)):
            decision_variable = offloading_decision[i]
            energy_consumed = self.energy_consumption(i, t, decision_variable, previous_task, self.resources)
            if parameter['residual_power'][i] > 0:
                parameter['residual_power'][i] -= energy_consumed
            else:
                parameter['residual_power'][i] = 0
            delay_calculated, offloading_time = self.delay_calculation(i, t, decision_variable, previous_task, self.task_time,self.resources)
            self.task_energy_consumption[i].append(energy_consumed)
            self.task_time[i].append(delay_calculated)
            self.offloading_time[i].append(offloading_time)
        ###############################################################################################################
        energy_values = self.check_device_energy(parameter['residual_power'])
        energy_violation = (energy_values < parameter['energy_threshold']).any()

        delay_violation = any(max(sublist) > parameter['application_delay'] for sublist in self.task_time)

        if energy_violation & delay_violation:
            print('energy----AND-----delay voilated')
            reward_e = 100
            reward_d = -100
            reward_threshold = 1
            voilated_array_energy = []
            for k in energy_values:
                if k < parameter['energy_threshold']:
                    voilated_value = k - parameter['energy_threshold']
                    voilated_array_energy.append(voilated_value)
            reward_e = sum(voilated_array_energy) * (reward_e)

            voilated_array_delay = []
            for sublist in self.task_time:
                if max(sublist) > parameter['application_delay']:
                    voilated_value = max(sublist) - parameter['application_delay']
                    voilated_array_delay.append(voilated_value)
            reward_d = sum(voilated_array_delay) * (reward_d)
            self.reward = (reward_e + reward_d)
            self.reward = -100
            energy = self.task_energy_consumption
            time = self.task_time
            total_time = sum([item[t] for item in time])
            average_energy = sum([item[t] for item in energy])
            self.average_time = (total_time) / (self.user_devices)
            transmission_time = self.offloading_time
            total_transmission_time = sum([item[t] for item in transmission_time])
            self.average_transmission_time = (total_transmission_time)/self.user_devices          
            self.average_energy = average_energy / (self.user_devices)
            return self.reward, reward_threshold,self.resources

        elif delay_violation:
            print("delay violated")
            self.reward = -100
            reward_threshold = 1
            voilated_array_delay = []
            for sublist in self.task_time:
                if max(sublist) > parameter['application_delay']:
                    voilated_value = max(sublist) - parameter['application_delay']
                    voilated_array_delay.append(voilated_value)
            self.reward = sum(voilated_array_delay) * (self.reward)
            self.reward = -100
            energy = self.task_energy_consumption
            time = self.task_time
            transmission_time = self.offloading_time
            total_transmission_time = sum([item[t] for item in transmission_time])
            self.average_transmission_time = (total_transmission_time)/self.user_devices  
            total_time = sum([item[t] for item in time])
            average_energy = sum([item[t] for item in energy])
            self.average_time = (total_time) / (self.user_devices)
            self.average_energy = average_energy / (self.user_devices)
            return self.reward, reward_threshold, self.resources

        elif energy_violation:
            print("energy violated")
            self.reward = 100
            reward_threshold = 1
            voilated_array_energy = []
            for k in energy_values:
                if k < parameter['energy_threshold']:
                    voilated_value = k - parameter['energy_threshold']
                    voilated_array_energy.append(voilated_value)
            self.reward = sum(voilated_array_energy) * (self.reward)
            self.reward = -100
            energy = self.task_energy_consumption
            time = self.task_time
            transmission_time = self.offloading_time
            total_transmission_time = sum([item[t] for item in transmission_time])
            self.average_transmission_time = (total_transmission_time)/self.user_devices              
            total_time = sum([item[t] for item in time])
            average_energy = sum([item[t] for item in energy])
            self.average_time = (total_time) / (self.user_devices)
            self.average_energy = average_energy / (self.user_devices)
            return self.reward, reward_threshold, self.resources
        else:
            self.reward = 100
            reward_threshold = 0
            energy = self.task_energy_consumption
            time = self.task_time
            total_time = sum([item[t] for item in time])
            average_energy = sum([item[t] for item in energy])
            transmission_time = self.offloading_time
            total_transmission_time = sum([item[t] for item in transmission_time])
            self.average_transmission_time = (total_transmission_time)/self.user_devices              
            self.average_time = (total_time) / (self.user_devices)
            self.average_energy = average_energy / (self.user_devices)
            # weighted_reward_e, weighted_reward_t = self.reward_scaler.scale(time, energy, t, self.user_devices)
            # current_reward = (0.1 * (1 / weighted_reward_e)) + (0.9 * (1 / weighted_reward_t))
            # self.reward = (current_reward ) + self.reward
            # current_reward = (( 0.5 * self.average_time) + (0.5 * self.average_energy)) #* 100
            # Assuming energy and delay are calculated at each step, and w_energy + w_delay = 1
            w_energy = 0.5
            w_delay = 0.5

            # Update max_energy and max_delay if current values are greater
            self.max_energy = max(self.max_energy, self.average_energy)
            self.max_delay_norm = max(self.max_delay_norm, self.average_time)

            energy_norm = self.average_energy / self.max_energy
            delay_norm = self.average_time / self.max_delay_norm

            etc = (w_energy * energy_norm + w_delay * delay_norm) 
            reward = etc * 0.01
         
            self.reward = 1/reward
            # self.intial_reward = current_reward

            if not os.path.isdir(text_data_folder):
                # If not, create it
                os.makedirs(text_data_folder)
            file_path = os.path.join(text_data_folder, "ETC_MDTORA.txt")
            with open(file_path, 'a') as f:
                f.write(f"current_reward :  {np.round(etc, 2)}, T = {t}\n")

            file_path = os.path.join(text_data_folder, "energy_reward.txt")
            with open(file_path, 'a') as f:
                f.write(f"energy_reward :  {np.round(self.average_energy, 4)}, T = {t}\n")
            file_path = os.path.join(text_data_folder, "delay_reward.txt")
            with open(file_path, 'a') as f:
                f.write(f"delay_reward :  {np.round(self.average_time, 4)}, T = {t}\n")

            return self.reward, reward_threshold, self.resources

    def offload_local(self, t):
        i = 0
        offloading_decision_local = [0] * len(self.current_task)
        resources_local = [self.max_freq_local] * len(self.current_task)
        for offloaded_value in self.completed_task_local:
            offloaded_value.append(0)
        for i in range(len(self.current_task)):
            decision_variable = offloading_decision_local[i]
            energy_consumed = self.energy_consumption(i, t, decision_variable, self.completed_task_local, resources_local)
            if self.residual_energy_local[i] > 0:
                self.residual_energy_local[i] -= energy_consumed
            else:
                self.residual_energy_local[i] = 0
            delay_calculated ,offloading_time_local= self.delay_calculation(i, t, decision_variable, self.completed_task_local,
                                                      self.task_time_all_local, resources_local)
            self.task_energy_consumption_all_local[i].append(energy_consumed)
            self.task_time_all_local[i].append(delay_calculated)
            self.offloading_time_local[i].append(offloading_time_local)
        energy_all_local = self.task_energy_consumption_all_local
        time_all_local = self.task_time_all_local
        energy_values = self.check_device_energy(self.residual_energy_local)
        energy_violation = (energy_values < parameter['energy_threshold']).any()
        transmission_time_l = self.offloading_time_local
        # if energy_violation:
        #      self.average_time_all_local = 0
        #      self.average_energy_all_local = 0
        # else:
        total_transmission_time_l = sum([item[t] for item in transmission_time_l])
        self.average_transmission_time_l = (total_transmission_time_l)/self.user_devices  
        total_time_all_local = sum([item[t] for item in time_all_local])
        average_energy_all_local = sum([item[t] for item in energy_all_local])
        self.average_time_all_local = total_time_all_local / self.user_devices
        self.average_energy_all_local = average_energy_all_local / self.user_devices
        # ETC_local = (( 0.5 * self.average_time_all_local) + (0.5 * self.average_energy_all_local))
        w_energy = 0.5
        w_delay = 0.5

        # Update max_energy and max_delay if current values are greater
        self.max_energy_local = max(self.max_energy_local, self.average_energy_all_local)
        self.max_delay_local = max(self.max_delay_local, self.average_time_all_local)

        energy_norm = self.average_energy_all_local / self.max_energy_local
        delay_norm = self.average_time_all_local / self.max_delay_local

        ETC_local = (w_energy * energy_norm + w_delay * delay_norm) 
        if not os.path.isdir(text_data_folder):
            # If not, create it
            os.makedirs(text_data_folder)
        file_path = os.path.join(text_data_folder, "ETC_local.txt")
        with open(file_path, 'a') as f:
            f.write(f"ETC_local :  {np.round(ETC_local, 2)}, T = {t}\n")
        return offloading_decision_local

    def offload_edge(self, t):
        i = 0
        if t == 0:
            resources_edge = [self.max_freq_local] * len(self.current_task)
            resources_edge_sending = [x / parameter['local_device_freq'] for x in resources_edge]

        else:
            resources_edge = [self.max_freq] * len(self.current_task)
            resources_edge_sending = [x / parameter['edge_server_freq'] for x in resources_edge]
        if t == 0:
            offloading_decision_edge = [0] * len(self.current_task)
            for offloaded_value in self.completed_task_edge:
                offloaded_value.append(0)
        else:
            offloading_decision_edge = []
            for _ in range(len(self.current_task)):
                offloading_value = random.choice(self.servers_ID)
                offloading_decision_edge.append(offloading_value)
                self.completed_task_edge[_].append(offloading_value)
            # print("offloading_decision_edge", offloading_decision_edge)
            from collections import Counter
            task_count_per_server = Counter(offloading_decision_edge)
            cpu_resource_per_task = {}
            for server_id, task_count in task_count_per_server.items():
                cpu_resource_per_task[server_id] = self.max_freq / task_count
            cpu_resources_for_tasks = []
            for server_id in offloading_decision_edge:
                cpu_resources_for_tasks.append(cpu_resource_per_task[server_id])
            
            cpu_resource_per_task, task_count_per_server
            resources_edge = cpu_resources_for_tasks
        for i in range(len(self.current_task)):
            decision_variable = offloading_decision_edge[i]
            energy_consumed = self.energy_consumption(i, t, decision_variable, self.completed_task_edge,resources_edge)
            if self.residual_energy_edge[i] > 0:
                self.residual_energy_edge[i] -= energy_consumed
            else:
                self.residual_energy_edge[i] = 0
            delay_calculated, offloading_time_edge  = self.delay_calculation(i, t, decision_variable, self.completed_task_edge,
                                                      self.task_time_all_edge, resources_edge)
            self.task_energy_consumption_all_edge[i].append(energy_consumed)
            self.task_time_all_edge[i].append(delay_calculated)
            self.offloading_edge[i].append(offloading_time_edge)
        energy_values = self.check_device_energy(self.residual_energy_edge)
        energy_violation = (energy_values < parameter['energy_threshold']).any()
        # if energy_violation:
        #     self.average_time_all_edge = 0
        #     self.average_energy_all_edge = 0
        # else:
        energy_all_edge = self.task_energy_consumption_all_edge
        time_all_edge = self.task_time_all_edge
        transmission_time_e = self.offloading_edge
        total_transmission_time_e = sum([item[t] for item in transmission_time_e])
        self.average_transmission_time_e = (total_transmission_time_e)/self.user_devices  
        total_time_all_edge = sum([item[t] for item in time_all_edge])
        average_energy_all_edge = sum([item[t] for item in energy_all_edge])
        self.average_time_all_edge = total_time_all_edge / self.user_devices
        self.average_energy_all_edge = average_energy_all_edge / self.user_devices
        # ETC_edge = (( 0.5 * self.average_time_all_edge) + (0.5 * self.average_energy_all_edge))
        w_energy = 0.5
        w_delay = 0.5

        # Update max_energy and max_delay if current values are greater
        self.max_energy_edge = max(self.max_energy_edge, self.average_energy_all_edge)
        self.max_delay_edge = max(self.max_delay_edge, self.average_time_all_edge)

        energy_norm = self.average_energy_all_edge / self.max_energy_edge
        delay_norm = self.average_time_all_edge / self.max_delay_edge

        ETC_edge = (w_energy * energy_norm + w_delay * delay_norm) 
        if not os.path.isdir(text_data_folder):
            # If not, create it
            os.makedirs(text_data_folder)
        file_path = os.path.join(text_data_folder, "ETC_edge.txt")
        with open(file_path, 'a') as f:
            f.write(f"ETC_local :  {np.round(ETC_edge, 2)}, T = {t}\n")

        return offloading_decision_edge, resources_edge_sending

    def offload_random(self, t, offloading_decision):
        if t == 0 :
            resources_random = [self.max_freq_local] * len(self.current_task)
            offloading_decision_random = offloading_decision
            for offloaded_value in self.completed_task_random:
                offloaded_value.append(0)
        else:
            resources_random = [self.max_freq] * len(self.current_task)
            offloading_decision_random = []
            for _ in range(len(self.current_task)):
                # offloading_value = random.choice([0] + self.servers_ID)
                # offloading_decision_random.append(offloading_value)
                # self.completed_task_random[_].append(offloading_value)
                if _ == 0:
                    offloading_value = 0
                    offloading_decision_random.append(offloading_value)
                    self.completed_task_random[_].append(offloading_value)
                else:
                    offloading_value = random.choice(self.servers_ID)
                    offloading_decision_random.append(offloading_value)
                    self.completed_task_random[_].append(offloading_value)

            # Calculate CPU Resources Per Task for Edge Servers
            from collections import Counter
            task_count_per_server = Counter([d for d in offloading_decision_random if d != 0])
            cpu_resource_per_task = {server_id: self.max_freq / task_count for server_id, task_count in task_count_per_server.items()}

            # Update resources_random with the calculated CPU resources
            for i in range(len(self.current_task)):
                server_id = offloading_decision_random[i]
                if server_id == 0:
                    resources_random[i] = self.max_freq_local  # Full frequency for local execution
                else:
                    resources_random[i] = cpu_resource_per_task.get(server_id, self.max_freq)  # Allocated frequency for edge serv

        for i in range(len(self.current_task)):
            decision_variable = offloading_decision_random[i]
            energy_consumed = self.energy_consumption(i, t, decision_variable, self.completed_task_random, resources_random)
            if self.residual_energy_random[i] > 0:
                self.residual_energy_random[i] -= energy_consumed
            else:
                self.residual_energy_random[i] = 0
            delay_calculated, offloading_time_random = self.delay_calculation(i, t, decision_variable, self.completed_task_random,
                                                      self.task_time_random, resources_random)
            self.task_energy_consumption_random[i].append(energy_consumed)
            self.task_time_random[i].append(delay_calculated)
            self.offloading_random[i].append(offloading_time_random)
        energy_random = self.task_energy_consumption_random
        time_random = self.task_time_random
        transmission_time_r = self.offloading_random
        energy_values = self.check_device_energy(self.residual_energy_random)
        energy_violation = (energy_values < parameter['energy_threshold']).any()
        # if energy_violation:
        #     self.average_time_random = 0
        #     self.average_energy_random = 0
        # else:
        total_transmission_time_r = sum([item[t] for item in transmission_time_r])
        self.average_transmission_time_r = (total_transmission_time_r)/self.user_devices  
        total_time_random = sum([item[t] for item in time_random])
        average_energy_random = sum([item[t] for item in energy_random])
        self.average_time_random = total_time_random / self.user_devices
        self.average_energy_random = average_energy_random / self.user_devices
        w_energy = 0.5
        w_delay = 0.5

        # Update max_energy and max_delay if current values are greater
        self.max_energy_random = max(self.max_energy_random, self.average_energy_random)
        self.max_delay_random = max(self.max_delay_random, self.average_time_random)

        energy_norm = self.average_energy_random / self.max_energy_random
        delay_norm = self.average_time_random / self.max_delay_random

        ETC_random = (w_energy * energy_norm + w_delay * delay_norm) 
        if not os.path.isdir(text_data_folder):
            # If not, create it
            os.makedirs(text_data_folder)
        file_path = os.path.join(text_data_folder, "ETC_random.txt")
        with open(file_path, 'a') as f:
            f.write(f"ETC_local :  {np.round(ETC_random, 2)}, T = {t}\n")
        return offloading_decision_random
    # '''
    def return_useful_data(self):
        return (self.average_time, self.average_energy, self.average_time_all_local, self.average_energy_all_local,
                self.average_time_all_edge,
                self.average_energy_all_edge, self.average_time_random, self.average_energy_random,
                parameter['residual_power'], self.residual_energy_local,
                self.residual_energy_edge, self.residual_energy_random, self.average_transmission_time ,self.average_transmission_time_l, self.average_transmission_time_e,self.average_transmission_time_r)

    def observe(self, t):
        self.very_next_state = []
        parameter['channel_gain'] = np.zeros(self.user_devices)
        for i in range(self.user_devices):
            self.channel_gain = np.random.randint(self.min_channel_gain, self.max_channel_gain)
            self.device_power_value = parameter['residual_power'][i]
            # self.task_state = np.round(self.user_info[i]['GNN_state'][t].item(), 2)
            parameter['channel_gain'][i] = self.channel_gain
            # Normalize the channel gain
            channel_gain_normalized = np.round((self.channel_gain - (-50)) / ((-8) - (-50)), 2)
            # Normalize the device power value
            device_power_value_normalized = np.round((self.device_power_value - 0) / (self.max_device_power - 0), 2)

            self.very_next_state.extend([channel_gain_normalized, device_power_value_normalized]) #self.task_state])

        # Normalize the number of offloaded tasks
        offloaded_tasks_normalized = np.round((len(self.offloaded_tasks) - 0) / (100 - 0), 2)

        self.very_next_state.extend([offloaded_tasks_normalized])
        return self.very_next_state

    def is_done(self):
        if len(self.offloaded_tasks) == parameter['all_tasks']:
            # print("All Offloaded")
            # self.reward = 100
            return self.reward, True
        return self.reward, False

    def uplink_data_rate(self, p, h, bandwidth, sigma):  # power, channel gain
        h_value = 10 ** (h / 10)  # convert to linear scale from db
        h = h_value *1000
        data_rate_uplink = bandwidth * math.log2(1 + (p * h) / sigma ** 2)
        data_rate_uplink = parameter['uplink_datarate']    #3Mbps
        return data_rate_uplink  # bps, divide by 10**6 = Mbps

    def downlink_data_rate(self, p, h, bandwidth, sigma):
        h_d_value = 10 ** (h / 10)
        h = h_d_value * 1000
        data_rate_downlink = bandwidth * math.log2(1 + (p * h) / sigma ** 2)
        data_rate_downlink = parameter['downlink_datarate'] #3Mbps
        return data_rate_downlink

    def offloading_transmission_time(self, output, data_rate_uplink):  # output
        uplink_transmission_time = ((output * 8) / data_rate_uplink) # here the data is in bytes so convert to bits and datarate is in Mbps    
        return uplink_transmission_time     #  output is seconds

    def downloading_transmission_time(self, output, data_rate_downlink):  
        downlink_transmission_time = ((output * 8) / data_rate_downlink) # here the data is in bytes so convert to bits and datarate is in Mbps    
        return downlink_transmission_time   #  output is seconds

    def downlink_transmission_energy(self, p, downlink_transmission_time):
        downloading_energy = p * downlink_transmission_time # P is watt and time is seconds so energy is in Joules
        return downloading_energy   

    def uplink_transmission_energy(self, p, uplink_transmission_time):
        uploading_energy = p * uplink_transmission_time # P is watt and time is seconds so energy is in Joules
        return uploading_energy  
    def dependency_check(self, i, t):
        dep = self.user_info[i]['new_adj'][t]
        if len(dep) != 0:
            Pre = 1
        else:
            Pre = 0
        return Pre

    def local_device(self, i, t, resource_value):

        exec_time = (self.user_info[i]['workload'][t] / resource_value[i]) # this value is seconds
        energy_consum = self.k * self.user_info[i]['workload'][t] * (resource_value[i] ** 2)  # this value is in Joules
        energy_consum = energy_consum 
        return exec_time, energy_consum

    def check_device_energy(self, residual_energy):
        device_residual_energy = (residual_energy * 100) / parameter['residual_power_max']
        # print(device_residual_energy)
        return device_residual_energy

    def edge_server(self, i, t, resource_value):

        exec_time = (self.user_info[i]['workload'][t] / (resource_value[i])) # this value is seconds
        energy_consum = self.k * self.user_info[i]['workload'][t] * resource_value[i] ** 2 # this value is in Joules
        energy_consum = energy_consum #* 1000 # convert to mJ
        return exec_time, energy_consum

    def energy_consumption(self, i, t, decision_variable, completed_task, resource_value):
        total_server_energy = []
        total_uplink_energy = []
        total_downlink_energy = []

        Pre = self.dependency_check(i, t)
        if decision_variable == 0:

            exec_time, total_energy, = self.local_device(i, t, resource_value)
            return total_energy

        else:
            if Pre == 1:
                length_of_previous_tasks = len(self.user_info[i]['new_adj'][t])
                values_of_previous_tasks = self.user_info[i]['new_adj'][t]
                for j in range(length_of_previous_tasks):
                    if completed_task[i][values_of_previous_tasks[j]] == 0:
                        uplink_data = self.uplink_data_rate(parameter['uplink_power'], parameter['channel_gain'][i], parameter['W'],
                                                            parameter['sigma'])
                        uplink_time = (self.offloading_transmission_time(
                            self.user_info[i]['data'][values_of_previous_tasks[j]],
                            uplink_data)) 
                        uplink_energy = self.uplink_transmission_energy(parameter['uplink_power'], uplink_time)
                        total_uplink_energy.append(uplink_energy)
                        # total_energy = sum(total_uplink_energy)
                        # self.task_energy_consumption[i].append(total_energy)
                        # parameter['residual_power'][i] -= sum(total_uplink_energy)#/3600)
                    # elif self.completed_task[values_of_previous_tasks[j]] == 1:
                    else:
                        total_energy_server = 0.00001
                        total_server_energy.append(total_energy_server)
                        # total_energy = sum(total_server_energy)
                total_energy_combined = sum(total_uplink_energy) + sum(total_server_energy)

                return total_energy_combined

    def delay_calculation(self, i, t, decision_variable, completed_task, previous_task_time, resource_value):

        start_time_current = []
        offload_time = 0  # Initialize offload_time
        Pre = self.dependency_check(i, t)
        if decision_variable == 0:

            if Pre == 0:
                exec_time, _ = self.local_device(i, t, resource_value)
                return exec_time, offload_time
            elif Pre == 1:
                length_of_previous_tasks = len(self.user_info[i]['new_adj'][t])
                values_of_previous_tasks = self.user_info[i]['new_adj'][t]
                for j in range(length_of_previous_tasks):
                    if completed_task[i][values_of_previous_tasks[j]] == 0:
                        start_time = previous_task_time[i][values_of_previous_tasks[j]]
                        start_time_current.append(start_time)
                    else:
                        downlink_data = self.downlink_data_rate(parameter['downlink_power'], parameter['channel_gain'][i],
                                                                parameter['downlink_bandwidth'],
                                                                parameter['sigma'])
                        start_time = previous_task_time[i][values_of_previous_tasks[j]] + self.downloading_transmission_time(
                            self.user_info[i]['data'][values_of_previous_tasks[j]], downlink_data)
                        start_time = previous_task_time[i][values_of_previous_tasks[j]] + 0
                        start_time_current.append(start_time)
                start_time_task = max(start_time_current)
                completion_time, _ = self.local_device(i, t, resource_value)
                self.completion_time = start_time_task + completion_time
                return self.completion_time, offload_time

        else:

            if Pre == 0:
                self.completion_time, _ = self.edge_server(i, t, resource_value)
                return self.completion_time, offload_time

            elif Pre == 1:
                length_of_previous_tasks = len(self.user_info[i]['new_adj'][t])
                values_of_previous_tasks = self.user_info[i]['new_adj'][t]
                for j in range(length_of_previous_tasks):
                    if completed_task[i][values_of_previous_tasks[j]] == 0:
                        uplink_data = self.uplink_data_rate(parameter['uplink_power'], parameter['channel_gain'][i], parameter['W'],
                                                            parameter['sigma'])
                        offload_time = self.offloading_transmission_time(self.user_info[i]['data'][values_of_previous_tasks[j]], uplink_data)
                        start_time = previous_task_time[i][values_of_previous_tasks[j]] + offload_time
                        start_time_current.append(start_time)
                    else:
                        start_time = previous_task_time[i][values_of_previous_tasks[j]]
                        start_time_current.append(start_time)
                start_time_task = max(start_time_current)
                self.completion_time, _ = self.edge_server(i, t, resource_value)
                self.completion_time = start_time_task + self.completion_time
                return self.completion_time, offload_time
