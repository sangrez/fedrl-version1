from user_info import*

def calculate_reward(self, delay_weight=0.5, energy_weight=0.5, penalty_weight=1.0, energy_violation=False, delay_violation=False):
        
        # Normalize delays and energies
        normalized_delays = [delay / parameter['max_delay'] for sublist in self.task_time for delay in sublist]
        normalized_energies = [energy / parameter['max_energy'] for sublist in self.task_energy_consumption for energy in sublist]
        
        # Calculate average normalized delay and energy
        average_normalized_delay = sum(normalized_delays) / len(normalized_delays)
        average_normalized_energy = sum(normalized_energies) / len(normalized_energies)
        
        # Initialize reward
        reward = 0.0
        
        # Penalize for delay and energy
        reward -= (delay_weight * average_normalized_delay + energy_weight * average_normalized_energy)
        
        # Add penalties for violations
        if energy_violation:
            reward -= penalty_weight * 10  # Arbitrary penalty value for energy violation
        if delay_violation:
            reward -= penalty_weight * 10  # Arbitrary penalty value for delay violation
        
        return reward