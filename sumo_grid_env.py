import gymnasium as gym
from gymnasium import spaces
import traci # SUMO's Traffic Control Interface
import numpy as np

class SumoGridEnv(gym.Env):
    def __init__(self, sumo_config="C:/Users/lucas/OneDrive/Documents/Coding/Traffic/network/fremont.sumocfg"):
        # Initialize SUMO
        self.sumoConfig = sumo_config
        self.sumoCmd = ["sumo", "-c", self.sumoConfig]
        self.sumo_running = False  # Initialize the SUMO running flag

        # Start SUMO to retrieve information on intersections and lighta
        traci.start(self.sumoCmd)
        self.sumo_running = True
        self.traffic_light_ids = traci.trafficlight.getIDList()
        self.lane_ids = traci.lane.getIDList()
        # Done retrieving information
        traci.close()
        self.sumo_running = False

        total_traffic_lights = len(self.traffic_light_ids)
        num_queue_lengths = len(self.lane_ids)
        self.current_step = 0  # Initialize the step counter

        # Define the action space
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(total_traffic_lights,), dtype=np.float32)

        # Define the observation space (queue lengths and traffic light states)
        total_observation_space_size = num_queue_lengths + total_traffic_lights
        self.observation_space = spaces.Box(low=np.float32(0), 
                                            high=np.float32(np.inf), 
                                            shape=(total_observation_space_size,), 
                                            dtype=np.float32)
        
    def calculate_reward(self, queue_lengths):
        """
        Calculate the reward, which is the negative sum of the number of cars waiting at all intersections.

        :param queue_lengths: A list containing the number of cars waiting at each traffic light.
        :return: A float representing the calculated reward.
        """
        # Testing an inverse reward function
        reward = -sum(1.0 / (1 + length) for length in queue_lengths)
        # Penalty for each emergency stop
        emergency_stop_penalty = -10
        # Check for emergency stops
        reward -= emergency_stop_penalty * traci.simulation.getEmergencyStoppingVehiclesNumber()
        return reward
    
    def check_if_done(self):
        """
        Check if the episode should be terminated. This could be based on various criteria,
        such as elapsed time or specific events in the simulation.

        :return: A boolean indicating whether the episode is done.
        """
        # Example termination condition: End the episode after a fixed number of steps
        max_steps = 1000  # Define the maximum number of steps per episode
        if self.current_step >= max_steps:
            return True
        else:
            return False

    def step(self, action):
        # Ensure SUMO is running
        if not self.sumo_running:
            traci.start(self.sumoCmd)
            self.sumo_running = True

        # Process the action for each traffic light
        for i, light_id in enumerate(self.traffic_light_ids):
            # Discretize the action value: round to convert to either 0 or 1
            discrete_action = int(round(action[i]))
            if discrete_action == 1:  # If the action is to change the phase
                current_phase = traci.trafficlight.getPhase(light_id)
                total_phases = len(traci.trafficlight.getAllProgramLogics(light_id)[0].getPhases())
                new_phase = (current_phase + 1) % total_phases
                traci.trafficlight.setPhase(light_id, new_phase)
            
        # Advance the SUMO simulation by one step
        traci.simulationStep()

        # Gather new state information
        new_queue_lengths = [traci.lane.getLastStepHaltingNumber(lane) for lane in self.lane_ids]
        new_traffic_light_states = [traci.trafficlight.getPhase(light) for light in self.traffic_light_ids]
        # Combine queue lengths and traffic light states into the new observation
        observation = np.concatenate([
            np.array(new_queue_lengths, dtype=np.float32),
            np.array(new_traffic_light_states, dtype=np.float32)
        ])        
        # Calculate the reward
        reward = self.calculate_reward(new_queue_lengths)
        # Check if the episode is done
        done = self.check_if_done()
        # For truncated, you might need additional logic depending on your environment
        # For example, if you have a maximum number of steps:
        max_steps = 1000
        truncated = self.current_step >= max_steps
        # Increment the step counter
        self.current_step += 1

        return observation, reward, done, truncated, {}

    def reset(self, seed=None):
        if self.sumo_running:
            traci.close()  # Close the current TraCI session if it's running
            self.sumo_running = False

        traci.start(self.sumoCmd)  # Start SUMO with the provided command
        self.sumo_running = True
        self.current_step = 0  # Reset the step counter
        self.traffic_light_ids = traci.trafficlight.getIDList()
        self.lane_ids = traci.lane.getIDList()

        # Generate the initial observation based on the current state of the simulation
        initial_queue_lengths = [traci.lane.getLastStepHaltingNumber(lane) for lane in self.lane_ids]
        initial_traffic_light_states = [traci.trafficlight.getPhase(light) for light in self.traffic_light_ids]   
        # Convert the observation components to a NumPy array and concatenate them
        observation = np.concatenate([
            np.array(initial_queue_lengths, dtype=np.float32),
            np.array(initial_traffic_light_states, dtype=np.float32)
        ])
        # Return a tuple of the observation and the info dictionary
        return observation, {}
    
    def render(self, mode='human'):
        # Optional: Implement rendering for visualization
        pass

    def close(self):
        # Stop SUMO if it is running
        if self.sumo_running:
            traci.close()
            self.sumo_running = False