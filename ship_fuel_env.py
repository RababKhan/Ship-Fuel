
import gym
from gym import spaces
import numpy as np
import pandas as pd
import random

class ShipFuelEnv(gym.Env):
    def __init__(self, df):
        super(ShipFuelEnv, self).__init__()

        self.df = df.copy().reset_index(drop=True)
        self.current_index = 0

        # Define action space (choose fuel type)
        self.fuel_types = ['HFO', 'Diesel']
        self.action_space = spaces.Discrete(len(self.fuel_types))

        # Define observation space (encoded features)
        self.ship_types = ['Oil Service Boat', 'Fishing Trawler', 'Surfer Boat', 'Tanker Ship']
        self.route_ids = ['Warri-Bonny', 'Port Harcourt-Lagos', 'Lagos-Apapa', 'Escravos-Lagos']
        self.months = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December']
        self.weather_conditions = ['Stormy', 'Moderate', 'Calm']

        # Observation space: [ship_type, route_id, month, weather, distance, engine_efficiency]
        low = np.array([0] * 4 + [0.0, 0.0])
        high = np.array([len(self.ship_types)-1, len(self.route_ids)-1,
                         len(self.months)-1, len(self.weather_conditions)-1,
                         df['distance'].max(), df['engine_efficiency'].max()])
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)

    def encode_state(self, row):
        return np.array([
            self.ship_types.index(row['ship_type']),
            self.route_ids.index(row['route_id']),
            self.months.index(row['month']),
            self.weather_conditions.index(row['weather_conditions']),
            row['distance'],
            row['engine_efficiency']
        ], dtype=np.float32)

    def reset(self):
        self.current_index = random.randint(0, len(self.df) - 1)
        row = self.df.iloc[self.current_index]
        return self.encode_state(row)

    def step(self, action):
        fuel_choice = self.fuel_types[action]
        row = self.df.iloc[self.current_index]

        # Penalty if wrong fuel selected
        if row['fuel_type'] != fuel_choice:
            fuel_consumption = row['fuel_consumption'] * 1.2  # simulate inefficiency
            CO2_emissions = row['CO2_emissions'] * 1.3
        else:
            fuel_consumption = row['fuel_consumption']
            CO2_emissions = row['CO2_emissions']

        # Reward: minimize both
        reward = - (0.6 * fuel_consumption + 0.4 * CO2_emissions)

        # Move to next sample
        self.current_index = random.randint(0, len(self.df) - 1)
        next_row = self.df.iloc[self.current_index]
        next_state = self.encode_state(next_row)

        done = False  # episodic setting can be added if needed
        info = {}

        return next_state, reward, done, info

    def render(self, mode='human'):
        print("Rendering not implemented")

# Example use:
# df = pd.read_csv("ship_fuel_efficiency.csv")
# env = ShipFuelEnv(df)
# obs = env.reset()
# obs, reward, done, info = env.step(env.action_space.sample())
