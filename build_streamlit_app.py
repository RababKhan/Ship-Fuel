
import streamlit as st
import torch
import pandas as pd
import numpy as np
import torch.nn as nn

from ship_fuel_env import ShipFuelEnv

class DQNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(DQNetwork, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim)
        )

    def forward(self, x):
        return self.fc(x)

# Load data and environment
df = pd.read_csv("ship_fuel_efficiency.csv")
env = ShipFuelEnv(df)

# Load trained model
state_dim = 6
action_dim = env.action_space.n
policy_net = DQNetwork(state_dim, action_dim)
policy_net.load_state_dict(torch.load("dqn_shipfuel_model.pth", map_location=torch.device('cpu')))
policy_net.eval()

# Streamlit UI
st.title("Ship Fuel Optimization Assistant")

ship_type = st.selectbox("Select Ship Type", sorted(df["ship_type"].unique()))
route_distance = st.slider("Route Distance (nautical miles)", 50.0, 500.0, 100.0)
weather = st.selectbox("Weather Condition", sorted(df["weather_conditions"].unique()))
efficiency = st.slider("Engine Efficiency (%)", 70.0, 100.0, 85.0)

if st.button("Get Fuel Recommendation"):
    row = {
        "ship_type": ship_type,
        "route_id": "Generic-Route",
        "month": "January",
        "distance": route_distance,
        "fuel_type": "",
        "fuel_consumption": 0.0,
        "CO2_emissions": 0.0,
        "weather_conditions": weather,
        "engine_efficiency": efficiency
    }

    state = env.encode_state(pd.Series(row))
    with torch.no_grad():
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        action = torch.argmax(policy_net(state_tensor)).item()

    recommended_fuel = env.fuel_types[action]
    predicted_consumption = max(1000.0, 5000.0 - efficiency * 30 + route_distance * 2)
    predicted_emission = predicted_consumption * 2.8

    st.subheader("Recommendation")
    st.write(f"**Recommended Fuel Type:** {recommended_fuel}")
    st.metric("Predicted Fuel Consumption (liters)", f"{predicted_consumption:.2f}")
    st.metric("Estimated CO₂ Emissions (kg)", f"{predicted_emission:.2f}")
