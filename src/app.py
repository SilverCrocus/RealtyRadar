import streamlit as st
import plotly.express as px
import pandas as pd

import json

# Initialize an empty list to store the parsed JSON data
data_list = []

# Read the JSON file line by line
with open('data/cleaned_v1.3.json', 'r') as file:
    for line in file:
        try:
            json_data = json.loads(line)
            data_list.append(json_data)
        except json.JSONDecodeError:
            # Skip lines that cannot be parsed as JSON
            continue

# Convert the parsed data into a DataFrame
data = pd.DataFrame(data_list)

# Streamlit app layout
st.title('Housing Data Dashboard')

# Sidebar for variable selection
variables = ['Price', 'Bedrooms', 'Bathrooms', 'Garage', 'Land_Area', 'Floor_Area']
selected_vars = st.sidebar.multiselect('Select variables', variables)

# Filter data based on selected variables
filtered_data = data[selected_vars]

# Create interactive visualizations
if len(selected_vars) == 1:
    # Histogram for a single variable
    fig = px.histogram(filtered_data, x=selected_vars[0])
    st.plotly_chart(fig)
elif len(selected_vars) == 2:
    # Scatter plot for two variables
    fig = px.scatter(filtered_data, x=selected_vars[0], y=selected_vars[1])
    st.plotly_chart(fig)
else:
    # Display a message if more than two variables are selected
    st.write('Please select one or two variables for visualization.')

# Display summary statistics
if len(selected_vars) > 0:
    st.write('Summary Statistics')
    st.write(filtered_data.describe())

# House price prediction
st.header('House Price Prediction')
# Add input fields for prediction features
# Load and use the pre-trained model for predictions
# Display the predicted house prices