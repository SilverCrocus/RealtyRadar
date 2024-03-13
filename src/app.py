import streamlit as st
import plotly.express as px
import pandas as pd

# Custom CSS styling
st.markdown("""
<style>
body {
    background-color: #1c1c1c;
    color: white;
}
.sidebar .sidebar-content {
    background-color: #333333;
    padding: 20px;
    border-radius: 5px;
}
h1 {
    color: white;
    font-size: 36px;
    margin-bottom: 20px;
}
.sidebar-title {
    color: white;
    font-size: 24px;
    margin-bottom: 10px;
}
.stSelectbox {
    margin-bottom: 10px;
    color: white;
}
.stButton {
    background-color: #ffd700;
    color: black;
}
.stButton:hover {
    background-color: #ffdf80;
}
</style>
""", unsafe_allow_html=True)

# Streamlit app layout
st.title('📊 Housing Data Dashboard')

# Load Parquet file
data = pd.read_parquet('data/cleaned_v1.6.parquet')

# Sidebar for variable selection
st.sidebar.title('⚙️ Settings')

# Collapsible section for variable types in the sidebar
with st.sidebar.expander('📊 Variable Types'):
    variable_types = {}
    for var in data.columns:
        if data[var].dtype == 'object':
            var_type = 'Categorical'
        else:
            var_type = 'Numeric'
        variable_types[var] = var_type
        st.write(f'- {var}: {var_type}')

selected_vars = st.sidebar.multiselect('🔍 Select variables', list(variable_types.keys()), format_func=lambda x: str(x))

# Filter data based on selected variables
filtered_data = data[selected_vars]

# Graph type selection
graph_type = st.sidebar.selectbox('📈 Select graph type', ['Histogram', 'Scatter Plot', 'Box Plot', 'Violin Plot'])

# Key variable selection for color coding
key_variable = st.sidebar.selectbox('🎨 Select key variable for color coding', ['None'] + selected_vars, format_func=lambda x: str(x))

# Create interactive visualizations
if len(selected_vars) >= 1:
    if graph_type == 'Histogram':
        # Histogram for a single variable
        fig = px.histogram(filtered_data, x=selected_vars[0], color=key_variable if key_variable != 'None' else None)
        fig.update_layout(xaxis_title=selected_vars[0], yaxis_title='Count', template='plotly_dark')
        st.plotly_chart(fig)
    elif graph_type == 'Scatter Plot' and len(selected_vars) >= 2:
        # Scatter plot for two variables
        fig = px.scatter(filtered_data, x=selected_vars[0], y=selected_vars[1], color=key_variable if key_variable != 'None' else None)
        fig.update_layout(xaxis_title=selected_vars[0], yaxis_title=selected_vars[1], template='plotly_dark')
        st.plotly_chart(fig)
    elif graph_type == 'Box Plot':
        # Box plot for selected variables
        fig = px.box(filtered_data, y=selected_vars, color=key_variable if key_variable != 'None' else None)
        fig.update_layout(yaxis_title='Value', template='plotly_dark')
        st.plotly_chart(fig)
    elif graph_type == 'Violin Plot':
        # Violin plot for selected variables
        fig = px.violin(filtered_data, y=selected_vars, color=key_variable if key_variable != 'None' else None)
        fig.update_layout(yaxis_title='Value', template='plotly_dark')
        st.plotly_chart(fig)
else:
    # Display a message if no variables are selected
    st.write('Please select at least one variable for visualization.')

# Display summary statistics
if len(selected_vars) > 0:
    st.write('## 📊 Summary Statistics')
    st.write(filtered_data.describe())

# House price prediction
st.header('🏠 House Price Prediction')
# Add input fields for prediction features
# Load and use the pre-trained model for predictions
# Display the predicted house prices