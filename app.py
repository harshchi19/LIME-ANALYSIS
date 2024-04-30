import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

# Indian Places dataset
indian_places_data = {
    "Destination": ["Goa", "Jaipur", "Shimla", "Kerala", "Varanasi"],
    "Accommodation Type": ["Hotel", "Resort", "Guest House", "Homestay", "Hostel"],
    "Travel Mode": ["Flight", "Train", "Bus", "Car", "Bike"],
    "Budget": [2000, 3000, 2500, 4000, 1500],
    "Duration": [3, 4, 5, 7, 2]
}

indian_places_df = pd.DataFrame(indian_places_data)

# Sidebar options
st.sidebar.title("Customize Your Itinerary")

# Select destinations
selected_destinations = st.sidebar.multiselect("Select Destinations", indian_places_df["Destination"])

# Create empty lists to store accommodation, travel mode, and budget options
selected_accommodation = []
selected_travel_mode = []
selected_budgets = []

# For each selected destination, add multiselect widgets for accommodation, travel mode, and budget
for destination in selected_destinations:
    selected_accommodation.append(st.sidebar.multiselect(f"Accommodation Type for {destination}", indian_places_df["Accommodation Type"]))
    selected_travel_mode.append(st.sidebar.multiselect(f"Travel Mode for {destination}", indian_places_df["Travel Mode"]))
    selected_budgets.append(st.sidebar.slider(f"Budget for {destination}", min_value=1000, max_value=5000, step=500, value=2500))

# Select duration
selected_duration = st.sidebar.slider("Select Duration (in days)", min_value=1, max_value=10, value=3)

# Prediction button
predict_button = st.sidebar.button("Predict")

# Generate dataset based on sidebar options
@st.cache_data
def generate_dataset(destinations, accommodations, travel_modes, budgets, duration):
    data = {
        "Destination": [],
        "Accommodation Type": [],
        "Travel Mode": [],
        "Budget": [],
        "Duration": []
    }
    for destination, accs, modes, b in zip(destinations, accommodations, travel_modes, budgets):
        for acc, mode in zip(accs, modes):
            data["Destination"].append(destination)
            data["Accommodation Type"].append(acc)
            data["Travel Mode"].append(mode)
            data["Budget"].append(b)
            data["Duration"].append(duration)
    return pd.DataFrame(data)

input_df = generate_dataset(selected_destinations, selected_accommodation, selected_travel_mode, selected_budgets, selected_duration)

if predict_button:
    predicted_attractions = np.random.choice(["Beach", "Fort", "Hill Station", "Temple", "Market"], size=len(selected_destinations))
    st.markdown("## **Predicted Attractions:**")
    for destination, attraction in zip(selected_destinations, predicted_attractions):
        st.write(f"- {destination}: **{attraction}**")


    # Generate feature importance graph
    def generate_feature_importance_graph():
        feature_names = ["Destination", "Accommodation Type", "Travel Mode", "Budget", "Duration"]
        feature_importances = np.random.rand(len(feature_names))  # Replace with actual feature importances
        feature_fig = go.Figure(data=[go.Bar(
            x=feature_names,
            y=feature_importances
        )])
        feature_fig.update_layout(title_text='Feature Importances')
        st.plotly_chart(feature_fig)

    # Generate LIME explanation graph
    def generate_lime_explanation_graph():
        lime_explanation = {
            "Feature": ["Destination", "Accommodation Type", "Travel Mode", "Budget", "Duration"],
            "Contribution": np.random.rand(5)  # Replace with actual LIME explanations
        }
        lime_df = pd.DataFrame(lime_explanation)
        lime_fig = go.Figure(data=[go.Bar(
            x=lime_df["Feature"],
            y=lime_df["Contribution"]
        )])
        lime_fig.update_layout(title_text='LIME Explanation')
        st.plotly_chart(lime_fig)

    # Additional data visualizations
    st.subheader("Additional Data Visualizations")
    
    # Generate and display feature importance graph
    st.subheader("Feature Importances")
    generate_feature_importance_graph()

    # Generate and display LIME explanation graph
    st.subheader("LIME Explanation")
    generate_lime_explanation_graph()

# Display the input itinerary
st.subheader("Your Input Itinerary:")
st.write(input_df)

# Add more unique data visualizations

# Scatter plot showing relationship between budget and duration for different destinations
st.subheader("Relationship Between Budget and Duration")
if predict_button:
    fig = go.Figure()
    for destination, budget in zip(selected_destinations, selected_budgets):
        color = '#%02x%02x%02x' % tuple(np.random.choice(range(256), size=3))
        fig.add_trace(go.Scatter(
            x=[budget],
            y=[selected_duration],
            mode='markers',
            marker=dict(color=color, size=12),
            name=destination
        ))
    fig.update_layout(title_text='Relationship Between Budget and Duration',
                      xaxis_title='Budget',
                      yaxis_title='Duration',
                      showlegend=True)
    st.plotly_chart(fig)




# Bar chart showing distribution of budgets for different destinations
st.subheader("Distribution of Budgets for Different Destinations")
if predict_button:
    # Concatenate the filtered data for all selected destinations into a single DataFrame
    filtered_data = pd.concat([indian_places_df[indian_places_df["Destination"] == destination] for destination in selected_destinations])

    fig = go.Figure()
    for destination in selected_destinations:
        # Filter the data based on the selected destination
        filtered_data = indian_places_df[indian_places_df["Destination"] == destination]

        fig.add_trace(go.Bar(
            x=filtered_data["Accommodation Type"],
            y=filtered_data["Budget"],
            name=f'{destination}',
            marker_color=np.random.choice(range(256), size=3)  # Random color for each destination
        ))
    fig.update_layout(title_text=f'Budget Distribution for Different Destinations',
                      xaxis_title='Accommodation Type',
                      yaxis_title='Budget',
                      barmode='group')
    st.plotly_chart(fig)

# 3D Scatter Plot with Multioptions
st.subheader("3D Scatter Plot")

if predict_button:
    # Select columns for x, y, and z axes
    selected_x = st.sidebar.selectbox("Select X-axis", ["Budget", "Duration"])
    selected_y = st.sidebar.selectbox("Select Y-axis", ["Budget", "Duration"])
    selected_z = st.sidebar.selectbox("Select Z-axis", ["Budget", "Duration"])

    fig = go.Figure(data=[go.Scatter3d(
        x=input_df[selected_x],
        y=input_df[selected_y],
        z=input_df[selected_z],  
        mode='markers',
        marker=dict(
            size=8,
            color=np.random.choice(range(256), size=len(input_df)),  # Random color for each point
            opacity=0.8
        ),
        text=input_df["Destination"],
        hoverinfo='text'
    )])

    fig.update_layout(scene=dict(
        xaxis_title=selected_x,
        yaxis_title=selected_y,
        zaxis_title=selected_z
    ))

    st.plotly_chart(fig)

# Parallel Coordinates Plot
st.subheader("Parallel Coordinates Plot")

if predict_button:
    # Select the numerical columns for visualization
    numerical_columns = ["Budget", "Duration"]

    # Filter the input DataFrame based on selected destinations
    filtered_data = input_df[input_df["Destination"].isin(selected_destinations)]

    # Create a Parallel Coordinates Plot
    fig = go.Figure(data=go.Parcoords(
        line=dict(color=np.random.rand(len(filtered_data)),  # Assign random color to each line
                  colorscale='Viridis',
                  showscale=True,
                  reversescale=True),
        dimensions=[dict(label=col, values=filtered_data[col]) for col in numerical_columns]
    ))

    # Update layout
    fig.update_layout(title="Parallel Coordinates Plot",
                      xaxis=dict(title="Variable"),
                      yaxis=dict(title="Value"))

    st.plotly_chart(fig)




# Destination Preference Radar Chart
# Assuming the user can input their preferences using sliders for different attributes
st.subheader("Destination Preference Radar Chart")
if predict_button:
    # Placeholder for user-defined preferences (replace with actual values)
    user_preferences = {
        "Cost": 7,
        "Scenic Beauty": 8,
        "Cultural Significance": 6,
        "Adventure": 5,
        "Food": 9
    }

    attributes = list(user_preferences.keys())
    values = list(user_preferences.values())

    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=values,
        theta=attributes,
        fill='toself',
        name='User Preferences'
    ))
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 10]  # Assuming preferences are on a scale of 0 to 10
            )),
        showlegend=True
    )
    st.plotly_chart(fig)

# Budget Allocation Pie Chart
st.subheader("Budget Allocation")
if predict_button:
    # Placeholder for budget allocation (replace with actual values)
    budget_allocation = {
        "Accommodation": 40,
        "Transportation": 30,
        "Food": 20,
        "Activities": 10
    }

    labels = list(budget_allocation.keys())
    values = list(budget_allocation.values())

    fig = go.Figure(data=[go.Pie(labels=labels, values=values)])
    fig.update_layout(title_text='Budget Allocation')
    st.plotly_chart(fig)

# Top Attractions Bar Chart
# Assuming you have a dataset of attractions with ratings or popularity scores
st.subheader("Top Attractions")
if predict_button:
    attractions_data = {
        "Attraction": ["Beach", "Historical Site", "National Park", "Museum", "Market"],
        "Popularity Score": [8.5, 7.9, 9.2, 8.0, 7.5]  # Example popularity scores
    }
    attractions_df = pd.DataFrame(attractions_data)

    fig = go.Figure(go.Bar(
        x=attractions_df["Attraction"],
        y=attractions_df["Popularity Score"],
        marker_color='skyblue'
    ))
    fig.update_layout(title_text='Top Attractions',
                      xaxis_title='Attraction',
                      yaxis_title='Popularity Score')
    st.plotly_chart(fig)

# Travel Mode Preference Pie Chart
st.subheader("Travel Mode Preference")
if predict_button:
    # Placeholder for travel mode preference (replace with actual values)
    travel_modes = ["Flight", "Train", "Bus", "Car", "Bike"]
    preference_distribution = [25, 20, 15, 30, 10]  # Example distribution

    fig = go.Figure(data=[go.Pie(labels=travel_modes, values=preference_distribution)])
    fig.update_layout(title_text='Travel Mode Preference')
    st.plotly_chart(fig)

# Budget vs. Attractions Heatmap
# Assuming you have data on the number of attractions available in each destination
st.subheader("Budget vs. Attractions Heatmap")
if predict_button:
    # Placeholder for number of attractions (replace with actual values)
    attractions_count = [10, 15, 8, 12, 20]  # Example attractions count

    fig = go.Figure(data=go.Heatmap(
        z=[attractions_count],
        x=selected_destinations,
        y=['Attractions'],
        colorscale='Viridis',
        colorbar=dict(title='Number of Attractions')
    ))
    fig.update_layout(title_text="Budget vs. Attractions Heatmap",
                      xaxis_title="Destination",
                      yaxis_title="",
                      height=400)
    st.plotly_chart(fig)


