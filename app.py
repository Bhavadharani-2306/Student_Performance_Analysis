import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# Load the dataset
data = pd.read_csv(r'C:\Users\R.BHAVADHARANI\Downloads\archive (6)\StudentsPerformance.csv')

# Add a column for the average score
data['average score'] = data[['math score', 'reading score', 'writing score']].mean(axis=1)

# Prepare the features (X) and target (y)
categorical_features = ['gender', 'parental level of education', 'lunch', 'test preparation course']
encoded_data = pd.get_dummies(data, columns=categorical_features, drop_first=True)

X = encoded_data.drop(columns=['math score', 'reading score', 'writing score', 'average score'])
y = encoded_data['average score']

# Train the model
model = LinearRegression()
model.fit(X, y)

# Streamlit interface for user input
st.title("Student Performance Analysis")
st.write("The model will automatically calculate the average score based on the student's features.")

# Let the user input categorical features
gender = st.selectbox("Gender", ["Male", "Female"])
lunch = st.selectbox("Lunch", ["Standard", "Free/Reduced"])
parental_education = st.selectbox("Parental Level of Education", ["some college", "associate's degree", "bachelor's degree", "high school", "master's degree"])
test_preparation = st.selectbox("Test Preparation Course", ["none", "completed"])

# Convert categorical inputs to match the dataset format
new_data = pd.DataFrame(columns=X.columns)

# Set all values to 0 initially
new_data.loc[0] = 0
new_data.loc[0, f"gender_{gender.lower()}"] = 1
new_data.loc[0, f"lunch_{lunch.lower()}"] = 1
new_data.loc[0, f"parental level of education_{parental_education.lower()}"] = 1
new_data.loc[0, f"test preparation course_{test_preparation.lower()}"] = 1

# Make prediction based on the inputs
predicted_avg_score = model.predict(new_data)

# Display the predicted result
st.write(f"Predicted Average Score: {predicted_avg_score[0]:.2f}")

# Visualization of the correlation heatmap
st.write("### Correlation Heatmap of Features")

# One-hot encode categorical features
encoded_data = pd.get_dummies(data, columns=categorical_features, drop_first=True)

# Calculate correlation for only numerical columns
numerical_features = ['math score', 'reading score', 'writing score', 'average score'] + [col for col in encoded_data.columns if col not in categorical_features]
correlation = encoded_data[numerical_features].corr()

# Plotting the heatmap
fig, ax = plt.subplots(figsize=(10, 8))
sns.heatmap(correlation, annot=True, cmap='coolwarm', fmt=".2f", ax=ax)
st.pyplot(fig)
