import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# Load the dataset
data = pd.read_csv(r'C:\Users\R.BHAVADHARANI\Downloads\archive (1)\StudentsPerformance.csv')

# Add a column for the average score
data['average score'] = data[['math score', 'reading score', 'writing score']].mean(axis=1)

# One-hot encode all categorical features
categorical_features = ['gender', 'race/ethnicity', 'parental level of education', 'lunch', 'test preparation course']
encoded_data = pd.get_dummies(data, columns=categorical_features, drop_first=True)

# Prepare features (X) and target (y)
X = encoded_data.drop(columns=['math score', 'reading score', 'writing score', 'average score'])
y = encoded_data['average score']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict
predictions = model.predict(X_test)

# Streamlit App UI
st.title("📊 Student Performance Analysis")

st.write("### Dataset Preview")
st.dataframe(data.head())

st.write("### Average Score Distribution")
fig, ax = plt.subplots()
sns.histplot(data['average score'], kde=True, ax=ax)
st.pyplot(fig)

st.write("### Model Performance")
st.write(f"R-squared Score on Test Data: **{model.score(X_test, y_test):.2f}**")

# Optional: User input for prediction
st.write("### Predict Average Score")
user_input = {}

for col in X.columns:
    if encoded_data[col].nunique() == 2 and set(encoded_data[col].unique()) == {0, 1}:
        user_input[col] = st.radio(f"{col.replace('_', ' ').title()}?", [0, 1])
    else:
        user_input[col] = st.number_input(f"{col.replace('_', ' ').title()}:", value=0.0)

input_df = pd.DataFrame([user_input])
if st.button("Predict"):
    pred = model.predict(input_df)[0]
    st.success(f"Predicted Average Score: **{pred:.2f}**")
