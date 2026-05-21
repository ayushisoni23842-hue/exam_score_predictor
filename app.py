import streamlit as st
import pandas as pd
import numpy as np
import pickle

# Load files
model = pickle.load(open("model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))
columns = pickle.load(open("columns.pkl", "rb"))

st.title("🎓 Student Exam Score Predictor")

# Inputs
age = st.number_input("Age", 10, 50)
gender = st.selectbox("Gender", ["male", "female", "other"])
internet = st.selectbox("Internet Access", ["yes", "no"])
sleep = st.selectbox("Sleep Quality", ["poor", "average", "good"])
difficulty = st.selectbox("Exam Difficulty", ["easy", "moderate", "hard"])
course = st.text_input("Course (example: science / commerce etc.)")
study = st.text_input("Study Method (example: online / offline etc.)")

# Convert input to dataframe
input_data = pd.DataFrame({
    'age': [age],
    'gender': [gender],
    'internet_access': [internet],
    'sleep_quality': [sleep],
    'exam_difficulty': [difficulty],
    'course': [course],
    'study_method': [study]
})

# Preprocessing function
def preprocess(df):
    df['gender'] = df['gender'].replace({'female': 0, 'male': 1, 'other': 2})
    df['internet_access'] = df['internet_access'].replace({'no': 0, 'yes': 1})
    df['sleep_quality'] = df['sleep_quality'].replace({'poor': 0, 'average': 0.5, 'good': 1})
    df['exam_difficulty'] = df['exam_difficulty'].replace({'easy': 0, 'moderate': 1, 'hard': 2})

    df = pd.get_dummies(df)

    # Match training columns
    df = df.reindex(columns=columns, fill_value=0)

    return df

# Prediction
if st.button("Predict Score"):
    processed = preprocess(input_data)
    scaled = scaler.transform(processed)
    prediction = model.predict(scaled)

    st.success(f"Predicted Exam Score: {prediction[0]:.2f}")