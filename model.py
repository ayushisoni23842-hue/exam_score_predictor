import pandas as pd
import numpy as np
import pickle

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# Load dataset
df = pd.read_csv("train.csv")

# Drop unnecessary columns
df.drop(columns=['facility_rating', 'id'], inplace=True)

# Encoding
df['gender'] = df['gender'].replace({'female': 0, 'male': 1, 'other': 2})
df['internet_access'] = df['internet_access'].replace({'no': 0, 'yes': 1})
df['sleep_quality'] = df['sleep_quality'].replace({'poor': 0, 'average': 0.5, 'good': 1})
df['exam_difficulty'] = df['exam_difficulty'].replace({'easy': 0, 'moderate': 1, 'hard': 2})

# One-hot encoding
df = pd.get_dummies(df, columns=['course'], drop_first=True)
df = pd.get_dummies(df, columns=['study_method'], drop_first=True)

# Split
X = df.drop('exam_score', axis=1)
y = df['exam_score']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)

# Model
model = LinearRegression()
model.fit(X_train, y_train)

# Save model + scaler + columns
pickle.dump(model, open("model.pkl", "wb"))
pickle.dump(scaler, open("scaler.pkl", "wb"))
pickle.dump(X.columns, open("columns.pkl", "wb"))

print("Model saved successfully!")