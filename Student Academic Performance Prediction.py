# Task 41: Student Academic Performance Prediction

# Step 1: Import Required Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

# Step 2: Load Student Dataset
df = pd.read_csv("StudentsPerformance_1000.csv")

print("\nSTUDENT PERFORMANCE DATASET")
print("-" * 40)

print("First Five Records:")
print(df.head())

print("\nDataset Shape:")
print(df.shape)

# Step 3: Check Dataset Information
print("\nDataset Information:")
print(df.info())

print("\nMissing Values:")
print(df.isnull().sum())

# Step 4: Calculate Average Marks
df["average_score"] = (
    df["math score"] +
    df["reading score"] +
    df["writing score"]
) / 3

print("\nAverage Student Marks:")
print(df["average_score"].head())

# Step 5: Prepare Input and Target
# Predict writing marks using student details
# and math and reading marks.

X = df[
    [
        "gender",
        "race/ethnicity",
        "parental level of education",
        "lunch",
        "test preparation course",
        "math score",
        "reading score"
    ]
]

y = df["writing score"]

# Step 6: Identify Column Types

categorical_columns = [
    "gender",
    "race/ethnicity",
    "parental level of education",
    "lunch",
    "test preparation course"
]

numerical_columns = [
    "math score",
    "reading score"
]

# Remove rows where target marks are missing
df = df.dropna(subset=["writing score"])

X = df[
    categorical_columns + numerical_columns
]

y = df["writing score"]

# Step 7: Handle Missing Values

categorical_pipeline = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ]
)

numerical_pipeline = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="median"))
    ]
)

# Step 8: Prepare Data

preprocessor = ColumnTransformer(
    transformers=[
        ("categorical", categorical_pipeline, categorical_columns),
        ("numerical", numerical_pipeline, numerical_columns)
    ]
)

# Step 9: Create Machine Learning Model

model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        (
            "regressor",
            RandomForestRegressor(
                n_estimators=100,
                random_state=42
            )
        )
    ]
)

# Step 9: Split Dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

print("\nTraining Records:", len(X_train))
print("Testing Records:", len(X_test))

# Step 10: Train Model
model.fit(X_train, y_train)

print("\nStudent Performance Model Trained Successfully!")

# Step 11: Make Predictions
y_pred = model.predict(X_test)

print("\nActual Writing Marks:")
print(y_test.head(10).to_list())

print("\nPredicted Writing Marks:")
print(np.round(y_pred[:10], 2))

# Step 12: Evaluate Model
mae = mean_absolute_error(y_test, y_pred)

mse = mean_squared_error(y_test, y_pred)

rmse = np.sqrt(mse)

r2 = r2_score(y_test, y_pred)

print("\nMODEL EVALUATION RESULTS")
print("-" * 40)

print("Mean Absolute Error:", round(mae, 2))
print("Mean Squared Error:", round(mse, 2))
print("Root Mean Squared Error:", round(rmse, 2))
print("R2 Score:", round(r2, 4))

# Step 13: Predict New Student Marks
new_student = pd.DataFrame({
    "gender": ["male"],
    "race/ethnicity": ["group C"],
    "parental level of education": ["bachelor's degree"],
    "lunch": ["standard"],
    "test preparation course": ["completed"],
    "math score": [80],
    "reading score": [85]
})

predicted_marks = model.predict(new_student)

print("\nNEW STUDENT PERFORMANCE")
print("-" * 40)

print("Math Marks: 80")
print("Reading Marks: 85")
print("Predicted Writing Marks:", round(predicted_marks[0], 2))

# Step 14: Visualize Actual vs Predicted Marks
plt.figure(figsize=(8, 5))

plt.scatter(
    y_test,
    y_pred,
    alpha=0.6
)

plt.xlabel("Actual Writing Marks")
plt.ylabel("Predicted Writing Marks")

plt.title("Actual vs Predicted Student Marks")

plt.plot(
    [0, 100],
    [0, 100],
    linestyle="--"
)

plt.grid(True)
plt.tight_layout()
plt.show()

# Step 15: Display Final Result
print("\nProject Completed Successfully!")
