🎓 Student Academic Performance Prediction

📌 Project Overview

This project uses Machine Learning to predict students' academic performance based on their personal and academic information.

The project uses a dataset containing 1,000 student records and applies the Random Forest Regression algorithm to predict students' writing scores.

It also performs data cleaning, missing value handling, model training, prediction, and evaluation.

🎯 Project Objectives

- Analyse student academic performance.
- Handle missing values in the dataset.
- Convert categorical data into numerical format.
- Train a Machine Learning model.
- Predict student writing scores.
- Evaluate model performance using different metrics.
- Visualize actual and predicted marks.

🛠️ Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Scikit-learn
- Random Forest Regression

📂 Dataset Information

Dataset Name: StudentsPerformance_1000.csv

Total Records: 1,000 Students

Total Features: 8 Columns

Dataset Columns

Column Name| Description
gender| Student gender
race/ethnicity| Student group
parental level of education| Parent education
lunch| Lunch type
test preparation course| Preparation status
math score| Mathematics marks
reading score| Reading marks
writing score| Writing marks

Note: This project uses a synthetically generated dataset for educational and practice purposes.

⚙️ Project Workflow

1. Import required Python libraries.
2. Load the student dataset.
3. Explore dataset information.
4. Check and handle missing values.
5. Calculate average student marks.
6. Separate input features and target variable.
7. Encode categorical variables using OneHotEncoder.
8. Impute missing values using SimpleImputer.
9. Split the dataset into training and testing data.
10. Train the Random Forest Regression model.
11. Predict student writing marks.
12. Evaluate model performance.
13. Visualize actual vs predicted marks.

🤖 Machine Learning Algorithm

Random Forest Regression

Random Forest Regression is an ensemble learning algorithm that combines multiple decision trees to predict numerical values.

In this project, it is used to predict students' writing marks based on their academic and personal information.

📊 Model Evaluation Metrics

The model performance is evaluated using:

- Mean Absolute Error (MAE): Measures average prediction error.
- Mean Squared Error (MSE): Measures the average squared prediction error.
- Root Mean Squared Error (RMSE): Measures prediction error in the original marks scale.
- R² Score: Measures how well the model explains variations in the target values.

📈 Data Visualization

The project generates an Actual vs Predicted Writing Marks scatter plot using Matplotlib.

This visualization helps compare actual student marks with predicted marks.

💻 Installation

Install the required Python libraries:

pip install pandas numpy matplotlib scikit-learn

▶️ How to Run the Project

1. Download or clone this repository.
2. Place "StudentsPerformance_1000.csv" in the same folder as the Python program.
3. Open the project in Python IDLE or VS Code.
4. Run the Python program.

python "Student Academic Performance Prediction.py"

📁 Project Structure

Student-Academic-Performance-Prediction/
│
├── Student Academic Performance Prediction.py
├── StudentsPerformance_1000.csv
└── README.md

🔍 Sample Prediction

The project tests a new student with:

- Math Marks: 80
- Reading Marks: 85
- Gender: Male
- Race/Ethnicity: Group C
- Parental Education: Bachelor's Degree
- Lunch: Standard
- Test Preparation: Completed

The trained model predicts the student's writing marks.

🚀 Learning Outcomes

Through this project, I learned:

- Data preprocessing and cleaning.
- Handling missing values.
- Feature encoding.
- Machine Learning model development.
- Random Forest Regression.
- Model evaluation techniques.
- Data visualization.
- Making predictions using trained models.

🏁 Conclusion

This project demonstrates how Machine Learning can be used to analyse student academic performance and predict writing scores.

It helped me gain practical experience in Python programming, data preprocessing, model training, prediction, and evaluation.

---

Project: Task 41 – Student Academic Performance Prediction

Domain: Machine Learning

Developed using: Python 