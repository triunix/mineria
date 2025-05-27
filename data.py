import pandas as pd
import numpy as np
import tkinter as tk
from tkinter import ttk
from tkinter import messagebox
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from ttkthemes import ThemedStyle

# Load the dataset (assuming the dataset is in the same directory as this script)
file_path = "student-mat.csv"
data_sucia = pd.read_csv(file_path, sep=';')

with open("data_sucia.txt", "w", encoding="utf-8") as archivo:
    archivo.write(data_sucia.to_string())







# Handle Missing Values
data = data_sucia.dropna().copy()


# Remove Duplicates
#data.drop_duplicates(subset='id_student', inplace=True)  # Drop duplicates based on 'StudentID' 

# Data preprocessing - handle missing values or categorical variables
# For categorical variables, we'll use one-hot encoding

# Convert categorical variables to one-hot encoding
data = pd.get_dummies(data, columns=['school', 'sex', 'address', 'famsize', 'Pstatus',
                                     'Mjob', 'Fjob', 'reason', 'guardian', 'schoolsup',
                                     'famsup', 'paid', 'activities', 'nursery', 'higher',
                                     'internet', 'romantic'], drop_first=True)
# Select features and target variable
features = data.drop(columns=['G3'])  # Features: all columns except 'G3' (final grade)
target = data['G3']  # Target variable: 'G3' (final grade)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

# Create and train the Linear Regression model
model = linear_model.LinearRegression()
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

with open("data_limpia.txt", "w", encoding="utf-8") as archivo:
    archivo.write(data.to_string())

# Function to perform the prediction and display the result on the GUI
def predict_grade():
    def predict():
        try:
            G1_input = int(g1_entry.get())
            G2_input = int(g2_entry.get())
            study_time_input = int(studytime_entry.get())

            new_student_features = pd.DataFrame({
                'G1': [G1_input],      # First-period grade
                'G2': [G2_input],      # Second-period grade
                'studytime': [study_time_input], # Weekly study time (hours)
            })
                        # Perform one-hot encoding for the new student data and align with training data
            new_student_features_encoded = pd.get_dummies(new_student_features, drop_first=True)
            new_student_features_encoded = new_student_features_encoded.align(features, join='right', axis=1, fill_value=0)[0]

            predicted_grade = model.predict(new_student_features_encoded)

            print("Predicted Final Grade for the New Student:", predicted_grade[0])
            predicted_label.config(text=f"Predicted Final Grade: {predicted_grade[0]:.2f}")
        except ValueError:
            print("Error: Please enter valid numeric values for G1, G2, and study time.")
            predicted_label.config(text="Please enter valid numeric values for G1, G2, and study time.")

    # Create the tkinter window
    window = tk.Tk()
    window.title("Student Grade Predictor")

    # Set a fixed window size
    window.geometry("400x300")  # Adjust the size as needed

    # Apply a themed style to the window
    style = ThemedStyle(window)
    style.theme_use("arc")  # You can change the theme here (try "clam", "equilux", etc.)

    # Create and pack a description label
    description_label = ttk.Label(window, text="Welcome to the Student Grade Predictor!\n"
                                               "Please enter the student's information below:")
    description_label.pack(pady=20)

    # Create and pack input fields
    g1_label = ttk.Label(window, text="G1 (first-period grade):")
    g1_label.pack()
    g1_entry = ttk.Entry(window)
    g1_entry.pack()

    g2_label = ttk.Label(window, text="G2 (second-period grade):")
    g2_label.pack()
    g2_entry = ttk.Entry(window)
    g2_entry.pack()

    studytime_label = ttk.Label(window, text="Weekly study time (hours):")
    studytime_label.pack()
    studytime_entry = ttk.Entry(window)
    studytime_entry.pack()

    # Create and pack the Predict button
    predict_button = ttk.Button(window, text="Predict", command=predict)
    predict_button.pack(pady=20)

    # Create and pack the label to display the predicted grade
    predicted_label = ttk.Label(window, text="", background=style.lookup("TLabel", "background"))
    predicted_label.pack()

    # Start the tkinter main loop
    window.mainloop()

# Call the function to predict grade
predict_grade()
