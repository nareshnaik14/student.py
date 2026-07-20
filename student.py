#import libraries

import pandas as pd
import numpy as np
import streamlit as st

#visualization
import matplotlib.pyplot as plt
import seaborn as sns

# machine learning model
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import plot_tree
#evaluation
from sklearn.metrics import (accuracy_score,
                             confusion_matrix,
                             classification_report
)

# Load Dataset
df = pd.read_csv('student_performance.csv')

print(df)
# Display first 5 rows
df.head()

#Step 3 — Dataset Information
# Dataset Shape

print("Dataset Shape:")
print(df.shape)

# Dataset Info

print("\nDataset Info:")
print(df.info())

# Missing Values

print("\nMissing Values:")
print(df.isnull().sum())

#4 statistical summary
print(df.describe())



#5Data visualization

plt.figure(figsize=(10,7))

# 1. study_hours vs result
plt.subplot(2,2,1)

sns.boxplot(
    x='result',
    y='study_hours',
    data=df
)

plt.title("study_hours vs result")


# 2. attendance vs result
plt.subplot(2,2,2)

sns.boxplot(
    x='result',
    y='attendance',
    data=df
)

plt.title("attendance vs result")


# 3. assignment_score vs result
plt.subplot(2,2,3)

sns.countplot(
    x='result',
    hue='assignment_score',
    data=df
)


plt.title("assignment_score vs result")

#4. sleep_hours vs result
plt.subplot(2,2,4)

sns.boxplot(
    x='result',
    y='sleep_hours',
    data=df
)

plt.title("sleep_hours vs result")



plt.tight_layout()
plt.show()


# ENCODING

le = LabelEncoder()

df['result'] = le.fit_transform(df['result'])


#Step 6 — Define Features and Target

X = df.drop('result', axis=1)

print("Training Columns:", X.columns.tolist())

y = df['result']

#Step 7 — Train Test Split
# Split Dataset

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print("Training Shape:", X_train.shape)
print("Testing Shape:", X_test.shape)

#Step 8 — Underfitting Model


underfit_model = DecisionTreeClassifier(
    max_depth=1,
    random_state=42
)

underfit_model.fit(X_train, y_train)

#Step 9 — Underfitting Evaluation
# Predictions

y_train_under = underfit_model.predict(X_train)

y_test_under = underfit_model.predict(X_test)

# Accuracy

train_acc_under = accuracy_score(y_train, y_train_under)

test_acc_under = accuracy_score(y_test, y_test_under)

print("Underfitting Model")

print("Training Accuracy:", train_acc_under)

print("Testing Accuracy:", test_acc_under)


#Step 10 — Overfitting Model
# Overfitting Model

overfit_model = DecisionTreeClassifier(
    random_state=42
)

overfit_model.fit(X_train, y_train)

#Step 11 — Overfitting Evaluation

# Predictions

y_train_over = overfit_model.predict(X_train)

y_test_over = overfit_model.predict(X_test)

# Accuracy

train_acc_over = accuracy_score(y_train, y_train_over)

test_acc_over = accuracy_score(y_test, y_test_over)

print("Overfitting Model")

print("Training Accuracy:", train_acc_over)

print("Testing Accuracy:", test_acc_over)

#Step 12 — Balanced/Tuned Model

# Tuned Decision Tree

balanced_model = DecisionTreeClassifier(
    criterion='entropy',
    max_depth=5,
    min_samples_split=10,
    min_samples_leaf=5,
    random_state=42
)

balanced_model.fit(X_train, y_train)

#Step 13 — Balanced Model Evaluation

# Predictions

y_train_bal = balanced_model.predict(X_train)

y_test_bal = balanced_model.predict(X_test)

# Accuracy

train_acc_bal = accuracy_score(y_train, y_train_bal)

test_acc_bal = accuracy_score(y_test, y_test_bal)

print("Balanced Model")

print("Training Accuracy:", train_acc_bal)

print("Testing Accuracy:", test_acc_bal)

#Step 14 — Compare All Models

# Comparison Table

comparison = pd.DataFrame({

    'Model': [
        'Underfitting',
        'Overfitting',
        'Balanced'
    ],

    'Training Accuracy': [
        train_acc_under,
        train_acc_over,
        train_acc_bal
    ],

    'Testing Accuracy': [
        test_acc_under,
        test_acc_over,
        test_acc_bal
    ]
})

print(comparison)



#Step 15 — Confusion Matrix

# Confusion Matrix for Balanced Model

cm = confusion_matrix(y_test, y_test_bal)

plt.figure(figsize=(6,5))

sns.heatmap(cm, annot=True, fmt='d')

plt.title("Balanced Model Confusion Matrix")

plt.xlabel("Predicted")

plt.ylabel("Actual")

plt.show()

#Step 16 — Classification Report



print(classification_report(y_test, y_test_bal))

#Step 17 — Decision Tree Visualization

# Underfitting Tree

plt.figure(figsize=(10,5))

plot_tree(
    underfit_model,
    feature_names=X.columns,
    class_names=['Fail', 'Pass'],
    filled=True
)

plt.show()

# Overfitting Tree

plt.figure(figsize=(25,12))

plot_tree(
    overfit_model,
    feature_names=X.columns,
    class_names=['Fail', 'Pass'],
    filled=True
)

plt.show()

# Balanced Tree

plt.figure(figsize=(15,8))

plot_tree(
    balanced_model,
    feature_names=X.columns,
    class_names=['Fail', 'Pass'],
    filled=True
)

plt.show()





st.header("🔮 Predict Student Result")

study_hours = st.number_input(
    "Study Hours",
    min_value=0.0,
    max_value=24.0,
    value=5.0
)

attendance = st.number_input(
    "Attendance (%)",
    min_value=0,
    max_value=100,
    value=80
)

assignment_score = st.number_input(
    "Assignment Score",
    min_value=0,
    max_value=100,
    value=75
)

sleep_hours = st.number_input(
    "Sleep Hours",
    min_value=0.0,
    max_value=12.0,
    value=7.0
)

if st.button("Predict"):

      new_data = pd.DataFrame(
        [[study_hours, attendance, assignment_score, sleep_hours]],
        columns=X.columns
    )

        prediction = balanced_model.predict(new_data)


    result = le.inverse_transform(prediction)

    st.subheader("Prediction Result")

    if result[0] == "Pass":
        st.success("🎉 Student is likely to PASS")
    else:
        st.error("❌ Student is likely to FAIL")
