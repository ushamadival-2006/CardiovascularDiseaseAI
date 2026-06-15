# HeartDiseaseAI

## Overview

HeartDiseaseAI is a machine learning-based cardiovascular disease risk prediction system designed to identify individuals at risk of heart disease using demographic, lifestyle, and clinical health indicators.

The system analyzes patient information such as age, blood pressure, cholesterol level, glucose level, BMI, smoking habits, alcohol consumption, and physical activity to predict cardiovascular disease risk.

Unlike traditional prediction systems that stop after generating a risk score, HeartDiseaseAI aims to support preventive healthcare through risk assessment, health benchmarking, personalized guidance, and progress tracking.

---

# Problem Statement

Cardiovascular diseases are among the leading causes of death worldwide.

Many individuals remain unaware of their cardiovascular health risks until severe complications occur. Existing systems primarily focus on disease prediction and provide limited support for preventive healthcare, personalized recommendations, and continuous monitoring.

This project aims to develop an AI-powered system that not only predicts heart disease risk but also helps users understand their health condition and take preventive actions.

---

# Objectives

* Predict heart disease risk using machine learning.
* Identify major risk factors contributing to cardiovascular disease.
* Categorize users into Low, Medium, or High risk groups.
* Compare user health parameters with healthy benchmark values.
* Provide personalized health recommendations.
* Enable follow-up assessments and progress tracking.
* Promote early detection and preventive healthcare.

---

# Dataset

Dataset Used: Cardiovascular Disease Dataset

Dataset Characteristics:

* 70,000 patient records
* 11 health-related features
* Binary classification problem
* Target variable indicating presence or absence of cardiovascular disease

Features:

* Age
* Gender
* Height
* Weight
* Systolic Blood Pressure
* Diastolic Blood Pressure
* Cholesterol Level
* Glucose Level
* Smoking Status
* Alcohol Consumption
* Physical Activity

Target:

* Cardio

  * 0 = No Heart Disease
  * 1 = Heart Disease

---

# Project Workflow

1. Data Collection
2. Data Analysis
3. Data Preprocessing
4. Feature Engineering
5. Feature Selection
6. Train-Test Split
7. Feature Scaling
8. Model Training
9. Model Evaluation
10. Heart Disease Risk Prediction
11. Risk Score Generation
12. Risk Category Classification
13. Explainable AI Analysis (SHAP)
14. Personalized Health Recommendations
15. Healthy Benchmark Comparison
16. Progress Tracking
17. Follow-up Risk Assessment

---

# Feature Engineering

## Age Conversion

The dataset stores age in days.

Age is converted into years for better interpretability and model understanding.

## BMI Calculation

BMI is calculated using:

BMI = Weight / Height²

Body Mass Index is an important cardiovascular risk indicator and helps identify obesity-related risks.

---

# Machine Learning Models

Two machine learning models were trained and evaluated.

## Random Forest Classifier

Random Forest is an ensemble learning algorithm that combines multiple decision trees.

Advantages:

* Handles complex relationships
* Reduces overfitting
* Provides feature importance

## XGBoost Classifier

XGBoost is a gradient boosting algorithm that builds trees sequentially and learns from previous mistakes.

Advantages:

* High predictive performance
* Efficient training
* Handles tabular healthcare data effectively

---

# Model Evaluation Metrics

The models were evaluated using:

* Accuracy
* Precision
* Recall
* F1 Score
* Confusion Matrix
* Classification Report

---

# Results

## Random Forest

* Accuracy: 70.97%
* F1 Score: 70.14%

## XGBoost

* Accuracy: 73.16%
* F1 Score: 71.36%

## Best Model

XGBoost was selected as the final deployment model because it achieved the highest overall performance.

---

# Post-Prediction Features

## Risk Categorization

Based on the predicted probability, users are categorized into:

* Low Risk
* Medium Risk
* High Risk

## Healthy Benchmark Comparison

The system compares the user's health parameters with recommended healthy values such as:

* BMI
* Blood Pressure
* Cholesterol
* Physical Activity

This helps users understand where improvements are needed.

## Personalized Recommendations

The system is designed to provide:

* Lifestyle improvement suggestions
* Exercise recommendations
* Dietary guidance
* Smoking reduction advice
* Alcohol reduction advice

## Progress Tracking

Users can periodically re-enter their health information.

The system compares previous and current assessments to identify:

* Risk reduction
* Risk increase
* Health improvement trends

## Follow-up Assessment

The platform supports continuous monitoring by allowing users to perform future health assessments and track cardiovascular health over time.

---

# Project Structure

HeartDiseaseAI/

├── data/

│ ├── cardio_train.csv

│ └── processed/

│ └── cleaned_data.csv

│

├── models/

│ ├── scaler.pkl

│ ├── random_forest.pkl

│ ├── xgboost.pkl

│ └── best_model.pkl

│

├── notebooks/

│ ├── 01_data_analysis.ipynb

│ ├── 02_preprocessing.ipynb

│ ├── 03_feature_selection_and_train_test_split.ipynb

│ └── 04_model_training.ipynb

│

├── app/

│

├── requirements.txt

│

└── README.md

---

# Technologies Used

* Python
* Pandas
* NumPy
* Scikit-Learn
* XGBoost
* Matplotlib
* Seaborn
* Joblib
* Jupyter Notebook

---

# Current Status

Completed:

* Data Analysis
* Data Preprocessing
* Feature Engineering
* Feature Selection
* Train-Test Split
* Feature Scaling
* Random Forest Training
* XGBoost Training
* Model Evaluation
* Model Saving
* Risk Prediction
* Terminal-Based Testing

Planned:

* SHAP Explainable AI
* LLM-Based Health Explanation
* Agentic AI Recommendation System
* User Dashboard
* Health Benchmark Visualization
* Progress Tracking
* Follow-up Assessment System

---

# Future Scope

* Explainable AI using SHAP
* Medical report upload and analysis
* Family history integration
* Personalized healthcare recommendations
* Multi-disease prediction
* Agentic AI-powered healthcare assistant
* Real-time health monitoring
* Mobile application deployment

---

# Author

Usha

Engineering Student | Machine Learning Enthusiast | AI Enthusiast
