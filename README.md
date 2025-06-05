# Predicting Heart Disease Status

### Description & Objective
##### Background
Cardiovascular disease is the leading cause of death worldwide, according to the World Health Organization. Many of these deaths are preventable through early diagnosis and medical intervention. However, identifying individuals at risk can be challenging, especially when symptoms are not obviously apparent.

Data analysis can be a powerful tool for analyzing patient data and uncovering patterns that may not be obvious. By applying classification algorithm to medical datasets, we can create models that assist in the early detetction of heart disease, potentially improving outcomes and reducing the burden on healthcare systems.

##### Objective
This project aims to evaluate and compare the performance of several classification models to identify the most suitable method for predicting heart disease status. Using a given dataset containing patient's health features, we aim to predict whether a patient is likely to have heart disease.

This task is a binary classification problem, where the response variable (e.g. heart disease status) indicates the **presence = 1** or **absence = 0** of heart disease. The goal is to propose the most accurate classifier based on performance metrices such as **True Positive Rate (TPR)**, **Precision**, and **Area Under the ROC Curve (AUC value)**.

### Dataset
##### Overview
The dataset used is attached, named `heart-disease-status.csv` and consists of **300 observations** (rows), each representing an individual patient's record. Each observation includes **12 input features** (independent variables) related to demographic and clinical health information, along with the response variable that indicates the presence or absence of heart disease.

##### Variables
|      Variable        |            Description         | Type of variables |
|----------------------|--------------------------------|-------------------|
|`age`                 |age of the patient in years     |      Numerical    |
|`sex`                 |
|`chest pain`          |
|`blood pressure (bp)` |
|`cholesterol (chol)`  |
|`fbs`                 |
|`rest.ecg`            |
|`heart rate`          |
|`angina`              |
|`st. depression`      |
|`vessels`             |
|`blood disorder`      |

##### Response Variable
`1` indicates the presence of heart disease and `0` indicates the absence of heart disease.

### Plan of Analysis
The analysis will follow a structured and systematic flow ro ensure comprehensive exploration, modeling, and evaluation of the dataset, including:
1. **Exploratory Data Analysis (EDA)**

   Exploring the variables and examining the associations between the response variable and each input variable.
   
3. **Implementation of methods**:
   - **K-Nearest Neighbors (KNN)**: A simple, instance-based learning method that classifies a sample based on the majority vote of its neighbors.
   - **Decision Tree**: A tree-structured model that splits features to create decision paths. It is highly interpretable and captures non-linear relationships.
   - **Logistic Regression**: A linear model used as a baseline classifier. It predicts the probability of heart disease _(response variable = 1)_ using a logistic function.
   
   Each model is trained using the same train-test split and evaluated on the test set for fair comparison.
   
5. **Evaluation and comparison of classifiers**
   To assess the performance of the models, we use a variety of evaluation metrics, particularly those that are crucial in medical diagnostics.
   - **True Positive Rate (TPR) / Sensitivity**: as a measurement on how well the model predicts actual cases of heart disease.
   - **Precision**: to measure the proportion of predicted positive cases that are actually positive.
   - **ROC Curve**: visual representation of the trade-off between sensitivity and specificity. 
   - **AUC Value (Area Under the Curve)**: summary of model performance across all classification threshold, higher AUC value means better model.
   
   The performance of all models will be compiled into a comparison table and ROC curves will be plotted to visually assess and compare their effectiveness.

### Conclusion
Based on the comparison of the True Positive Rate (TPR), Precision, and AUC values across the three classifiers, the **Decision Tree** tends to be the most effective model for predicting heart disease status. It has the highest TPR and Precision, which indicates a strong performance in identifying true positive cases and minimizing false positives.

While the Logistic Regression model has the highest AUC value (which indicates that this model is stronger in overall classification capability), the Decision Tree model still more reliable in a medical context where correctness in identifying positive cases is very crucial. Thus, the Decision Tree model demonstrates the best balance between sensitivity and precision, making it the most suitable choice for this task.
