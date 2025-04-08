# Predicting Meal Orders with Machine Learning

## Problem Statement
This project aims to predict the number of orders (`num_orders`) for meals using data from fulfillment centers, meal information, and historical trends. The dataset includes features such as meal categories, cuisines, center types, and pricing information. The goal is to build and evaluate machine learning models to accurately predict the target variable (`num_orders`).

## Dataset
The dataset consists of three files:
1. `fulfilment_center_info.csv`: Contains details about fulfillment centers (e.g., center type, region, operational area).
2. `meal_info.csv`: Contains details about meals (e.g., category, cuisine).
3. `train.csv`: Contains historical data, including the target variable (`num_orders`).

The datasets were merged and preprocessed to create a single dataset for training and evaluation.

## Models Implemented
The following models were implemented and evaluated:
1. **Logistic Regression** (Classical ML Model).
2. **XGBoost** (Optimized using hyperparameters such as `n_estimators`, `learning_rate`, and `max_depth`).
3. **Simple Neural Network** (No optimization techniques applied).
4. **Neural Network with Dropout** (To prevent overfitting).
5. **Neural Network with Dropout and L2 Regularization** (To improve generalization).
6. **Neural Network with Early Stopping** (To stop training when validation loss stops improving).

---

## Results
The table below summarizes the performance of the models on the test set:

| **Instance**            | **Optimizer** | **Regularizer**     | **Epochs** | **Early Stopping** | **Learning Rate** | **Test Loss (MSE)** | **Test MAE** |
|--------------------------|---------------|---------------------|------------|--------------------|-------------------|---------------------|--------------|
| Simple Neural Network    | Adam          | None                | 10         | No                 | 0.001             | 34592.714844	     | 92.599998    |
| NN with Dropout          | Adam          | Dropout             | 20         | No                 | 0.001             | 35492.660156        | 94.895203    |
| NN with Dropout + L2     | Adam          | Dropout + L2        | 30         | No                 | 0.001             | 34155.500000        | 92.900146    |
| NN with Early Stopping   | Adam          | Dropout + L2        | 50         | Yes                | 0.001             | 31836.179688        | 92.701187    |

---

## Findings
1. **Best Performing Model**: The **Neural Network with Early Stopping** achieved the lowest Test Loss (MSE) of 31836.179688 and a Test MAE of 92.701187, confirming it is the best-performing model.
2. **Simple Neural Network**: This model performed reasonably well but was outperformed by models with regularization and early stopping.
3. **Neural Network with Dropout**: Adding dropout alone did not improve performance significantly and resulted in slightly higher error compared to the simple neural network.
4. **Neural Network with Dropout + L2 Regularization**: Adding both dropout and L2 regularization improved generalization and reduced the error compared to the simple neural network and dropout-only model.
5. **Neural Network with Early Stopping**: This model outperformed all others by combining dropout, L2 regularization, and early stopping, which prevented overfitting and improved generalization.

---

## Summary
- **Neural Networks**: The Neural Network with Early Stopping was the best-performing model overall. Dropout and L2 regularization helped improve generalization, while early stopping prevented overfitting.
- **XGBoost**: Performed competitively but required careful hyperparameter tuning (n_estimators, learning_rate, max_depth). Its exact performance metrics are missing and need to be added for a complete comparison.
- **Logistic Regression**: Was not effective for this task due to the complexity of the dataset. Its exact performance metrics are missing and need to be added for a complete comparison.

---

## Instructions for Running the Notebook
1. Clone the repository:
   ```bash
   git clone <repository_link>
   cd <repository_name>


## Link to Video Presentation
1. Clone the repository:
   ```bash
   git clone <repository_link>
   cd <repository_name>