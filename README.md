# Predicting Meal Orders with Machine Learning

## Problem Statement
This project predicts the number of orders (`num_orders`) for meals using data from fulfillment centers, meal information, and historical trends.

## Dataset
The dataset includes:
- `fulfilment_center_info.csv`
- `meal_info.csv`
- `train.csv`

## Models Implemented
1. Logistic Regression
2. XGBoost
3. Simple Neural Network
4. Neural Network with Dropout
5. Neural Network with Dropout and L2 Regularization
6. Neural Network with Early Stopping

## Results
| Instance                | Optimizer | Regularizer     | Epochs | Early Stopping | Learning Rate | Test Loss (MSE) | Test MAE |
|-------------------------|-----------|-----------------|--------|----------------|---------------|------------------|----------|
| Simple NN              | Adam      | None            | 10     | No             | 0.001         | ...              | ...      |
| NN with Dropout        | Adam      | Dropout         | 20     | No             | 0.001         | ...              | ...      |
| NN with Dropout + L2   | Adam      | Dropout + L2    | 30     | No             | 0.001         | ...              | ...      |
| NN with Early Stopping | Adam      | Dropout + L2    | 50     | Yes            | 0.001         | ...              | ...      |

## Summary
- The optimized neural network with early stopping achieved the best performance.
- XGBoost also performed well with proper hyperparameter tuning.
- Logistic Regression was the least effective model for this regression task.

## Instructions
1. Clone the repository.
2. Install the required libraries.
3. Run the notebook to reproduce the results.