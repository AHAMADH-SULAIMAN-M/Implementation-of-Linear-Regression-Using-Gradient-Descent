# Implementation-of-Linear-Regression-Using-Gradient-Descent

## AIM:
To write a program to predict the profit of a city using the linear regression model with gradient descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the required library and read the dataframe.
2. Write a function computeCost to generate the cost function.
3. Perform iterations og gradient steps with learning rate.
4. Plot the Cost function using Gradient Descent and generate the required graph.


## Program:
```

Program to implement the linear regression using gradient descent.
Developed by: AHAMADH SULAIMAN M
RegisterNumber: 212224230009

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

def linear_regression(X1, y, learning_rate=0.01, num_iters=1000):
    X = np.c_[np.ones(len(X1)), X1]
    theta = np.zeros(X.shape[1]).reshape(-1,1)
    for _ in range(num_iters):
        predictions = (X).dot(theta).reshape(-1, 1)
        errors = (predictions - y).reshape(-1,1)
        theta -= learning_rate * (1 / len(X1)) * X.T.dot(errors)

    return theta
data = pd.read_csv('/content/50_Startups.csv', header=None)
print(data.head())
X = (data.iloc[1:, :-2].values)
print(X)
X1 = X.astype(float)
scaler = StandardScaler()
y = (data.iloc[1:, -1].values).reshape(-1,1)
print(y)
X1_Scaled = scaler.fit_transform(X1)
y1_Scaled = scaler.fit_transform(y)
print('Name: RITHIKA L')
print('Register No.:212224230231')
print(X1_Scaled)
theta = linear_regression(X1_Scaled, y1_Scaled)
new_data = np.array([165349.2, 136897.8, 471784.1]).reshape(-1,1)
new_scaled = scaler.fit_transform(new_data)
prediction = np.dot(np.append(1, new_scaled), theta)
prediction = prediction.reshape(-1,1)
pre = scaler.inverse_transform(prediction)
print(f"Predicted value: {pre}")
```

## Output:

<img width="710" height="923" alt="image" src="https://github.com/user-attachments/assets/a5ef364e-b35c-48e8-9521-4f0000aa4b9f" />

<img width="422" height="914" alt="image" src="https://github.com/user-attachments/assets/928cad37-965d-4b6f-b451-9eecaff2e855" />

<img width="592" height="917" alt="image" src="https://github.com/user-attachments/assets/005af7ff-77db-4a92-b30d-2eb66697beed" />

<img width="497" height="642" alt="image" src="https://github.com/user-attachments/assets/3f70ce9d-d6be-4239-b866-dd549d2bcb01" />

## Result:
Thus the program to implement the linear regression using gradient descent is written and verified using python programming.
