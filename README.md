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

<img width="857" height="929" alt="image" src="https://github.com/user-attachments/assets/69c35215-9442-4600-8be1-6450cdfaec35" />

<img width="458" height="935" alt="image" src="https://github.com/user-attachments/assets/32480ed0-01e2-45c1-8f89-1c1a204f1ab1" />

<img width="595" height="961" alt="image" src="https://github.com/user-attachments/assets/4f145172-65ad-48de-8636-f9616b1992dd" />

<img width="569" height="618" alt="image" src="https://github.com/user-attachments/assets/eb935ead-7b21-4438-9e68-05804f9d6cec" />


## Result:
Thus the program to implement the linear regression using gradient descent is written and verified using python programming.
