
from cmath import cos
import numpy as np
import math
import pandas as pd
from sklearn import linear_model

def predict_using_sklearn():
    df = pd.read_csv('test_scores.csv')
    reg = linear_model.LinearRegression()
    reg.fit(df[['math']], df.cs)
    return reg.coef_,reg.intercept_

def gradient_descent(x,y):
    m_curr = b_curr = 0
    n = len(x)
    learning_range = 0.0002
    cost_previous = 0
    for i in range(10000):
        y_predict = m_curr*x + b_curr
        cost = (1/n) * sum(val**2 for val in (y-y_predict))
        md = -(2/n) * sum(x*(y-y_predict))
        bd = -(2/n) * sum(y-y_predict)
        m_curr = m_curr - learning_range*md
        b_curr = b_curr - learning_range*bd
        if math.isclose(cost,cost_previous,rel_tol=1e-20):
            break
        cost_previous = cost
        print(f"m {m_curr}, b {b_curr}, iteration {i} and cost {cost}")
    return m_curr, b_curr

if __name__ == "__main__":
    df = pd.read_csv('test_scores.csv')
    x = np.array(df.math)
    y = np.array(df.cs)

    m, b = gradient_descent(x,y)
    print(f"Using gradient descent coeff {m} and intercept {b}")

    m_sklearn, b_sklearn = predict_using_sklearn()
    print(f"Using sklearn coeff {m_sklearn} and intercept {b_sklearn}")

