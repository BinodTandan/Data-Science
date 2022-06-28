import numpy as np

def gradient_descent(x,y):
    m_curr = b_curr = 0
    iter = 10000

    n = len(x)
    learning_rate= 0.084
    for i in range(iter):
        y_predict = m_curr * x + b_curr
        cost = (1/n) * sum(val**2 for val in (y-y_predict))
        md = (-2/n) * sum(x*(y-y_predict))
        bd = (-2/n) * sum(y-y_predict)
        m_curr =m_curr - learning_rate * md
        b_curr =b_curr - learning_rate * bd
        print(f"m {m_curr}, b {b_curr},iteration {i}, cost {cost}")
x = np.array([1,2,3,4,5])
y = np.array([5,7,9,11,13])

gradient_descent(x,y)