"""
Script defining different functions used for demonstration.

Author: Sivakumar Balasubramanian
Date: 06 March 2024
"""

import numpy as np


class Circle:
    def __init__(self, xmin: np.array):
        self.name = 'Circle'
        self.title = r"$(\mathbf{x - \mathbf{x}^*})^\top(\mathbf{x} - \mathbf{x}^*)$"
        self.xmin = xmin.reshape(-1, 1)
    
    def func(self, x1, x2):
        _x1 = x1 - self.xmin[0, 0]
        _x2 = x2 - self.xmin[1, 0]
        return 0.5 * (_x1 ** 2 + _x2 ** 2)
    
    def grad(self, x1, x2):
        _x1 = x1 - self.xmin[0, 0]
        _x2 = x2 - self.xmin[1, 0]
        return np.array([[_x1],
                         [_x2]])  


class Ellipse:
    def __init__(self, xmin: np.array, Q: np.array):
        self.name = 'Ellipse'
        self.title = r"$(\mathbf{x - \mathbf{x}^*})^\top\mathbf{Q}(\mathbf{x} -  - \mathbf{x}^*)$"
        self.xmin = xmin.reshape(-1, 1)
        self.Q = Q
    
    def func(self, x1, x2):
        _x1 = x1 - self.xmin[0, 0]
        _x2 = x2 - self.xmin[1, 0]
        return (self.Q[0, 0] * _x1 ** 2
                + self.Q[1, 1] * _x2 ** 2
                + (self.Q[1, 0] + self.Q[0, 1]) * _x1 * _x2)
    
    def grad(self, x1, x2):
        _x1 = x1 - self.xmin[0, 0]
        _x2 = x2 - self.xmin[1, 0]
        return np.array([
            [2 * self.Q[0, 0] * _x1 + (self.Q[1, 0] + self.Q[0, 1]) * _x2],
            [2 * self.Q[1, 1] * _x2 + (self.Q[1, 0] + self.Q[0, 1]) * _x1]
        ])  


class Rosenbrock:
    def __init__(self, a: float, b: float):
        self.name = 'Rosenbrock'
        self.title = r"$(a - x_1)^2 + b (x_2 - x_1^2)^2$"
        self.xmin = np.array([[a], [a ** 2]])
        self.a, self.b = a, b
    
    def func(self, x1, x2):
        return (self.a - x1)**2 + self.b * (x2 - x1**2)**2
    
    def grad(self, x1, x2):
        return np.array([
            [(-2 * (self.a - x1)) - (4 * self.b * x1 * (x2 - x1**2))],
            [2 * self.b * (x2 - x1**2)]
        ])