import numpy as np
import matplotlib.pyplot as plt


def step_function(x):
    return np.array(x > 0, dtype=np.int)


X = np.arange(-5.0, 5.0, 0.1)
print(X)
