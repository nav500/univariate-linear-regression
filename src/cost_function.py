import numpy as np

def cost_function(w, b, x_train, y_train):
    m = x_train.shape[0]
    
    # calculating cost value
    cost_value = 0

    # intialization of squared difference sum
    squared_diff_sum = 0
    for i in range(m):
        squared_diff = (((w * x_train[i]) + b) - y_train[i])**2
        squared_diff_sum = squared_diff_sum + squared_diff
    
    # taking average of squared differences
    cost_value = squared_diff_sum / (2 * m)

    return cost_value
