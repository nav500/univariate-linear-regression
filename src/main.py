import numpy as np
import gradient_descent as gd
import matplotlib.pyplot as plt

# The program uses gradient descent algorithm to train the model
# The feature x consists of size of houses in 1000 sq feets, for example input 2 means 2000 sq. feets.
# The target value y conists of prices in 1000s of dollars
# training set
x_train = np.array([1.0, 2.0, 3.0, 4.0])
y_train = np.array([300.0, 500.0, 600.0, 700.0])

# The model is linear. f(x) = w * x + b
# Initial w and b
w_init = 0
b_init = 0

# preffered cost value is 0.01
pref_cost_value = 0.01

# Final w, b and cost value using gradient descent algorithm
# w_final, b_final, cost = gradient_descent(w_init, b_init, pref_cost_value)
w_curr, b_curr, cost_value = gd.gradient_descent(w_init, b_init, pref_cost_value, x_train, y_train, 1.0e-2)

print("optimum value of w: ", w_curr)
print("optimum value of b: ", b_curr)
print("cost value: ", cost_value)

# Plotting training set points
plt.scatter(x_train, y_train)

# plotting features(x-axix) and estimated target values(y-axis) by the trained model
x_feature = np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])
y_est_target = x_feature * w_curr + b_curr
plt.plot(x_feature, y_est_target)
plt.xlabel("Size in 1000 sq feet")
plt.ylabel("Price in 1000$ dollars")
plt.title("Size of house vs Price")
plt.show()
