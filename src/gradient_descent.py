import cost_function as cf
import gradient as grdt

def gradient_descent(w_init, b_init, pref_cost_value, x_train, y_train, alpha):

    # initializing current w, b values with initial w, b values
    w_curr = w_init
    b_curr = b_init
    
    cost_value = cf.cost_function(w_init, b_init, x_train, y_train)

    iterations = 0

    while iterations < 10000:
        dj_dw, dj_db = grdt.gradient(x_train, y_train, w_curr, b_curr)

        w_curr = w_curr - alpha*dj_dw
        b_curr = b_curr - alpha*dj_db

        print("w_curr: ", w_curr)
        print("b_curr: ", b_curr)
        cost_value = cf.cost_function(w_curr, b_curr, x_train, y_train)
        # print(cost_value)
        if (dj_dw == 0.0 and dj_db == 0.0):
            break
        iterations = iterations + 1
    return w_curr, b_curr, cost_value
