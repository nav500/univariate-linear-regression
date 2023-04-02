# This function calculates gradient
def gradient(x_train, y_train, w, b):

    m = x_train.shape[0]
    # print("x_train: ", x_train)
    # print("y_train: ", y_train)
    # partial derivative of cost function with respect to w
    dj_dw = 0
    for i in range(m):
        dj_dw_i = ((w * x_train[i] + b) - y_train[i]) * x_train[i]
        dj_dw = dj_dw + dj_dw_i
        dj_dw = round(dj_dw, 2)
    dj_dw = dj_dw / m

    # partial derivative of cost function with respect to b
    dj_db = 0
    for i in range(m):
        dj_db_i = ((w * x_train[i] + b) - y_train[i]) 
        dj_db = dj_db + dj_db_i
        dj_db = round(dj_db, 2)
    dj_db = dj_db / m

    # print("dj_dw :", dj_dw)
    # print("dj_db :", dj_db)
    return dj_dw, dj_db
