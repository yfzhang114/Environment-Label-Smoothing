import numpy as np
import matplotlib.pyplot as plt

def get_train_data(data_size=100):
    data_label = np.zeros((2 * data_size, 1))
    # class 1
    x1 = np.reshape(np.random.normal(1, 0.6, 
    data_size), (data_size, 1))
    y1 = np.reshape(np.random.normal(1, 0.8, 
    data_size), (data_size, 1))
    data_train = np.concatenate((x1, y1), axis=1)
    data_label[0:data_size, :] = -1
    # class 2
    x2 = np.reshape(np.random.normal(-1, 0.3, 
    data_size), (data_size, 1))
    y2 = np.reshape(np.random.normal(-1, 0.5, 
    data_size), (data_size, 1))
    data_train = np.concatenate((data_train, 
    np.concatenate((x2, y2), axis=1)), axis=0)
    data_label[data_size:2 * data_size, :] = 1
    return data_train, data_label

def get_test_data(data_size=10):
    testdata_label = np.zeros((2 * data_size, 1))
    # class 1
    x1 = np.reshape(np.random.normal(1, 0.6, 
    data_size), (data_size, 1))
    y1 = np.reshape(np.random.normal(1, 0.8, 
    data_size), (data_size, 1))
    data_test = np.concatenate((x1, y1), axis=1)
    testdata_label[0:data_size, :] = -1
    # class 2
    x2 = np.reshape(np.random.normal(-1, 0.3, 
    data_size), (data_size, 1))
    y2 = np.reshape(np.random.normal(-1, 0.5, 
    data_size), (data_size, 1))
    data_test = np.concatenate((data_test, 
    np.concatenate((x2, y2), axis=1)), axis=0)
    testdata_label[data_size:2 * data_size, :] = 1
    return data_test, testdata_label


x_tr, y_tr = get_train_data()
x_te, y_te = get_test_data()

# 线性回归
 
def LinearRegression(data, label):
  
    values = np.ones(200) 
    X = np.matrix(np.insert(data, 2, values, axis=1))
    y = label
    X_t = np.matrix(X.T)
    x1 = np.matmul(X_t, X)
    x2 = np.linalg.inv(x1)
    x3 = np.matmul(x2, X_t)
    w_pre = np.matmul(x3, y)

ax = plt.axes(projection='3d')
