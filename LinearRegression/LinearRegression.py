# -*- coding: utf-8 -*-
"""
Created on Thu May 04 21:44:02 2017

@author: SUNFC
"""
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
class Linear_regression(object):
    x = []
    def solve_weight_as_matrix(self, X, y):
        x_new = np.mat(X)
        y_new = np.mat(y)
        self.w = ((x_new.T*x_new).I)*(x_new.T)*y_new
    def solve_weight_as_SGD(self, X, y):
        self.epoch = 100
        self.alpha = 0.0005
        x_new = np.mat(X)
        y_new = np.mat(y)
        print x_new.shape
        print y_new.shape
        self.w = np.random.random(x_new.shape[1])
        for i in range(self.epoch):
            self.w = self.w + self.alpha * (((y_new - (x_new*self.w)).T)*x_new)
            print U'迭代: w=%.5f' % self.w 

    def fit_mat(self, X, y):
        self.solve_weight_as_matrix(X, y)
        
    def fit(self, X, y):
        self.solve_weight_as_SGD(X, y)
        
    def predict(self, X):
        return X*self.w

if __name__=='__main__':
    X_train = [[6], [8], [10], [14], [18]]
    y_train = [[7], [9], [13], [17.5], [18]]
    X_test = [[6], [8], [11], [16]]
    y_test = [[8], [12], [15], [18]]
    
    model = Linear_regression()
    model.fit(X_train, y_train)
    plt.plot(X_train, y_train, 'bo')
    xx = np.linspace(0, 26, 100)
    print model.w
    plt.plot(X_test, model.predict(X_test), 'r-')
    plt.title(u'二维矩阵求解')
    plt.show()
    
    # -------------------------------------------------------------------------
    xx, yy = np.meshgrid(np.linspace(0,10,10), np.linspace(0,10,10))
    zz = 1.0 * xx + 3.5 * yy + np.random.randint(0,10,(10,10)) 
    # 构建成特征、值的形式
    X_train, y_train = np.column_stack((xx.flatten(),yy.flatten())), zz.flatten()
    # 转化为list
    X_train = X_train.tolist()
    y_train = y_train.reshape(-1,1)
    y_train = y_train.tolist()
    
    model.fit_mat(X_train, y_train)
    fig = plt.figure()
    ax = fig.gca(projection='3d')  
    # 1.画出真实的点
    ax.scatter(xx, yy, zz)
    # 2.画出拟合的平面
    ax.plot_wireframe(xx, yy, model.predict(X_train).reshape(10,10))
    ax.plot_surface(xx, yy, model.predict(X_train).reshape(10,10), alpha=0.3)
    plt.title(u'三维矩阵求解')
    plt.show()
    # -------------------------------------------------------------------------
    