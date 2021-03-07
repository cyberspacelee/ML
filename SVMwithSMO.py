# -*- coding:utf-8 -*-
# @Time:2021/3/6 18:48
# @Author: explorespace
# @Email: cyberspacecloner@qq.com
# @File: SVMwithSMO.py
# software: PyCharm

import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import make_blobs, make_circles, make_moons
from sklearn.preprocessing import StandardScaler


class SMO:
    def __init__(self, X, y, C, kernel, Lambdas, b, errors, eps, tol):
        self.X = X  # training data vector
        self.y = y  # class label vector
        self.C = C  # regularization parameter
        self.kernel = kernel  # kernel function
        self.Lambdas = Lambdas  # lagrange multiplier vector
        self.b = b  # scalar bias term
        self.errors = errors  # error cache
        self._obj = []  # record of objective function value
        self.m = len(self.X)  # store size of training set
        self.eps = eps
        self.tol = tol


def linear_kernel(x, y, b=1):
    return x @ y.T + b


def gaussian_kernel(x, y, sigma=1):
    if np.ndim(x) == 1 and np.ndim(y) == 1:
        # np.ndim() 获取 array 的维度数值
        result = np.exp(- (np.linalg.norm(x - y, 2)) ** 2 / (2 * sigma ** 2))
    elif (np.ndim(x) > 1 and np.ndim(y) == 1) or (np.ndim(x) == 1 and np.ndim(y) > 1):
        result = np.exp(- (np.linalg.norm(x - y, 2, axis=1) ** 2) / (2 * sigma ** 2))
        # np.linalg.norm() 范数，ord=2，l2 范数，axis=1，列向量
    elif np.ndim(x) > 1 and np.ndim(y) > 1:
        result = np.exp(- (np.linalg.norm(x[:, np.newaxis] - y[np.newaxis, :], 2, axis=-1) ** 2) / (2 * sigma ** 2))
    return result


def objective_function(Lambdas, target, kernel, X_train):
    return np.sum(Lambdas) - 0.5 * np.sum(
        (target[:, None] * target[None, :]) * kernel(X_train, X_train) * (Lambdas[:, None] * Lambdas[None, :]))
        # [:,None]，None 表示该维不进行切片，而是将该维整体作为数组元素处理，即增加一个维度，和 [:,np.newaxis] 作用一致


def decision_function(Lambdas, target, kernel, X_train, x_test, b):
    result = (Lambdas * target) @ kernel(X_train, x_test) - b
    return result


def plot_decision_boundary(model, ax, resolution=100, colors=('b', 'k', 'r'), levels=(-1, 0, 1)):
    """Plots the model's decision boundary on the input axes object.
    Range of decision boundary grid is determined by the training data.
    Returns decision boundary grid and axes object (`grid`, `ax`)."""

    # Generate coordinate grid of shape [resolution x resolution]
    # and evaluate the model over the entire space
    x_range = np.linspace(model.X[:, 0].min(), model.X[:, 0].max(), resolution)
    y_range = np.linspace(model.X[:, 1].min(), model.X[:, 1].max(), resolution)
    grid = [[decision_function(model.Lambdas, model.y, model.kernel, model.X, np.array([xr, yr]), model.b) for xr in x_range] for yr in y_range]
    grid = np.array(grid).reshape(len(x_range), len(y_range))

    # Plot decision contours using grid and
    # make a scatter plot of training data
    ax.contour(x_range, y_range, grid, levels=levels, linewidths=(1, 1, 1),
               linestyles=('--', '-', '--'), colors=colors)
    ax.scatter(model.X[:, 0], model.X[:, 1],
               c=model.y, cmap=plt.cm.viridis, lw=0, alpha=0.25)

    # Plot support vectors (non-zero Lambdas)
    # as circled points (linewidth > 0)
    mask = np.round(model.Lambdas, decimals=2) != 0.0
    ax.scatter(model.X[mask, 0], model.X[mask, 1],
               c=model.y[mask], cmap=plt.cm.viridis, lw=1, edgecolors='k')

    return grid, ax


def take_step(i_1, i_2, model):
    # Skip if chosen Lambdas are the same
    if i_1 == i_2:
        return 0, model

    Lambda_1 = model.Lambdas[i_1]
    Lambda_2 = model.Lambdas[i_2]
    y_1 = model.y[i_1]
    y_2 = model.y[i_2]
    E_1 = model.errors[i_1]
    E_2 = model.errors[i_2]
    s = y_1 * y_2

    # Compute L & H, the bounds on new possible alpha values
    if (y_1 != y_2):
        L = max(0, Lambda_2 - Lambda_1)
        H = min(model.C, model.C + Lambda_2 - Lambda_1)
    elif (y_1 == y_2):
        L = max(0, Lambda_1 + Lambda_2 - model.C)
        H = min(model.C, Lambda_1 + Lambda_2)
    if (L == H):
        return 0, model

    # Compute kernel & 2nd derivative eta
    k_11 = model.kernel(model.X[i_1], model.X[i_1])
    k_12 = model.kernel(model.X[i_1], model.X[i_2])
    k_22 = model.kernel(model.X[i_2], model.X[i_2])
    eta = 2 * k_12 - k_11 - k_22

    # Compute new alpha 2 (Lamb_2) if eta is negative
    if (eta < 0):
        Lamb_2 = Lambda_2 - y_2 * (E_1 - E_2) / eta
        # Clip Lamb_2 based on bounds L & H
        if L < Lamb_2 < H:
            Lamb_2 = Lamb_2
        elif (Lamb_2 <= L):
            Lamb_2 = L
        elif (Lamb_2 >= H):
            Lamb_2 = H

    # If eta is non-negative, move new Lamb_2 to bound with greater objective function value
    else:
        Lambdas_adj = model.Lambdas.copy()
        Lambdas_adj[i_2] = L
        # objective function output with Lamb_2 = L
        Lobj = objective_function(Lambdas_adj, model.y, model.kernel, model.X)
        Lambdas_adj[i_2] = H
        # objective function output with Lamb_2 = H
        Hobj = objective_function(Lambdas_adj, model.y, model.kernel, model.X)
        if Lobj > (Hobj + model.eps):
            Lamb_2 = L
        elif Lobj < (Hobj - model.eps):
            Lamb_2 = H
        else:
            Lamb_2 = Lambda_2

    # Push Lamb_2 to 0 or C if very close
    if Lamb_2 < 1e-8:
        Lamb_2 = 0.0
    elif Lamb_2 > (model.C - 1e-8):
        Lamb_2 = model.C

    # If examples can't be optimized within model.epsilon (model.eps), skip this pair
    if (np.abs(Lamb_2 - Lambda_2) < model.eps * (Lamb_2 + Lambda_2 + model.eps)):
        return 0, model

    # Calculate new alpha 1 (Lamb_1)
    Lamb_1 = Lambda_1 + s * (Lambda_2 - Lamb_2)

    # Update threshold b to reflect newly calculated Lambdas
    # Calculate both possible thresholds
    b_1 = E_1 + y_1 * (Lamb_1 - Lambda_1) * k_11 + y_2 * (Lamb_2 - Lambda_2) * k_12 + model.b
    b_2 = E_2 + y_1 * (Lamb_1 - Lambda_1) * k_12 + y_2 * (Lamb_2 - Lambda_2) * k_22 + model.b

    # Set new threshold based on if Lamb_1 or Lamb_2 is bound by L and/or H
    if 0 < Lamb_1 and Lamb_1 < model.C:
        b_new = b_1
    elif 0 < Lamb_2 and Lamb_2 < model.C:
        b_new = b_2
    # Average thresholds if both are bound
    else:
        b_new = (b_1 + b_2) * 0.5

    # Update model object with new Lambdas & threshold
    model.Lambdas[i_1] = Lamb_1
    model.Lambdas[i_2] = Lamb_2

    # Update error cache
    # Error cache for optimized Lambdas is set to 0 if they're unbound
    for index, alph in zip([i_1, i_2], [Lamb_1, Lamb_2]):
        if 0.0 < alph < model.C:
            model.errors[index] = 0.0

    # Set non-optimized errors based on equation 12.11 in Platt's book
    non_opt = [n for n in range(model.m) if (n != i_1 and n != i_2)]
    model.errors[non_opt] = model.errors[non_opt] + \
                            y_1 * (Lamb_1 - Lambda_1) * model.kernel(model.X[i_1], model.X[non_opt]) + \
                            y_2 * (Lamb_2 - Lambda_2) * model.kernel(model.X[i_2], model.X[non_opt]) + model.b - b_new

    # Update model threshold
    model.b = b_new

    return 1, model


def examine_example(i_2, model):
    y_2 = model.y[i_2]
    Lambda_2 = model.Lambdas[i_2]
    E_2 = model.errors[i_2]
    r_2 = E_2 * y_2

    # Proceed if error is within specified tolerance (tol)
    if ((r_2 < -model.tol and Lambda_2 < model.C) or (r_2 > model.tol and Lambda_2 > 0)):

        if len(model.Lambdas[(model.Lambdas != 0) & (model.Lambdas != model.C)]) > 1:
            # Use 2nd choice heuristic is choose max difference in error
            if model.errors[i_2] > 0:
                i_1 = np.argmin(model.errors)
            elif model.errors[i_2] <= 0:
                i_1 = np.argmax(model.errors)
            step_result, model = take_step(i_1, i_2, model)
            if step_result:
                return 1, model

        # Loop through non-zero and non-C Lambdas, starting at a random point
        for i_1 in np.roll(np.where((model.Lambdas != 0) & (model.Lambdas != model.C))[0],
                          np.random.choice(np.arange(model.m))):
            step_result, model = take_step(i_1, i_2, model)
            if step_result:
                return 1, model

        # loop through all Lambdas, starting at a random point
        for i_1 in np.roll(np.arange(model.m), np.random.choice(np.arange(model.m))):
            step_result, model = take_step(i_1, i_2, model)
            if step_result:
                return 1, model

    return 0, model


def train(model):
    num_changed = 0
    examine_all = 1

    while (num_changed > 0) or (examine_all):
        num_changed = 0
        if examine_all:
            # loop over all training examples
            for i in range(model.Lambdas.shape[0]):
                examine_result, model = examine_example(i, model)
                num_changed += examine_result
                if examine_result:
                    obj_result = objective_function(model.Lambdas, model.y, model.kernel, model.X)
                    model._obj.append(obj_result)
        else:
            # loop over examples where Lambdas are not already at their limits
            for i in np.where((model.Lambdas != 0) & (model.Lambdas != model.C))[0]:
                examine_result, model = examine_example(i, model)
                num_changed += examine_result
                if examine_result:
                    obj_result = objective_function(model.Lambdas, model.y, model.kernel, model.X)
                    model._obj.append(obj_result)
        if examine_all == 1:
            examine_all = 0
        elif num_changed == 0:
            examine_all = 1

    return model

if __name__ == '__main__':
    X_train, y = make_blobs(n_samples=1000, centers=2,
                            n_features=2, random_state=1)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train, y)
    y[y == 0] = -1
    # Set model parameters and initial values
    C = 1000.0
    m = len(X_train_scaled)
    initial_Lambdas = np.zeros(m)
    initial_b = 0.0

    # Set tolerances
    tol = 0.01  # error tolerance
    eps = 0.01  # alpha tolerance

    # Instantiate model
    # model = SMO(X_train_scaled, y, C, linear_kernel,
    #                  initial_Lambdas, initial_b, np.zeros(m), eps, tol)
    #
    # # Initialize error cache
    # initial_error = decision_function(model.Lambdas, model.y, model.kernel,
    #                                   model.X, model.X, model.b) - model.y
    # model.errors = initial_error
    # np.random.seed(0)
    # output = train(model)
    # fig, ax = plt.subplots()
    # grid, ax = plot_decision_boundary(output, ax)
    # plt.show()

    def guass_kernel():
        X_train, y = make_circles(n_samples=500, noise=0.1,
                                  factor=0.1,
                                  random_state=1)
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train, y)
        y[y == 0] = -1
        # Set model parameters and initial values
        C = 1.0
        m = len(X_train_scaled)
        initial_Lambdas = np.zeros(m)
        initial_b = 0.0

        # Instantiate model
        model = SMO(X_train_scaled, y, C, gaussian_kernel,
                         initial_Lambdas, initial_b, np.zeros(m), eps, tol)

        # Initialize error cache
        initial_error = decision_function(model.Lambdas, model.y, model.kernel,
                                          model.X, model.X, model.b) - model.y
        model.errors = initial_error
        output = train(model)
        fig, ax = plt.subplots()
        grid, ax = plot_decision_boundary(output, ax)
        plt.show()

    def moon():
        X_train, y = make_moons(n_samples=500, noise=0.1,
                                random_state=1)
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train, y)
        y[y == 0] = -1
        # Set model parameters and initial values
        C = 1.0
        m = len(X_train_scaled)
        initial_Lambdas = np.zeros(m)
        initial_b = 0.0

        # Instantiate model
        model = SMO(X_train_scaled, y, C, lambda x, y: gaussian_kernel(x, y, sigma=0.5),
                         initial_Lambdas, initial_b, np.zeros(m), eps, tol)

        # Initialize error cache
        initial_error = decision_function(model.Lambdas, model.y, model.kernel,
                                          model.X, model.X, model.b) - model.y
        model.errors = initial_error
        output = train(model)
        fig, ax = plt.subplots()
        grid, ax = plot_decision_boundary(output, ax)
        plt.show()

    # guass_kernel()
    moon()