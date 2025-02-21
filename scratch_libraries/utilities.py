import matplotlib.pyplot as plt  # visualization
import seaborn as sns  # statistical visualizations and aesthetics
from collections import Counter
import numpy as np  # linear algebra


def plot_theta_gd(X, y, model, cost_history, theta_history, index_t0=0, index_t1=1): # plot column 0 (index_t0=0) vs column 1 (index_t1=1)
    """
    This function visualizes the progression of gradient descent on the cost function as the model’s parameters (theta) update 
    over iterations. It provides both a 3D surface plot and a 2D contour plot of the cost function, showing how the parameters
    converge towards optimal values.
    """
    # Setup of meshgrid of theta values
    T0, T1 = np.linspace(theta_history[:, index_t0].min(), theta_history[:, index_t0].max(), 100), \
        np.linspace(theta_history[:, index_t1].min(), theta_history[:, index_t1].max(), 100)

    # Computing the cost function for each theta combination
    idx = np.random.randint(1000, size=100)
    zs = []
    for i, t0 in enumerate(T0):
        for q, t1 in enumerate(T1):
            model.theta[0] = t0
            model.theta[1] = t1
            h = X.dot(model.theta)
            j = (h - y)
            J = j.dot(j) / 2 / (len(X))
            zs.append(J)
    # Reshaping the cost values
    T0, T1 = np.meshgrid(T0, T1)
    Z = np.array(zs).reshape(T0.shape)

    anglesx = np.array(theta_history[:, index_t0])[1:] - np.array(theta_history[:, index_t0])[:-1]
    anglesy = np.array(theta_history[:, index_t1])[1:] - np.array(theta_history[:, index_t1])[:-1]

    fig = plt.figure(figsize=(16, 8))

    # Surface plot
    ax = fig.add_subplot(1, 2, 1, projection='3d')
    ax.plot_surface(T0, T1, Z, rstride=5, cstride=5, cmap='jet', alpha=0.5)
    ax.plot(theta_history[:, index_t0], theta_history[:, index_t1], cost_history, marker='*',
            color='r', alpha=.4, label='Gradient descent')

    ax.set_xlabel('theta 0')
    ax.set_ylabel('theta 1')
    ax.set_zlabel('Cost function')
    ax.set_title('Gradient descent: Root at {}'.format(model.theta.ravel()))
    ax.view_init(45, 45)

    # Contour plot
    ax = fig.add_subplot(1, 2, 2)
    ax.contour(T0, T1, Z, 70, cmap='jet')
    ax.quiver(theta_history[:, index_t0][:-1], theta_history[:, index_t1][:-1], anglesx, anglesy,
              scale_units='xy', angles='xy', scale=1, color='r', alpha=.9)
    plt.show()


def plot_skew(df):
    for feat in df.columns.tolist():
        skew = df[feat].skew()
        sns.displot(df[feat], kde=False, label='Skew = %.3f' % (skew), bins=30)
        plt.legend(loc='best')
        plt.show()


def outlier_hunt(df):
    outlier_indices = []

    for col in df.columns.tolist():
        Q1 = np.percentile(df[col], 25)
        Q3 = np.percentile(df[col], 75)
        IRQ = Q3 - Q1
        outlier_step = 1.5 * IRQ

        outlier_list_col = df[(df[col] < Q1 - outlier_step) |
                              (df[col] > Q3 + outlier_step)].index
        outlier_indices.extend(outlier_list_col)

    outlier_indices = Counter(outlier_indices)
    multiple_outliers = list(k for k, v in outlier_indices.items() if v > 2)

    return multiple_outliers


def plot_learning_curve(train_sizes, train_scores, test_scores, title, alpha=0.1):
    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    test_mean = np.mean(test_scores, axis=1)
    test_std = np.std(test_scores, axis=1)
    plt.plot(train_sizes, train_mean, label='train score', color='blue', marker='o')
    plt.fill_between(train_sizes,train_mean + train_std,
                    train_mean - train_std, color='blue', alpha=alpha)
    plt.plot(train_sizes, test_mean, label='test score', color='red',marker='o')
    plt.fill_between(train_sizes,test_mean + test_std, test_mean - test_std , color='red', alpha=alpha)
    plt.title(title)
    plt.xlabel('Number of training points')
    plt.ylabel('Accuracy')
    plt.grid(ls='--')
    plt.legend(loc='best')
    plt.show()