import numpy as np
from numpy.linalg import norm
from utils.metric import accuracy
from sklearn.model_selection import train_test_split


def split_train_val_sets(X, y, val_size, random_state=10):
    """
    Split data into training and validation sets
    """
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=val_size,
                                                      random_state=random_state, shuffle=False)

    return [X_train, X_val, y_train, y_val]


def initialize_weights(K, n, random_state=None):
    """
    Initialize n weights_list with random values in range [-1,1]
    :param K: number of classes
    :param n: number of features
    :param random_state: random state
    :return: weights
    """
    if random_state is not None:
        np.random.seed(random_state)

    weights = np.zeros((K, n))
    for k in range(K):
        weights[k] = (np.random.rand(n) - .5) * 2

    return weights


def predict(x, weights):

    b = np.amax(x @ weights.T, axis=1)  # for numerical stability of the softmax function

    a = x @ weights.T
    numerator = np.exp((a.T - b).T)
    denominator = 1 / np.sum(np.exp((a.T - b).T), axis=1)
    h3 = (numerator.T * denominator).T

    return h3


def predictions(x, weights):

    h = predict(x, weights)

    return h.argmax(axis=1)


def cross_entropy(weights, x, t, alpha):
    """
    Return the value of cross entropy error function in case of multiclass classification
    """

    N = x.shape[0]  # number of inputs

    h = predict(x, weights)  # predictions

    # compute cost function
    epsilon = 1e-14
    loss = np.sum(np.multiply(t, np.log(h + epsilon)))

    # regularization
    K = t.shape[1]
    reg = 0
    for k in range(K):
        reg += weights[k] @ weights[k]

    return (-1 / N) * loss + .5 * alpha * reg


def cross_entropy_gradient(weights, x, t, alpha):
    """
    Return the gradient of cross entropy error function
    """

    N = x.shape[0]  # number of inputs

    h = predict(x, weights)  # predictions

    # compute gradient
    w_grad = (h - t).T @ x

    # regularization
    r_grad = alpha * weights

    # return grad
    return (1 / N) * w_grad + r_grad


def step_gold_search(fun, w, direction, s, e, alpha, x, t, tol, fixed_range):
    """
    Find suitable learning_rate using golden section search method
    """

    # initialization
    gg = (np.sqrt(5) - 1) / 2
    a1 = gg * s + (1 - gg) * e
    a2 = gg * e + (1 - gg) * s

    weights = np.copy(w)
    fe = fun(weights + e * direction, x, t, alpha)
    weights = np.copy(w)
    f1 = fun(weights + a1 * direction, x, t, alpha)
    weights = np.copy(w)
    f2 = fun(weights + a2 * direction, x, t, alpha)
    i = 0
    while True and not fixed_range:
        if fe >= f2:
            break

        if fe < f2:
            a1 = a2
            f1 = f2
            a2 = e
            f2 = fe
            e = (e - (1 - gg) * s) / gg
            weights = np.copy(w)
            fe = fun(weights + e * direction, x, t, alpha)
        i += 1

    i = 0
    while (abs(a1 - a2) / a1) > tol:
        if f1 >= f2:
            s = a1
            a1 = a2
            f1 = f2
            a2 = gg * e + (1 - gg) * s
            weights = np.copy(w)
            f2 = fun(weights + a2 * direction, x, t, alpha)
        else:
            e = a2
            a2 = a1
            f2 = f1
            a1 = gg * s + (1 - gg) * e
            weights = np.copy(w)
            f1 = fun(weights + a1 * direction, x, t, alpha)
        i += 1

    if f2 < f1:
        x = a2
    else:
        x = a1

    return x


def gradient_descent(weights, x, t, iter_max, alpha):
    """
    Update weights_list using gradient descent
    """

    tol_grad = 1e-6
    tol_error = 1e-10
    e_old = 1e20
    iteration = 0
    while True:
        grad = cross_entropy_gradient(weights, x, t, alpha)  # gradient

        if norm(grad) <= tol_grad or iteration >= iter_max:  # stopping condition
            break

        direction = -grad  # search direction

        # find step size
        s = 0
        e = 1e-3
        tol = 1e-4
        fixed_range = 0
        step = step_gold_search(cross_entropy, weights, direction, s, e, alpha, x, t, tol, fixed_range)

        # update weights_list
        weights += step * direction

        e = cross_entropy(weights, x, t, alpha)
        if abs(e - e_old) <= tol_error:
            break
        e_old = np.copy(e)

        iteration += 1


def cross_validation(x_train, t_train, x_val, t_val, x_test, t_test, iter_max, random_state, alpha_vals):

    K = t_train.shape[1]  # number of classes
    n = x_train.shape[1]  # number of features

    # initialize arrays
    train_error = []
    val_error = []

    i = 0
    # try different values of alpha
    for alpha in alpha_vals:
        print('--> alpha = %f' % alpha)

        # initialize weights_list
        weights = initialize_weights(K, n, random_state)

        # gradient descent
        gradient_descent(weights=weights, x=x_train, t=t_train, iter_max=iter_max, alpha=alpha)

        train_error.append(cross_entropy(weights, x_train, t_train, alpha))
        val_error.append(cross_entropy(weights, x_val, t_val, 0))

        if i == 0:  # initial optimal values
            alpha_optimal = alpha
            weights_optimal = weights

        if len(train_error) != 1:  # skip first alpha
            if val_error[i - 1] < val_error[i]:
                break
            else:
                alpha_optimal = alpha
                weights_optimal = weights
        i += 1

    print('\n --> Training set size: ' + str(x_train.shape))
    print('=== After applying logistic regression: ===')
    print('Optimal alpha = ' + str(alpha_optimal))
    print('Train: Error = ' + str(cross_entropy(weights_optimal, x_train, t_train, alpha_optimal)))

    print('Val: Error = ' + str(cross_entropy(weights_optimal, x_val, t_val, 0)))
    print('Val: Accuracy = ' + str(accuracy(t_val.argmax(axis=1), predictions(x_val, weights_optimal))))

    print('Test: Error = ' + str(cross_entropy(weights_optimal, x_test, t_test, 0)))
    print('Test: Accuracy = ' + str(accuracy(t_test.argmax(axis=1), predictions(x_test, weights_optimal))))

    return [weights_optimal, train_error, val_error]


def fit(x, t, x_test, t_test, iter_max, n_images_list, random_state, alpha_vals):
    """ Training logistic regression"""

    K = t.shape[1]  # number of classes

    print('--> iter_max = %d' % iter_max)
    print('--> random_state = %d' % random_state)
    print('--> alpha in ' + str(alpha_vals))
    print('--> K = %d' % K)
    print('--> number of images in the training set: ' + str([5 * (2 ** i) for i in n_images_list]) + '\n')

    # split data into training and validation sets
    val_size = 0.20  # validation set percentage
    x_train_all, x_val, t_train_all, t_val = split_train_val_sets(x, t, val_size)

    # balance data set
    indices0 = []
    indices1 = []
    indices2 = []
    indices3 = []
    indices4 = []
    indices5 = []
    indices6 = []
    indices7 = []
    indices8 = []
    indices9 = []

    for i in range(x_train_all.shape[0]):
        if np.where(t_train_all[i] == 1)[0] == 0:
            indices0.append(i)
        if np.where(t_train_all[i] == 1)[0] == 1:
            indices1.append(i)
        if np.where(t_train_all[i] == 1)[0] == 2:
            indices2.append(i)
        if np.where(t_train_all[i] == 1)[0] == 3:
            indices3.append(i)
        if np.where(t_train_all[i] == 1)[0] == 4:
            indices4.append(i)
        if np.where(t_train_all[i] == 1)[0] == 5:
            indices5.append(i)
        if np.where(t_train_all[i] == 1)[0] == 6:
            indices6.append(i)
        if np.where(t_train_all[i] == 1)[0] == 7:
            indices7.append(i)
        if np.where(t_train_all[i] == 1)[0] == 8:
            indices8.append(i)
        if np.where(t_train_all[i] == 1)[0] == 9:
            indices9.append(i)

    x_train = np.zeros((5 * (2 ** n_images_list[0]), x_train_all.shape[1]))
    t_train = np.zeros((5 * (2 ** n_images_list[0]), K))
    sizes = np.linspace(0, 5 * (2 ** n_images_list[0]), K + 1)

    n = int(5 * (2 ** n_images_list[0]) / 10)
    for i in range(K):
        # print(i, ' ', int(sizes[i]), ' ', int(sizes[i + 1]))
        if i == 0:
            x_train[int(sizes[i]):int(sizes[i + 1])] = x_train_all[indices0[0:n]]
            t_train[int(sizes[i]):int(sizes[i + 1])] = t_train_all[indices0[0:n]]
        if i == 1:
            x_train[int(sizes[i]):int(sizes[i + 1])] = x_train_all[indices1[0:n]]
            t_train[int(sizes[i]):int(sizes[i + 1])] = t_train_all[indices1[0:n]]
        if i == 2:
            x_train[int(sizes[i]):int(sizes[i + 1])] = x_train_all[indices2[0:n]]
            t_train[int(sizes[i]):int(sizes[i + 1])] = t_train_all[indices2[0:n]]
        if i == 3:
            x_train[int(sizes[i]):int(sizes[i + 1])] = x_train_all[indices3[0:n]]
            t_train[int(sizes[i]):int(sizes[i + 1])] = t_train_all[indices3[0:n]]
        if i == 4:
            x_train[int(sizes[i]):int(sizes[i + 1])] = x_train_all[indices4[0:n]]
            t_train[int(sizes[i]):int(sizes[i + 1])] = t_train_all[indices4[0:n]]
        if i == 5:
            x_train[int(sizes[i]):int(sizes[i + 1])] = x_train_all[indices5[0:n]]
            t_train[int(sizes[i]):int(sizes[i + 1])] = t_train_all[indices5[0:n]]
        if i == 6:
            x_train[int(sizes[i]):int(sizes[i + 1])] = x_train_all[indices6[0:n]]
            t_train[int(sizes[i]):int(sizes[i + 1])] = t_train_all[indices6[0:n]]
        if i == 7:
            x_train[int(sizes[i]):int(sizes[i + 1])] = x_train_all[indices7[0:n]]
            t_train[int(sizes[i]):int(sizes[i + 1])] = t_train_all[indices7[0:n]]
        if i == 8:
            x_train[int(sizes[i]):int(sizes[i + 1])] = x_train_all[indices8[0:n]]
            t_train[int(sizes[i]):int(sizes[i + 1])] = t_train_all[indices8[0:n]]
        if i == 9:
            x_train[int(sizes[i]):int(sizes[i + 1])] = x_train_all[indices9[0:n]]
            t_train[int(sizes[i]):int(sizes[i + 1])] = t_train_all[indices9[0:n]]

    cross_validation(x_train[:5 * (2 ** n_images_list[0])], t_train[:5 * (2 ** n_images_list[0])],
                     x_val, t_val, x_test, t_test, iter_max, random_state, alpha_vals)



