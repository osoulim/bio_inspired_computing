from sklearn import datasets
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

batch_index = 0
reg_lambda = 0.0001
epsilon = 0.0003
num_examples = 8000
batch_size = 40
batch_num = num_examples / batch_size
n2, n3, n4 = 8, 12, 2
X, y = datasets.make_moons(num_examples, noise=0.1)

n1 = X.shape[1]
W1 = np.random.rand(n1, n2) / np.sqrt(n1)
W2 = np.random.rand(n2, n3) / np.sqrt(n2)
W3 = np.random.rand(n3, n4) / np.sqrt(n3)
b1 = np.random.rand(1, n2)
b2 = np.random.rand(1, n3)
b3 = np.random.rand(1, n4)


def calculate_loss():
    z1 = X.dot(W1) + b1
    a1 = np.tanh(z1)
    z2 = a1.dot(W2) + b2
    a2 = np.tanh(z2)
    z3 = a2.dot(W3) + b3
    exp_scores = np.exp(z3)
    probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
    corect_logprobs = -np.log(probs[range(num_examples), y])
    data_loss = np.sum(corect_logprobs)
    data_loss += reg_lambda/2 * (np.sum(np.square(W1)) + np.sum(np.square(W2)) + np.sum(np.square(W3)))
    return 1./num_examples * data_loss


def predict(x):
    z1 = x.dot(W1) + b1
    a1 = np.tanh(z1)
    z2 = a1.dot(W2) + b2
    a2 = np.tanh(z2)
    z3 = a2.dot(W3) + b3
    exp_scores = np.exp(z3)
    probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
    return np.argmax(probs, axis=1)

iter_num = 20000
np.random.seed(0)

for i in tqdm(range(iter_num)):
    x = X[batch_index : batch_index + batch_size]
    Y = y[batch_index : batch_index + batch_size]
    batch_index += 40
    if batch_index >= num_examples - 1:
        batch_index = 0
    z1 = x.dot(W1) + b1
    a1 = np.tanh(z1)
    z2 = a1.dot(W2) + b2
    a2 = np.tanh(z2)
    z3 = a2.dot(W3) + b3
    exp_scores = np.exp(z3)
    probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
    # Backpropagation
    delta3 = probs
    delta3[range(40), Y] -= 1
    dW3 = (a2.T).dot(delta3)
    db3 = np.sum(delta3, axis=0, keepdims=True)
    delta2 = delta3.dot(W3.T) * (1 - np.power(a2, 2))
    dW2 = (a1.T).dot(delta2)
    db2 = np.sum(delta2, axis=0, keepdims=True)
    delta1 = delta2.dot(W2.T) * (1 - np.power(a1, 2))
    dW1 = np.dot(x.T, delta1)
    db1 = np.sum(delta1, axis=0)

    dW3 += reg_lambda * W3
    dW2 += reg_lambda * W2
    dW1 += reg_lambda * W1
    W1 += -epsilon * dW1
    b1 += -epsilon * db1
    W2 += -epsilon * dW2
    b2 += -epsilon * db2
    W3 += -epsilon * dW3
    b3 += -epsilon * db3

x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
h = 0.01
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

Z = predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral)
plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Spectral)
plt.show()