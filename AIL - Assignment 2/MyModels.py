import numpy as np


class NeuralNetwork:
    def __init__(self, num_of_input_features, num_of_outputs = 5, num_of_nodes_hidden = 10, learning_rate = 0.001):
        self.W1 = np.random.rand(num_of_nodes_hidden, num_of_input_features)
        self.b1 = np.random.rand(self.W1.shape[0], 1)
        self.W2 = np.random.rand(num_of_outputs, self.W1.shape[0])
        self.b2 = np.random.rand(self.W2.shape[0], 1)
        self.learning_rate = learning_rate

    def sigmoid(self, Z):
        return 1 / (1 + np.exp(-Z))

    def forward_prop(self, X):
        Z1 = self.W1.dot(X) + self.b1
        A1 = self.sigmoid(Z1)
        Z2 = self.W2.dot(A1) + self.b2
        A2 = self.sigmoid(Z2)
        return Z1, A1, Z2, A2

    def derivative_sigmoid(self, Z):
        temp = self.sigmoid(Z)
        return temp * (1 - temp)

    def back_prop(self, Y, A2, Z2, A1, Z1, X):
        num_of_examples = X.shape[1]
        dZ2 = A2 - Y
        dW2 = dZ2.dot(A1.T) / num_of_examples
        db2 = np.sum(dZ2, axis=1, keepdims=True) / num_of_examples
        dZ1 = self.W2.T.dot(dZ2) * self.derivative_sigmoid(Z1)
        dW1 = dZ1.dot(X.T) / num_of_examples
        db1 = np.sum(dZ1, axis=1, keepdims=True) / num_of_examples
        return dW2, db2, dW1, db1

    def update_parameters(self, dW2, db2, dW1, db1):
        self.W2 -= self.learning_rate * dW2
        self.b2 -= self.learning_rate * db2
        self.W1 -= self.learning_rate * dW1
        self.b1 -= self.learning_rate * db1

    def get_predictions(self, A2):
        return np.argmax(A2, 0)

    def get_accuracy(self, predictions, Y):
        print(predictions, np.argmax(Y, 0))
        return np.sum(predictions == np.argmax(Y, 0)) / Y.shape[1]

    def gradient_decent(self, X, Y, num_of_iterations=5000):
        for i in range(num_of_iterations):
            Z1, A1, Z2, A2 = self.forward_prop(X)
            dW2, db2, dW1, db1 = self.back_prop(Y, A2, Z2, A1, Z1, X)
            self.update_parameters(dW2, db2, dW1, db1)
            if i % 500 == 0:
                self.learning_rate /= 1.1
                print('Iteration', i)
                print('Accuracy: ', self.get_accuracy(self.get_predictions(A2), Y))
        return self.W1, self.b1, self.W2, self.b2

    def make_predictions(self, X):
        _, _, _, A2 = self.forward_prop(X)
        predictions = self.get_predictions(A2)
        return predictions


class LinearRegression:
    def __init__(self, num_of_input_features, learning_rate=0.01):
        self.W = np.random.rand(1, num_of_input_features)
        self.b = np.random.rand(1, 1)
        self.learning_rate = learning_rate

    def forward_prop(self, X):
        return self.W.dot(X) + self.b

    def compute_cost(self, X, y):
        num_of_examples = X.shape[1]
        return 1 / (2 * num_of_examples) * np.sum(np.square(X - y))

    def get_accuracy(self, X, y):
        predictions = self.forward_prop(X)
        return np.sum(np.round(predictions) == y) / y.shape[1]

    def gradient_decent(self, X, y, num_of_iterations=1000):
        num_of_examples = X.shape[1]
        for i in range(num_of_iterations):
            prediction = self.forward_prop(X)
            if i % 100 == 0:
                self.learning_rate /= 1.1
                print('Iteration:', i)
                print('Cost:', self.compute_cost(prediction, y))
                print('Accuracy:', self.get_accuracy(X, y))

            dW = np.sum(X * (prediction - y), axis=1) / num_of_examples
            db = np.sum(prediction - y) / num_of_examples
            self.W -= dW * self.learning_rate
            self.b -= db * self.learning_rate
        return self.W, self.b
