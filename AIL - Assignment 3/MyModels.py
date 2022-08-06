import numpy as np

class NeuralNetwork:
    def __init__(self, num_of_input_features, num_of_outputs = 5,
                 num_of_nodes_hidden_1 = 20, num_of_nodes_hidden_2 = 10, learning_rate = 0.001):
        self.W1 = np.random.rand(num_of_nodes_hidden_1, num_of_input_features)
        self.b1 = np.random.rand(self.W1.shape[0], 1)
        self.W2 = np.random.rand(num_of_nodes_hidden_2, self.W1.shape[0])
        self.b2 = np.random.rand(self.W2.shape[0], 1)
        self.W3 = np.random.rand(num_of_outputs, self.W2.shape[0])
        self.b3 = np.random.rand(self.W3.shape[0], 1)
        self.learning_rate = learning_rate

    def sigmoid(self, Z):
        return 1 / (1 + np.exp(-Z))

    def softmax(self, Z):
        return np.exp(Z) / np.sum(np.exp(Z), axis=0)

    def forward_prop(self, X):
        Z1 = self.W1.dot(X) + self.b1
        A1 = self.sigmoid(Z1)
        Z2 = self.W2.dot(A1) + self.b2
        A2 = self.sigmoid(Z2)
        Z3 = self.W3.dot(A2) + self.b3
        A3 = self.softmax(Z3)
        return Z1, A1, Z2, A2, Z3, A3

    def derivative_sigmoid(self, Z):
        temp = self.sigmoid(Z)
        return temp * (1 - temp)

    def derivative_softmax(self, Z):
        temp = self.softmax(Z)
        return temp * (1 - temp)

    def back_prop(self, Y, A3, Z3, A2, Z2, A1, Z1, X):
        num_of_examples = X.shape[1]

        dZ3 = A3 - Y
        dW3 = dZ3.dot(A2.T) / num_of_examples
        db3 = np.sum(dZ3, axis=1, keepdims=True) / num_of_examples

        dZ2 = self.W3.T.dot(dZ3) * self.derivative_sigmoid(Z2)
        dW2 = dZ2.dot(A1.T) / num_of_examples
        db2 = np.sum(dZ2, axis=1, keepdims=True) / num_of_examples

        dZ1 = self.W2.T.dot(dZ2) * self.derivative_sigmoid(Z1)
        dW1 = dZ1.dot(X.T) / num_of_examples
        db1 = np.sum(dZ1, axis=1, keepdims=True) / num_of_examples

        return dW3, db3, dW2, db2, dW1, db1

    def update_parameters(self, dW3, db3, dW2, db2, dW1, db1):
        self.W3 -= self.learning_rate * dW3
        self.b3 -= self.learning_rate * db3
        self.W2 -= self.learning_rate * dW2
        self.b2 -= self.learning_rate * db2
        self.W1 -= self.learning_rate * dW1
        self.b1 -= self.learning_rate * db1

    def get_predictions(self, A3):
        return np.argmax(A3, 0)

    def get_accuracy(self, predictions, Y):
        print('Predictions:', predictions)
        print('Actual', np.argmax(Y, 0))
        return np.sum(predictions == np.argmax(Y, 0)) / Y.shape[1]

    def gradient_decent(self, X, Y, num_of_iterations=5000):
        num_of_examples = X.shape[1]
        for i in range(num_of_iterations):
            Z1, A1, Z2, A2, Z3, A3 = self.forward_prop(X)
            dW3, db3, dW2, db2, dW1, db1 = self.back_prop(Y, A3, Z3, A2, Z2, A1, Z1, X)
            self.update_parameters(dW3, db3, dW2, db2, dW1, db1)
            if i % 500 == 0:
                self.learning_rate /= 1.05
            if i % 500 == 0:
                loss = - np.log(A3) * Y - np.log(1 - A3) * (1 - Y)

                print('\nIteration', i)
                print('Accuracy: ', self.get_accuracy(self.get_predictions(A3), Y))
                print('Loss: ', np.sum(loss) / num_of_examples)
        return self.W1, self.b1, self.W2, self.b2, self.W3, self.b3

    def make_predictions(self, X):
        _, _, _, _, _, A3 = self.forward_prop(X)
        predictions = self.get_predictions(A3)
        return predictions
