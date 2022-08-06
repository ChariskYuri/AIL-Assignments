import MyModels
import pandas as pd
import numpy as np

# Get the dataset from 'Preprocessed data'
data_frame = pd.read_csv('Preprocessed square data.csv')
np_data = data_frame.to_numpy()
np_data = np.delete(np_data, 0, axis=1)

# Train, test split
np.random.shuffle(np_data)
train_data, test_data = np_data[0:2900].T, np_data[2900:].T
X_train, Y_train = train_data[:30], train_data[30:]
X_test, Y_test = test_data[:30], test_data[30:]

# Train and predict
my_model = MyModels.NeuralNetwork(num_of_input_features=X_train.shape[0],
                                  num_of_nodes_hidden_1=30, num_of_nodes_hidden_2=10, learning_rate=1)
my_model.gradient_decent(X_train, Y_train, num_of_iterations=30000)
print('Test accuracy:', my_model.get_accuracy(my_model.make_predictions(X_test), Y_test))
