import MyModels
import pandas as pd
import numpy as np

# Get the dataset from 'Preprocessed data'
data_frame = pd.read_csv('Preprocessed data LR.csv')
np_data = data_frame.to_numpy()
np_data = np.delete(np_data, 0, axis=1)

# Train, test split
np.random.shuffle(np_data)
train_data, test_data = np_data[0:2600].T, np_data[2600:].T
X_train, Y_train = train_data[0:15], train_data[15:]
X_test, Y_test = test_data[0:15], test_data[15:]

# Train and predict
my_model = MyModels.LinearRegression(num_of_input_features=15, learning_rate=0.01)
my_model.gradient_decent(X_train, Y_train, num_of_iterations=5000)
print('Predictions: ', np.round(my_model.forward_prop(X_test)), Y_test)
