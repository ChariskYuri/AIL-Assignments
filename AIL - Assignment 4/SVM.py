from sklearn import svm
from sklearn import model_selection
import pandas as pd
import numpy as np

# Get the dataset from 'Preprocessed data'
data_frame = pd.read_csv('Preprocessed SVM data.csv')
np_data = data_frame.to_numpy()
np_data = np.delete(np_data, 0, axis=1)

# Train, test split
X, y = np_data[:, :15], np_data[:, 15]
X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, train_size=0.8, random_state=123)

# Train and predict (random): C = 1, max_iter = 1000
# model = svm.SVC(kernel='linear', probability=True, random_state=123)
# model.fit(X_train, y_train)
# print('Accuracy:', model.score(X_test, y_test))
# Accuracy: 0.3931034482758621 --> Squared.
# Accuracy: 0.39448275862068966 --> Normal.

# Using GridSearchCV
model = svm.SVC(kernel='linear', probability=True, random_state=123)
search_space = {
    "C": [10, 1, 0.1, 0.01, 0.001],
    "max_iter": [1000, 10000, 50000, 100000]
}
GS = model_selection.GridSearchCV(
    estimator=model,
    param_grid=search_space,
    scoring=['accuracy', 'roc_auc_ovr'],
    refit='accuracy',
    cv=5,
    verbose=4
)
GS.fit(X_train, y_train)
print('Best hyperparameters:\n', GS.best_params_)
print('Best score:\n', GS.best_score_)  # accuracy

df = pd.DataFrame(GS.cv_results_)
df.sort_values(['rank_test_accuracy', 'rank_test_roc_auc_ovr'], inplace=True)
df.to_csv('GS_Result.csv')