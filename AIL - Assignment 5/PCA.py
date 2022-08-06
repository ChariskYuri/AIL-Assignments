from sklearn.decomposition import PCA
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Get the data.
df = pd.read_csv('Preprocessed SVM data.csv', index_col=False)
df.drop(columns=['Unnamed: 0'], inplace=True)

np_df = df.to_numpy()
x_train = np_df[:, :15]
print(x_train.shape)

# Apply PCA into the data.
pca = PCA(n_components=None, copy=True, whiten=False, svd_solver='full', random_state=123)
pca.fit(x_train)
explained_VR = pca.explained_variance_ratio_
print('Explained VR:', explained_VR)
df = pd.DataFrame(pca.components_)
df.to_csv('PCA Components.csv')

# Plot out the explained variance ratio.
plt.plot(range(1, 16, 1), np.cumsum(explained_VR))
plt.xlabel('Number of PCs')
plt.ylabel('Explained variance ratio')
plt.grid(axis='x')
plt.axhline(y=0.9, color='r', linestyle='-')
plt.axhline(y=0.95, color='b', linestyle='-')
plt.axhline(y=0.98, color='g', linestyle='-')
plt.axhline(y=0.99, color='m', linestyle='-')
plt.show()

# Inspect data with 2 PCs.
eig_vec = pca.components_[:, :2]
trans_x = np.dot(x_train, eig_vec)
print(trans_x.shape)
y = np_df[:, -1]

# Plot out the inspection.
color_list = ['r', 'b', 'g', 'k', 'm']
for i in range(len(color_list)):
    x1 = []
    x2 = []
    for j in range(len(y)):
        if y[j] == 5 - i:
            x1.append(trans_x[int(j)][0])
            x2.append(trans_x[int(j)][1])
    plt.scatter(x1, x2, c=color_list[i], label=i+1)

plt.legend()
plt.show()
