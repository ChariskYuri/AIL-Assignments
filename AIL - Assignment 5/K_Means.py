import sklearn.cluster as cls
from sklearn import metrics
import pandas as pd
import matplotlib.pyplot as plt

# Get the data.
df = pd.read_csv('Preprocessed SVM data.csv', index_col=False)
df.drop(columns=['Unnamed: 0'], inplace=True)

np_df = df.to_numpy()
x_train = np_df[:, :15]
print(x_train.shape)

nums_of_clusters = []
distances = []
sil_scores = []

# Train models with 2 to 49 clusters.
for k in range(2, 50):
    print('K is now:', k)
    model = cls.KMeans(n_clusters=k, init='random', n_init=10,
                       max_iter=300, tol=0.0001, verbose=0, random_state=123,
                       copy_x=True, algorithm='lloyd')

    model.fit(x_train)
    nums_of_clusters.append(k)
    distances.append(model.inertia_)
    sil_score = metrics.silhouette_score(X=x_train, labels=model.labels_,
                                         metric='euclidean', sample_size=None, random_state=123)
    sil_scores.append(sil_score)

# Plot out Elbow.png.
plt.plot(nums_of_clusters, distances)
plt.xlabel('Number of clusters')
plt.ylabel('Sum of distances')
plt.show()

# Plot out Sil_score.png.
plt.plot(nums_of_clusters, sil_scores)
plt.xlabel('Number of clusters')
plt.ylabel('Silhouette score')
plt.show()
