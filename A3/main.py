import numpy as np
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt

#load dataset
iris_data = load_iris()
X = iris_data.data
y = iris_data.target


def kmeans(X,K,max_iter=200):
    #choose random centroids
    centroids = X[np.random.choice(X.shape[0], K, replace=False)]

    for _ in range(max_iter):
        #assign data to the nearest centroid
        distances = np.sqrt(((X - centroids[:, np.newaxis]) ** 2).sum(axis=2))
        labels = np.argmin(distances, axis=0)

        #calculate the new centroid
        new_centroids = np.array([X[labels == k].mean(axis=0) for k in range(K)])

        #check if the centroids are stabilized
        if np.all(centroids == new_centroids):
            break

        #update the centroids
        centroids = new_centroids

    return centroids, labels


K = 3
centroids, cluster_lbl = kmeans(X, K)

#print the value of each centroid
print("Centroids:")
print(centroids)

id_0 = 0 #sepal length
id_1 = 1 #sepal width
labels= ['Iris-setosa', 'Iris-versicolor', 'Iris-virginica']
colors = ['blue', 'orange', 'pink']

"""
plt.scatter(X[y == 0,0], X[y ==0, 1], s=80, c='pink', label='Iris-versicolour')
plt.scatter(X[y == 1,0], X[y == 1, 1], s=80, c='green', label='Iris-virginica')
plt.scatter(X[y == 2,0], X[y == 2,1], s=80, c='blue', label='Iris-setosa')
"""

for label in np.unique(cluster_lbl):
    plt.scatter(X[cluster_lbl == label, id_0], X[cluster_lbl == label, id_1], c=colors[label], label=f'Cluster {label}')
plt.scatter(centroids[:, id_0], centroids[:, id_1], marker='o', s=200, c='red', label='Centroids')
plt.title("K-means Clustering of IRIS Dataset")
plt.legend()
plt.show()
